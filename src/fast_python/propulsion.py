# src/fast_python/propulsion.py

"""Propulsion architecture utilities ported from FAST PropulsionPkg."""

import inspect
from copy import deepcopy

import numpy as np

from fast_python.atmosphere import standard_atmosphere
from fast_python.battery import discharging
from fast_python.engine import (
    simple_off_design,
    turbofan_nonlinear_sizing,
    turboprop_nonlinear_sizing,
)
from fast_python.markers import MatlabExpression, parse_constant_matrix_expression
from fast_python.regression import nlgpr, search_db


class PropulsionError(ValueError):
    """Report invalid propulsion architecture data."""


def eval_split(split_fun, split_val=None):
    """Evaluate an operational split function.

    Inputs:
        split_fun: Callable split matrix factory or an already-materialized
            split matrix.
        split_val: Scalar or sequence of split values.

    Outputs:
        Numeric split matrix as a NumPy array.

    Assumptions:
        This mirrors MATLAB EvalSplit by passing as many scalar split values as
        the callable declares. Non-callable matrices are returned as-is.
    """

    if isinstance(split_fun, MatlabExpression):
        return as_array(parse_constant_matrix_expression(split_fun))

    if not callable(split_fun):
        return as_array(split_fun)

    values = normalize_split_values(split_val)
    narg = callable_arg_count(split_fun)

    if narg > 17:
        raise PropulsionError(
            "Split evaluation is unavailable for more than 17 split arguments."
        )

    return as_array(split_fun(*values[:narg]))


def power_flow(power, arch, split, eta, direct, tol=1.0e-6):
    """Propagate power through a propulsion architecture matrix.

    Inputs:
        power: Initial component power vector.
        arch: Architecture adjacency matrix.
        split: Upstream or downstream operational split matrix.
        eta: Upstream or downstream efficiency matrix.
        direct: +1 for upstream propagation, -1 for downstream propagation.
        tol: Convergence tolerance.

    Outputs:
        Updated component power vector.

    Assumptions:
        The iterative update follows FAST PropulsionPkg.PowerFlow exactly:
        only entries whose propagated value is above tolerance are overwritten.
    """

    pwr = as_vector(power).astype(float)
    arch = as_array(arch)
    split = as_array(split)
    eta = as_array(eta)
    previous = np.zeros(len(pwr))

    if direct == 1:
        matrix = (split * arch * eta).T
    elif direct == -1:
        matrix = (split * arch / eta).T
    else:
        raise PropulsionError("direct must be +1 for upstream or -1 for downstream.")

    iteration = 0

    while np.linalg.norm(previous - pwr) > tol and iteration < len(pwr):
        propagated = matrix @ pwr
        previous = pwr.copy()
        update = np.abs(propagated) > tol
        pwr[update] = propagated[update]
        iteration += 1

    return pwr


def power_supplement_check(preq, arch, split, eta, trn_type, eta_fan):
    """Return supplemental power for series and parallel transmitter links.

    Inputs:
        preq: Required component power matrix in W, rows by mission points.
        arch: Propulsion architecture adjacency matrix.
        split: Operational split matrix for the active propagation direction.
        eta: Efficiency matrix paired with split.
        trn_type: Transmitter type vector, where gas turbines are 1 and thrust
            sinks are 2 in FAST's convention.
        eta_fan: Fan or propeller efficiency used for parallel assistance.

    Outputs:
        Supplemental power matrix in W with the same row count and transmitter
        column count as preq.
    """

    preq = as_2d(preq)
    arch = as_array(arch)
    split = as_array(split)
    eta = as_array(eta)
    trn_type = as_vector(trn_type)
    npnt, ntrn = preq.shape
    psupp = np.zeros((npnt, ntrn))

    gtes = np.where(trn_type == 1)[0]

    for gte in gtes:
        cur_row = arch[gte, :]
        conns = (cur_row == 1) & (trn_type != 2)

        if not np.any(conns):
            continue

        factors = split[conns, gte] / eta[conns, gte]
        psupp[:, gte] -= preq[:, conns] @ factors

    parallel = np.where(np.sum(arch, axis=0) > 1)[0]

    for icomp in parallel:
        connected = arch[:, icomp] > 0
        driving = np.where(connected & (trn_type == 1))[0]

        if len(driving) != 1:
            continue

        helping = np.where(connected & (trn_type == 0))[0]

        if len(helping) > 0:
            psupp[:, driving[0]] += np.sum(preq[:, helping], axis=1) * eta_fan

    return psupp


def prop_arch_connections(aircraft):
    """Identify parallel engine/electric-motor connections in an aircraft dict.

    Inputs:
        aircraft: Aircraft dictionary with Specs.Propulsion.PropArch matrices.

    Outputs:
        A deep-copied aircraft dictionary with PropArch.ParConns populated as
        lists of electric transmitter indices assisting each gas turbine.

    Assumptions:
        Component indices follow FAST's concatenated source/transmitter/sink
        ordering. The stored helper indices remain MATLAB-style component
        positions because downstream FAST port code expects that convention.
    """

    aircraft = deepcopy(aircraft)
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    arch = as_array(prop_arch["Arch"])
    src_type = as_vector(prop_arch["SrcType"])
    trn_type = as_vector(prop_arch["TrnType"])
    nsrc = len(src_type)
    ntrn = len(trn_type)
    prop_arch["ParConns"] = [[] for _ in range(ntrn)]
    transmitter_indices = np.arange(nsrc, nsrc + ntrn)
    sub_arch = arch[np.ix_(transmitter_indices, transmitter_indices)]
    any_parallel = np.where(np.sum(sub_arch, axis=0) > 1)[0] + nsrc

    for icomp in any_parallel:
        incoming = arch[transmitter_indices, icomp] > 0
        driving = np.where(incoming & (trn_type == 1))[0]
        helping = np.where(incoming & (trn_type == 0))[0] + nsrc

        if len(driving) > 0 and len(helping) > 0:
            prop_arch["ParConns"][driving[0]].extend(helping.tolist())

    return aircraft


def create_prop_arch(aircraft):
    """Create built-in propulsion architecture matrices for an aircraft dict.

    Inputs:
        aircraft: Dictionary with Specs.TLAR, Specs.Power, and
            Specs.Propulsion fields.

    Outputs:
        Updated aircraft dictionary with PropArch matrices, source types,
        transmitter types, and split argument counts populated.

    Assumptions:
        Conventional ("C"), electric ("E"), parallel hybrid ("PHE"), series
        hybrid ("SHE"), turboelectric ("TE"), and partially turboelectric
        ("PE") architectures are ported. Custom ("O") architectures are
        validated and returned with their prescribed matrices.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    prop = specs["Propulsion"]
    prop_arch = prop["PropArch"]
    arch_name = prop_arch.get("Type", "O")

    if isinstance(arch_name, str):
        arch_name = arch_name.upper()

    if arch_name == "O":
        validate_custom_prop_arch(prop_arch)
        set_split_arg_counts(aircraft)
        return aircraft

    if arch_name not in ("C", "E", "PHE", "SHE", "TE", "PE"):
        raise PropulsionError(
            f"Built-in propulsion architecture {arch_name!r} is not ported yet."
        )

    num_eng = int(prop["NumEngines"])
    aircraft_class = specs["TLAR"]["Class"]
    eta_ts = get_thrust_sink_efficiency(specs, aircraft_class)
    eta_em = specs["Power"]["Eta"]["EM"]
    eta = specs["Power"]["Eta"]
    architecture_builders = {
        "PHE": lambda: parallel_hybrid_architecture(num_eng, eta_em, eta_ts),
        "SHE": lambda: series_hybrid_architecture(
            num_eng, eta_em, eta["EG"], eta_ts
        ),
        "TE": lambda: turboelectric_architecture(num_eng, eta_em, eta["EG"], eta_ts),
        "PE": lambda: partial_turboelectric_architecture(
            num_eng, eta_em, eta["EG"], eta_ts
        ),
    }

    if arch_name in architecture_builders:
        arch, oper_ups, oper_dwn, eta_ups, eta_dwn, src_type, trn_type = architecture_builders[arch_name]()
        prop_arch["Arch"] = arch.tolist()
        prop_arch["OperUps"] = make_split_callable(oper_ups)
        prop_arch["OperDwn"] = make_split_callable(oper_dwn)
        prop_arch["EtaUps"] = eta_ups.tolist()
        prop_arch["EtaDwn"] = eta_dwn.tolist()
        prop_arch["SrcType"] = src_type
        prop_arch["TrnType"] = trn_type
        set_split_arg_counts(aircraft)
        return aircraft

    arch, oper_ups, oper_dwn = simple_source_transmitter_architecture(num_eng)
    eta_ups = np.ones_like(arch)
    eta_dwn = np.ones_like(arch)

    engine_idx = np.arange(1, 1 + num_eng)
    thrust_sink_idx = np.arange(1 + num_eng, 1 + 2 * num_eng)

    if arch_name == "C":
        for engine, sink in zip(engine_idx, thrust_sink_idx):
            eta_ups[engine, sink] = eta_ts
            eta_dwn[sink, engine] = eta_ts

        src_type = [1]
        trn_type = [1] * num_eng + [2] * num_eng

    else:
        for motor in engine_idx:
            eta_ups[0, motor] = eta_em
            eta_dwn[motor, 0] = eta_em

        for motor, sink in zip(engine_idx, thrust_sink_idx):
            eta_ups[motor, sink] = eta_ts
            eta_dwn[sink, motor] = eta_ts

        src_type = [0]
        trn_type = [0] * num_eng + [2] * num_eng

    prop_arch["Arch"] = arch.tolist()
    prop_arch["OperUps"] = lambda: oper_ups.tolist()
    prop_arch["OperDwn"] = lambda: oper_dwn.tolist()
    prop_arch["EtaUps"] = eta_ups.tolist()
    prop_arch["EtaDwn"] = eta_dwn.tolist()
    prop_arch["SrcType"] = src_type
    prop_arch["TrnType"] = trn_type
    set_split_arg_counts(aircraft)
    return aircraft


def simple_source_transmitter_architecture(num_eng):
    """Return architecture and split matrices for C/E-style layouts."""

    ncomp = 2 * num_eng + 2
    arch = np.zeros((ncomp, ncomp))
    oper_ups = np.zeros((ncomp, ncomp))
    oper_dwn = np.zeros((ncomp, ncomp))
    source = 0
    engine_idx = np.arange(1, 1 + num_eng)
    sink_idx = np.arange(1 + num_eng, 1 + 2 * num_eng)
    final_sink = ncomp - 1
    arch[source, engine_idx] = 1
    oper_ups[source, engine_idx] = 1 / num_eng
    oper_dwn[engine_idx, source] = 1

    for engine, sink in zip(engine_idx, sink_idx):
        arch[engine, sink] = 1
        oper_ups[engine, sink] = 1
        oper_dwn[sink, engine] = 1
        arch[sink, final_sink] = 1
        oper_ups[sink, final_sink] = 1
        oper_dwn[final_sink, sink] = 1 / num_eng

    return arch, oper_ups, oper_dwn


def parallel_hybrid_architecture(num_eng, eta_em, eta_ts):
    """Return FAST's built-in parallel hybrid architecture matrices."""

    ncomp = 3 * num_eng + 3
    arch = np.zeros((ncomp, ncomp))
    fuel = 0
    battery = 1
    engine_idx = np.arange(2, 2 + num_eng)
    motor_idx = np.arange(2 + num_eng, 2 + 2 * num_eng)
    sink_idx = np.arange(2 + 2 * num_eng, 2 + 3 * num_eng)
    final_sink = ncomp - 1
    arch[fuel, engine_idx] = 1
    arch[battery, motor_idx] = 1

    for engine, motor, sink in zip(engine_idx, motor_idx, sink_idx):
        arch[engine, sink] = 1
        arch[motor, sink] = 1
        arch[sink, final_sink] = 1

    eta_ups = np.ones_like(arch)
    eta_dwn = np.ones_like(arch)
    eta_ups[battery, motor_idx] = eta_em
    eta_dwn[motor_idx, battery] = eta_em

    for engine, motor, sink in zip(engine_idx, motor_idx, sink_idx):
        eta_ups[engine, sink] = eta_ts
        eta_ups[motor, sink] = eta_ts
        eta_dwn[sink, engine] = eta_ts
        eta_dwn[sink, motor] = eta_ts

    src_type = [1, 0]
    trn_type = [1] * num_eng + [0] * num_eng + [2] * num_eng

    def oper_ups(lam):
        """Return upstream operational splits for a PHE architecture."""

        matrix = np.zeros_like(arch)
        matrix[fuel, engine_idx] = 1 / num_eng
        matrix[battery, motor_idx] = 1 / num_eng

        for engine, motor, sink in zip(engine_idx, motor_idx, sink_idx):
            matrix[engine, sink] = 1
            matrix[motor, sink] = lam
            matrix[sink, final_sink] = 1

        return matrix

    def oper_dwn(lam):
        """Return downstream operational splits for a PHE architecture."""

        matrix = np.zeros_like(arch)
        matrix[engine_idx, fuel] = 1
        matrix[motor_idx, battery] = 1

        for engine, motor, sink in zip(engine_idx, motor_idx, sink_idx):
            matrix[sink, engine] = 1 - lam
            matrix[sink, motor] = lam

        matrix[final_sink, sink_idx] = 1 / num_eng
        return matrix

    return arch, oper_ups, oper_dwn, eta_ups, eta_dwn, src_type, trn_type


def series_hybrid_architecture(num_eng, eta_em, eta_eg, eta_ts):
    """Return FAST's built-in series hybrid architecture matrices."""

    ncomp = 5 * num_eng + 3
    arch = np.zeros((ncomp, ncomp))
    fuel = 0
    battery = 1
    engine_idx = np.arange(2, 2 + num_eng)
    generator_idx = np.arange(2 + num_eng, 2 + 2 * num_eng)
    cable_idx = np.arange(2 + 2 * num_eng, 2 + 3 * num_eng)
    motor_idx = np.arange(2 + 3 * num_eng, 2 + 4 * num_eng)
    sink_idx = np.arange(2 + 4 * num_eng, 2 + 5 * num_eng)
    final_sink = ncomp - 1
    arch[fuel, engine_idx] = 1
    arch[battery, cable_idx] = 1

    for engine, generator, cable, motor, sink in zip(
        engine_idx,
        generator_idx,
        cable_idx,
        motor_idx,
        sink_idx,
    ):
        arch[engine, generator] = 1
        arch[generator, motor] = 1
        arch[cable, motor] = 1
        arch[motor, sink] = 1
        arch[sink, final_sink] = 1

    eta_ups = np.ones_like(arch)
    eta_dwn = np.ones_like(arch)

    for engine, generator, cable, motor, sink in zip(
        engine_idx,
        generator_idx,
        cable_idx,
        motor_idx,
        sink_idx,
    ):
        eta_ups[engine, generator] = eta_eg
        eta_ups[generator, motor] = eta_em
        eta_ups[cable, motor] = eta_em
        eta_ups[motor, sink] = eta_ts
        eta_dwn[generator, engine] = eta_eg
        eta_dwn[motor, generator] = eta_em
        eta_dwn[motor, cable] = eta_em
        eta_dwn[sink, motor] = eta_ts

    src_type = [1, 0]
    trn_type = (
        [1] * num_eng
        + [3] * num_eng
        + [4] * num_eng
        + [0] * num_eng
        + [2] * num_eng
    )

    def oper_ups(lam):
        """Return upstream operational splits for an SHE architecture."""

        matrix = np.zeros_like(arch)
        matrix[fuel, engine_idx] = 1 / num_eng
        matrix[battery, cable_idx] = 1 / num_eng

        for engine, generator, cable, motor, sink in zip(
            engine_idx,
            generator_idx,
            cable_idx,
            motor_idx,
            sink_idx,
        ):
            matrix[engine, generator] = 1
            matrix[generator, motor] = 1
            matrix[cable, motor] = lam
            matrix[motor, sink] = 1
            matrix[sink, final_sink] = 1

        return matrix

    def oper_dwn(lam):
        """Return downstream operational splits for an SHE architecture."""

        matrix = np.zeros_like(arch)
        matrix[engine_idx, fuel] = 1
        matrix[cable_idx, battery] = 1

        for engine, generator, cable, motor, sink in zip(
            engine_idx,
            generator_idx,
            cable_idx,
            motor_idx,
            sink_idx,
        ):
            matrix[generator, engine] = 1
            matrix[motor, generator] = 1 - lam
            matrix[motor, cable] = lam
            matrix[sink, motor] = 1

        matrix[final_sink, sink_idx] = 1 / num_eng
        return matrix

    return arch, oper_ups, oper_dwn, eta_ups, eta_dwn, src_type, trn_type


def turboelectric_architecture(num_eng, eta_em, eta_eg, eta_ts):
    """Return FAST's built-in turboelectric architecture matrices."""

    ncomp = 4 * num_eng + 2
    arch = np.zeros((ncomp, ncomp))
    fuel = 0
    engine_idx = np.arange(1, 1 + num_eng)
    generator_idx = np.arange(1 + num_eng, 1 + 2 * num_eng)
    motor_idx = np.arange(1 + 2 * num_eng, 1 + 3 * num_eng)
    sink_idx = np.arange(1 + 3 * num_eng, 1 + 4 * num_eng)
    final_sink = ncomp - 1
    arch[fuel, engine_idx] = 1

    for engine, generator, motor, sink in zip(
        engine_idx,
        generator_idx,
        motor_idx,
        sink_idx,
    ):
        arch[engine, generator] = 1
        arch[generator, motor] = 1
        arch[motor, sink] = 1
        arch[sink, final_sink] = 1

    eta_ups = np.ones_like(arch)
    eta_dwn = np.ones_like(arch)

    for engine, generator, motor, sink in zip(
        engine_idx,
        generator_idx,
        motor_idx,
        sink_idx,
    ):
        eta_ups[engine, generator] = eta_eg
        eta_ups[generator, motor] = eta_em
        eta_ups[motor, sink] = eta_ts
        eta_dwn[generator, engine] = eta_eg
        eta_dwn[motor, generator] = eta_em
        eta_dwn[sink, motor] = eta_ts

    src_type = [1]
    trn_type = [1] * num_eng + [3] * num_eng + [0] * num_eng + [2] * num_eng

    oper_ups = np.zeros_like(arch)
    oper_dwn = np.zeros_like(arch)
    oper_ups[fuel, engine_idx] = 1 / num_eng
    oper_dwn[engine_idx, fuel] = 1

    for engine, generator, motor, sink in zip(
        engine_idx,
        generator_idx,
        motor_idx,
        sink_idx,
    ):
        oper_ups[engine, generator] = 1
        oper_ups[generator, motor] = 1
        oper_ups[motor, sink] = 1
        oper_ups[sink, final_sink] = 1
        oper_dwn[generator, engine] = 1
        oper_dwn[motor, generator] = 1
        oper_dwn[sink, motor] = 1

    oper_dwn[final_sink, sink_idx] = 1 / num_eng
    return arch, oper_ups, oper_dwn, eta_ups, eta_dwn, src_type, trn_type


def partial_turboelectric_architecture(num_eng, eta_em, eta_eg, eta_ts):
    """Return FAST's built-in partially turboelectric architecture matrices."""

    ncomp = 5 * num_eng + 2
    arch = np.zeros((ncomp, ncomp))
    fuel = 0
    engine_idx = np.arange(1, 1 + num_eng)
    generator_idx = np.arange(1 + num_eng, 1 + 2 * num_eng)
    motor_idx = np.arange(1 + 2 * num_eng, 1 + 3 * num_eng)
    inboard_idx = np.arange(1 + 3 * num_eng, 1 + 4 * num_eng)
    outboard_idx = np.arange(1 + 4 * num_eng, 1 + 5 * num_eng)
    final_sink = ncomp - 1
    arch[fuel, engine_idx] = 1

    for engine, generator, motor, inboard, outboard in zip(
        engine_idx,
        generator_idx,
        motor_idx,
        inboard_idx,
        outboard_idx,
    ):
        arch[engine, generator] = 1
        arch[engine, outboard] = 1
        arch[generator, motor] = 1
        arch[motor, inboard] = 1
        arch[inboard, final_sink] = 1
        arch[outboard, final_sink] = 1

    eta_ups = np.ones_like(arch)
    eta_dwn = np.ones_like(arch)

    for engine, generator, motor, inboard, outboard in zip(
        engine_idx,
        generator_idx,
        motor_idx,
        inboard_idx,
        outboard_idx,
    ):
        eta_ups[engine, generator] = eta_eg
        eta_ups[engine, outboard] = eta_ts
        eta_ups[generator, motor] = eta_em
        eta_ups[motor, inboard] = eta_ts
        eta_dwn[generator, engine] = eta_eg
        eta_dwn[motor, generator] = eta_em
        eta_dwn[inboard, motor] = eta_ts
        eta_dwn[outboard, engine] = eta_ts

    src_type = [1]
    trn_type = (
        [1] * num_eng
        + [3] * num_eng
        + [0] * num_eng
        + [2] * (2 * num_eng)
    )

    def oper_ups(lam):
        """Return upstream operational splits for a PE architecture."""

        matrix = np.zeros_like(arch)
        matrix[fuel, engine_idx] = 1 / num_eng

        for engine, generator, motor, inboard, outboard in zip(
            engine_idx,
            generator_idx,
            motor_idx,
            inboard_idx,
            outboard_idx,
        ):
            matrix[engine, generator] = 1 - lam
            matrix[engine, outboard] = lam
            matrix[generator, motor] = 1
            matrix[motor, inboard] = 1
            matrix[inboard, final_sink] = 1
            matrix[outboard, final_sink] = 1

        return matrix

    def oper_dwn(lam):
        """Return downstream operational splits for a PE architecture."""

        matrix = np.zeros_like(arch)
        matrix[engine_idx, fuel] = 1

        for engine, generator, motor, inboard, outboard in zip(
            engine_idx,
            generator_idx,
            motor_idx,
            inboard_idx,
            outboard_idx,
        ):
            matrix[generator, engine] = 1
            matrix[motor, generator] = 1
            matrix[inboard, motor] = 1
            matrix[outboard, engine] = 1

        matrix[final_sink, inboard_idx] = (1 - lam) / num_eng
        matrix[final_sink, outboard_idx] = lam / num_eng
        return matrix

    return arch, oper_ups, oper_dwn, eta_ups, eta_dwn, src_type, trn_type


def make_split_callable(split):
    """Return a callable wrapper while preserving MATLAB nargin behavior.

    Inputs:
        split: Either a constant split matrix or a one-argument split function.

    Outputs:
        Callable returning nested Python lists so JSON-facing structures remain
        serializable after architecture construction.

    Assumptions:
        Built-in architectures currently need at most one operational split
        argument. Custom callables are counted elsewhere by callable_arg_count().
    """

    if callable(split):
        return lambda lam: split(lam).tolist()

    return lambda: split.tolist()


def get_thrust_sink_efficiency(specs, aircraft_class):
    """Return fan or propeller efficiency for built-in architectures."""

    if aircraft_class.lower() == "turbofan":
        return specs["Propulsion"]["Engine"]["EtaPoly"]["Fan"]

    if aircraft_class.lower() in ("turboprop", "piston"):
        return specs["Power"]["Eta"]["Propeller"]

    raise PropulsionError(f"Invalid aircraft class: {aircraft_class}")


def validate_custom_prop_arch(prop_arch):
    """Validate required fields for a custom propulsion architecture.

    Inputs:
        prop_arch: Custom "O" architecture dictionary.

    Outputs:
        None. Raises PropulsionError for missing fields or inconsistent matrix
        dimensions.

    Assumptions:
        Sources are nodes with no incoming edges and sinks are nodes with no
        outgoing edges, matching FAST's architecture graph interpretation.
    """

    required_fields = ("Arch", "OperUps", "OperDwn", "EtaUps", "EtaDwn", "SrcType", "TrnType")

    for field_name in required_fields:
        if field_name not in prop_arch:
            raise PropulsionError(f"Custom PropArch is missing {field_name}.")

    arch = as_array(prop_arch["Arch"])

    for field_name in ("OperUps", "OperDwn", "EtaUps", "EtaDwn"):
        value = eval_split(prop_arch[field_name]) if callable(prop_arch[field_name]) else as_array(prop_arch[field_name])

        if value.shape[0] != value.shape[1]:
            raise PropulsionError(f"Custom PropArch {field_name} matrix must be square.")

        if value.shape != arch.shape:
            raise PropulsionError(f"Custom PropArch {field_name} shape must match Arch.")

    if arch.shape[0] != arch.shape[1]:
        raise PropulsionError("Custom PropArch Arch matrix must be square.")

    nsrc = int(np.sum(np.sum(arch, axis=0) == 0))
    nsnk = int(np.sum(np.sum(arch, axis=1) == 0))
    ntrn = arch.shape[0] - nsrc - nsnk

    if nsrc != len(as_vector(prop_arch["SrcType"])):
        raise PropulsionError("Custom PropArch has an incorrect number of sources.")

    if ntrn != len(as_vector(prop_arch["TrnType"])):
        raise PropulsionError("Custom PropArch has an incorrect number of transmitters.")


def set_split_arg_counts(aircraft):
    """Store split callable argument counts in aircraft settings.

    Inputs:
        aircraft: Aircraft dictionary with PropArch.OperUps and OperDwn.

    Outputs:
        None. Settings.nargOperUps and Settings.nargOperDwn are updated in
        place because initialization routines read those counts immediately
        after architecture creation.
    """

    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    settings = aircraft.setdefault("Settings", {})
    settings["nargOperUps"] = callable_arg_count(prop_arch["OperUps"]) if callable(prop_arch["OperUps"]) else 0
    settings["nargOperDwn"] = callable_arg_count(prop_arch["OperDwn"]) if callable(prop_arch["OperDwn"]) else 0


def engine_lapse(sls, aircraft_class, rho):
    """Estimate lapsed thrust or power from sea-level-static output.

    Inputs:
        sls: Sea-level-static thrust or power value.
        aircraft_class: FAST TLAR class string.
        rho: Air density in kg/m^3 at one or more mission points.

    Outputs:
        Lapsed thrust/power with scalar/list shape restored.

    Assumptions:
        Turbofan thrust scales with density ratio; turboprop and piston power
        retain the MATLAB EngineLapse exponent of zero in this FAST model.
    """

    _, _, rho_sl = standard_atmosphere(0)
    rho_ratio = as_array(rho) / rho_sl

    if aircraft_class.lower() == "turbofan":
        exponent = 1.0
    elif aircraft_class.lower() in ("turboprop", "piston"):
        exponent = 0.0
    else:
        raise PropulsionError(f"Invalid aircraft class: {aircraft_class}")

    result = as_array(sls) * rho_ratio ** exponent
    return restore_scalar_or_list(result)


def power_available(aircraft):
    """Compute available thrust power for a mission segment.

    Inputs:
        aircraft: Dictionary containing a mission segment, mission history, and
            propulsion architecture fields.

    Outputs:
        The same aircraft dictionary with Power.Pav, Power.Tav, and Power.TV
        populated over the active segment.

    Assumptions:
        This ports the matrix propagation behavior of PropulsionPkg.PowerAvailable
        for dictionary data. Engine off-design lapse is currently the same
        density-ratio model as FAST EngineLapse.
    """

    aircraft = deepcopy(aircraft)
    profile = aircraft["Mission"]["Profile"]
    seg_id = int(profile["SegsID"]) - 1
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    history = aircraft["Mission"]["History"]["SI"]
    tas = as_vector(history["Performance"]["TAS"][seg_beg:seg_end])
    rho = as_vector(history["Performance"]["Rho"][seg_beg:seg_end])
    aclass = aircraft["Specs"]["TLAR"]["Class"]
    prop = aircraft["Specs"]["Propulsion"]
    prop_arch = prop["PropArch"]
    trn_type = as_vector(prop_arch["TrnType"])
    npnt = len(tas)
    arch = as_array(prop_arch["Arch"])
    eta_ups = as_array(prop_arch["EtaUps"])
    oper_ups = prop_arch["OperUps"]
    lam_ups = as_2d(history["Power"]["LamUps"][seg_beg:seg_end])
    ncomp = arch.shape[0]
    nsrc = len(as_vector(prop_arch["SrcType"]))
    ntrn = len(trn_type)
    nsnk = ncomp - nsrc - ntrn
    isnk = np.arange(nsrc + ntrn, ncomp)
    thrust_av = np.tile(as_vector(prop["SLSThrust"]), (npnt, 1))
    power_av = np.tile(as_vector(prop["SLSPower"]), (npnt, 1))
    sls_power = as_vector(prop["SLSPower"])
    itrn = np.arange(nsrc, nsrc + ntrn)

    for jtrn in range(ntrn):
        if trn_type[jtrn] == 1:
            if aclass.lower() == "turbofan":
                thrust_av[:, jtrn] = as_vector(
                    engine_lapse(thrust_av[:, jtrn], aclass, rho)
                )
                power_av[:, jtrn] = thrust_av[:, jtrn] * tas
            elif aclass.lower() in ("turboprop", "piston"):
                power_av[:, jtrn] = as_vector(
                    engine_lapse(power_av[:, jtrn], aclass, rho)
                )
            else:
                raise PropulsionError(f"Invalid aircraft class: {aclass}")
        elif trn_type[jtrn] not in (0, 2, 3, 4):
            raise PropulsionError(f"Invalid transmitter type at position {jtrn + 1}.")

    pav = np.zeros((npnt, ncomp))
    idx = np.arange(nsrc, ncomp)

    for ipnt in range(npnt):
        split = eval_split(oper_ups, lam_ups[ipnt, :])
        up_trn = np.where((np.sum(split[np.ix_(itrn, itrn)], axis=0) > 0) | (trn_type == 2))[0]
        power_av[ipnt, up_trn] = 0
        pav[ipnt, :] = np.concatenate(
            [np.zeros(nsrc), power_av[ipnt, :], np.zeros(nsnk)]
        )
        pav[ipnt, idx] = power_flow(
            pav[ipnt, idx],
            arch[np.ix_(idx, idx)],
            split[np.ix_(idx, idx)],
            eta_ups[np.ix_(idx, idx)],
            1,
        )
        overload = pav[ipnt, itrn] > sls_power

        if np.any(overload):
            temp_idx = np.where(overload)[0]
            pav[ipnt, temp_idx + nsrc] = sls_power[temp_idx]
            temp_trn = np.where(arch[:, isnk].any(axis=1))[0]
            recompute_idx = np.concatenate([temp_trn, isnk])
            pav[ipnt, recompute_idx] = power_flow(
                pav[ipnt, recompute_idx],
                arch[np.ix_(recompute_idx, recompute_idx)],
                split[np.ix_(recompute_idx, recompute_idx)],
                eta_ups[np.ix_(recompute_idx, recompute_idx)],
                1,
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        tav = pav / tas[:, None]
    tv = np.sum(pav[:, nsrc + ntrn:], axis=1)
    ensure_power_history_arrays(history, "Pav", ncomp, len(history["Performance"]["TAS"]))
    ensure_power_history_arrays(history, "Tav", ncomp, len(history["Performance"]["TAS"]))
    ensure_power_history_vector(history, "TV", len(history["Performance"]["TAS"]))
    history["Power"]["Pav"][seg_beg:seg_end] = pav.tolist()
    history["Power"]["Tav"][seg_beg:seg_end] = tav.tolist()
    history["Power"]["TV"][seg_beg:seg_end] = tv.tolist()
    return aircraft


def propulsion_sizing(aircraft):
    """Size propulsion component power, thrust, and weights.

    Inputs:
        aircraft: Dictionary with processed specs, propulsion architecture, and
            historical engine data.

    Outputs:
        A deep-copied aircraft dictionary with SLSPower, SLSThrust,
        supplemental power/thrust, and propulsion component weights populated.

    Assumptions:
        This ports PropulsionPkg.PropulsionSizing's graph power split, weight
        bookkeeping, and turbofan nonlinear engine sizing call.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    aclass = specs["TLAR"]["Class"]
    tko_vel = specs["Performance"]["Vels"]["Tko"]
    prop = specs["Propulsion"]
    power_specs = specs["Power"]
    prop_arch = prop["PropArch"]
    trn_type = as_vector(prop_arch["TrnType"])
    engines = trn_type == 1
    motors = trn_type == 0
    generators = trn_type == 3
    cables = trn_type == 4
    p_wem = power_specs["P_W"]["EM"]
    p_weg = power_specs["P_W"]["EG"]
    arch = as_array(prop_arch["Arch"])
    lam_dwn = power_specs["LamDwn"]["SLS"]
    eta_dwn = as_array(prop_arch["EtaDwn"])
    eta_fan = transmitter_fan_efficiency(specs, aclass)

    if aclass.lower() == "turbofan":
        p0 = 0.95 * prop["Thrust"]["SLS"] * tko_vel
    elif aclass.lower() in ("turboprop", "piston"):
        p0 = power_specs["SLS"]
    else:
        raise PropulsionError(f"Invalid aircraft class: {aclass}")

    split = eval_split(prop_arch["OperDwn"], lam_dwn)
    nsrc = len(as_vector(prop_arch["SrcType"]))
    ntrn = len(trn_type)
    ncomp = len(arch)
    flow_idx = np.arange(nsrc, ncomp)
    initial = np.concatenate([np.zeros(ntrn), [p0]])
    pdwn = power_flow(
        initial,
        arch[np.ix_(flow_idx, flow_idx)].T,
        split[np.ix_(flow_idx, flow_idx)],
        eta_dwn[np.ix_(flow_idx, flow_idx)],
        -1,
    )
    trn_idx = np.arange(nsrc, nsrc + ntrn)
    psupp = power_supplement_check(
        pdwn[:-1].reshape(1, -1),
        arch[np.ix_(trn_idx, trn_idx)],
        split[np.ix_(trn_idx, trn_idx)],
        eta_dwn[np.ix_(trn_idx, trn_idx)],
        trn_type,
        eta_fan,
    )
    tdwn = pdwn / tko_vel
    tsupp = psupp / tko_vel
    prop["SLSPower"] = pdwn[:-1].tolist()
    prop["PowerSupp"] = psupp.reshape(-1).tolist()
    prop["SLSThrust"] = tdwn[:-1].tolist()
    prop["ThrustSupp"] = tsupp.reshape(-1).tolist()

    if np.any((trn_type > 0) & (trn_type != 2)):
        weng = engine_weights_for_sizing(
            aircraft,
            aclass,
            engines,
            pdwn,
            tdwn,
            psupp.reshape(-1),
            tsupp.reshape(-1),
        )
    else:
        weng = np.asarray([0.0])

    specs["Weight"]["Cables"] = cable_weight_for_sizing(
        aircraft,
        cables,
        pdwn,
    )
    specs["Weight"]["Engines"] = float(np.sum(weng))
    specs["Weight"]["EM"] = safe_component_weight(np.sum(pdwn[:-1][motors]), p_wem)
    specs["Weight"]["EG"] = safe_component_weight(np.sum(pdwn[:-1][generators]), p_weg)
    return aircraft


def recompute_splits(aircraft, seg_beg, seg_end):
    """Recompute downstream splits for full-throttle parallel connections.

    Inputs:
        aircraft: Dictionary with PropArch.ParConns and mission power history.
        seg_beg: One-based beginning mission-history row.
        seg_end: One-based ending mission-history row.

    Outputs:
        A deep-copied aircraft dictionary with Power.LamDwn updated.

    Assumptions:
        The Python ParConns convention stores zero-based component indices for
        the supplemental transmitter, matching prop_arch_connections().
    """

    aircraft = deepcopy(aircraft)
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    parallel = [
        index
        for index, conns in enumerate(prop_arch.get("ParConns", []))
        if len(conns) > 0
    ]

    if not parallel:
        return aircraft

    start = int(seg_beg) - 1
    stop = int(seg_end)
    nsrc = len(as_vector(prop_arch["SrcType"]))
    power = aircraft["Mission"]["History"]["SI"]["Power"]
    pav = as_array(power["Pav"][start:stop])
    lam_ups = as_2d(power["LamUps"][start:stop])
    lam_dwn = as_2d(power["LamDwn"][start:stop]).copy()
    active_rows = np.any(lam_ups > 0, axis=1)
    nsplit = int(aircraft["Settings"].get("nargOperDwn", lam_dwn.shape[1]))
    tmp_split = lam_dwn[0, :].copy()
    oper_dwn = eval_split(prop_arch["OperDwn"], tmp_split)

    for main_relative in parallel:
        main_component = main_relative + nsrc
        supplement_components = prop_arch["ParConns"][main_relative]
        use_split = np.zeros(nsplit, dtype=bool)

        for split_index in range(nsplit):
            tmp_split[split_index] += 0.01
            oper_new = eval_split(prop_arch["OperDwn"], tmp_split)
            use_split[split_index] = np.any(
                oper_dwn[supplement_components, main_component]
                != oper_new[supplement_components, main_component]
            )
            tmp_split[split_index] -= 0.01

        total_power = np.sum(
            pav[np.ix_(active_rows, [main_component] + supplement_components)],
            axis=1,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            split_values = pav[np.ix_(active_rows, supplement_components)] / total_power[:, None]

        split_values[~np.isfinite(split_values)] = 0
        lam_dwn[np.ix_(active_rows, use_split)] = split_values

    power["LamDwn"][start:stop] = lam_dwn.tolist()
    return aircraft


def prop_analysis(aircraft):
    """Analyze propulsion power flow over the active mission segment.

    Inputs:
        aircraft: Dictionary with mission history, propulsion architecture, and
            active Mission.Profile.SegsID/MissID fields.

    Outputs:
        A deep-copied aircraft dictionary with component power, thrust, energy,
        SOC, fuel-burn, and mass histories updated over the active segment.

    Assumptions:
        This ports PropulsionPkg.PropAnalysis's matrix power bookkeeping,
        detailed and simple battery energy accounting, the default SimpleOffDesign
        turbofan fuel-flow path, and the default turboprop nonlinear
        fuel-flow path.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    prop = specs["Propulsion"]
    prop_arch = prop["PropArch"]
    history = aircraft["Mission"]["History"]["SI"]
    profile = aircraft["Mission"]["Profile"]
    aclass = specs["TLAR"]["Class"]
    efuel = specs["Power"]["SpecEnergy"]["Fuel"]
    arch = as_array(prop_arch["Arch"])
    oper_dwn = prop_arch["OperDwn"]
    eta_dwn = as_array(prop_arch["EtaDwn"])
    src_type = as_vector(prop_arch["SrcType"])
    trn_type = as_vector(prop_arch["TrnType"])
    nsrc = len(src_type)
    ntrn = len(trn_type)
    ncomp = len(arch)
    seg_id = int(profile["SegsID"]) - 1
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    preq_snk = as_vector(history["Power"]["Req"][seg_beg:seg_end])
    pav = history_matrix(history["Power"]["Pav"][seg_beg:seg_end], ncomp)
    lam_dwn = history_matrix(history["Power"]["LamDwn"][seg_beg:seg_end], max(1, aircraft["Settings"].get("nargOperDwn", 1)))
    mass = as_vector(history["Weight"]["CurWeight"][seg_beg:seg_end])
    time = as_vector(history["Performance"]["Time"][seg_beg:seg_end])
    tas = as_vector(history["Performance"]["TAS"][seg_beg:seg_end])
    alt = as_vector(history["Performance"]["Alt"][seg_beg:seg_end])
    mach = as_vector(history["Performance"]["Mach"][seg_beg:seg_end])
    dt = np.diff(time)
    npnt = len(time)
    preq = np.zeros((npnt, ncomp))
    pout = np.zeros((npnt, ncomp))
    psupp = np.zeros((npnt, ncomp))
    preq[:, -1] = preq_snk
    trn_snk_idx = np.arange(nsrc, ncomp)
    src_trn_idx = np.arange(0, nsrc + ntrn)
    trn_idx = np.arange(nsrc, nsrc + ntrn)

    for ipnt in range(npnt):
        if np.isinf(preq[ipnt, -1]):
            preq[ipnt, :] = np.inf
            continue

        split = eval_split(oper_dwn, lam_dwn[ipnt, :])
        preq[ipnt, trn_snk_idx] = power_flow(
            preq[ipnt, trn_snk_idx],
            arch[np.ix_(trn_snk_idx, trn_snk_idx)].T,
            split[np.ix_(trn_snk_idx, trn_snk_idx)],
            eta_dwn[np.ix_(trn_snk_idx, trn_snk_idx)],
            -1,
        )

    pout[:, trn_snk_idx] = preq[:, trn_snk_idx]
    pout_test = pout[:, trn_snk_idx]
    pav_test = pav[:, trn_snk_idx]
    exceeds = pout_test - pav_test > 1.0e-6
    pout_test[exceeds] = pav_test[exceeds]
    pout[:, trn_snk_idx] = pout_test

    for ipnt in range(npnt):
        split = eval_split(oper_dwn, lam_dwn[ipnt, :])
        pout[ipnt, src_trn_idx] = power_flow(
            pout[ipnt, src_trn_idx],
            arch[np.ix_(src_trn_idx, src_trn_idx)].T,
            split[np.ix_(src_trn_idx, src_trn_idx)],
            eta_dwn[np.ix_(src_trn_idx, src_trn_idx)],
            -1,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        tout = pout / tas[:, None]
        treq = preq / tas[:, None]

    eta_fan = transmitter_fan_efficiency(specs, aclass)

    for ipnt in range(npnt):
        split = eval_split(oper_dwn, lam_dwn[ipnt, :])
        psupp[ipnt, trn_idx] = power_supplement_check(
            pout[ipnt, trn_idx],
            arch[np.ix_(trn_idx, trn_idx)],
            split[np.ix_(trn_idx, trn_idx)],
            eta_dwn[np.ix_(trn_idx, trn_idx)],
            trn_type,
            eta_fan,
        )

    fuel = src_type == 1
    batt = src_type == 0
    engines = trn_type == 1
    e_es = np.zeros((npnt, nsrc))
    fburn = np.zeros(npnt)
    soc = np.ones((npnt, nsrc)) * 100
    voltage = np.zeros((npnt, nsrc))
    current = np.zeros((npnt, nsrc))
    capacity = np.zeros((npnt, nsrc))
    c_rate = np.zeros((npnt, nsrc))
    eleft_es = history_matrix(history["Energy"]["Eleft_ES"][seg_beg:seg_end], nsrc)

    if seg_beg > 0:
        fburn[:] = history["Weight"]["Fburn"][seg_beg]
        e_es[:] = history_matrix(history["Energy"]["E_ES"][seg_beg:seg_end], nsrc)[0, :]

        if is_detailed_battery(specs):
            soc[:] = history_matrix(history["Power"]["SOC"][seg_beg:seg_end], nsrc)[0, :]

    update_battery_energy(
        aircraft,
        pout,
        dt,
        batt,
        engines,
        nsrc,
        e_es,
        eleft_es,
        soc,
        voltage,
        current,
        capacity,
        c_rate,
        lam_dwn,
        mass,
        seg_beg,
    )
    sfc = np.zeros((npnt, ntrn))
    mdot_fuel = np.zeros((npnt, ntrn))
    dmdt = np.zeros((npnt, nsrc))

    if np.any(fuel):
        estimate_fuel_use(
            aircraft,
            pout,
            psupp,
            dt,
            fuel,
            engines,
            nsrc,
            ntrn,
            efuel,
            alt,
            mach,
            tout,
            fburn,
            mass,
            e_es,
            eleft_es,
            sfc,
            mdot_fuel,
            dmdt,
        )

    assign_history_matrix(history["Power"], "Preq", preq, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Treq", treq, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Pout", pout, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Tout", tout, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "SOC", soc, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Voltage", voltage, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Current", current, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "Capacity", capacity, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "C_rate", c_rate, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "LamDwn", lam_dwn, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "E_ES", e_es, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    assign_history_matrix(history["Propulsion"], "TSFC", sfc, seg_beg, seg_end)
    assign_history_matrix(history["Propulsion"], "MDotFuel", mdot_fuel, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "Fburn", fburn, seg_beg, seg_end)
    return aircraft


def update_battery_energy(
    aircraft,
    pout,
    dt,
    batt,
    engines,
    nsrc,
    e_es,
    eleft_es,
    soc,
    voltage,
    current,
    capacity,
    c_rate,
    lam_dwn,
    mass,
    seg_beg,
):
    """Update battery energy and optional detailed cell histories.

    Inputs:
        aircraft: Aircraft dictionary whose history flags may be updated.
        pout: Component output-power history in W, mutated in place.
        dt: Segment time-step vector in seconds.
        batt: Boolean source mask for battery columns.
        engines: Boolean transmitter mask for gas-turbine columns.
        nsrc: Number of source columns.
        e_es: Source energy used in FAST mission energy units, mutated in place.
        eleft_es: Remaining source energy, mutated in place.
        soc, voltage, current, capacity, c_rate: Detailed battery histories
            mutated in place when cell sizing is prescribed.
        lam_dwn: Downstream split history, set to zero after battery cutoff.
        mass: Mission mass history, included for parity with the MATLAB call
            signature.
        seg_beg: Zero-based beginning row of the active mission segment.

    Outputs:
        None. History arrays are updated in place to match PropAnalysis.

    Assumptions:
        Off-design non-electric missions shut battery contribution off when SOC
        or available energy is exhausted; electric-only missions keep the
        requested power history because no alternate fuel source exists.
    """

    if not np.any(batt):
        return

    detailed = is_detailed_battery(aircraft["Specs"])
    power_battery = aircraft["Specs"]["Power"].get("Battery", {})

    for icol in np.where(batt)[0]:
        if detailed and len(dt) > 0:
            (
                voltage_values,
                current_values,
                _,
                capacity_values,
                soc_values,
                c_rate_values,
            ) = discharging(
                aircraft,
                pout[:-1, icol],
                dt,
                soc[0, icol],
                power_battery["ParCells"],
                power_battery["SerCells"],
            )
            voltage[:-1, icol] = voltage_values
            current[:-1, icol] = current_values
            capacity[1:, icol] = capacity_values
            soc[:, icol] = soc_values
            c_rate[:-1, icol] = c_rate_values
            stop_soc_rows = np.where(soc[:, icol] < 20)[0]
            arch_type = aircraft["Specs"]["Propulsion"]["PropArch"].get("Type", "")
            analysis_type = aircraft["Settings"]["Analysis"]["Type"]

            if (
                len(stop_soc_rows) > 0
                and str(arch_type).upper() != "E"
                and analysis_type < 0
            ):
                stop_soc = stop_soc_rows[0]
                pout[stop_soc:, icol] = 0
                lam_dwn[stop_soc:, :] = 0
                soc[stop_soc:, icol] = soc[max(0, stop_soc - 1), icol]

                if "Flags" in aircraft["Mission"]["History"]:
                    miss_id = int(aircraft["Mission"]["Profile"].get("MissID", 1)) - 1
                    aircraft["Mission"]["History"]["Flags"]["SOCOff"][miss_id] = 1

        used = np.cumsum(pout[:-1, icol] * dt)
        e_es[1:, icol] = e_es[0, icol] + used
        eleft_es[1:, icol] = eleft_es[0, icol] - used
        stop_rows = np.where(eleft_es[:, icol] < 0)[0]

        if len(stop_rows) == 0 or aircraft["Settings"]["Analysis"]["Type"] >= 0:
            continue

        stop = max(0, stop_rows[0] - 1)
        engine_cols = np.where(engines)[0] + nsrc

        if len(engine_cols) > 0:
            extra = pout[stop:-1, icol] / len(engine_cols)
            pout[stop:-1, engine_cols] += extra[:, None]

        pout[stop:, icol] = 0
        used = np.cumsum(pout[:-1, icol] * dt)
        e_es[1:, icol] = e_es[0, icol] + used
        eleft_es[1:, icol] = eleft_es[0, icol] - used
        lam_dwn[stop:, :] = 0
        soc[stop:, icol] = soc[max(0, stop - 1), icol]

        if "Flags" in aircraft["Mission"]["History"]:
            miss_id = int(aircraft["Mission"]["Profile"].get("MissID", 1)) - 1
            aircraft["Mission"]["History"]["Flags"]["SOCOff"][miss_id] = 1


def estimate_fuel_use(
    aircraft,
    pout,
    psupp,
    dt,
    fuel,
    engines,
    nsrc,
    ntrn,
    efuel,
    alt,
    mach,
    tout,
    fburn,
    mass,
    e_es,
    eleft_es,
    sfc,
    mdot_fuel,
    dmdt,
):
    """Estimate fuel use for engines with native or supplied fuel models.

    Inputs:
        aircraft: Aircraft dictionary with propulsion class and engine data.
        pout, psupp: Component output and supplemental power histories in W.
        dt: Segment time steps in seconds.
        fuel, engines: Boolean masks for fuel sources and engine transmitters.
        nsrc, ntrn: Number of source and transmitter components.
        efuel: Fuel specific energy in FAST mission energy units.
        alt, mach, tout: Flight condition and thrust histories.
        fburn, mass, e_es, eleft_es, sfc, mdot_fuel, dmdt: History arrays
            mutated in place.

    Outputs:
        None. Fuel burn, source energy, mass, SFC, and mass-flow histories are
        updated in place for the active segment.

    Assumptions:
        A custom FuelFlowModel takes precedence. Otherwise turbofan fuel uses
        SimpleOffDesign and turboprop/piston fuel uses the nonlinear sizing
        path, matching FAST's class-dependent branch.
    """

    fuel_model = aircraft["Specs"]["Propulsion"].get("FuelFlowModel")
    aircraft_class = aircraft["Specs"]["TLAR"]["Class"]
    arch = as_array(aircraft["Specs"]["Propulsion"]["PropArch"]["Arch"])
    trn_type = as_vector(aircraft["Specs"]["Propulsion"]["PropArch"]["TrnType"])

    if (
        not callable(fuel_model)
        and aircraft_class.lower() not in ("turbofan", "turboprop", "piston")
    ):
        return

    fuel_cols = np.where(fuel)[0]
    engine_cols = np.where(engines)[0]

    for ipnt in range(len(dt)):
        for rel_engine in engine_cols:
            component = rel_engine + nsrc

            if callable(fuel_model):
                flow, sfc_value = fuel_model(
                    aircraft,
                    ipnt,
                    component,
                    pout[ipnt, component],
                    psupp[ipnt, component],
                )
            elif aircraft_class.lower() == "turbofan":
                thrust = engine_thrust_requirement(
                    arch,
                    trn_type,
                    nsrc,
                    tout,
                    component,
                    ipnt,
                )

                if thrust < 1:
                    thrust = 0.05 * aircraft["Specs"]["Propulsion"]["Thrust"]["SLS"]

                off_params = {
                    "FlightCon": {
                        "Mach": mach[ipnt],
                        "Alt": alt[ipnt],
                    },
                    "Thrust": thrust,
                }
                output = simple_off_design(
                    aircraft,
                    off_params,
                    psupp[ipnt, component],
                    component,
                    # MATLAB passes the segment-local point index here, even
                    # though SimpleOffDesign reads the full Tav history.
                    ipnt,
                )
                flow = output["Fuel"]
                sfc_value = output["TSFC"]
            else:
                req_power = pout[ipnt, component]

                if req_power < 1:
                    req_power = 0.05 * aircraft["Specs"]["Power"]["SLS"]

                aircraft["Specs"]["Propulsion"]["Engine"]["ReqPower"] = req_power
                engine_spec = deepcopy(aircraft["Specs"]["Propulsion"]["Engine"])
                engine_spec["ReqPower"] = req_power
                output = turboprop_nonlinear_sizing(
                    engine_spec,
                    psupp[ipnt, component],
                )
                flow = output["Fuel"]["MDot"]
                sfc_value = output["BSFC"]

            mdot_cf = aircraft["Specs"]["Propulsion"].get("MDotCF", 1)
            mdot_fuel[ipnt, rel_engine] = flow * mdot_cf
            sfc[ipnt, rel_engine] = sfc_value * mdot_cf

        if len(fuel_cols) > 0:
            dmdt[ipnt, fuel_cols[0]] = np.sum(mdot_fuel[ipnt, engine_cols])

    if len(fuel_cols) == 0:
        return

    fuel_col = fuel_cols[0]

    if len(dt) > 0:
        fuel_used = np.cumsum(dmdt[:len(dt), fuel_col] * dt)
        energy_used = np.cumsum(dmdt[:len(dt), fuel_col] * efuel * dt)
        fburn[1:] = fburn[0] + fuel_used
        mass[1:] = mass[0] - fuel_used
        e_es[1:, fuel_col] = e_es[1:, fuel_col] + energy_used
        eleft_es[1:, fuel_col] = eleft_es[0, fuel_col] - energy_used


def engine_thrust_requirement(arch, trn_type, nsrc, tout, component, point):
    """Return thrust handled by the sink connected to an engine component."""

    thrust_components = np.where(trn_type == 2)[0] + nsrc

    if len(thrust_components) == 0:
        return 0

    connected = thrust_components[arch[component, thrust_components] > 0]

    if len(connected) == 0:
        return 0

    thrust = tout[point, connected[0]]

    if not np.isfinite(thrust):
        return 0

    return thrust


def is_detailed_battery(specs):
    """Return True when detailed battery cells are prescribed."""

    battery = specs["Power"]["Battery"]

    try:
        return not np.isnan(battery["SerCells"]) and not np.isnan(battery["ParCells"])
    except TypeError:
        return False


def transmitter_fan_efficiency(specs, aircraft_class):
    """Return fan efficiency used for supplemental power bookkeeping."""

    if aircraft_class.lower() == "turbofan":
        return specs["Propulsion"]["Engine"]["EtaPoly"]["Fan"]

    if aircraft_class.lower() in ("turboprop", "piston"):
        return 1

    raise PropulsionError(f"Invalid aircraft class: {aircraft_class}")


def engine_weights_for_sizing(aircraft, aircraft_class, engines, pdwn, tdwn, psupp, tsupp):
    """Return gas-turbine engine weights for PropulsionSizing."""

    prop = aircraft["Specs"]["Propulsion"]

    if aircraft_class.lower() == "turbofan":
        data = aircraft["HistData"]["Eng"]
        target = tdwn[:-1][engines].reshape(-1, 1)
        weights, _ = nlgpr(data, [["Thrust_Max"], ["DryWeight"]], target)
        first_engine = np.where(engines)[0][0]
        prop["Engine"]["Alt"] = 0
        prop["Engine"]["Mach"] = 0.05

        if tsupp[first_engine] > 0:
            prop["Engine"]["DesignThrust"] = tdwn[:-1][first_engine] + tsupp[first_engine]
        else:
            prop["Engine"]["DesignThrust"] = tdwn[:-1][first_engine]

        prop["Engine"]["Sizing"] = 1
        prop["SizedEngine"] = turbofan_nonlinear_sizing(
            prop["Engine"],
            psupp[first_engine],
        )
        prop["Engine"]["Sizing"] = 0
        prop["SizedEngine"]["Specs"]["Sizing"] = 0
        return weights

    if aircraft_class.lower() in ("turboprop", "piston"):
        data = aircraft["HistData"]["Eng"]
        _, weight_rows = search_db(data, ["DryWeight"])
        _, power_rows = search_db(data, ["Power_SLS"])
        dry_weight = as_vector([row[1] for row in weight_rows])
        power_sls = as_vector([row[1] for row in power_rows])
        valid = (~np.isnan(dry_weight)) & (~np.isnan(power_sls))
        fit = np.polyfit(power_sls[valid], dry_weight[valid], 1)
        return np.polyval(fit, pdwn[:-1][engines] / 1000)

    raise PropulsionError(f"Invalid aircraft class: {aircraft_class}")


def cable_weight_for_sizing(aircraft, cables, pdwn):
    """Return cable weight for architectures with cable transmitters."""

    if not np.any(cables):
        return 0.0

    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]

    if "CableConns" not in prop_arch or "CableLengths" not in prop_arch:
        return 0.0

    cable_conns = as_array(prop_arch["CableConns"])
    cable_lengths = as_array(prop_arch["CableLengths"])
    p_cab = np.tile(pdwn[:-1][cables], (cable_conns.shape[0], 1))
    w_pcab = aircraft["Specs"]["Power"]["P_W"]["Cables"]
    return float(np.sum(w_pcab * (p_cab / 1.0e6) * cable_conns * cable_lengths))


def safe_component_weight(power_value, power_to_weight):
    """Return component weight while preserving zero-power components as zero."""

    if abs(power_value) < 1.0e-12:
        return 0.0

    return float(power_value / power_to_weight)


def ensure_power_history_arrays(history, name, columns, rows):
    """Ensure a 2D power history array exists."""

    if name not in history["Power"]:
        history["Power"][name] = [[0.0] * columns for _ in range(rows)]


def ensure_power_history_vector(history, name, rows):
    """Ensure a 1D power history vector exists."""

    if name not in history["Power"]:
        history["Power"][name] = [0.0] * rows


def assign_history_matrix(section, name, values, start, stop):
    """Assign a matrix slice into a history section."""

    values = np.asarray(values, dtype=float)
    rows = stop
    columns = values.shape[1]

    if name not in section:
        section[name] = [[0.0] * columns for _ in range(rows)]

    section[name][start:stop] = values.tolist()


def assign_history_vector(section, name, values, start, stop):
    """Assign a vector slice into a history section."""

    values = np.asarray(values, dtype=float).reshape(-1)

    if name not in section:
        section[name] = [0.0 for _ in range(stop)]

    section[name][start:stop] = values.tolist()


def history_matrix(value, columns):
    """Return a mission-history value as a 2D float matrix."""

    array = as_array(value)

    if array.ndim == 1 and columns == 1:
        return array.reshape(-1, 1)

    if array.ndim == 1:
        return array.reshape(1, -1)

    return array


def normalize_split_values(split_val):
    """Return split values as a list."""

    if split_val is None:
        return []

    if isinstance(split_val, np.ndarray):
        return split_val.flatten().tolist()

    if isinstance(split_val, list) or isinstance(split_val, tuple):
        return list(split_val)

    return [split_val]


def callable_arg_count(func):
    """Return the number of positional arguments declared by a callable."""

    signature = inspect.signature(func)
    count = 0

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return 17

        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1

    return count


def as_array(value):
    """Return value as a float NumPy array."""

    if isinstance(value, MatlabExpression):
        value = parse_constant_matrix_expression(value)

    if hasattr(value, "value"):
        value = value.value

    return np.asarray(value, dtype=float)


def as_vector(value):
    """Return value as a one-dimensional float NumPy array."""

    array = as_array(value)

    if array.ndim == 0:
        return array.reshape(1)

    return array.reshape(-1)


def as_2d(value):
    """Return value as a two-dimensional float NumPy array."""

    array = as_array(value)

    if array.ndim == 1:
        return array.reshape(1, -1)

    return array


def restore_scalar_or_list(value):
    """Return a scalar for scalar arrays, otherwise a Python list."""

    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    if array.size == 1:
        return float(array.reshape(-1)[0])

    return array.tolist()


EvalSplit = eval_split
PowerFlow = power_flow
PowerSupplementCheck = power_supplement_check
PropArchConnections = prop_arch_connections
CreatePropArch = create_prop_arch
EngineLapse = engine_lapse
PowerAvailable = power_available
PropulsionSizing = propulsion_sizing
RecomputeSplits = recompute_splits
PropAnalysis = prop_analysis
