# src/fast_python/safety.py

"""Fault-tree analysis helpers ported from FAST SafetyPkg."""

from itertools import combinations, product

import numpy as np


class SafetyError(ValueError):
    """Report invalid safety-analysis inputs."""


def compare_cols(col1, col2):
    """Set entries in the second column to zero when they match the first."""

    col1 = np.asarray(col1, dtype=int).reshape(-1)
    result = np.asarray(col2, dtype=int).reshape(-1).copy()
    result[col1 == result] = 0
    return result


def enumerate_failures(fail_list):
    """Enumerate all failure combinations from AND-gate inputs."""

    matrices = [as_failure_matrix(item) for item in fail_list]

    if any(item.size == 0 for item in matrices):
        return np.zeros((0, 0), dtype=int)

    rows = [range(item.shape[0]) for item in matrices]
    output = []

    for combo in product(*rows):
        parts = [matrices[index][row, :] for index, row in enumerate(combo)]
        output.append(np.concatenate(parts))

    return np.asarray(output, dtype=int)


def idempotent_law(fail_modes, split_col=0):
    """Remove duplicate events within each failure mode."""

    fail_modes = as_failure_matrix(fail_modes).copy()

    if fail_modes.size == 0:
        return fail_modes

    nmode, ncomp = fail_modes.shape

    if split_col > 0:
        outer_indices = range(split_col - 1)

        def inner_indices(_):
            return range(split_col - 1, ncomp)
    else:
        outer_indices = range(max(0, ncomp - 1))

        def inner_indices(index):
            return range(index + 1, ncomp)

    for icomp in outer_indices:
        temp_col = fail_modes[:, icomp]

        for jcomp in inner_indices(icomp):
            fail_modes[:, jcomp] = compare_cols(temp_col, fail_modes[:, jcomp])

    width = int(np.max(np.sum(fail_modes > 0, axis=1)))
    new_modes = np.zeros((nmode, width), dtype=int)

    for imode in range(nmode):
        remaining = fail_modes[imode, fail_modes[imode, :] > 0]
        new_modes[imode, : len(remaining)] = remaining

    return new_modes


def law_of_absorption(fail_modes):
    """Remove failure modes absorbed by simpler modes."""

    fail_modes = as_failure_matrix(fail_modes).copy()

    if fail_modes.size == 0:
        return fail_modes

    _, ncomp = fail_modes.shape

    for icomp in range(1, ncomp + 1):
        row_sum = np.sum(fail_modes > 0, axis=1)
        baseline = np.where(row_sum == icomp)[0]

        for current_index in baseline:
            current_mode = fail_modes[current_index, :icomp]

            if np.sum(current_mode == 0) == icomp:
                continue

            common = np.isin(fail_modes, current_mode).sum(axis=1)
            absorbed = common == icomp
            absorbed[current_index] = False
            fail_modes[absorbed, :] = 0

    keep_row = np.any(fail_modes > 0, axis=1)
    keep_col = np.any(fail_modes > 0, axis=0)

    if not np.any(keep_row) or not np.any(keep_col):
        return np.zeros((0, 0), dtype=int)

    return fail_modes[np.ix_(keep_row, keep_col)]


def and_gate(dwn_fails, ndwn=None):
    """Enumerate and simplify failures entering an AND gate."""

    if ndwn is None:
        ndwn = len(dwn_fails)

    dwn_fails = [as_failure_matrix(item) for item in dwn_fails[:ndwn]]

    if any(item.size == 0 for item in dwn_fails):
        return np.zeros((0, 0), dtype=int)

    if ndwn == 1:
        return dwn_fails[0]

    final_fails = None
    temp_fails = [dwn_fails[0], dwn_fails[1]]
    split_col = 0

    for index in range(1, ndwn):
        if temp_fails[0].size > 0 and temp_fails[1].size > 0:
            final_fails = enumerate_failures(temp_fails)
            final_fails = idempotent_law(final_fails, split_col)
            final_fails = law_of_absorption(final_fails)
        elif temp_fails[0].size == 0:
            final_fails = temp_fails[1]
        else:
            final_fails = temp_fails[0]

        if index < ndwn - 1:
            split_col = final_fails.shape[1] + 1
            temp_fails = [final_fails, dwn_fails[index + 1]]

    return final_fails


def create_cut_sets(arch, components, icomp, ntrigger, ninput):
    """Recursively list minimum cut sets for one architecture component."""

    downstream = arch[icomp]
    fail_mode = components["FailMode"][icomp]

    if str(fail_mode).lower() != "":
        internal_fails = np.asarray([[components["FailID"][icomp]]], dtype=int)
        fail_flag = 1
    else:
        internal_fails = np.zeros((0, 0), dtype=int)
        fail_flag = 0

    dwn_fails = [
        create_cut_sets(arch, components, index, ntrigger, ninput)
        for index in downstream
    ]
    ndwn = len(downstream)

    if ndwn > 0:
        if ndwn == 1:
            final_fails = dwn_fails[0]
        elif ntrigger[icomp] == ninput[icomp]:
            final_fails = and_gate(dwn_fails, ndwn)
        else:
            final_fails = kn_gate(dwn_fails, int(ntrigger[icomp]))

        if fail_flag == 1:
            return append_failure_rows(final_fails, internal_fails)

        return final_fails

    return internal_fails


def kn_gate(dwn_fails, ntrigger):
    """Enumerate and simplify a K/N gate."""

    if ntrigger <= 0:
        return np.zeros((0, 0), dtype=int)

    final_fails = None

    for combo in combinations(dwn_fails, ntrigger):
        if ntrigger == 1:
            current = combo[0]
        else:
            current = and_gate(list(combo), ntrigger)

        if final_fails is None:
            final_fails = current
        else:
            final_fails = append_failure_rows(final_fails, current)

        final_fails = law_of_absorption(final_fails)

    if final_fails is None:
        return np.zeros((0, 0), dtype=int)

    return final_fails


def fault_tree_analysis(arch, components, remove_src=0):
    """Run FAST's fault-tree analysis on an architecture matrix.

    Inputs:
        arch: Square architecture matrix. Positive column entries indicate
            which upstream components feed a component and the K/N trigger
            count for that gate.
        components: Dict with Name, FailMode, and FailRate arrays.
        remove_src: 1 to remove source outgoing connections before analysis.

    Outputs:
        Tuple of system failure probability and failure-mode name matrix.
    """

    arch = np.asarray(arch, dtype=float).copy()

    if arch.ndim != 2 or arch.shape[0] != arch.shape[1]:
        raise SafetyError("FaultTreeAnalysis architecture matrix must be square.")

    components = normalize_components(components)
    ncomp = arch.shape[0]

    if len(components["Name"]) != ncomp:
        raise SafetyError(
            "FaultTreeAnalysis component count must match architecture size."
        )

    conn = arch > 0
    ninput = np.sum(conn, axis=0)
    noutput = np.sum(conn, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ntrigger = np.sum(arch, axis=0) / ninput

    ntrigger[~np.isfinite(ntrigger)] = 0
    sources = np.where((ninput == 0) & (noutput > 0))[0]
    sinks = np.where((ninput > 0) & (noutput == 0))[0]

    if len(sinks) > 1:
        raise SafetyError("FaultTreeAnalysis found multiple sinks.")

    if len(sinks) == 0:
        raise SafetyError("FaultTreeAnalysis requires one sink.")

    if remove_src == 1:
        arch[sources, :] = 0

    arch_conns = [np.where(arch[:, index] > 0)[0].tolist() for index in range(ncomp)]
    components["FailID"] = np.arange(1, ncomp + 1)
    enum_modes = create_cut_sets(
        arch_conns,
        components,
        int(sinks[0]),
        ntrigger,
        ninput,
    )
    enum_modes = idempotent_law(enum_modes)
    enum_modes = law_of_absorption(enum_modes)
    fail_rates = np.ones(enum_modes.shape)
    fail_modes = [["" for _ in range(enum_modes.shape[1])] for _ in range(enum_modes.shape[0])]

    for icomp in range(ncomp):
        matches = enum_modes == icomp + 1
        fail_rates[matches] = components["FailRate"][icomp]

        for row, col in zip(*np.where(matches)):
            fail_modes[row][col] = components["Name"][icomp]

    return float(np.sum(np.prod(fail_rates, axis=1))), fail_modes


def failure_model(base_rate, exposure):
    """Return failure probability for baseline rates and exposure times."""

    return (1 - np.exp(-np.asarray(base_rate) * np.asarray(exposure))).tolist()


def component_database(name):
    """Return FAST's nominal failure rate for an electrical component."""

    lookup = {
        "turbineengine": 2.67e-6,
        "acgenerator": 130e-6,
        "battery": 93.1e-6,
        "electricmotor": 92.4e-6,
    }
    key = str(name).lower()

    if key not in lookup:
        raise SafetyError("ComponentDatabase component not included.")

    return {
        "Name": name,
        "FailRate": lookup[key],
    }


def normalize_components(components):
    """Return component fields as mutable Python lists."""

    normalized = {}

    for key in ("Name", "FailMode", "FailRate"):
        if key not in components:
            raise SafetyError(f"FaultTreeAnalysis components missing {key}.")

        value = components[key]

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, list):
            value = [value]

        normalized[key] = value

    normalized["FailRate"] = [float(item) for item in normalized["FailRate"]]
    return normalized


def append_failure_rows(first, second):
    """Append failure-mode matrices with zero padding as needed."""

    first = as_failure_matrix(first)
    second = as_failure_matrix(second)

    if first.size == 0:
        return second

    if second.size == 0:
        return first

    width = max(first.shape[1], second.shape[1])
    first = pad_failure_width(first, width)
    second = pad_failure_width(second, width)
    return np.vstack([first, second])


def pad_failure_width(matrix, width):
    """Pad a failure-mode matrix to a requested column count."""

    matrix = as_failure_matrix(matrix)

    if matrix.shape[1] == width:
        return matrix

    padding = np.zeros((matrix.shape[0], width - matrix.shape[1]), dtype=int)
    return np.hstack([matrix, padding])


def as_failure_matrix(value):
    """Return a failure-mode value as a two-dimensional integer matrix."""

    array = np.asarray(value, dtype=int)

    if array.size == 0:
        return np.zeros((0, 0), dtype=int)

    if array.ndim == 0:
        return array.reshape(1, 1)

    if array.ndim == 1:
        return array.reshape(-1, 1)

    return array


AndGate = and_gate
CompareCols = compare_cols
ComponentDatabase = component_database
CreateCutSets = create_cut_sets
EnumerateFailures = enumerate_failures
FailureModel = failure_model
FaultTreeAnalysis = fault_tree_analysis
IdempotentLaw = idempotent_law
LawOfAbsorption = law_of_absorption
