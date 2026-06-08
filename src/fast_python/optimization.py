# src/fast_python/optimization.py

"""Numerical optimization helpers ported from FAST OptimizationPkg."""

import time

import numpy as np


class OptimizationError(ValueError):
    """Report invalid or ill-posed OptimizationPkg inputs."""


def check_flag(main_struct, var):
    """Return 1 when a dict flag exists and is set to one, otherwise zero."""

    return 1 if main_struct.get(var) == 1 else 0


def feas_step(ng, s, ps):
    """Return the maximum feasible slack-variable step size."""

    tau = 0.005
    amax = 1.0
    s = np.asarray(s, dtype=float).reshape(-1)
    ps = np.asarray(ps, dtype=float).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        max_step = (tau - 1) * s / ps

    for index in range(int(ng)):
        if max_step[index] > 0:
            amax = min(amax, max_step[index])

    return float(amax)


def gauss_elim(matrix, prow, pcol):
    """Perform one Gaussian-elimination pivot step.

    Inputs:
        matrix: Matrix before elimination.
        prow: One-based pivot row index, matching OptimizationPkg.GaussElim.
        pcol: One-based pivot column index, matching OptimizationPkg.GaussElim.

    Outputs:
        Updated matrix with the pivot row normalized and all other rows
        eliminated in the pivot column.

    Assumptions:
        FAST's MATLAB file currently uses `size(A)` where `size(A, 1)` was
        intended. The Python port follows the documented algorithm.
    """

    result = np.asarray(matrix, dtype=float).copy()
    row = int(prow) - 1
    col = int(pcol) - 1
    pivot = result[row, col]
    result[row, :] = result[row, :] / pivot

    for irow in range(result.shape[0]):
        if irow == row:
            continue

        result[irow, :] = result[irow, :] - result[irow, col] * result[row, :]

    return result


def hess_upd(hessian, s, y):
    """Update a Hessian approximation with FAST's damped BFGS formula."""

    hessian = np.asarray(hessian, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    sy = (s.T @ y).item()
    hs = hessian @ s
    qterm = (hs.T @ s).item()

    if sy >= 0.2 * qterm:
        theta = 1.0
    else:
        theta = 0.8 * qterm / (qterm - sy)

    residual = theta * y + (1 - theta) * hs
    updated = hessian + residual @ residual.T / (residual.T @ s).item()
    updated = updated - hs @ hs.T / qterm
    return updated


def merit_function(obj_fun, x, con_fun=None, s=None, mu=None):
    """Evaluate FAST's line-search merit function."""

    fvalue, _, info = call_objective(obj_fun, x)

    if con_fun is not None:
        g, h = call_constraints(con_fun, x, info)
        g = zero_if_empty(g)
        h = zero_if_empty(h)
    else:
        g = np.asarray([0.0])
        h = np.asarray([0.0])

    if s is not None and len(np.asarray(g).reshape(-1)) > 0:
        s = np.asarray(s, dtype=float).reshape(-1)
        slack_penalty = mu * np.sum(np.log(s))
        rho = 100 * mu
    else:
        s = np.asarray([0.0])
        slack_penalty = 0
        rho = 1

    con_penalty = 0.5 * rho * (
        np.linalg.norm(h) ** 2 + np.linalg.norm(g + s) ** 2
    )
    return float(fvalue - slack_penalty + con_penalty)


def golden_section(f, x, p, amax=1.0e6, g=None, delta=0.1, tol=1.0e-3):
    """Run FAST's phase-I golden-section line search."""

    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    need_cons = g is not None

    while True:
        new_bounds = False
        abound = np.cumsum(delta * 1.618 ** np.arange(3))

        if np.any(abound > amax):
            delta = delta / 1.618
            continue

        fvalue = np.zeros(3)
        con_vals = None

        for index in range(3):
            xcand = x + abound[index] * p
            fvalue[index] = f(xcand)

            if need_cons:
                gvals, hvals = g(xcand)
                values = np.concatenate(
                    [
                        np.asarray(gvals, dtype=float).reshape(-1),
                        np.abs(np.asarray(hvals, dtype=float).reshape(-1)),
                    ]
                )

                if con_vals is None:
                    con_vals = np.zeros((len(values), 3))

                con_vals[:, index] = values

                if np.any(con_vals[:, index] > 0):
                    amax = abound[index]
                    delta = delta / 1.618
                    new_bounds = True
                    break

            if index == 1 and fvalue[1] - fvalue[0] > tol:
                return float(abound[0]), float(fvalue[0])

        if not new_bounds:
            break

    index = 3

    while fvalue[1] - fvalue[2] > 1.0e-6:
        abound[0] = abound[1]
        abound[1] = abound[2]
        fvalue[0] = fvalue[1]
        fvalue[1] = fvalue[2]
        abound[2] = abound[1] + delta * 1.618 ** index

        if abound[2] > amax:
            abound[2] = amax

        fvalue[2] = f(x + abound[2] * p)

        if need_cons:
            gvals, hvals = g(x + abound[2] * p)
            values = np.concatenate(
                [
                    np.asarray(gvals, dtype=float).reshape(-1),
                    np.abs(np.asarray(hvals, dtype=float).reshape(-1)),
                ]
            )

            if np.any(values > 0):
                abound[2] = abound[1]
                fvalue[2] = fvalue[1]
                break

        index += 1

    return float((abound[0] + abound[2]) / 2), None


def simplex_solve(tableau):
    """Solve a FAST linear-programming tableau with the simplex method."""

    matrix = np.asarray(tableau, dtype=float).copy()
    nrow, ncol = matrix.shape
    nvar = ncol - 1
    ncon = nrow - 1
    eps06 = 1.0e-06
    is_basic = np.abs(matrix[-1, :nvar]) < eps06
    basic_loc = np.cumsum(is_basic).astype(int)
    nbasic = int(basic_loc[-1])

    if (nvar - nbasic) != (nvar - ncon):
        raise OptimizationError("SimplexSolve received an invalid basis.")

    iterations = 0

    while np.any(matrix[-1, :nvar] < 0) and iterations < 10000:
        pcol = int(np.argmin(matrix[-1, :nvar]))
        is_positive = matrix[:ncon, pcol] > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            constraint_ratio = matrix[:ncon, -1] / matrix[:ncon, pcol]

        constraint_ratio[~is_positive] = np.inf
        prow = int(np.argmin(constraint_ratio))

        if constraint_ratio[prow] > 1.0e12:
            raise OptimizationError(
                "SimplexSolve solution is likely unbounded."
            )

        leaving = np.flatnonzero(basic_loc == prow + 1)

        if len(leaving) == 0:
            raise OptimizationError("SimplexSolve could not identify a basis row.")

        basic_loc[leaving[0]] = 0
        basic_loc[pcol] = prow + 1
        matrix = gauss_elim(matrix, prow + 1, pcol + 1)
        iterations += 1

    xopt = np.zeros(nvar)

    for ivar in range(nvar):
        if basic_loc[ivar] > 0:
            xopt[ivar] = matrix[basic_loc[ivar] - 1, -1]

    return xopt, iterations


def simplex_setup(aircraft, ielem):
    """Build the operational power-split simplex tableau."""

    specs = aircraft["Specs"]
    power = specs["Power"]
    propulsion = specs["Propulsion"]
    weights = specs["Weight"]
    history = aircraft["Mission"]["History"]["SI"]
    indices = np.asarray(ielem, dtype=int).reshape(-1) - 1
    efuel = power["SpecEnergy"]["Fuel"]
    ebatt = power["SpecEnergy"]["Batt"]
    eta_prop = propulsion["Eta"]["Prop"]
    eta_em = power["Eta"]["EM"]
    phi_design = power["Phi"]["SLS"]
    pw_em = power["P_W"]["EM"]
    wem = weights["EM"]
    wbatt = weights["Batt"]
    arch = propulsion["Arch"]["Type"]
    pout = history_array(history["Power"]["Out"], indices)
    _ = history_array(history["Power"]["Av"], indices)
    tsfc = history_array(history["Propulsion"]["TSFC"], indices)
    _ = history_array(history["Performance"]["Alt"], indices)
    time = history_array(history["Performance"]["Time"], indices)
    tas = history_array(history["Performance"]["TAS"], indices)
    dt = np.diff(time)
    ebatt_coeff = pout[:-1] * dt / eta_prop / eta_em

    if arch == "PHE":
        _ = pout / eta_prop
    elif arch == "SHE":
        _ = pout / eta_prop / eta_em
    else:
        raise OptimizationError(
            "SimplexSetup requires a PHE or SHE propulsion architecture."
        )

    pem_coeff = pout / eta_prop / eta_em
    pem_max = pw_em * wem
    ebatt_max = ebatt * wbatt
    obj_fun = aircraft["PowerOpt"]["ObjFun"]
    segments = listify_segments(aircraft["PowerOpt"]["Segments"])
    nphi = len(dt)

    if any("Takeoff" in segment for segment in segments):
        tas[0] = 1

    ncon = 4
    nvar = 5
    acon = 2
    avar = 2
    tableau = np.zeros((ncon * nphi + acon + 1, nvar * nphi + avar + 1))

    for iphi in range(nphi):
        irow = ncon * iphi
        icol = ncon * iphi + nphi
        tableau[irow, iphi] = -1
        tableau[irow, icol] = 1

        irow += 1
        icol += 1
        tableau[irow, iphi] = 1
        tableau[irow, icol] = 1
        tableau[irow, -1] = 0.9 * phi_design

    for iphi in range(nphi):
        irow = ncon * iphi + 2
        icol = ncon * iphi + nphi + 2
        tableau[irow, iphi] = -pem_coeff[iphi]
        tableau[irow, icol] = 1

        irow += 1
        icol += 1
        tableau[irow, iphi] = pem_coeff[iphi]
        tableau[irow, icol] = 1
        tableau[irow, -1] = pem_max

    irow = ncon * nphi
    icol = nvar * nphi
    tableau[irow, :nphi] = -ebatt_coeff[:nphi]
    tableau[irow, icol] = 1

    irow += 1
    icol += 1
    tableau[irow, :nphi] = ebatt_coeff[:nphi]
    tableau[irow, icol] = 1
    tableau[irow, -1] = ebatt_max

    if obj_fun == "FuelBurn":
        coefficients = -pout[:-1] * tsfc[:-1] * dt / tas[:-1]
    elif obj_fun == "Energy":
        coefficients = pout[:-1] * dt
        coefficients *= 1 / eta_prop / eta_em - tsfc[:-1] * efuel / tas[:-1]
    else:
        raise OptimizationError(
            "SimplexSetup objective must be 'FuelBurn' or 'Energy'."
        )

    coefficients[np.isnan(coefficients)] = 0
    tableau[-1, :nphi] = coefficients
    return tableau


def history_array(values, indices):
    """Return selected one-based mission-history values as floats."""

    return np.asarray(values, dtype=float).reshape(-1)[indices]


def listify_segments(value):
    """Return PowerOpt segment labels as a flat string list."""

    if isinstance(value, str):
        return [value]

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.reshape(-1).tolist()]

    return [str(item) for item in value]


def simplex_post(aircraft, ielem, phi_opt):
    """Write optimized operational power splits into mission history."""

    indices = np.asarray(ielem, dtype=int).reshape(-1)
    phi = np.asarray(phi_opt, dtype=float).reshape(-1)
    nphi = len(indices) - 1
    history = aircraft["Mission"]["History"]["SI"]["Power"]["Phi"]

    for index, value in zip(indices[:nphi], phi[:nphi]):
        history[int(index) - 1] = value

    return aircraft


def get_splits(aircraft, seg_beg, seg_end, lam_ts, lam_tsps, lam_psps, lam_pses):
    """Return optimized split schedules for a mission segment."""

    power_opt = aircraft["PowerOpt"]
    settings = aircraft["Settings"]
    seg_index = np.asarray(power_opt["SegIndex"], dtype=int).reshape(-1)
    lam_index = np.asarray(power_opt["LamIndex"], dtype=int).reshape(-1)
    splits = np.asarray(power_opt["Splits"], dtype=float).reshape(-1)
    npoint = int(power_opt["npoint"])
    seg_pts = np.flatnonzero((seg_beg <= seg_index) & (seg_index < seg_end))
    lam_ts = two_dimensional(lam_ts)
    lam_tsps = two_dimensional(lam_tsps)
    lam_psps = two_dimensional(lam_psps)
    lam_pses = two_dimensional(lam_pses)
    tsplit = 0

    if len(seg_pts) > 0:
        if power_opt["Settings"].get("OperTS") == 1:
            nsplit = int(settings["nargTS"])
            lam_ts = fill_split_values(lam_ts, splits, lam_index, seg_pts, npoint, nsplit, tsplit)
            tsplit += nsplit

        if power_opt["Settings"].get("OperTSPS") == 1:
            nsplit = int(settings["nargTSPS"])
            lam_tsps = fill_split_values(lam_tsps, splits, lam_index, seg_pts, npoint, nsplit, tsplit)
            tsplit += nsplit

        if power_opt["Settings"].get("OperPSPS") == 1:
            nsplit = int(settings["nargPSPS"])
            lam_psps = fill_split_values(lam_psps, splits, lam_index, seg_pts, npoint, nsplit, tsplit)
            tsplit += nsplit

        if power_opt["Settings"].get("OperPSES") == 1:
            nsplit = int(settings["nargPSES"])
            lam_pses = fill_split_values(lam_pses, splits, lam_index, seg_pts, npoint, nsplit, tsplit)

    return lam_ts, lam_tsps, lam_psps, lam_pses


def obj_power_management(x, need_grad, aircraft, analysis_runner=None):
    """Evaluate FAST's operational power-management objective."""

    if analysis_runner is None:
        from fast_python.analysis import eap_analysis

        analysis_runner = eap_analysis

    splits = np.asarray(x, dtype=float).reshape(-1)
    power_opt = aircraft["PowerOpt"]
    power_opt["Splits"] = splits.copy()
    designed = run_power_management_analysis(aircraft, analysis_runner)
    power_opt.setdefault("Results", {})["FlownAC"] = designed
    fvalue = power_management_objective(designed, power_opt["ObjFun"])
    update_power_management_constraints(aircraft, designed)

    if need_grad == 1:
        eps = 1.0e-06
        constraints = power_opt.setdefault("Constraints", {})
        constraints["EPS"] = eps
        nvar = int(power_opt["ndvars"])
        seg_index = np.asarray(designed["PowerOpt"]["SegIndex"], dtype=int).reshape(-1)
        nmiss = len(seg_index)
        grad = np.zeros(nvar)
        sen_pem = np.zeros((nmiss, nvar))
        sen_pem_av = np.zeros(nvar)
        sen_pav_gt = np.zeros(nvar)
        sen_crs_pow = np.zeros(nvar)
        sen_pgt_av = np.zeros((nmiss, nvar))
        sen_pgt = np.zeros((nmiss, nvar))
        sen_ebatt_av = np.zeros(nvar)
        sen_ebatt = np.zeros(nvar)
        cruise_index = first_segment_index(designed["Mission"]["History"]["Segment"], "Cruise")

        for ivar in range(nvar):
            power_opt["Splits"][ivar] += eps
            sensitivity = run_power_management_analysis(aircraft, analysis_runner)
            delta = power_management_objective(sensitivity, power_opt["ObjFun"])
            grad[ivar] = (delta - fvalue) / eps
            sen_pem_av[ivar] = electric_motor_power_available(sensitivity)
            sen_pem[:, ivar] = one_based_history_values(
                sensitivity["Mission"]["History"]["SI"]["Power"]["EM"],
                seg_index,
            )
            sen_pav_gt[ivar] = sensitivity["Mission"]["History"]["SI"]["Power"]["AvGT"][cruise_index]
            sen_crs_pow[ivar] = sensitivity["Mission"]["History"]["SI"]["Power"]["GT"][cruise_index]
            sen_pgt_av[:, ivar] = one_based_history_values(
                sensitivity["Mission"]["History"]["SI"]["Power"]["AvGT"],
                seg_index,
            )
            sen_pgt[:, ivar] = one_based_history_values(
                sensitivity["Mission"]["History"]["SI"]["Power"]["GT"],
                seg_index,
            )
            sen_ebatt_av[ivar] = battery_energy_available(sensitivity)
            sen_ebatt[ivar] = sensitivity["Mission"]["History"]["SI"]["Energy"]["Batt"][-1]
            power_opt["Splits"][ivar] -= eps

        constraints["SenPavGT"] = sen_pav_gt
        constraints["SenCrsPow"] = sen_crs_pow
        constraints["SenPemAv"] = sen_pem_av
        constraints["SenPem"] = sen_pem
        constraints["SenPgtAv"] = sen_pgt_av
        constraints["SenPgt"] = sen_pgt
        constraints["SenEbattAv"] = sen_ebatt_av
        constraints["SenEbatt"] = sen_ebatt
    else:
        grad = np.asarray([])

    return fvalue, grad, aircraft


def des_optimize(aircraft, analysis_runner=None, optimizer=None):
    """Run FAST's design and operation split optimizer orchestration."""

    if optimizer is None:
        optimizer = interior_point

    prepare_design_optimization(aircraft)
    ndvars = int(aircraft["PowerOpt"]["ndvars"])
    splits = np.zeros(ndvars)
    aircraft["PowerOpt"].setdefault("Results", {})["FlownAC"] = []
    start_time = time.perf_counter()
    xopt, fopt, split_hist, fhist, optim, feas, aircraft = optimizer(
        lambda values, need_grad: obj_power_management(
            values,
            need_grad,
            aircraft,
            analysis_runner,
        ),
        splits,
        lambda values, need_grad, info=None: con_size_opt(
            values,
            need_grad,
            aircraft if info is None else info,
        ),
    )
    aircraft["PowerOpt"]["WallTime"] = time.perf_counter() - start_time
    obj_fun = aircraft["PowerOpt"]["ObjFun"]

    if obj_fun.lower() == "energy":
        fopt = fopt * 7.4335e11
    elif obj_fun.lower() == "fuelburn":
        fopt = fopt * 17207

    results = aircraft["PowerOpt"].setdefault("Results", {})
    results["ObjFunVal"] = fopt
    results["OptParams"] = np.asarray(xopt, dtype=float)
    results["Optimality"] = optim
    results["Feasiblity"] = feas
    results["ParamHist"] = split_hist
    results["ObjFnHist"] = fhist
    return aircraft


def ops_optimize(
    aircraft,
    profile_fxn=None,
    mission_runner=None,
    tableau_builder=None,
    simplex_solver=None,
    post_processor=None,
):
    """Run FAST's operational power-split optimization driver."""

    if tableau_builder is None:
        tableau_builder = simplex_setup

    if simplex_solver is None:
        simplex_solver = simplex_solve

    if post_processor is None:
        post_processor = simplex_post

    phi = aircraft["Specs"]["Power"].setdefault("Phi", {})

    for segment in ["Tko", "Clb", "Crs", "Des", "Lnd"]:
        phi[segment] = 0

    max_iter = int(aircraft["PowerOpt"]["MaxIter"])
    tol = aircraft["PowerOpt"]["Tol"]
    iteration = 0
    aircraft["PowerOpt"]["PhiCount"] = 0
    start_time = time.perf_counter()
    old_phi = None

    while iteration < max_iter:
        aircraft = run_ops_mission(aircraft, profile_fxn, mission_runner)
        aircraft["PowerOpt"]["PhiCount"] = 1

        if iteration < 1:
            ielem, nphi = ops_optimization_indices(aircraft)

        tableau = tableau_builder(aircraft, ielem)
        phi_opt = simplex_solver(tableau)

        if isinstance(phi_opt, tuple):
            phi_opt = phi_opt[0]

        aircraft = post_processor(aircraft, ielem, phi_opt)
        aircraft["PowerOpt"]["LastObjFunVal"] = operational_objective_value(aircraft)
        current_phi = one_based_history_values(
            aircraft["Mission"]["History"]["SI"]["Power"]["Phi"],
            np.asarray(ielem[:nphi], dtype=int),
        )

        if iteration > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.abs(current_phi - old_phi) / old_phi

            rel_err[np.isnan(rel_err)] = 0

            if not np.any(rel_err > tol):
                aircraft["PowerOpt"].pop("PhiHist", None)
                break

        iteration += 1
        old_phi = current_phi
        aircraft["PowerOpt"]["PhiHist"] = list(
            aircraft["Mission"]["History"]["SI"]["Power"]["Phi"]
        )

    aircraft["PowerOpt"]["WallTime"] = time.perf_counter() - start_time
    aircraft["PowerOpt"]["Iter"] = iteration
    return aircraft


def run_ops_mission(aircraft, profile_fxn, mission_runner):
    """Call FlyMission or an injected mission runner for OpsOptimize."""

    if mission_runner is None:
        from fast_python.mission import fly_mission

        return fly_mission(aircraft)

    if profile_fxn is None:
        return mission_runner(aircraft)

    return mission_runner(aircraft, profile_fxn)


def ops_optimization_indices(aircraft):
    """Return OpsOptimize one-based mission indices and optimized count."""

    history = aircraft["Mission"]["History"]
    segments = listify_segments(aircraft["PowerOpt"]["Segments"])
    segment_history = history["Segment"]
    selected = np.zeros(len(segment_history), dtype=bool)

    for segment in segments:
        selected = selected | np.asarray(
            [str(item) == segment for item in segment_history],
            dtype=bool,
        )

    nphi = int(np.count_nonzero(selected))
    npnt = len(history["SI"]["Performance"]["Alt"])
    selected_indices = (np.flatnonzero(selected) + 1).tolist()

    if nphi == 0:
        raise OptimizationError("OpsOptimize found no mission points in requested segments.")

    if nphi == npnt:
        ielem = selected_indices
    else:
        ielem = selected_indices + [selected_indices[-1] + 1]

    return ielem, nphi


def operational_objective_value(aircraft):
    """Return OpsOptimize's unscaled objective for the current mission."""

    obj_fun = aircraft["PowerOpt"]["ObjFun"].lower()
    history = aircraft["Mission"]["History"]["SI"]

    if obj_fun == "doc":
        return 0

    if obj_fun == "fuelburn":
        return history["Weight"]["Fburn"][-1]

    if obj_fun == "energy":
        return history["Energy"]["Fuel"][-1] + history["Energy"]["Batt"][-1]

    raise OptimizationError(
        "OpsOptimize objective must be 'DOC', 'FuelBurn', or 'Energy'."
    )


def prepare_design_optimization(aircraft):
    """Populate DesOptimize split counts and mission/control-point indices."""

    settings = aircraft["Settings"]
    opt = aircraft["PowerOpt"]
    opt_settings = opt["Settings"]
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    flags = {
        "DesnTS": check_flag(opt_settings, "DesnTS"),
        "OperTS": check_flag(opt_settings, "OperTS"),
        "DesnTSPS": check_flag(opt_settings, "DesnTSPS"),
        "OperTSPS": check_flag(opt_settings, "OperTSPS"),
        "DesnPSPS": check_flag(opt_settings, "DesnPSPS"),
        "OperPSPS": check_flag(opt_settings, "OperPSPS"),
        "DesnPSES": check_flag(opt_settings, "DesnPSES"),
        "OperPSES": check_flag(opt_settings, "OperPSES"),
    }
    validate_design_optimization_inputs(settings, prop_arch, flags)
    count_operational_splits(aircraft, flags)
    ndesns = 0

    for name, flag_name in [
        ("TS", "DesnTS"),
        ("TSPS", "DesnTSPS"),
        ("PSPS", "DesnPSPS"),
        ("PSES", "DesnPSES"),
    ]:
        if flags[flag_name] == 1:
            ndesns += int(settings.get(f"narg{name}", 0))

    opt["ndesns"] = ndesns
    opt["ndvars"] = int(opt["nopers"]) + ndesns
    return aircraft


def validate_design_optimization_inputs(settings, prop_arch, flags):
    """Check DesOptimize split arguments and powertrain connectivity."""

    split_names = ["TS", "TSPS", "PSPS", "PSES"]

    for name in split_names:
        if (
            flags.get(f"Desn{name}", 0) == 1
            or flags.get(f"Oper{name}", 0) == 1
        ) and int(settings.get(f"narg{name}", 0)) < 1:
            raise OptimizationError(f"DesOptimize requested {name} splits without inputs.")

    sum_ts = np.asarray(prop_arch["TSPS"], dtype=float).shape[0]
    sum_tsps = np.sum(np.asarray(prop_arch["TSPS"], dtype=float), axis=1)
    sum_psps = np.sum(np.asarray(prop_arch["PSPS"], dtype=float), axis=1)
    sum_pses = np.sum(np.asarray(prop_arch["PSES"], dtype=float), axis=1)

    if (flags["DesnTS"] == 1 or flags["OperTS"] == 1) and sum_ts < 2:
        raise OptimizationError("DesOptimize requested TS splits with only one thrust source.")

    if (flags["DesnTSPS"] == 1 or flags["OperTSPS"] == 1) and not np.any(sum_tsps > 1):
        raise OptimizationError("DesOptimize requested TSPS splits without multi-input rows.")

    if (flags["DesnPSPS"] == 1 or flags["OperPSPS"] == 1) and not np.any(sum_psps > 1):
        raise OptimizationError("DesOptimize requested PSPS splits without multi-input rows.")

    if (flags["DesnPSES"] == 1 or flags["OperPSES"] == 1) and not np.any(sum_pses > 1):
        raise OptimizationError("DesOptimize requested PSES splits without multi-input rows.")


def count_operational_splits(aircraft, flags):
    """Populate DesOptimize operational split counts and index histories."""

    settings = aircraft["Settings"]
    opt = aircraft["PowerOpt"]
    nsplit = 0

    for name, flag_name in [
        ("TS", "OperTS"),
        ("TSPS", "OperTSPS"),
        ("PSPS", "OperPSPS"),
        ("PSES", "OperPSES"),
    ]:
        if flags[flag_name] == 1:
            nsplit += int(settings.get(f"narg{name}", 0))

    if nsplit == 0:
        opt["npoint"] = 0
        opt["nopers"] = 0
        opt["SegIndex"] = []
        opt["LamIndex"] = []
        return

    mission = aircraft["Mission"]["Profile"]

    if "PowerOpt" not in mission:
        raise OptimizationError("DesOptimize needs Mission.Profile.PowerOpt for operational splits.")

    power_opt = np.asarray(mission["PowerOpt"], dtype=int).reshape(-1)
    seg_opt = np.flatnonzero(power_opt > 0)
    npoint = int(np.sum(power_opt[seg_opt]))
    seg_index = []
    lam_index = []
    lam_counter = 1

    for segment in seg_opt:
        ibeg = int(np.asarray(mission["SegBeg"]).reshape(-1)[segment])
        iend = int(np.asarray(mission["SegEnd"]).reshape(-1)[segment])
        npnt = int(np.asarray(mission["SegPts"]).reshape(-1)[segment])
        nctrl = int(power_opt[segment])
        xpnt = np.linspace(0, 1 - (1 / (npnt - 1)), npnt - 1)
        xctrl = np.linspace(0, 1 - (1 / nctrl), nctrl)
        local_lam = np.zeros(npnt - 1, dtype=int)

        for xctrl_value in xctrl:
            local_lam[xpnt >= xctrl_value] = lam_counter
            lam_counter += 1

        seg_index.extend(range(ibeg, iend))
        lam_index.extend(local_lam.tolist())

    opt["npoint"] = npoint
    opt["nopers"] = nsplit * npoint
    opt["SegIndex"] = np.tile(np.asarray(seg_index, dtype=int), nsplit).tolist()
    opt["LamIndex"] = np.tile(np.asarray(lam_index, dtype=int), nsplit).tolist()


def run_power_management_analysis(aircraft, analysis_runner):
    """Call the analysis routine with ObjPowerManagement's FAST arguments."""

    settings = aircraft["Settings"]["Analysis"]
    return analysis_runner(aircraft, settings["Type"], settings["MaxIter"])


def power_management_objective(aircraft, obj_fun):
    """Return the scaled objective value from a flown aircraft."""

    history = aircraft["Mission"]["History"]["SI"]
    name = obj_fun.lower()

    if name == "doc":
        return 0

    if name == "fuelburn":
        return history["Weight"]["Fburn"][-1] / 17207

    if name == "energy":
        energy = history["Energy"]["Fuel"][-1] + history["Energy"]["Batt"][-1]
        return energy / 7.4335e11

    raise OptimizationError(
        "ObjPowerManagement objective must be 'DOC', 'FuelBurn', or 'Energy'."
    )


def update_power_management_constraints(aircraft, designed):
    """Store ObjPowerManagement constraint values on the original aircraft."""

    constraints = aircraft["PowerOpt"].setdefault("Constraints", {})
    seg_index = np.asarray(designed["PowerOpt"]["SegIndex"], dtype=int).reshape(-1)
    history = designed["Mission"]["History"]
    si = history["SI"]
    cruise_index = first_segment_index(history["Segment"], "Cruise")
    constraints["DesPemAv"] = electric_motor_power_available(designed)
    constraints["DesPem"] = one_based_history_values(si["Power"]["EM"], seg_index)
    constraints["DesPgtAv"] = one_based_history_values(si["Power"]["AvGT"], seg_index)
    constraints["DesPgt"] = one_based_history_values(si["Power"]["GT"], seg_index)
    constraints["DesEbattAv"] = battery_energy_available(designed)
    constraints["DesEbatt"] = si["Energy"]["Batt"][-1]
    constraints["DesPavGT"] = si["Power"]["AvGT"][cruise_index]
    constraints["DesCrsPow"] = si["Power"]["GT"][cruise_index]


def electric_motor_power_available(aircraft):
    """Return available electric motor power used by ObjPowerManagement."""

    specs = aircraft["Specs"]
    return specs["Power"]["P_W"]["EM"] * specs["Weight"]["EM"]


def battery_energy_available(aircraft):
    """Return available battery energy used by ObjPowerManagement."""

    specs = aircraft["Specs"]
    return specs["Power"]["SpecEnergy"]["Batt"] * specs["Weight"]["Batt"]


def one_based_history_values(values, indices):
    """Return history values selected by MATLAB one-based indices."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.asarray(indices, dtype=int).reshape(-1) - 1]


def first_segment_index(segments, name):
    """Return the zero-based index of the first matching mission segment."""

    for index, segment in enumerate(segments):
        if str(segment).lower() == name.lower():
            return index

    raise OptimizationError(f"Mission history has no {name} segment.")


def con_size_opt(x, need_grad, aircraft):
    """Return FAST design/operation split sizing constraints."""

    x = np.asarray(x, dtype=float).reshape(-1)
    opt = aircraft["PowerOpt"]
    opt_settings = opt["Settings"]
    constraints = opt.get("Constraints", {})
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    eps = 1.0e-06
    opt_desn_ts = check_flag(opt_settings, "DesnTS")
    opt_oper_ts = check_flag(opt_settings, "OperTS")
    opt_desn_tsps = check_flag(opt_settings, "DesnTSPS")
    opt_oper_tsps = check_flag(opt_settings, "OperTSPS")
    opt_desn_psps = check_flag(opt_settings, "DesnPSPS")
    opt_oper_psps = check_flag(opt_settings, "OperPSPS")
    opt_desn_pses = check_flag(opt_settings, "DesnPSES")
    opt_oper_pses = check_flag(opt_settings, "OperPSES")
    design_active = any([opt_desn_ts, opt_desn_tsps, opt_desn_psps, opt_desn_pses])
    oper_active = any([opt_oper_ts, opt_oper_tsps, opt_oper_psps, opt_oper_pses])
    nopers = int(opt.get("nopers", 0))

    g01a, g01b = power_limit_constraints(
        prop_arch_has(prop_arch, "PSType", 0),
        constraints,
        "DesPem",
        "DesPemAv",
        eps,
    )
    g02a, g02b = power_limit_constraints(
        prop_arch_has(prop_arch, "PSType", 1),
        constraints,
        "DesPgt",
        "DesPgtAv",
        eps,
    )
    g03a, g03b = power_limit_constraints(
        prop_arch_has(prop_arch, "ESType", 0),
        constraints,
        "DesEbatt",
        "DesEbattAv",
        eps,
    )

    if design_active:
        g04a = -x[nopers:]
        g04b = x[nopers:] - 1
    else:
        g04a = np.asarray([])
        g04b = np.asarray([])

    if oper_active:
        g05a = -x[:nopers]
        split_blocks = operational_split_constraint_blocks(
            x,
            aircraft,
            [
                ("TS", opt_oper_ts, opt_desn_ts),
                ("TSPS", opt_oper_tsps, opt_desn_tsps),
                ("PSPS", opt_oper_psps, opt_desn_psps),
                ("PSES", opt_oper_pses, opt_desn_pses),
            ],
            eps,
        )
    else:
        g05a = np.asarray([])
        split_blocks = {
            "TS": np.asarray([]),
            "TSPS": np.asarray([]),
            "PSPS": np.asarray([]),
            "PSES": np.asarray([]),
        }

    if aircraft["Settings"]["Analysis"]["Type"] == 1:
        g06 = sanitize_values(
            np.asarray(constraints["DesCrsPow"], dtype=float)
            / np.asarray(constraints["DesPavGT"], dtype=float)
            - 1,
            eps,
        )
    else:
        g06 = np.asarray([])

    g = concatenate_vectors(
        [
            g01a,
            g01b,
            g02a,
            g02b,
            g03a,
            g03b,
            g04a,
            g04b,
            g05a,
            split_blocks["TS"],
            split_blocks["TSPS"],
            split_blocks["PSPS"],
            split_blocks["PSES"],
            g06,
        ]
    )
    h = np.asarray([])

    if need_grad == 1:
        dgdx = con_size_opt_gradient(
            x,
            aircraft,
            constraints,
            prop_arch,
            eps,
            g,
            g01a,
            g01b,
            g02a,
            g02b,
            g03a,
            g03b,
            g06,
            design_active,
            oper_active,
            [
                ("TS", opt_oper_ts, opt_desn_ts),
                ("TSPS", opt_oper_tsps, opt_desn_tsps),
                ("PSPS", opt_oper_psps, opt_desn_psps),
                ("PSES", opt_oper_pses, opt_desn_pses),
            ],
        )
        dhdx = np.zeros((0, len(x)))
    else:
        dgdx = np.asarray([])
        dhdx = np.asarray([])

    return g, h, dgdx, dhdx


def con_size_opt_gradient(
    x,
    aircraft,
    constraints,
    prop_arch,
    eps,
    g,
    g01a,
    g01b,
    g02a,
    g02b,
    g03a,
    g03b,
    g06,
    design_active,
    oper_active,
    split_flags,
):
    """Return ConSizeOpt gradients using FAST's finite-difference fields."""

    ndvars = int(aircraft["PowerOpt"]["ndvars"])
    ndesns = int(aircraft["PowerOpt"].get("ndesns", 0))
    nopers = int(aircraft["PowerOpt"].get("nopers", 0))
    gradients = []
    gradients += power_limit_gradients(
        prop_arch_has(prop_arch, "PSType", 0),
        constraints,
        "SenPem",
        "SenPemAv",
        g01a,
        g01b,
        eps,
        ndvars,
    )
    gradients += power_limit_gradients(
        prop_arch_has(prop_arch, "PSType", 1),
        constraints,
        "SenPgt",
        "SenPgtAv",
        g02a,
        g02b,
        eps,
        ndvars,
    )
    gradients += power_limit_gradients(
        prop_arch_has(prop_arch, "ESType", 0),
        constraints,
        "SenEbatt",
        "SenEbattAv",
        g03a,
        g03b,
        eps,
        ndvars,
    )

    if design_active:
        dg04a = np.zeros((ndesns, ndvars))
        dg04b = np.zeros((ndesns, ndvars))
        dg04a[:, nopers:nopers + ndesns] = -np.eye(ndesns)
        dg04b[:, nopers:nopers + ndesns] = np.eye(ndesns)
    else:
        dg04a = np.zeros((0, ndvars))
        dg04b = np.zeros((0, ndvars))

    gradients.append(dg04a)
    gradients.append(dg04b)

    if oper_active:
        dg05a = np.zeros((nopers, ndvars))
        dg05a[:nopers, :nopers] = -np.eye(nopers)
        split_gradients = operational_split_gradient_blocks(x, aircraft, split_flags)
    else:
        dg05a = np.zeros((0, ndvars))
        split_gradients = {
            "TS": np.zeros((0, ndvars)),
            "TSPS": np.zeros((0, ndvars)),
            "PSPS": np.zeros((0, ndvars)),
            "PSES": np.zeros((0, ndvars)),
        }

    gradients.append(sanitize_gradient(dg05a))
    gradients.append(split_gradients["TS"])
    gradients.append(split_gradients["TSPS"])
    gradients.append(split_gradients["PSPS"])
    gradients.append(split_gradients["PSES"])

    if aircraft["Settings"]["Analysis"]["Type"] == 1:
        changed = (
            np.asarray(constraints["SenCrsPow"], dtype=float)
            / np.asarray(constraints["SenPavGT"], dtype=float)
            - 1
        )
        changed = sanitize_values(changed, eps)
        dg06dx = ((changed.reshape(1, -1) - g06.reshape(-1, 1)) / eps).reshape(1, ndvars)
    else:
        dg06dx = np.zeros((0, ndvars))

    gradients.append(dg06dx)
    result = concatenate_matrices(gradients, ndvars)

    if result.shape[0] != len(g):
        raise OptimizationError("ConSizeOpt gradient row count does not match constraints.")

    return result


def power_limit_constraints(active, constraints, used_key, available_key, eps):
    """Return paired lower/upper power or energy limit constraints."""

    if not active:
        return np.asarray([]), np.asarray([])

    used = np.asarray(constraints[used_key], dtype=float)
    available = np.asarray(constraints[available_key], dtype=float)
    lower = sanitize_values(-used / available, eps)
    upper = sanitize_values(used / available - 1, eps)
    return lower, upper


def power_limit_gradients(active, constraints, used_key, available_key, g_lower, g_upper, eps, ndvars):
    """Return finite-difference gradients for paired power limit constraints."""

    if not active:
        return [np.zeros((0, ndvars)), np.zeros((0, ndvars))]

    used = np.asarray(constraints[used_key], dtype=float)
    available = np.asarray(constraints[available_key], dtype=float)
    changed_lower = sanitize_array(-used / available, eps)
    changed_upper = sanitize_array(used / available - 1, eps)
    lower = (as_gradient_block(changed_lower, ndvars) - g_lower.reshape(-1, 1)) / eps
    upper = (as_gradient_block(changed_upper, ndvars) - g_upper.reshape(-1, 1)) / eps
    return [lower, upper]


def operational_split_constraint_blocks(x, aircraft, split_flags, eps):
    """Return grouped constraints bounding operation splits by design splits."""

    result = {}
    npoint = int(aircraft["PowerOpt"]["npoint"])
    nopers = int(aircraft["PowerOpt"]["nopers"])
    split_index = 0

    for name, oper_active, design_active in split_flags:
        nsplit = int(aircraft["Settings"].get(f"narg{name}", 0))

        if oper_active == 1:
            group_values = []

            for isplit in range(nsplit):
                start = split_index * npoint
                stop = start + npoint

                if design_active == 1:
                    design_value = x[nopers + split_index]
                    group_values.append(x[start:stop] / design_value - 1)
                else:
                    lam_max = aircraft["Specs"]["Power"][f"Lam{name}"]["SLS"]
                    group_values.append(x[start:stop] / lam_max - 1)

                split_index += 1

            result[name] = sanitize_values(np.concatenate(group_values), eps)
        else:
            result[name] = np.asarray([])

    return result


def operational_split_gradient_blocks(x, aircraft, split_flags):
    """Return grouped gradients for operation split bound constraints."""

    result = {}
    npoint = int(aircraft["PowerOpt"]["npoint"])
    nopers = int(aircraft["PowerOpt"]["nopers"])
    ndvars = int(aircraft["PowerOpt"]["ndvars"])
    split_index = 0

    for name, oper_active, design_active in split_flags:
        nsplit = int(aircraft["Settings"].get(f"narg{name}", 0))

        if oper_active == 1:
            block = np.zeros((npoint * nsplit, ndvars))

            for isplit in range(nsplit):
                row_start = isplit * npoint
                row_stop = row_start + npoint
                x_start = split_index * npoint
                x_stop = x_start + npoint

                if design_active == 1:
                    design_col = nopers + split_index
                    design_value = x[design_col]
                    block[row_start:row_stop, x_start:x_stop] = np.eye(npoint) / design_value
                    block[row_start:row_stop, design_col] = -x[x_start:x_stop] / design_value ** 2
                else:
                    lam_max = aircraft["Specs"]["Power"][f"Lam{name}"]["SLS"]
                    block[row_start:row_stop, x_start:x_stop] = np.eye(npoint) / lam_max

                split_index += 1

            result[name] = sanitize_gradient(block)
        else:
            result[name] = np.zeros((0, ndvars))

    return result


def prop_arch_has(prop_arch, field_name, value):
    """Return whether a PropArch type vector contains the requested code."""

    if field_name not in prop_arch:
        return False

    return bool(np.any(np.asarray(prop_arch[field_name], dtype=float).reshape(-1) == value))


def sanitize_values(values, eps):
    """Replace NaN and Inf values the way FAST constraint code does."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = eps
    return array


def sanitize_array(values, eps):
    """Replace NaN and Inf values while preserving array shape."""

    array = np.asarray(values, dtype=float)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = eps
    return array


def sanitize_gradient(values):
    """Replace NaN and Inf entries in split gradients as MATLAB does."""

    array = np.asarray(values, dtype=float)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 1
    return array


def as_gradient_block(values, ndvars):
    """Return finite-difference values as an nconstraint-by-ndvars matrix."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if len(array) == ndvars:
            array = array.reshape(1, ndvars)
        else:
            array = array.reshape(-1, 1)

    if array.shape[1] == 1 and ndvars != 1:
        array = np.repeat(array, ndvars, axis=1)

    return array.reshape(array.shape[0], ndvars)


def concatenate_vectors(pieces):
    """Concatenate one-dimensional constraint pieces with empty support."""

    nonempty = [np.asarray(piece, dtype=float).reshape(-1) for piece in pieces if len(np.asarray(piece).reshape(-1)) > 0]

    if not nonempty:
        return np.asarray([])

    return np.concatenate(nonempty)


def concatenate_matrices(pieces, ncol):
    """Concatenate gradient blocks with empty support."""

    nonempty = [np.asarray(piece, dtype=float).reshape(-1, ncol) for piece in pieces if np.asarray(piece).size > 0]

    if not nonempty:
        return np.zeros((0, ncol))

    return np.vstack(nonempty)


def interior_point(obj_fun, x0, con_fun=None):
    """Optimize with FAST's interior-point quasi-Newton routine."""

    eps = 1.0e-03
    max_iter = 99
    have_cons = con_fun is not None
    mu = 10
    rho = 0.50
    iteration = 0
    x = np.asarray(x0, dtype=float).reshape(-1)
    nx = len(x)
    hessian = np.eye(nx)
    xhist = [x.copy()]
    fhist = []
    lam = np.asarray([])
    sig = np.asarray([])
    slack = np.asarray([])
    ng = 0
    nh = 0
    feas = 0
    optim = np.inf
    info = None
    fvalue = np.nan

    while iteration < max_iter:
        fvalue, dfdx, info = objective_with_gradient(obj_fun, x)
        fhist.append(fvalue)
        dfdx = np.asarray(dfdx, dtype=float).reshape(-1)

        if have_cons:
            g, h, dgdx, dhdx = constraints_with_gradient(con_fun, x, info)
            feas = 0

            if len(h) > 0:
                feas = max(feas, float(np.max(np.abs(h))))
        else:
            g = np.asarray([])
            h = np.asarray([])
            dgdx = np.zeros((0, nx))
            dhdx = np.zeros((0, nx))
            feas = 0

        if iteration < 1:
            ng = len(g)
            nh = len(h)
            lam = np.zeros(nh)
            sig = np.zeros(ng)
            slack = np.ones(ng)

        if ng > 0:
            dldx_g = dgdx.T @ sig
        else:
            dldx_g = 0

        if nh > 0:
            dldx_h = dhdx.T @ lam
        else:
            dldx_h = 0

        dldx = dfdx + dldx_g + dldx_h
        optim = float(np.max(np.abs(dldx)))

        if optim < eps and feas < eps:
            break

        ndim = nx + nh + 2 * ng
        matrix = np.zeros((ndim, ndim))
        rhs = np.zeros(ndim)
        matrix[:nx, :nx] = hessian
        rhs[:nx] = -dldx

        if nh > 0:
            matrix[nx:nx + nh, :nx] = dhdx
            matrix[:nx, nx:nx + nh] = dhdx.T
            rhs[nx:nx + nh] = -h

        if ng > 0:
            g_start = nx + nh
            s_start = nx + nh + ng
            matrix[g_start:g_start + ng, :nx] = dgdx
            matrix[:nx, g_start:g_start + ng] = dgdx.T
            matrix[g_start:g_start + ng, s_start:] = np.eye(ng)
            matrix[s_start:, g_start:g_start + ng] = np.eye(ng)
            matrix[s_start:, s_start:] = np.diag(sig / slack)
            rhs[g_start:g_start + ng] = -(g + slack)
            rhs[s_start:] = -(sig - mu / slack)

        step = np.linalg.solve(matrix, rhs)
        px = step[:nx]
        plam = step[nx:nx + nh]
        psig = step[nx + nh:nx + nh + ng]
        ps = step[nx + nh + ng:]
        amaxs = feas_step(ng, slack, ps) if ng > 0 else 1.0
        amaxsig = feas_step(ng, sig, psig) if ng > 0 else 1.0

        if ng > 0:
            merit = lambda candidate: merit_function(obj_fun, candidate, con_fun, slack, mu)
        elif nh > 0:
            merit = lambda candidate: merit_function(obj_fun, candidate, con_fun)
        else:
            merit = lambda candidate: merit_function(obj_fun, candidate)

        ax, _ = golden_section(merit, x, px, amaxs)
        newx = x + ax * px
        news = slack + ax * ps
        newlam = lam + amaxsig * plam
        newsig = sig + amaxsig * psig
        delx = newx - x
        newf, newdfdx, info = objective_with_gradient(obj_fun, newx)
        newdfdx = np.asarray(newdfdx, dtype=float).reshape(-1)

        if have_cons:
            _, _, newdgdx, newdhdx = constraints_with_gradient(con_fun, newx, info)

            if ng > 0:
                newdldx_g = newdgdx.T @ newsig
                olddldx_g = dgdx.T @ newsig
            else:
                newdldx_g = 0
                olddldx_g = 0

            if nh > 0:
                newdldx_h = newdhdx.T @ newlam
                olddldx_h = dhdx.T @ newlam
            else:
                newdldx_h = 0
                olddldx_h = 0
        else:
            newdldx_g = 0
            olddldx_g = 0
            newdldx_h = 0
            olddldx_h = 0

        del_lagrangian = (newdfdx + newdldx_g + newdldx_h) - (
            dfdx + olddldx_g + olddldx_h
        )

        if iteration % 5 == 0:
            hessian = np.eye(nx)
        else:
            hessian = hess_upd(hessian, delx, del_lagrangian)

        if have_cons:
            mu = mu * rho

        iteration += 1
        x = newx
        slack = news
        lam = newlam
        sig = newsig
        fvalue = newf
        xhist.append(x.copy())

    return (
        x,
        float(fvalue),
        np.column_stack(xhist),
        np.asarray(fhist),
        optim,
        feas,
        info,
    )


def objective_with_gradient(obj_fun, x):
    """Call a FAST objective with gradients requested."""

    output = obj_fun(x, 1)
    values = list(output) + [None, None]
    return values[0], values[1], values[2]


def constraints_with_gradient(con_fun, x, info):
    """Call a FAST constraint function with gradients requested."""

    if info is None:
        output = con_fun(x, 1)
    else:
        output = con_fun(x, 1, info)

    values = list(output) + [None, None, None, None]
    g = zero_if_empty_constraints(values[0])
    h = zero_if_empty_constraints(values[1])
    dgdx = gradient_matrix(values[2], len(g), len(x))
    dhdx = gradient_matrix(values[3], len(h), len(x))
    return g, h, dgdx, dhdx


def zero_if_empty_constraints(value):
    """Return an empty vector when a constraint output is empty."""

    if value is None:
        return np.asarray([])

    array = np.asarray(value, dtype=float).reshape(-1)

    if len(array) == 0:
        return np.asarray([])

    return array


def gradient_matrix(value, nrow, ncol):
    """Return a two-dimensional constraint gradient matrix."""

    if value is None or nrow == 0:
        return np.zeros((nrow, ncol))

    return np.asarray(value, dtype=float).reshape(nrow, ncol)


def fill_split_values(target, splits, lam_index, seg_pts, npoint, nsplit, offset):
    """Fill one split array using FAST's flattened optimized split vector."""

    result = target.copy()

    for isplit in range(nsplit):
        for ipnt, seg_pt in enumerate(seg_pts):
            split_index = (offset + isplit) * npoint + lam_index[seg_pt] - 1
            result[ipnt, isplit] = splits[split_index]

    return result


def two_dimensional(value):
    """Return a numeric split array with explicit row and column axes."""

    array = np.asarray(value, dtype=float).copy()

    if array.ndim == 1:
        return array.reshape(-1, 1)

    return array


def call_objective(obj_fun, x):
    """Call an objective function and normalize FAST-style outputs."""

    output = obj_fun(x, 0)

    if isinstance(output, tuple):
        values = list(output) + [None, None]
        return values[0], values[1], values[2]

    return output, None, None


def call_constraints(con_fun, x, info):
    """Call constraints with optional objective-provided info."""

    if info is None:
        output = con_fun(x, 0)
    else:
        output = con_fun(x, 0, info)

    if isinstance(output, tuple):
        values = list(output) + [None]
        return values[0], values[1]

    return output


def zero_if_empty(value):
    """Return a numeric zero vector when a constraint output is empty."""

    if value is None:
        return np.asarray([0.0])

    array = np.asarray(value, dtype=float).reshape(-1)

    if len(array) == 0:
        return np.asarray([0.0])

    return array


CheckFlag = check_flag
ConSizeOpt = con_size_opt
DesOptimize = des_optimize
FeasStep = feas_step
GaussElim = gauss_elim
GetSplits = get_splits
GoldenSection = golden_section
HessUpd = hess_upd
InteriorPoint = interior_point
MeritFunction = merit_function
ObjPowerManagement = obj_power_management
OpsOptimize = ops_optimize
SimplexPost = simplex_post
SimplexSetup = simplex_setup
SimplexSolve = simplex_solve
