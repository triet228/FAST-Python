# src/fast_python/engine.py

"""Engine model helpers ported from FAST EngineModelPkg."""

from copy import deepcopy

import numpy as np

from fast_python.atmosphere import standard_atmosphere
from fast_python.units import convert_temperature, convert_tsfc


def pt_ps(ps, mach, gamma):
    """Return stagnation pressure from static pressure and Mach number."""

    return ps * (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2) ** (
        gamma / (gamma - 1)
    )


def ps_pt(pt, mach, gamma):
    """Return static pressure from stagnation pressure and Mach number."""

    return pt / (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2) ** (
        gamma / (gamma - 1)
    )


def tt_ts(ts, mach, gamma):
    """Return stagnation temperature from static temperature and Mach number."""

    return ts * (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2)


def ts_tt(tt, mach, gamma):
    """Return static temperature from stagnation temperature and Mach number."""

    return tt / (1 + (gamma - 1) / 2 * np.asarray(mach) ** 2)


def astar_a(area, mach, gamma):
    """Return choked area from area, Mach number, and gamma."""

    ratio = ((gamma + 1) / 2) ** (-(gamma + 1) / (2 * gamma - 2)) * (
        (1 + (gamma - 1) / 2 * mach ** 2) ** ((gamma + 1) / (2 * gamma - 2))
        / mach
    )
    return area / ratio


def a_astar(area_star, mach, gamma):
    """Return flow area from choked area, Mach number, and gamma."""

    ratio = ((gamma + 1) / 2) ** (-(gamma + 1) / (2 * gamma - 2)) * (
        (1 + (gamma - 1) / 2 * mach ** 2) ** ((gamma + 1) / (2 * gamma - 2))
        / mach
    )
    return area_star * ratio


def mass_flow_parameter(mach, gamma):
    """Return FAST's nondimensional mass-flow parameter."""

    gas_constant = 287
    return (
        np.sqrt(gamma / gas_constant)
        * mach
        * (1 + (gamma - 1) / 2 * mach ** 2) ** ((gamma + 1) / (2 * gamma - 2))
    )


def off_design_nozzle(area_1, area_2, mach_1, gamma):
    """Return ComponentOffPkg.Nozzle's exit Mach estimate."""

    area_star = astar_a(area_1, mach_1, gamma)
    mach_2 = mach_1
    prime = 1
    iteration = 0

    while abs(prime) > 1.0e-5 and iteration < 10:
        area_guess = a_astar(area_star, mach_2, gamma)
        residual = (area_guess - area_2) ** 2
        factor = gamma / 2 - 1 / 2
        denom = (gamma / 2 + 1 / 2) ** ((gamma + 1) / (2 * gamma - 2))
        exponent = (gamma + 1) / (2 * gamma - 2)
        term = factor * mach_2 ** 2 + 1
        area_term = area_star * term ** exponent / (mach_2 * denom)
        slope_term = area_star * term ** exponent / (mach_2 ** 2 * denom)
        slope_term -= (
            2
            * area_star
            * factor
            * term ** (exponent - 1)
            * (gamma + 1)
            / ((2 * gamma - 2) * denom)
        )
        prime = 2 * slope_term * (area_2 - area_term)
        mach_2 = mach_2 - residual / prime
        iteration += 1

    if mach_2 > 1:
        mach_2 = 1

    if mach_2 < 0:
        mach_2 = mach_1

    return mach_2


def rhos_rhot(rhot, mach, gamma):
    """Return static density from stagnation density and Mach number."""

    return rhot / (1 + (gamma - 1) / 2 * mach ** 2) ** (1 / (gamma - 1))


def cp_air(t_low, t_high=None):
    """Return air Cp or integrated heat between two temperatures.

    Inputs:
        t_low: Temperature in K when one argument is supplied, or beginning
            temperature when two arguments are supplied.
        t_high: Optional ending temperature in K.

    Outputs:
        Cp in J/(kg K) for one input, or integrated specific heat in J/kg for
        two inputs.

    Assumptions:
        Uses the same S-curve fit and antiderivative as
        EngineModelPkg.SpecHeatPkg.CpAir.
    """

    length = 233.0
    rate = 1 / 210
    midpoint = 875
    offset = 993
    t_low = np.asarray(t_low, dtype=float)

    if t_high is None:
        return restore_scalar_or_list(
            length / (1 + np.exp(-rate * (t_low - midpoint))) + offset
        )

    t_high = np.asarray(t_high, dtype=float)
    return restore_scalar_or_list(
        cp_antiderivative(t_high, length, rate, midpoint, offset)
        - cp_antiderivative(t_low, length, rate, midpoint, offset)
    )


def cp_jeta(t_low, t_high):
    """Return integrated Jet-A specific heat between two temperatures."""

    length = 4600
    rate = 1 / 410
    midpoint = 500
    offset = 100
    t_low = np.asarray(t_low, dtype=float)
    t_high = np.asarray(t_high, dtype=float)
    return restore_scalar_or_list(
        cp_antiderivative(t_high, length, rate, midpoint, offset)
        - cp_antiderivative(t_low, length, rate, midpoint, offset)
    )


def cv_air(t_low):
    """Return air Cv from FAST's fitted S-curve."""

    length = 233.0
    rate = 1 / 210
    midpoint = 875
    offset = 993 - 287
    t_low = np.asarray(t_low, dtype=float)
    return restore_scalar_or_list(
        length / (1 + np.exp(-rate * (t_low - midpoint))) + offset
    )


def new_gamma(tt, mach, gamma):
    """Iterate thermally perfect static temperature, Cp, Cv, and gamma."""

    delta_gamma = 1
    iteration = 0
    cp = None
    cv = None
    gamma_new = gamma
    ts = ts_tt(tt, mach, gamma_new)

    while delta_gamma > 1.0e-3 and iteration < 10:
        ts = ts_tt(tt, mach, gamma_new)
        cp = cp_air(ts)
        cv = cv_air(ts)
        gamma_next = cp / cv
        delta_gamma = abs(gamma_next - gamma_new) / gamma_new
        gamma_new = gamma_next
        iteration += 1

    return ts, cp, cv, gamma_new


def local_efficiency(reynolds):
    """Return FAST's local efficiency fit as a function of Reynolds number."""

    high = 1
    low = 0.75
    growth_rate = 1 / 2
    inflection = 7
    return low + (high - low) / (
        1 + np.exp(-growth_rate * (np.log10(reynolds) - inflection))
    )


def local_reynolds(flow_state):
    """Return local Reynolds number for an engine flow-state dictionary."""

    gas_constant = 287
    ts = flow_state["Ts"]
    length = flow_state["Ro"] - flow_state["Ri"]
    rho = flow_state["Ps"] / gas_constant / ts
    sigma = 350e-12
    boltzmann = 1.380639e-23
    molecular_mass = 0.02869 / 6.022e23
    viscosity = (
        1.106
        * 5
        / 16
        / sigma ** 2
        * np.sqrt(boltzmann * molecular_mass * ts / np.pi)
    )
    velocity = flow_state["Mach"] * np.sqrt(flow_state["Gam"] * ts * gas_constant)
    return rho * velocity * length / viscosity


def newton_raphson_tt1(tt1, heat):
    """Invert CpAir for heat added to air starting at tt1."""

    tt3 = tt1 * 1.05
    iteration = 0

    while abs(cp_air(tt1, tt3) - heat) / heat > 1.0e-3 and iteration < 10:
        tt3 = tt3 - (cp_air(tt3, tt1) + heat) / cp_air_inverse_prime(tt3)
        iteration += 1

    return tt3


def newton_raphson_tt3(tt1, heat):
    """Invert CpAir for heat removed from air starting at tt1."""

    tt3 = tt1 * 0.95
    iteration = 0

    while abs(cp_air(tt3, tt1) - heat) / heat > 1.0e-3 and iteration < 10:
        tt3 = tt3 - (cp_air(tt3, tt1) - heat) / cp_air_inverse_prime(tt3)
        iteration += 1

    return tt3


def diffuser(old_state, desired_mach, inner_outer, eta_poly):
    """Compute FAST's on-design diffuser state update.

    Inputs:
        old_state: Flow-state dictionary before diffusion.
        desired_mach: Target Mach number after diffusion.
        inner_outer: Use "Inner" to keep inner radius fixed, "Outer" to keep
            outer radius fixed.
        eta_poly: Engine efficiency dictionary containing Diffusers.

    Outputs:
        New flow-state dictionary.

    Assumptions:
        Mirrors EngineModelPkg.ComponentOnPkg.Diffuser, including the simple
        total-pressure efficiency loss.
    """

    new_state = deepcopy(old_state)
    area1 = old_state["Area"]
    mach1 = old_state["Mach"]
    gamma = old_state["Gam"]
    area_star = astar_a(area1, mach1, gamma)
    area2 = a_astar(area_star, desired_mach, gamma)
    ro1 = old_state["Ro"]
    ri1 = old_state["Ri"]

    if inner_outer == "Inner":
        ri2 = ri1
        ro2 = np.sqrt(ri2 ** 2 + area2 / np.pi)
    elif inner_outer == "Outer":
        ro2 = ro1
        ri2_sq = ro2 ** 2 - area2 / np.pi

        if np.real(ri2_sq) < 0:
            raise ValueError("New diffuser area cannot fit within outer radius.")

        ri2 = np.sqrt(ri2_sq)
    else:
        raise ValueError("inner_outer must be Inner or Outer.")

    ts, cp, cv, gamma = new_gamma(new_state["Tt"], desired_mach, old_state["Gam"])
    new_state["Ts"] = ts
    new_state["Cp"] = cp
    new_state["Cv"] = cv
    new_state["Gam"] = gamma
    new_state["Pt"] = old_state["Pt"] * eta_poly["Diffusers"]
    new_state["Ps"] = ps_pt(new_state["Pt"], desired_mach, new_state["Gam"])
    new_state["Mach"] = desired_mach
    new_state["Area"] = area2
    new_state["Ro"] = ro2
    new_state["Ri"] = ri2
    return new_state


def burner(state31, tt4_max, lhv_fuel, eta_poly):
    """Compute FAST's on-design burner state update.

    Inputs:
        state31: Post-bleed compressor-exit state dictionary.
        tt4_max: Maximum total temperature after combustion in K.
        lhv_fuel: Fuel lower heating value in J/kg.
        eta_poly: Engine efficiency dictionary containing Combustor.

    Outputs:
        Tuple of post-combustion state, fuel mass flow, and fuel-air ratio.

    Assumptions:
        Mirrors EngineModelPkg.ComponentOnPkg.Burner, including the internal
        combustor diffuser and fixed total-pressure loss.
    """

    mass31 = state31["MDot"]
    area31 = state31["Area"]
    pt31 = state31["Pt"]
    tt31 = state31["Tt"]
    mach31 = state31["Mach"]
    gamma31 = state31["Gam"]
    gas_constant = 287
    ts31 = ts_tt(tt31, mach31, gamma31)
    diffuser_speed = 40

    for _ in range(10):
        mach32 = diffuser_speed / np.sqrt(gamma31 * gas_constant * ts31)
        ts31 = ts_tt(tt31, mach32, gamma31)
        cp32 = cp_air(ts31)
        cv32 = cv_air(ts31)
        gamma32 = cp32 / cv32
        gamma31 = gamma32

    area_star = astar_a(area31, mach31, gamma31)
    area32 = a_astar(area_star, mach32, gamma32)
    area_ratio = area32 / area31
    blockage = 0.05
    eta_dm = 0.965 - 2.72 * blockage
    eta_dopt = (
        1 - 2 * eta_dm + (eta_dm * area_ratio) ** 2
    ) / (eta_dm * area_ratio ** 2 - eta_dm)
    pi_d = 1 - (1 - 1 / area_ratio ** 2) * (1 - eta_dopt) / (
        1 + 2 / gamma31 * mach31 ** 2
    )
    pt32 = pt31 * pi_d
    tt32 = tt31
    mass32 = mass31
    mfuel = mass32 * cp_air(tt32, tt4_max) / (
        eta_poly["Combustor"] * lhv_fuel - cp_jeta(tt32, tt4_max)
    )
    mass39 = mass31 + mfuel
    fuel_air_ratio = mfuel / mass32
    pt39 = pt32 * 0.95
    state39 = deepcopy(state31)
    state39["MDot"] = mass39
    state39["Pt"] = pt39
    state39["Tt"] = tt4_max
    state39["Mach"] = mach32
    ts, cp, cv, gamma = new_gamma(state39["Tt"], state39["Mach"], gamma32)
    state39["Ts"] = ts
    state39["Cp"] = cp
    state39["Cv"] = cv
    state39["Gam"] = gamma
    state39["Ps"] = ps_pt(state39["Pt"], state39["Mach"], state39["Gam"])
    state39["Area"] = area32
    state39["Ro"] = state31["Ro"]
    state39["Ri"] = np.sqrt(state39["Ro"] ** 2 - state39["Area"] / np.pi)
    return state39, mfuel, fuel_air_ratio


def comp_stage(old_state, eta_poly, stage_pr, rpm, fan=False):
    """Compute one FAST on-design compressor or fan stage.

    Inputs:
        old_state: Flow-state dictionary entering the stage.
        eta_poly: Engine efficiency dictionary.
        stage_pr: Stage total-pressure ratio.
        rpm: Shaft speed in revolutions per minute.
        fan: True to use EtaPoly.Fan; otherwise use Compressors.

    Outputs:
        Tuple of updated state, required work, and total-temperature ratio.

    Assumptions:
        Mirrors EngineModelPkg.ComponentOnPkg.CompStg, including compressor
        map bookkeeping fields on the returned stage state.
    """

    efficiency = eta_poly["Fan"] if fan else eta_poly["Compressors"]
    gas_constant = 287
    pressure_std = 101325.353
    temperature_std = 288.15
    gamma1 = old_state["Gam"]
    mach1 = old_state["Mach"]
    area1 = old_state["Area"]
    tt1 = old_state["Tt"]
    pt1 = old_state["Pt"]
    ts1 = old_state["Ts"]
    rho1 = old_state["Ps"] / ts1 / gas_constant
    tau = stage_pr ** ((gamma1 - 1) / gamma1)
    mach3 = mach1 * np.sqrt(
        1 / (tau * (1 + (gamma1 - 1) * mach1 ** 2 / 2) - (gamma1 - 1) * mach1 ** 2 / 2)
    )
    tt3 = tau * tt1
    ts3, cp3, cv3, gamma3 = new_gamma(tt3, mach3, gamma1)
    pt3 = stage_pr * pt1
    ps3 = ps_pt(pt3, mach3, gamma3)
    rho3 = ps3 / gas_constant / ts3
    area3 = area1 * rho1 / rho3
    ro3 = old_state["Ro"]
    ri3_sq = ro3 ** 2 - area3 / np.pi

    if np.real(ri3_sq) < 0:
        raise ValueError("Compressor stage area cannot fit within outer radius.")

    ri3 = np.sqrt(ri3_sq)
    work = cp_air(tt1, tt3) * old_state["MDot"] / efficiency
    new_state = deepcopy(old_state)
    new_state["MDot"] = old_state["MDot"]
    new_state["Pt"] = pt3
    new_state["Tt"] = tt3
    new_state["Ts"] = ts3
    new_state["Mach"] = mach3
    new_state["Cp"] = cp3
    new_state["Cv"] = cv3
    new_state["Gam"] = gamma3
    new_state["Ps"] = ps3
    new_state["Area"] = area3
    new_state["Ro"] = ro3
    new_state["Ri"] = ri3
    new_state["Rp"] = (ro3 + ri3) / 2
    omega = rpm / 60 * 2 * np.pi
    blade_speed = omega * new_state["Rp"]
    if abs(work) < 1.0e-12:
        new_state["Eta"] = np.nan
    else:
        new_state["Eta"] = (
            ts1 * stage_pr ** ((gamma3 - 1) / gamma3) * cp_air(ts1)
            - ts1 * cp_air(ts1)
        ) / work * new_state["MDot"]
    new_state["Psi"] = cp_air(tt1, tt3) / blade_speed ** 2
    new_state["MNorm"] = (
        new_state["MDot"] * np.sqrt(ts3 / temperature_std) / (new_state["Ps"] / pressure_std)
    )
    new_state["NNorm"] = rpm / np.sqrt(ts3 / temperature_std)
    new_state["Phi"] = (
        gas_constant
        * 60
        / (2 * np.pi * new_state["Area"] * new_state["Rp"])
        * (new_state["MNorm"] / pressure_std * np.sqrt(temperature_std))
        / (new_state["NNorm"] / np.sqrt(temperature_std))
    )
    new_state["Zeta"] = new_state["Psi"] / new_state["Eta"]
    return new_state, work, tau


def compressor(state21, compressor_pr, rpm, eta_poly):
    """Compute FAST's on-design multi-stage compressor wrapper."""

    gas_constant = 287
    pressure_std = 101325.353
    temperature_std = 288.15
    cur_state = deepcopy(state21)
    cur_state["Rp"] = (cur_state["Ro"] + cur_state["Ri"]) / 2
    cur_state["Eta"] = np.nan
    cur_state["Psi"] = np.nan
    cur_state["MNorm"] = (
        cur_state["MDot"] * np.sqrt(cur_state["Ts"] / temperature_std)
        / (cur_state["Ps"] / pressure_std)
    )
    cur_state["NNorm"] = rpm / np.sqrt(cur_state["Ts"] / temperature_std)
    cur_state["Phi"] = (
        gas_constant
        * 60
        / (2 * np.pi * cur_state["Area"] * cur_state["Rp"])
        * (cur_state["MNorm"] * np.sqrt(temperature_std) / pressure_std)
        / (cur_state["NNorm"] / np.sqrt(temperature_std))
    )
    cur_state["Zeta"] = cur_state["Psi"] / cur_state["Eta"]
    comp_object = {
        "States": {
            "Entry": deepcopy(cur_state),
        }
    }
    cur_pi = 1
    stage_prs = []
    stage_work = []
    stage_taus = []
    stage_index = 1

    while cur_pi < compressor_pr and stage_index < 100:
        if compressor_pr / cur_pi > np.sqrt(2):
            stage_pr = np.sqrt(2)
        else:
            stage_pr = compressor_pr / cur_pi

        cur_state, work, tau = comp_stage(
            cur_state,
            eta_poly,
            stage_pr,
            rpm,
            False,
        )
        stage_prs.append(stage_pr)
        stage_work.append(work)
        stage_taus.append(tau)
        cur_pi = float(np.prod(stage_prs))
        comp_object["States"][f"Stage_{stage_index}"] = deepcopy(cur_state)
        stage_index += 1

    nstages = stage_index - 1
    comp_object["States"]["Exit"] = deepcopy(comp_object["States"][f"Stage_{nstages}"])
    pi_comp = float(np.prod(stage_prs))
    tau_comp = float(np.prod(stage_taus))
    comp_work = float(np.sum(stage_work))
    state3 = deepcopy(comp_object["States"]["Exit"])

    for field_name in ("Zeta", "Eta", "Psi", "Phi", "MNorm", "Rp", "NNorm"):
        state3.pop(field_name, None)

    comp_object["AdiabaticEfficiency"] = eta_poly["Compressors"]
    comp_object["NoStages"] = nstages
    comp_object["Pi"] = pi_comp
    comp_object["Tau"] = tau_comp
    comp_object["ReqWork"] = comp_work
    comp_object["RPM"] = rpm
    return state3, comp_object


def turb_stage(old_state, tt3, rpm, choked, eta_poly):
    """Compute one FAST on-design turbine stage."""

    gas_constant = 287
    mach1 = old_state["Mach"]
    gamma1 = old_state["Gam"]
    tt1 = old_state["Tt"]
    eta_turbine = eta_poly["Turbines"]
    ts1 = ts_tt(tt1, mach1, gamma1)
    u1 = mach1 * np.sqrt(gamma1 * gas_constant * ts1)

    if choked:
        mach2 = 1.1
    else:
        mach2 = 0.9

    tt2 = tt1
    gamma2 = gamma1
    omega = rpm / 60 * 2 * np.pi
    radius_pitch = 0.5 * (old_state["Ro"] + old_state["Ri"])
    psi = cp_air(tt3, tt2) / (omega * radius_pitch) ** 2
    tau = tt3 / tt2
    pi_stage = tau ** (gamma2 / (gamma2 - 1) * eta_turbine)
    velocity_prime = np.sqrt(cp_air(tt1) * tt1)
    velocity2 = velocity_prime * np.sqrt(
        (gamma1 - 1) * mach2 ** 2 / (1 + (gamma1 - 1) / 2 * mach2 ** 2)
    )
    alpha2 = np.arccos(u1 / velocity2)
    mach3 = mach2 * np.cos(alpha2) / np.sqrt(
        1
        - (1 - tau)
        * (1 - psi / 2)
        * (1 + (gamma2 - 1) / 2 * mach2 ** 2)
    )
    pt2 = old_state["Pt"]
    pt3 = pi_stage * pt2
    new_state = deepcopy(old_state)
    new_state["Tt"] = tt3
    new_state["Pt"] = pt3
    new_state["Mach"] = mach3
    ts, cp, cv, gamma = new_gamma(new_state["Tt"], new_state["Mach"], gamma2)
    new_state["Ts"] = ts
    new_state["Cp"] = cp
    new_state["Cv"] = cv
    new_state["Gam"] = gamma
    new_state["Ps"] = ps_pt(new_state["Pt"], new_state["Mach"], new_state["Gam"])
    u3 = mach3 * np.sqrt(new_state["Ts"] * new_state["Gam"] * gas_constant)
    rho3 = new_state["Ps"] / new_state["Ts"] / gas_constant
    new_state["Area"] = old_state["MDot"] / u3 / rho3
    new_state["Ri"] = old_state["Ri"]
    new_state["Ro"] = np.sqrt(new_state["Ri"] ** 2 + new_state["Area"] / np.pi)
    return new_state, pi_stage, tau


def turbine(old_state, compressor_object, eta_poly, ambient=None):
    """Compute FAST's on-design multi-stage turbine wrapper."""

    rpm = compressor_object["RPM"]
    tt1 = old_state["Tt"]
    radius_pitch = 0.5 * (old_state["Ro"] + old_state["Ri"])
    omega = rpm * 2 * np.pi / 60

    if ambient is not None:
        pi_turb = ambient["Pt"] / old_state["Pt"]
        gamma = old_state["Gam"]
        tau_turb = pi_turb ** ((gamma - 1) / gamma)
        tt3 = tau_turb * old_state["Tt"]
        turbine_work = cp_air(tt3, tt1) / eta_poly["Turbines"]
    else:
        turbine_work = (
            compressor_object["ReqWork"] / old_state["MDot"] / eta_poly["Turbines"]
        )
        tt3 = newton_raphson_tt3(tt1, turbine_work)

    loading = turbine_work / (omega * radius_pitch) ** 2
    nstages = int(np.real(np.ceil(loading / 2)))

    if nstages > 7:
        nstages = 7

    if nstages < 1:
        nstages = 1

    dtemp = (tt1 - tt3) / nstages
    cur_state = deepcopy(old_state)
    turbine_object = {
        "States": {
            "Entry": deepcopy(cur_state),
        }
    }
    pis = []
    taus = []
    cur_state, pi_stage, tau_stage = turb_stage(
        cur_state,
        tt1 - dtemp,
        rpm,
        True,
        eta_poly,
    )
    pis.append(pi_stage)
    taus.append(tau_stage)
    turbine_object["States"]["Stage_1"] = deepcopy(cur_state)

    for stage_index in range(2, nstages + 1):
        stage_tt = cur_state["Tt"] - dtemp
        cur_state, pi_stage, tau_stage = turb_stage(
            cur_state,
            stage_tt,
            rpm,
            False,
            eta_poly,
        )
        pis.append(pi_stage)
        taus.append(tau_stage)
        turbine_object["States"][f"Stage_{stage_index}"] = deepcopy(cur_state)

    turbine_object["States"]["Exit"] = deepcopy(
        turbine_object["States"][f"Stage_{nstages}"]
    )
    turbine_object["NoStages"] = nstages
    turbine_object["CPR"] = float(np.prod(pis))
    turbine_object["CTR"] = float(np.prod(taus))
    turbine_object["DelivWork"] = turbine_work * cur_state["MDot"]
    turbine_object["RPM"] = rpm
    return cur_state, turbine_object


def perf_ex_nozzle(old_state, ambient, eta_poly, nozzle_type):
    """Compute FAST's on-design perfect-expansion nozzle approximation."""

    if nozzle_type == "Bypass":
        npr = 1.1
    elif nozzle_type == "Core":
        npr = 1.3
    elif nozzle_type == "Prop":
        npr = 1
    else:
        raise ValueError("nozzle_type must be Core, Bypass, or Prop.")

    gas_constant = 287
    ambient_ps = ps_pt(ambient["Pt"], ambient["Mach"], ambient["Gam"])
    ps9 = ambient_ps * npr
    tt9 = old_state["Tt"]
    pt5 = old_state["Pt"]
    gamma = old_state["Gam"]
    cp = old_state["Cp"]
    efficiency = eta_poly.get(
        "Nozzles",
        eta_poly.get(f"{nozzle_type}Nozzle"),
    )

    if efficiency is None:
        raise ValueError("EtaPoly must define Nozzles or a type-specific nozzle efficiency.")

    ts9_ideal = tt9 * (ps9 / pt5) ** ((gamma - 1) / gamma)
    u9_ideal = np.sqrt((tt9 - ts9_ideal) * 2 * cp)

    if np.real(u9_ideal ** 2) < 0:
        raise ValueError("Cannot expand air to atmospheric pressure.")

    u9 = u9_ideal * efficiency
    ts9 = tt9 - u9 ** 2 / 2 / cp
    pt9 = ps9 * (tt9 / ts9) ** (gamma / (gamma - 1))
    mach9 = u9 / np.sqrt(gamma * gas_constant * ts9)

    if mach9 > 1:
        mach9 = 1
        u9 = mach9 * np.sqrt(gamma * gas_constant * ts9)
        ts9 = tt9 - u9 ** 2 / 2 / cp
        ps9 = pt9 / (tt9 / ts9) ** (gamma / (gamma - 1))

    new_state = deepcopy(old_state)
    new_state["Tt"] = tt9
    new_state["Ts"] = ts9
    new_state["Pt"] = pt9
    new_state["Ps"] = ps9
    new_state["Mach"] = mach9
    new_state["Cp"] = cp_air(ts9)
    new_state["Cv"] = cv_air(ts9)
    new_state["Gam"] = new_state["Cp"] / new_state["Cv"]
    rho9 = ps9 / ts9 / gas_constant
    new_state["Area"] = new_state["MDot"] / u9 / rho9
    thrust = new_state["MDot"] * u9 + (ps9 - ambient_ps) * new_state["Area"]

    if nozzle_type == "Bypass":
        new_state["Ro"] = np.sqrt(new_state["Area"] / np.pi - new_state["Ri"] ** 2)
    else:
        new_state["Ri"] = 0
        new_state["Ro"] = np.sqrt(new_state["Area"] / np.pi)

    return new_state, thrust


def fan_system(state1, fan_sys_object, opr, spools, bpr, eta_poly):
    """Compute FAST's on-design turbofan fan/LPC system.

    Inputs:
        state1: Post-inlet flow-state dictionary.
        fan_sys_object: Fan architecture dictionary with geared/boosted flags.
        opr: Overall pressure ratio requested by the engine spec.
        spools: Dictionary containing Count and RPM entries.
        bpr: Engine bypass ratio.
        eta_poly: Engine efficiency dictionary.

    Outputs:
        Tuple of station 2, station 21, station 13, station 25, and updated
        fan-system object dictionaries.

    Assumptions:
        Mirrors ComponentOnPkg.FanSystem, including the empirical half-factor
        used when sizing fan-shaft LPC/booster compression.
    """

    fan_sys_object = deepcopy(fan_sys_object)
    fan_pressure_ratio = fan_sys_object["FanObject"]["Pi"]
    cur_state, fan_work, tau_fan = comp_stage(
        state1,
        eta_poly,
        fan_pressure_ratio,
        fan_sys_object["FanObject"]["RPM"],
        True,
    )
    state2 = deepcopy(cur_state)
    state21 = deepcopy(cur_state)
    state21["MDot"] = cur_state["MDot"] / (1 + bpr)
    state21["Area"] = cur_state["Area"] / (1 + bpr)
    state13 = deepcopy(state21)
    state13["MDot"] = state21["MDot"] * bpr
    state13["Area"] = state21["Area"] * bpr
    state21["Ro"] = np.sqrt(state21["Area"] / np.pi)
    state21["Ri"] = 0
    state13["Ri"] = state21["Ro"]
    state13["Ro"] = np.sqrt(state13["Area"] / np.pi + state13["Ri"] ** 2)
    fan_sys_object["FanObject"]["NoStages"] = 1
    fan_sys_object["FanObject"]["ReqWork"] = fan_work
    fan_sys_object["FanObject"]["Tau"] = tau_fan
    pi_n1 = 0.5 * (opr / fan_pressure_ratio) ** (1 / spools["Count"])

    if fan_sys_object["Boosted"] and fan_sys_object["Geared"]:
        raise ValueError(
            "A turbofan cannot have both a booster and a geared fan."
        )

    if fan_sys_object["Boosted"]:
        booster_rpm = fan_sys_object["BoosterObject"]["RPM"]
        state25, fan_sys_object["BoosterObject"] = compressor(
            state21,
            pi_n1,
            booster_rpm,
            eta_poly,
        )
        fan_sys_object["ReqWork"] = (
            fan_sys_object["FanObject"]["ReqWork"]
            + fan_sys_object["BoosterObject"]["ReqWork"]
        )
        fan_sys_object["Tau"] = (
            fan_sys_object["FanObject"]["Tau"]
            * fan_sys_object["BoosterObject"]["Tau"]
        )
        fan_sys_object["Pi"] = (
            fan_sys_object["FanObject"]["Pi"]
            * fan_sys_object["BoosterObject"]["Pi"]
        )
        fan_sys_object["RPM"] = booster_rpm
    elif fan_sys_object["Geared"]:
        lpc_rpm = fan_sys_object["LPCObject"]["RPM"]
        state25, fan_sys_object["LPCObject"] = compressor(
            state21,
            pi_n1,
            lpc_rpm,
            eta_poly,
        )
        fan_sys_object["ReqWork"] = (
            fan_sys_object["FanObject"]["ReqWork"]
            + fan_sys_object["LPCObject"]["ReqWork"]
        )
        fan_sys_object["Tau"] = (
            fan_sys_object["FanObject"]["Tau"]
            * fan_sys_object["LPCObject"]["Tau"]
        )
        fan_sys_object["Pi"] = (
            fan_sys_object["FanObject"]["Pi"]
            * fan_sys_object["LPCObject"]["Pi"]
        )
        fan_sys_object["RPM"] = lpc_rpm
    else:
        state25 = deepcopy(state21)
        fan_sys_object["ReqWork"] = fan_sys_object["FanObject"]["ReqWork"]
        fan_sys_object["Tau"] = fan_sys_object["FanObject"]["Tau"]
        fan_sys_object["Pi"] = fan_sys_object["FanObject"]["Pi"]
        fan_sys_object["RPM"] = fan_sys_object["FanObject"]["RPM"]

    return state2, state21, state13, state25, fan_sys_object


def make_fan_system_object(engine_spec, spools):
    """Return the primitive FanSysObject used by the turbofan cycle."""

    gear_ratio = engine_spec.get("FanGearRatio", np.nan)
    fan_sys_object = {
        "Geared": False,
        "Boosted": False,
        "GearRatio": 1,
        "FanObject": {},
        "BoosterObject": {},
        "LPCObject": {},
    }

    if is_nan_scalar(gear_ratio) or gear_ratio == 1:
        fan_sys_object["Geared"] = False
        fan_sys_object["GearRatio"] = 1
        fan_sys_object["FanObject"]["RPM"] = spools["RPM"][0]
        fan_sys_object["LPCObject"] = "Nonexistent"
    else:
        fan_sys_object["Geared"] = True
        fan_sys_object["GearRatio"] = gear_ratio
        fan_sys_object["FanObject"]["RPM"] = spools["RPM"][0] / gear_ratio
        fan_sys_object["LPCObject"]["RPM"] = spools["RPM"][0]

    if engine_spec.get("FanBoosters", False):
        fan_sys_object["Boosted"] = True
        fan_sys_object["BoosterObject"]["RPM"] = spools["RPM"][0]
    else:
        fan_sys_object["Boosted"] = False
        fan_sys_object["BoosterObject"] = "Nonexistent"

    return fan_sys_object


def turbofan_on_design_cycle(engine_spec, mdot_input, elec_power=None):
    """Run FAST's turbofan on-design thermodynamic cycle.

    Inputs:
        engine_spec: Turbofan engine specification dictionary.
        mdot_input: Total streamtube air mass flow in kg/s.
        elec_power: Optional electric power supplied to the low-pressure
            turbine shaft in watts.

    Outputs:
        Engine object dictionary with flow states, turbomachinery objects,
        thrust, fuel, efficiency, TSFC, EGT, and fan diameter.

    Assumptions:
        Mirrors CycleModelPkg.TurbofanOnDesignCycle. The three-spool IPT
        branch follows the MATLAB implementation exactly, including its use
        of the HPC object when computing the intermediate turbine.
    """

    rpm = as_vector(engine_spec["RPMs"])

    if engine_spec["NoSpools"] != len(rpm):
        raise ValueError("Number of spools must match number of input RPMs.")

    spools = {
        "Count": int(engine_spec["NoSpools"]),
        "RPM": rpm,
    }
    eta_poly = engine_spec["EtaPoly"]
    opr = engine_spec["OPR"]
    tt4_max = engine_spec["Tt4Max"]
    lhv_fuel = 43.17e6
    gas_constant = 287
    mach0 = engine_spec["Mach"]
    ts0, ps0, rho0 = standard_atmosphere(engine_spec["Alt"])
    cp0 = cp_air(ts0)
    cv0 = cv_air(ts0)
    gamma0 = cp0 / cv0
    pt0 = pt_ps(ps0, mach0, gamma0)
    tt0 = tt_ts(ts0, mach0, gamma0)
    mdot0 = mdot_input
    u0 = mach0 * np.sqrt(gamma0 * ts0 * gas_constant)
    area0 = mdot0 / rho0 / u0
    ro0 = np.sqrt(area0 / np.pi)
    state0 = {
        "MDot": mdot0,
        "Pt": pt0,
        "Ps": ps0,
        "Tt": tt0,
        "Ts": ts0,
        "Mach": mach0,
        "Cp": cp0,
        "Cv": cv0,
        "Gam": gamma0,
        "Area": area0,
        "Ro": ro0,
        "Ri": 0,
    }
    ram_drag = u0 * mdot0
    state1 = diffuser(state0, 0.5, "Inner", eta_poly)
    fan_sys_object = make_fan_system_object(engine_spec, spools)
    fan_sys_object["FanObject"]["Pi"] = engine_spec["FPR"]
    state2, state21, state13, state25, fan_sys_object = fan_system(
        state1,
        fan_sys_object,
        opr,
        spools,
        engine_spec["BPR"],
        eta_poly,
    )
    pi_fan = fan_sys_object["Pi"]

    if spools["Count"] == 3:
        pi_ipc = (opr / pi_fan) ** (1 / (spools["Count"] - 1))
        state26, ipc_object = compressor(state25, pi_ipc, spools["RPM"][1], eta_poly)
        pi_ipc = ipc_object["Pi"]
    else:
        state26 = deepcopy(state25)
        ipc_object = "Nonexistent"
        pi_ipc = 1

    pi_hpc = opr / pi_fan / pi_ipc
    state3, hpc_object = compressor(state26, pi_hpc, spools["RPM"][-1], eta_poly)
    opr_actual = fan_sys_object["Pi"] * pi_ipc * hpc_object["Pi"]
    core_flow = engine_spec["CoreFlow"]
    pax_bleed = core_flow["PaxBleed"]
    leakage = core_flow["Leakage"]
    cooling = core_flow["Cooling"]
    state31 = deepcopy(state3)
    state31["MDot"] = state3["MDot"] * (1 - cooling - pax_bleed - leakage)
    state_coolant = deepcopy(state3)
    state_coolant["MDot"] = state_coolant["MDot"] * cooling
    state39, mdot_fuel, far = burner(state31, tt4_max, lhv_fuel, eta_poly)
    state4 = diffuser(state39, 0.4, "Outer", eta_poly)
    state41 = deepcopy(state4)
    state41["MDot"] = state41["MDot"] + state_coolant["MDot"]
    mixed_guess = (
        state4["Tt"] * state4["MDot"] / state41["MDot"]
        + state_coolant["Tt"] * state_coolant["MDot"] / state41["MDot"]
    )
    enthalpy41 = (
        state4["Cp"] * state4["Tt"] * state4["MDot"]
        + state_coolant["Cp"] * state_coolant["Tt"] * state_coolant["MDot"]
    )
    state41["Tt"] = enthalpy41 / state41["MDot"] / cp_air(mixed_guess)
    state41["Pt"] = state41["Pt"] * 0.95
    state41["Cp"] = cp_air(state41["Tt"])
    state41["Cv"] = cv_air(state41["Tt"])
    state41["Gam"] = state41["Cp"] / state41["Cv"]
    state5, hpt_object = turbine(state41, hpc_object, eta_poly)

    if spools["Count"] == 3:
        state55, ipt_object = turbine(state5, hpc_object, eta_poly)
    else:
        ipt_object = "Nonexistent"
        state55 = deepcopy(state5)

    fan_sys_object["TotalWork"] = fan_sys_object["ReqWork"]

    if elec_power is None:
        fan_sys_object["ElecWork"] = 0
    else:
        fan_sys_object["ElecWork"] = elec_power
        fan_sys_object["ReqWork"] = fan_sys_object["ReqWork"] - elec_power

        if fan_sys_object["ReqWork"] < 0:
            raise ValueError("Electric power input exceeds low pressure turbine power.")

    state6, lpt_object = turbine(state55, fan_sys_object, eta_poly)
    state19, bypass_thrust = perf_ex_nozzle(state13, state0, eta_poly, "Bypass")
    state9, core_thrust = perf_ex_nozzle(state6, state0, eta_poly, "Core")
    thrust = {
        "Net": bypass_thrust + core_thrust - ram_drag,
        "Bypass": bypass_thrust,
        "Core": core_thrust,
        "RamDrag": -ram_drag,
    }
    u9 = state9["Mach"] * np.sqrt(state9["Gam"] * gas_constant * state9["Ts"])
    u19 = state19["Mach"] * np.sqrt(state19["Gam"] * gas_constant * state19["Ts"])
    kinetic_power = (
        state9["MDot"] * u9 ** 2
        + state19["MDot"] * u19 ** 2
        - state0["MDot"] * u0 ** 2
    ) / 2
    thermal_efficiency = kinetic_power / lhv_fuel / mdot_fuel
    propulsive_efficiency = thrust["Net"] * u0 / kinetic_power
    efficiency = {
        "Thermal": thermal_efficiency,
        "Propulsive": propulsive_efficiency,
        "Overall": thermal_efficiency * propulsive_efficiency,
    }
    lam = fan_sys_object["ReqWork"] / fan_sys_object["TotalWork"]
    thrust_corr = (thrust["Core"] + lam * thrust["Bypass"]) / (
        thrust["Core"] + thrust["Bypass"]
    )
    tsfc = mdot_fuel / thrust["Net"]

    return {
        "States": {
            "StreamTube": state0,
            "Station1": state1,
            "Station2": state2,
            "Station21": state21,
            "Station25": state25,
            "Station26": state26,
            "Station3": state3,
            "Station31": state31,
            "Station39": state39,
            "Station4": state4,
            "Station41": state41,
            "Station5": state5,
            "Station55": state55,
            "Station6": state6,
            "Station13": state13,
            "Station9": state9,
            "Station19": state19,
        },
        "FanSysObject": fan_sys_object,
        "IPCObject": ipc_object,
        "HPCObject": hpc_object,
        "HPTObject": hpt_object,
        "IPTObject": ipt_object,
        "LPTObject": lpt_object,
        "Efficiency": efficiency,
        "Thrust": thrust,
        "MDotAir": state0["MDot"],
        "Fuel": {
            "MDot": mdot_fuel,
            "FAR": far,
        },
        "TSFC": tsfc,
        "TSFC_Imperial": convert_tsfc(tsfc, "SI", "Imp"),
        "TSFC_Adj_Lam": convert_tsfc(tsfc, "SI", "Imp") / lam,
        "TSFC_Adj_Thrust": convert_tsfc(tsfc, "SI", "Imp") / thrust_corr,
        "EGT_Celsius": convert_temperature(state6["Ts"], "K", "C"),
        "FanDiam": state1["Ro"] * 2,
        "OPRActual": opr_actual,
        "Specs": deepcopy(engine_spec),
    }


def turboprop_on_design_cycle(engine_spec, mdot_input, elec_power=None):
    """Run FAST's turboprop on-design thermodynamic cycle."""

    rpm = as_vector(engine_spec["RPMs"])

    if engine_spec["NoSpools"] != len(rpm):
        raise ValueError("Number of spools must match number of input RPMs.")

    eta_poly = engine_spec["EtaPoly"]
    opr = engine_spec["OPR"]
    tt4_max = engine_spec["Tt4Max"]
    lhv_fuel = 43.17e6
    cooling = 0.0
    gas_constant = 287
    mach0 = engine_spec["Mach"]
    ts0, ps0, rho0 = standard_atmosphere(engine_spec["Alt"])
    cp0 = cp_air(ts0)
    cv0 = cv_air(ts0)
    gamma0 = cp0 / cv0
    pt0 = pt_ps(ps0, mach0, gamma0)
    tt0 = tt_ts(ts0, mach0, gamma0)
    mdot0 = mdot_input
    u0 = mach0 * np.sqrt(gamma0 * ts0 * gas_constant)
    area0 = mdot0 / rho0 / u0
    ro0 = np.sqrt(area0 / np.pi)
    state0 = {
        "MDot": mdot0,
        "Pt": pt0,
        "Tt": tt0,
        "Mach": mach0,
        "Cp": cp0,
        "Cv": cv0,
        "Gam": gamma0,
        "Area": area0,
        "Ro": ro0,
        "Ri": 0,
        "Ps": ps0,
        "Ts": ts0,
    }
    ram_drag = u0 * mdot0
    state1 = diffuser(state0, 0.5, "Inner", eta_poly)

    if len(rpm) > 1:
        state3, hpc_object = compressor(state1, opr, rpm[-2], eta_poly)
    else:
        state3, hpc_object = compressor(state1, opr, rpm[0], eta_poly)

    opr_actual = hpc_object["Pi"]
    state31 = deepcopy(state3)
    state31["MDot"] = state3["MDot"] * (1 - cooling - 0.04)
    state_coolant = deepcopy(state3)
    state_coolant["MDot"] = cooling * state3["MDot"]
    state39, mdot_fuel, _ = burner(state31, tt4_max, lhv_fuel, eta_poly)
    state4 = diffuser(state39, 0.4, "Outer", eta_poly)
    state41 = deepcopy(state4)
    state41["MDot"] = state41["MDot"] + state_coolant["MDot"]
    mixed_guess = (
        state4["Tt"] * state4["MDot"] / state41["MDot"]
        + state_coolant["Tt"] * state_coolant["MDot"] / state41["MDot"]
    )
    enthalpy41 = (
        state4["Cp"] * state4["Tt"] * state4["MDot"]
        + state_coolant["Cp"] * state_coolant["Tt"] * state_coolant["MDot"]
    )
    state41["Tt"] = enthalpy41 / state41["MDot"] / cp_air(mixed_guess)
    state41["Pt"] = state41["Pt"] * 0.95
    state41["Cp"] = cp_air(state41["Tt"])
    state41["Cv"] = cv_air(state41["Tt"])
    state41["Gam"] = state41["Cp"] / state41["Cv"]
    state41["Ps"] = ps_pt(state41["Pt"], state41["Mach"], state41["Gam"])
    state41["Ts"] = ts_tt(state41["Tt"], state41["Mach"], state41["Gam"])

    if len(rpm) > 1:
        state5, hpt_object = turbine(state41, hpc_object, eta_poly)
    else:
        hpc_object["CompWork"] = hpc_object["ReqWork"]
        hpc_object["TotalWork"] = hpc_object["ReqWork"] + engine_spec["ReqPower"]

        if elec_power is None:
            hpc_object["ReqWork"] = hpc_object["TotalWork"]
        else:
            hpc_object["ReqWork"] = (
                hpc_object["CompWork"] + engine_spec["ReqPower"] - elec_power
            )

        state5, hpt_object = turbine(state41, hpc_object, eta_poly)

    state55 = deepcopy(state5)

    if len(rpm) > 1:
        free_shaft_object = {
            "TotalWork": engine_spec["ReqPower"],
            "ReqWork": engine_spec["ReqPower"],
            "RPM": rpm[-1],
        }
        state7, ft_object = turbine(state55, free_shaft_object, eta_poly, state0)
    else:
        free_shaft_object = "Nonexistent"
        ft_object = "Nonexistent"
        state7 = deepcopy(state55)

    state9, core_thrust = perf_ex_nozzle(state7, state0, eta_poly, "Prop")
    jet_power = 0

    if elec_power is None:
        if len(rpm) > 1:
            power = ft_object["DelivWork"] + jet_power
        else:
            power = hpt_object["DelivWork"] - hpc_object["CompWork"] + jet_power
    else:
        if len(rpm) > 1:
            power = ft_object["DelivWork"] + jet_power + elec_power
        else:
            power = (
                hpt_object["DelivWork"]
                - hpc_object["CompWork"]
                + jet_power
                + elec_power
            )

    return {
        "States": {
            "StreamTube": state0,
            "Station1": state1,
            "Station3": state3,
            "Station31": state31,
            "Station39": state39,
            "Station4": state4,
            "Station41": state41,
            "Station5": state5,
            "Station55": state55,
            "Station7": state7,
            "Station9": state9,
        },
        "FreeShaftObject": free_shaft_object,
        "HPCObject": hpc_object,
        "HPTObject": hpt_object,
        "FTObject": ft_object,
        "Power": power,
        "MDotAir": state0["MDot"],
        "BSFC": mdot_fuel / power,
        "BSFC_Imp": mdot_fuel / power * 3.6e3 / 0.00134102 * 2.20462,
        "JetThrust": core_thrust - ram_drag,
        "OPR": opr_actual,
        "Fuel": {
            "MDot": mdot_fuel,
            "FAR": mdot_fuel / state3["MDot"],
        },
    }


def turboprop_nonlinear_sizing(engine_spec, elec_power=None):
    """Iterate FAST's turboprop cycle on mass flow to required power."""

    desired_power = engine_spec["ReqPower"]
    initial_guess = turboprop_linear_sizing(engine_spec)
    mdot0 = initial_guess["MDot0"]
    mdot1 = mdot0 * 1.005
    engine0 = turboprop_on_design_cycle(engine_spec, mdot0, elec_power)
    engine1 = turboprop_on_design_cycle(engine_spec, mdot1, elec_power)
    power0 = engine0["Power"]
    power1 = engine1["Power"]
    iteration = 1
    engine2 = None

    while abs(power1 - desired_power) / desired_power > 1.0e-5 and iteration < 10:
        mdot2 = mdot1 * (1 - (power1 - desired_power) / desired_power)

        if np.imag(mdot2) > 0 or mdot2 < 0:
            raise ValueError("Non-physical value for mass flow.")

        engine2 = turboprop_on_design_cycle(engine_spec, mdot2, elec_power)
        mdot0 = mdot1
        mdot1 = mdot2
        power0 = power1
        power1 = engine2["Power"]
        iteration += 1

    if engine2 is None:
        return engine1

    return engine2


def turbofan_nonlinear_sizing(engine_spec, elec_power=None):
    """Iterate FAST's turbofan cycle on mass flow to design thrust."""

    design_thrust = engine_spec["DesignThrust"]
    initial_guess = turbofan_linear_sizing(engine_spec)
    mdot0 = initial_guess["MDot0"]
    mdot1 = mdot0 * 1.005
    engine0 = turbofan_on_design_cycle(engine_spec, mdot0, elec_power)
    engine1 = turbofan_on_design_cycle(engine_spec, mdot1, elec_power)
    thrust0 = engine0["Thrust"]["Net"]
    thrust1 = engine1["Thrust"]["Net"]
    iteration = 1
    max_iter = int(engine_spec.get("MaxIter", 300))
    engine2 = None

    while (
        abs(np.real(thrust1) - design_thrust) / design_thrust > 1.0e-3
        and iteration < max_iter
    ):
        weight = 1 / iteration
        mdot2 = mdot1 - weight * (thrust1 - design_thrust) * (
            mdot1 - mdot0
        ) / (thrust1 - thrust0)

        if np.imag(mdot2) > 0 or mdot2 < 0:
            raise ValueError("Non-physical value for mass flow.")

        engine2 = turbofan_on_design_cycle(engine_spec, mdot2, elec_power)
        mdot0 = mdot1
        mdot1 = mdot2
        thrust0 = thrust1
        thrust1 = engine2["Thrust"]["Net"]
        iteration += 1

    if engine2 is None:
        return engine1

    return engine2


def cp_air_inverse_prime(temperature):
    """Return FAST's derivative expression used by Newton-Raphson helpers."""

    length = 233.0
    rate = 1 / 210
    midpoint = 875
    offset = 993
    return -(
        offset
        + length / (np.exp(rate * (midpoint - temperature)) + 1)
        - 2 * length
    )


def cp_antiderivative(temperature, length, rate, midpoint, offset):
    """Return the antiderivative used by FAST's fitted Cp curves."""

    temp = np.asarray(temperature, dtype=float)
    return temp * (offset + length) + length * np.log1p(
        np.exp(rate * (midpoint - temp))
    ) / rate


def simple_off_design(aircraft, off_params, electric_load, engine_idx, mission_idx):
    """Evaluate FAST's BADA-style simple turbofan off-design model.

    Inputs:
        aircraft: Aircraft dictionary with engine coefficients and mission
            power-available history.
        off_params: Dictionary containing FlightCon.Alt, FlightCon.Mach, and
            required Thrust in newtons.
        electric_load: Supplemental electric power in watts.
        engine_idx: Zero-based propulsion component index for the engine.
        mission_idx: Zero-based mission-history row.

    Outputs:
        Dictionary with fuel flow, thrust, TSFC, imperial TSFC, and HE
        coefficient.

    Assumptions:
        This mirrors EngineModelPkg.SimpleOffDesign, with Python indices
        converted to the component-relative engine entries used by FAST.
    """

    flight = off_params["FlightCon"]
    altitude = flight["Alt"]
    mach = flight["Mach"]
    temperature, _, _ = standard_atmosphere(altitude)
    tas = mach * np.sqrt(1.4 * 287 * temperature)
    thrust_req = off_params["Thrust"]

    with np.errstate(divide="ignore", invalid="ignore"):
        thrust_supp = electric_load / tas

    if not np.isfinite(thrust_supp):
        thrust_supp = 0

    thrust_req = thrust_req - thrust_supp
    tav = aircraft["Mission"]["History"]["SI"]["Power"]["Tav"][mission_idx][engine_idx]

    if thrust_req < -1.0e-6:
        thrust_req = 0
    elif thrust_req > tav:
        thrust_req = tav

    thrust_req_kn = thrust_req / 1000
    prop = aircraft["Specs"]["Propulsion"]
    prop_arch = prop["PropArch"]
    nsrc = len(as_vector(prop_arch["SrcType"]))
    rel_engine = engine_idx - nsrc
    thrust_supp_sls = np.asarray(
        prop.get("ThrustSupp", np.zeros(len(prop["SLSThrust"]))),
        dtype=float,
    ).reshape(-1)
    sls_thrust = np.asarray(prop["SLSThrust"], dtype=float).reshape(-1)
    supplement = thrust_supp_sls[rel_engine]

    if supplement < 0:
        supplement = 0

    sls_thrust_conv = (sls_thrust[rel_engine] + supplement) / 1000
    engine = prop["Engine"]
    coeff = engine["HEcoeff"]
    thrust_frac = thrust_req_kn / (coeff * sls_thrust_conv)
    fuel_flow = (
        engine["Cff3"] * thrust_frac ** 3
        + engine["Cff2"] * thrust_frac ** 2
        + engine["Cff1"] * thrust_frac
        + engine["Cffch"] * thrust_req_kn * altitude
    )

    if thrust_req_kn <= 0:
        tsfc = 0
    else:
        tsfc = fuel_flow / (thrust_req_kn * 1000)

    return {
        "Fuel": fuel_flow,
        "Thrust": thrust_req_kn * 1000,
        "TSFC": tsfc,
        "TSFC_Imperial": convert_tsfc(tsfc, "SI", "Imp"),
        "C": coeff,
    }


def turboprop_linear_sizing(engine_spec):
    """Return FAST's linear turboprop sizing estimate.

    Inputs:
        engine_spec: Dictionary with Mach, Alt, OPR, Tt4Max, ReqPower, NPR,
            and EtaPoly entries.

    Outputs:
        Dictionary matching EngineModelPkg.TurbopropLinearSizing's scalar
        outputs: MDot0, BSFC, BSFC_g_kW_hr, m2, mfuel, and Tt7.

    Assumptions:
        The model is the low-fidelity thermodynamic cycle used by FAST to seed
        the nonlinear turboprop sizing iteration.
    """

    gl = 7 / 5
    gh = 4 / 3
    gas_constant = 287
    cpl = gl * gas_constant / (gl - 1)
    cph = gh * gas_constant / (gh - 1)
    lower_heating_value = 43.17e6
    mach = engine_spec["Mach"]
    altitude = engine_spec["Alt"]
    opr = engine_spec["OPR"]
    tt4 = engine_spec["Tt4Max"]
    req_power = engine_spec["ReqPower"]
    eta = engine_spec["EtaPoly"]
    eta3 = eta["Compressors"]
    eta4 = eta["Combustor"]
    eta49 = eta["Turbines"]
    eta7 = eta["Turbines"]
    ts0, ps0, _ = standard_atmosphere(altitude)
    pt0 = pt_ps(ps0, mach, gl)
    tt0 = tt_ts(ts0, mach, gl)
    tt2 = tt0
    pt3 = opr * pt0
    ideal_tt3 = tt2 * opr ** ((gl - 1) / gl)
    tt3 = (ideal_tt3 - tt2) / eta3 + tt2
    compressor_work = cpl * (tt3 - tt2)
    mass_leak = 0.01
    mass_bleed = 0.03
    mass_cooling = 0.06
    mass31 = 1 - mass_leak - mass_bleed - mass_cooling
    tt31 = tt3
    pt31 = pt3
    pt4 = pt31 * 0.95
    mfuel = mass31 * cp_air(tt31, tt4) / (
        eta4 * lower_heating_value - cp_jeta(tt31, tt4)
    )
    mass4 = mass31 + mfuel
    tt49_ideal = tt4 - compressor_work / mass4 / cph
    pt49 = pt4 * (1 + (tt49_ideal - tt4) / tt4 / eta49) ** (gh / (gh - 1))
    tt49 = tt4 - compressor_work / mass4 / cph / eta49
    mass495 = mass4 + mass_cooling
    tt495 = (mass4 * cph * tt49 + mass_cooling * cpl * tt31) / cph / mass495
    pt495 = pt49 * (tt495 / tt49) ** (gh / (gh - 1))
    pt7 = ps0
    tt7 = tt495 * (pt7 / pt495) ** ((gh - 1) / gh)
    specific_power = mass495 * cph * eta7 * (tt495 - tt7)
    mass2 = req_power / specific_power
    mfuel = mfuel * mass2
    bsfc = mfuel / req_power

    return {
        "MDot0": mass2,
        "BSFC": bsfc,
        "BSFC_g_kW_hr": bsfc * 3.6e9,
        "m2": mass2,
        "mfuel": mfuel,
        "Tt7": tt7,
    }


def turbofan_linear_sizing(engine_spec):
    """Return FAST's linear turbofan sizing estimate.

    Inputs:
        engine_spec: Dictionary with Mach, Alt, OPR, BPR, FPR, Tt4Max,
            DesignThrust, and EtaPoly entries.

    Outputs:
        Dictionary matching EngineModelPkg.TurbofanLinearSizing's core scalar
        outputs used by FAST's nonlinear turbofan sizing path.

    Assumptions:
        The model is the low-fidelity thermodynamic cycle used by FAST to seed
        nonlinear turbofan sizing.
    """

    gl = 7 / 5
    gh = 4 / 3
    gas_constant = 287
    cpl = gl * gas_constant / (gl - 1)
    cph = gh * gas_constant / (gh - 1)
    lower_heating_value = 43.17e6
    mach = engine_spec["Mach"]
    altitude = engine_spec["Alt"]
    cpr = engine_spec["OPR"]
    bpr = engine_spec["BPR"]
    fpr = engine_spec["FPR"]
    tt4 = engine_spec["Tt4Max"]
    design_thrust = engine_spec["DesignThrust"]
    eta = engine_spec["EtaPoly"]
    eta1 = eta["Inlet"]
    eta2 = eta["Fan"]
    eta3 = eta["Compressors"]
    eta19 = eta["BypassNozzle"]
    eta4 = eta["Combustor"]
    eta49 = eta["Turbines"]
    eta5 = eta["Turbines"]
    eta9 = eta["CoreNozzle"]
    ts0, ps0, rho0 = standard_atmosphere(altitude)
    pt0 = pt_ps(ps0, mach, gl)
    tt0 = tt_ts(ts0, mach, gl)
    mass0 = 1 + bpr
    u0 = mach * np.sqrt(gl * gas_constant * ts0)
    area0 = mass0 / rho0 / u0
    mass1 = mass0
    pt1 = pt0 * eta1
    tt1 = tt0
    mach1 = 0.4
    ps1 = ps_pt(pt1, mach1, gl)
    ts1 = ts_tt(tt1, mach1, gl)
    rho1 = ps1 / gas_constant / ts1
    u1 = mach1 * np.sqrt(gl * gas_constant * ts1)
    area1 = mass1 / rho1 / u1
    pt2 = fpr * pt1
    ideal_tt2 = tt1 * fpr ** (1 - 1 / gl)
    tt2 = (ideal_tt2 - tt1) / eta2 + tt1
    fan_work = (1 + bpr) * cpl * (tt2 - tt1)
    mass13 = bpr
    pt13 = pt2
    tt13 = tt2
    mass19 = mass13
    ps19 = ps0
    tt19 = tt13
    ts19_ideal = tt19 * (ps19 / pt13) ** ((gl - 1) / gl)
    u19_ideal = np.sqrt((tt19 - ts19_ideal) * 2 * cpl)
    u19 = u19_ideal * eta19
    ts19 = tt19 - u19 ** 2 / 2 / cpl
    pt19 = ps19 * (tt19 / ts19) ** (gl / (gl - 1))
    pt3 = cpr * pt2
    ideal_tt3 = tt2 * cpr ** ((gl - 1) / gl)
    tt3 = (ideal_tt3 - tt2) / eta3 + tt2
    compressor_work = cpl * (tt3 - tt2)
    mass_leak = 0.01
    mass_bleed = 0.03
    mass_cooling = 0.06
    mass31 = 1 - mass_leak - mass_bleed - mass_cooling
    tt31 = tt3
    pt31 = pt3
    pt4 = pt31 * 0.95
    mfuel = mass31 * cp_air(tt31, tt4) / (
        eta4 * lower_heating_value - cp_jeta(tt31, tt4)
    )
    fuel_air_ratio = (tt4 / tt3 - 1) / (
        (eta4 * lower_heating_value) / cp_air(tt31, tt4) - tt4 / tt3
    )
    mass4 = mass31 + mfuel
    tt49_ideal = tt4 - compressor_work / mass4 / cph
    pt49 = pt4 * (1 + (tt49_ideal - tt4) / tt4 / eta49) ** (gh / (gh - 1))
    tt49 = tt4 - compressor_work / mass4 / cph / eta49
    mass495 = mass4 + mass_cooling
    tt495 = (mass4 * cph * tt49 + mass_cooling * cpl * tt31) / cph / mass495
    pt495 = pt49 * (tt495 / tt49) ** (gh / (gh - 1))
    tt5_ideal = tt495 - fan_work / mass495 / cph
    pt5 = pt495 * (1 + (tt5_ideal - tt495) / tt495 / eta5) ** (
        gh / (gh - 1)
    )
    tt5 = tt495 - fan_work / mass495 / cph / eta5
    mass9 = mass495
    ps9 = ps0
    tt9 = tt5
    ts9_ideal = tt9 * (ps9 / pt5) ** ((gh - 1) / gh)
    u9_ideal = np.sqrt((tt9 - ts9_ideal) * 2 * cph)
    u9 = u9_ideal * eta9
    ts9 = tt9 - u9 ** 2 / 2 / cph
    pt9 = ps9 * (tt9 / ts9) ** (gh / (gh - 1))
    specific_thrust = mass9 * u9 + mass19 * u19 - mass0 * u0
    mass2 = design_thrust / specific_thrust
    area1 = area1 * mass2
    fan_diameter = 2 * np.sqrt(area1 / np.pi)
    mass0 = mass0 * mass2
    mass3 = mass2
    mass31 = mass31 * mass2
    mass4 = mass4 * mass2
    mass49 = mass4
    mass495 = mass495 * mass2
    mass5 = mass495
    mass9 = mass9 * mass2
    mass13 = mass13 * mass2
    mass19 = mass19 * mass2
    mfuel = mfuel * mass2

    return {
        "TSFC": mfuel / design_thrust,
        "MDot0": mass0,
        "MFuel": mfuel,
        "compwork": compressor_work,
        "Tt49": tt5,
        "DFan": fan_diameter,
        "wdot": mass2 * (1 + bpr),
        "u9": u9,
        "f": fuel_air_ratio,
        "TGT_Stagnation": tt5,
        "A0": area0 * mass2,
        "A1": area1,
        "Pt19": pt19,
        "Pt9": pt9,
        "m3": mass3,
        "m31": mass31,
        "m4": mass4,
        "m49": mass49,
        "m495": mass495,
        "m5": mass5,
        "m9": mass9,
        "m13": mass13,
        "m19": mass19,
    }


SimpleOffDesign = simple_off_design
Pt_Ps = pt_ps
Ps_Pt = ps_pt
Tt_Ts = tt_ts
Ts_Tt = ts_tt
Astar_A = astar_a
A_Astar = a_astar
MassFlowParam = mass_flow_parameter
Nozzle = off_design_nozzle
Rhos_Rhot = rhos_rhot
CpAir = cp_air
CpJetA = cp_jeta
CvAir = cv_air
NewGamma = new_gamma
LocalEfficiency = local_efficiency
LocalReynolds = local_reynolds
NewtonRaphsonTt1 = newton_raphson_tt1
NewtonRaphsonTt3 = newton_raphson_tt3
Diffuser = diffuser
Burner = burner
CompStg = comp_stage
Compressor = compressor
TurbStg = turb_stage
Turbine = turbine
PerfExNozzle = perf_ex_nozzle
FanSystem = fan_system
TurbofanOnDesignCycle = turbofan_on_design_cycle
TurbofanNonlinearSizing = turbofan_nonlinear_sizing
TurbopropOnDesignCycle = turboprop_on_design_cycle
TurbopropNonlinearSizing = turboprop_nonlinear_sizing
TurbopropLinearSizing = turboprop_linear_sizing
TurbofanLinearSizing = turbofan_linear_sizing


def as_vector(value):
    """Return marker-aware values as a one-dimensional float array."""

    if hasattr(value, "value"):
        value = value.value

    return np.asarray(value, dtype=float).reshape(-1)


def restore_scalar_or_list(value):
    """Return scalar NumPy results as floats and arrays as lists."""

    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    if array.size == 1:
        return float(array.reshape(-1)[0])

    return array.tolist()


def is_nan_scalar(value):
    """Return True when a scalar-like value is NaN."""

    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
