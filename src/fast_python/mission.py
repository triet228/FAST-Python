# src/fast_python/mission.py

"""Mission profile processing helpers ported from MissionSegsPkg."""

import warnings
from copy import deepcopy

import numpy as np

from fast_python.atmosphere import gravity as gravity_at_altitude
from fast_python.atmosphere import standard_atmosphere
from fast_python.battery import discharging
from fast_python.data_struct import clear_mission
from fast_python.propulsion import (
    is_detailed_battery,
    power_available,
    prop_analysis,
    recompute_splits,
)


class MissionProfileError(ValueError):
    """Report invalid mission profile data."""


def compute_flight_conditions(altitude, disa, velocity_type, velocity):
    """Compute flight condition quantities from altitude and speed.

    Inputs:
        altitude: Scalar or list of altitudes in meters.
        disa: Standard-atmosphere temperature deviation in K.
        velocity_type: TAS, EAS, or Mach.
        velocity: Speed in m/s for TAS/EAS, or Mach number.

    Outputs:
        A tuple of EAS, TAS, Mach, temperature, pressure, density, and air
        viscosity. Scalars return scalars; vector inputs return lists.

    Assumptions:
        Constants, viscosity polynomial, and conversion formulas mirror
        MissionSegsPkg.ComputeFltCon.
    """

    gamma = 1.4
    gas_constant = 287
    alt = np.asarray(altitude, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    disa = np.asarray(disa, dtype=float)
    _, _, rho_sl_std = standard_atmosphere(0)
    t_isa, p_isa, _ = standard_atmosphere(to_python_shape(alt))
    t_isa = np.asarray(t_isa, dtype=float)
    p_isa = np.asarray(p_isa, dtype=float)
    t_inf = t_isa + disa
    p_inf = p_isa
    rho_inf = p_inf / (gas_constant * t_inf)
    sound_speed = np.sqrt(gamma * gas_constant * t_inf)
    visc = (
        -1.51e-19 * alt ** 3
        + 1.64e-14 * alt ** 2
        - 4.67e-10 * alt
        + 1.81e-5
    )
    vel_type = velocity_type.lower()

    if vel_type == "mach":
        mach = vel
        tas = vel * sound_speed
        eas = tas * np.sqrt(rho_inf / rho_sl_std)
    elif vel_type == "tas":
        tas = vel
        eas = tas * np.sqrt(rho_inf / rho_sl_std)
        mach = tas / sound_speed
    elif vel_type == "eas":
        eas = vel
        tas = eas * np.sqrt(rho_sl_std / rho_inf)
        mach = tas / sound_speed
    else:
        raise MissionProfileError("velocity_type must be TAS, EAS, or Mach.")

    return tuple(
        restore_scalar_or_list(item)
        for item in (eas, tas, mach, t_inf, p_inf, rho_inf, visc)
    )


def process_profile(aircraft):
    """Validate and annotate an aircraft mission profile.

    Inputs:
        aircraft: Dictionary with Aircraft.Mission.Profile and Settings fields.

    Outputs:
        The same aircraft dictionary with SegBeg, SegEnd, and SegPts populated
        under Mission.Profile.

    Assumptions:
        This mirrors FAST's profile validation and segment indexing. It does
        not fly the mission; mission segment evaluation is ported separately.
    """

    mission = aircraft["Mission"]["Profile"]
    eps06 = 1.0e-6
    mission["Target"]["Valu"] = list_or_scalar_to_list(mission["Target"]["Valu"])
    mission["Target"]["Type"] = list_or_scalar_to_list(mission["Target"]["Type"])

    segments = mission["Segs"]
    nsegs = len(segments)
    seg_beg = [1] * nsegs
    seg_end = [1] * nsegs
    seg_pts = [1] * nsegs

    for index, segment_name in enumerate(segments):
        if segment_name in ("Takeoff", "DetailedTakeoff"):
            seg_pts[index] = aircraft["Settings"]["TkoPoints"]
        elif segment_name == "Climb":
            seg_pts[index] = aircraft["Settings"]["ClbPoints"]
        elif segment_name in ("Cruise", "CruiseBRE"):
            seg_pts[index] = aircraft["Settings"]["CrsPoints"]
        elif segment_name == "Descent":
            seg_pts[index] = aircraft["Settings"]["DesPoints"]
        elif segment_name == "Landing":
            seg_pts[index] = 2
        else:
            raise MissionProfileError(
                f"Mission segment {index + 1} has an invalid name: {segment_name}."
            )

        if index > 0:
            seg_beg[index] = seg_beg[index - 1] + seg_pts[index - 1] - 1

        seg_end[index] = seg_beg[index] + seg_pts[index] - 1

    target = mission["Target"]

    if len(target["Valu"]) != len(target["Type"]):
        raise MissionProfileError("Mission target values and types must match.")

    for index, value in enumerate(target["Valu"]):
        if is_nan(value):
            continue

        if value < eps06:
            raise MissionProfileError(
                f"Mission target {index + 1} must be positive."
            )

        if target["Type"][index] not in ("Dist", "Time"):
            raise MissionProfileError(
                f"Mission target type {index + 1} must be Dist or Time."
            )

    if max(mission["ID"]) > len(target["Valu"]):
        raise MissionProfileError("Mission ID exceeds the number of mission targets.")

    if any(item < 1 for item in mission["ID"]):
        raise MissionProfileError("Mission IDs must be integers greater than or equal to 1.")

    require_non_negative(mission["AltBeg"], "beginning altitudes", eps06)
    require_non_negative(mission["AltEnd"], "ending altitudes", eps06)
    require_non_negative(mission["VelBeg"], "beginning airspeeds", eps06)
    require_non_negative(mission["VelEnd"], "ending airspeeds", eps06)

    segment_fields = [
        "AltBeg",
        "AltEnd",
        "VelBeg",
        "VelEnd",
        "TypeBeg",
        "TypeEnd",
        "ClbRate",
    ]

    for field_name in segment_fields:
        if len(mission[field_name]) != nsegs:
            raise MissionProfileError(
                f"Mission field {field_name} must have {nsegs} entries."
            )

    for field_name in ("TypeBeg", "TypeEnd"):
        for index, speed_type in enumerate(mission[field_name]):
            if speed_type not in ("TAS", "EAS", "Mach"):
                raise MissionProfileError(
                    f"{field_name}[{index}] must be TAS, EAS, or Mach."
                )

    mission["SegBeg"] = seg_beg
    mission["SegEnd"] = seg_end
    mission["SegPts"] = seg_pts
    aircraft["Mission"]["Profile"] = mission
    return aircraft


def eval_takeoff(aircraft):
    """Evaluate a takeoff mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history and an active
            Mission.Profile.SegsID.

    Outputs:
        A deep-copied aircraft dictionary with takeoff trajectory, energy, and
        propulsion histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalTakeoff: one minute, constant
        acceleration, maximum available power, and zero climb during the ground
        roll.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_end = profile["AltEnd"][seg_id]
    v_to = profile["VelEnd"][seg_id]
    vtype = profile["TypeEnd"][seg_id]
    gravity = 9.81
    takeoff_time = 60
    disa = 0
    time = np.linspace(0, takeoff_time, npoint)
    dh_dt = np.zeros(npoint)
    fpa = np.zeros(npoint)
    alt = np.ones(npoint) * specs["Performance"]["Alts"]["Tko"]
    mass = np.ones(npoint) * specs["Weight"]["MTOW"]
    eleft_es = initial_energy_remaining(specs, npoint)
    _, tas_takeoff, _, _, _, _, _ = compute_flight_conditions(
        alt_end,
        disa,
        vtype,
        v_to,
    )
    dv_dt = (tas_takeoff - 0) / takeoff_time
    dist = 0.5 * dv_dt * time ** 2
    velocity = dv_dt * time
    eas, tas, mach, _, _, rho, _ = compute_flight_conditions(
        alt.tolist(),
        disa,
        "TAS",
        velocity.tolist(),
    )
    tas = np.asarray(tas, dtype=float)
    eas = np.asarray(eas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    rho = np.asarray(rho, dtype=float)
    ke = 0.5 * mass * tas ** 2
    pe = mass * gravity * alt
    set_split_history(history, specs, "Tko", seg_beg, seg_end)
    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Rho", rho, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
    aircraft = power_available(aircraft)
    aircraft = recompute_splits(aircraft, seg_beg + 1, seg_end)
    history = aircraft["Mission"]["History"]["SI"]
    preq = np.ones(npoint) * np.inf
    ps = np.zeros(npoint)
    assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    aircraft = prop_analysis(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", np.ones(npoint) * dv_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Takeoff" for _ in range(npoint)
    ]
    return aircraft


def eval_detailed_takeoff(aircraft):
    """Evaluate a physics-based takeoff mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history, wing loading, and
            active Mission.Profile.SegsID.

    Outputs:
        A deep-copied aircraft dictionary with detailed takeoff trajectory,
        aerodynamic state, and propulsion histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalDetailedTakeoff. It keeps FAST's
        hard-coded takeoff aerodynamics and computes time and distance from
        available thrust rather than prescribing a one-minute ground roll.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    mass_takeoff = specs["Weight"]["MTOW"]
    wing_loading = specs["Aero"]["W_S"]["SLS"]
    wing_area = mass_takeoff / wing_loading
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_end = profile["AltEnd"][seg_id]
    v_to = profile["VelEnd"][seg_id]
    vtype = profile["TypeEnd"][seg_id]
    gravity = 9.81
    disa = 0
    dh_dt = np.zeros(npoint)
    fpa = np.zeros(npoint)
    alt = np.ones(npoint) * specs["Performance"]["Alts"]["Tko"]
    mass = np.ones(npoint) * mass_takeoff
    preq = np.ones(npoint) * np.inf
    _, tas_takeoff, _, _, _, rho_takeoff, _ = compute_flight_conditions(
        alt_end,
        disa,
        vtype,
        v_to,
    )
    velocity = np.linspace(0, tas_takeoff, npoint)
    set_split_history(history, specs, "Tko", seg_beg, seg_end)
    assign_history_vector(history["Performance"], "TAS", velocity, seg_beg, seg_end)
    assign_history_vector(
        history["Performance"],
        "Rho",
        np.ones(npoint) * rho_takeoff,
        seg_beg,
        seg_end,
    )
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(
        history["Energy"],
        "Eleft_ES",
        initial_energy_remaining(specs, npoint),
        seg_beg,
        seg_end,
    )
    cl_max = 2 * mass_takeoff * gravity / (
        rho_takeoff * (tas_takeoff / 1.1) ** 2 * wing_area
    )
    cd0 = 0.0017
    k_uc = 3.16e-5
    delta_cd0 = wing_loading * k_uc * mass_takeoff ** -0.215
    k1 = 0.02
    k3 = 1 / (np.pi * 0.9 * 10)
    ground_effect = 0.6
    cd = cd0 + delta_cd0 + (k1 + ground_effect * k3) * cl_max ** 2
    lift_drag = np.ones(npoint) * (cl_max / cd)
    drag = np.zeros(npoint)
    acceleration = np.zeros(npoint)
    dtime = np.zeros(npoint)
    ddist = np.zeros(npoint)
    aircraft = power_available(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    pav = np.asarray(history["Power"]["TV"][seg_beg:seg_end], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        thrust = pav / velocity

    thrust[0] = np.nan
    lift = 0.5 * rho_takeoff * velocity[1:] ** 2 * cl_max * wing_area
    friction = 0.02 * (mass_takeoff * gravity - lift)
    friction[friction < 0] = 0
    drag[1:] = 0.5 * rho_takeoff * velocity[1:] ** 2 * cd * wing_area
    acceleration[1:] = (thrust[1:] - drag[1:] - friction) / mass_takeoff
    dtime[1:] = np.diff(velocity) / acceleration[1:]
    ddist[1:] = np.diff(velocity ** 2) / (2 * acceleration[1:])
    drag_power = drag * velocity
    ps = (pav - drag_power) / (mass * gravity)
    time = np.cumsum(dtime)
    dist = np.cumsum(ddist)
    eas, tas, mach, _, _, _, _ = compute_flight_conditions(
        alt.tolist(),
        disa,
        "TAS",
        velocity.tolist(),
    )
    eas = np.asarray(eas, dtype=float)
    tas = np.asarray(tas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    pe = mass * gravity * alt
    ke = 0.5 * mass * tas ** 2
    assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
    aircraft = prop_analysis(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", acceleration, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "LD", lift_drag, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Takeoff" for _ in range(npoint)
    ]
    return aircraft


def eval_landing(aircraft):
    """Evaluate a landing mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history and active
            Mission.Profile.SegsID.

    Outputs:
        A deep-copied aircraft dictionary with landing trajectory, energy, and
        propulsion histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalLanding: 30 seconds, two control
        points, level ground roll, and 30 percent of available power used for
        reverse-thrust demand.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    seg_id = int(profile["SegsID"]) - 1
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_land = profile["AltBeg"][seg_id]
    vel_land = profile["VelBeg"][seg_id]
    vtype = profile["TypeBeg"][seg_id]
    landing_time = 30
    npoint = 2
    gravity = 9.81
    disa = 0
    dh_dt = np.zeros(npoint)
    fpa = np.zeros(npoint)
    dist = np.zeros(npoint)
    time = np.zeros(npoint)

    if seg_beg > 0:
        mass = np.ones(npoint) * history["Weight"]["CurWeight"][seg_beg]
        dist[:] = history["Performance"]["Dist"][seg_beg]
        time[0] = history["Performance"]["Time"][seg_beg]
        eleft_es = np.tile(
            np.asarray(history["Energy"]["Eleft_ES"][seg_beg], dtype=float),
            (npoint, 1),
        )
    else:
        mass = np.ones(npoint) * specs["Weight"]["MTOW"]
        eleft_es = initial_energy_remaining(specs, npoint)

    set_split_history(history, specs, "Lnd", seg_beg, seg_end)
    dtime = np.diff(np.linspace(0, landing_time, npoint))
    time[1:] = time[0] + np.cumsum(dtime)
    _, tas_land, _, _, _, rho_land, _ = compute_flight_conditions(
        alt_land,
        disa,
        vtype,
        vel_land,
    )
    alt = np.ones(npoint) * alt_land
    tas = np.linspace(tas_land, 0, npoint)
    eas, _, mach, _, _, _, _ = compute_flight_conditions(
        alt_land,
        disa,
        "TAS",
        tas.tolist(),
    )
    eas = np.asarray(eas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
    pe = mass * gravity * alt
    ke = 0.5 * mass * tas ** 2
    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Rho", np.ones(npoint) * rho_land, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    aircraft = power_available(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    pav = np.asarray(history["Power"]["TV"][seg_beg:seg_end], dtype=float)
    preq = 0.3 * pav
    preq[-1] = 0
    ps = np.zeros(npoint)
    assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
    aircraft = prop_analysis(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", dvelocity_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Landing" for _ in range(npoint)
    ]
    return aircraft


def eval_cruise(aircraft):
    """Evaluate a cruise mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history and active
            Mission.Profile.SegsID/CrsTarget fields.

    Outputs:
        A deep-copied aircraft dictionary with cruise trajectory, power
        required, and energy histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalCruise. Mass iteration converges after
        propulsion analysis updates CurWeight; with the current native
        PropAnalysis, all-electric cases converge in one pass.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    dh_dt_max = specs["Performance"]["RCMax"]
    lift_drag = specs["Aero"]["L_D"]["Crs"]
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_beg = profile["AltBeg"][seg_id]
    alt_end = profile["AltEnd"][seg_id]
    vel_beg = profile["VelBeg"][seg_id]
    vel_end = profile["VelEnd"][seg_id]
    type_beg = profile["TypeBeg"][seg_id]
    type_end = profile["TypeEnd"][seg_id]
    target = profile["CrsTarget"]
    gravity = 9.81
    disa = 0
    tolerance = 1.0e-6
    max_iter = 10
    _, tas_beg, _, _, _, _, _ = compute_flight_conditions(
        alt_beg,
        disa,
        type_beg,
        vel_beg,
    )
    _, tas_end, _, _, _, _, _ = compute_flight_conditions(
        alt_end,
        disa,
        type_end,
        vel_end,
    )
    alt = np.linspace(alt_beg, alt_end, npoint)
    tas = np.linspace(tas_beg, tas_end, npoint)
    dist = np.zeros(npoint)
    time = np.zeros(npoint)

    if seg_beg > 0:
        mass = np.ones(npoint) * history["Weight"]["CurWeight"][seg_beg]
        dist[0] = history["Performance"]["Dist"][seg_beg]
        time[0] = history["Performance"]["Time"][seg_beg]
        eleft_es = np.tile(
            np.asarray(history["Energy"]["Eleft_ES"][seg_beg], dtype=float),
            (npoint, 1),
        )
    else:
        mass = np.ones(npoint) * specs["Weight"]["MTOW"]
        eleft_es = initial_energy_remaining(specs, npoint)

    set_split_history(history, specs, "Crs", seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    eas, _, mach, _, _, rho, _ = compute_flight_conditions(
        alt.tolist(),
        disa,
        "TAS",
        tas.tolist(),
    )
    eas = np.asarray(eas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    rho = np.asarray(rho, dtype=float)
    dist = np.linspace(dist[0], target, npoint)
    ddist = np.diff(dist)
    dtime = ddist / tas[:-1]
    dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
    dh_dt = np.concatenate([np.diff(alt) / dtime, [0]])
    high_climb = dh_dt > dh_dt_max
    high_descent = dh_dt < -dh_dt_max

    if np.any(high_climb) or np.any(high_descent):
        dh_dt[high_climb] = dh_dt_max
        dh_dt[high_descent] = -dh_dt_max
        dtime = np.diff(alt) / dh_dt[:-1]
        dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])

    time[1:] = time[0] + np.cumsum(dtime)

    with np.errstate(divide="ignore", invalid="ignore"):
        fpa = np.degrees(np.arcsin(dh_dt / tas))

    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Rho", rho, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
    aircraft = power_available(aircraft)
    history = aircraft["Mission"]["History"]["SI"]
    pav = np.asarray(history["Power"]["TV"][seg_beg:seg_end], dtype=float)

    for _ in range(max_iter):
        lift = mass * gravity * np.cos(np.radians(fpa))
        drag = lift / lift_drag
        drag_power = drag * tas
        ps = (pav - drag_power) / (mass * gravity)
        pe = mass * gravity * alt
        ke = 0.5 * mass * tas ** 2
        dpe_dt = mass * gravity * dh_dt
        dke_dt = mass * tas * dvelocity_dt
        preq = dpe_dt + dke_dt + drag_power
        mass_old = mass.copy()
        assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
        assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
        aircraft = prop_analysis(aircraft)
        history = aircraft["Mission"]["History"]["SI"]
        mass = np.asarray(history["Weight"]["CurWeight"][seg_beg:seg_end], dtype=float)
        mass_check = np.abs(mass - mass_old) / mass_old

        if not np.any(mass_check > tolerance):
            break

    time[1:] = time[0] + np.cumsum(dtime)
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", dvelocity_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Cruise" for _ in range(npoint)
    ]
    return aircraft


def eval_cruise_breguet(aircraft):
    """Evaluate FAST's Breguet-equation cruise segment.

    Inputs:
        aircraft: Dictionary with processed mission history and an active
            CruiseBRE segment.

    Outputs:
        A deep-copied aircraft dictionary with Breguet cruise trajectory,
        fuel/battery energy, mass, and aggregate power histories populated.

    Assumptions:
        This ports the active range-to-final-weight branch of
        MissionSegsPkg.EvalCruiseBRE. The commented MATLAB branch that solves
        range from a prescribed landing weight remains intentionally omitted.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_beg = profile["AltBeg"][seg_id]
    alt_end = profile["AltEnd"][seg_id]
    vel_beg = profile["VelBeg"][seg_id]
    vel_end = profile["VelEnd"][seg_id]
    type_beg = profile["TypeBeg"][seg_id]
    type_end = profile["TypeEnd"][seg_id]
    target = profile["CrsTarget"]

    if abs(vel_beg - vel_end) > 1.0e-6:
        raise MissionProfileError(
            "CruiseBRE initial and final airspeeds must be the same."
        )

    if type_beg != type_end:
        raise MissionProfileError(
            "CruiseBRE initial and final airspeed types must be the same."
        )

    if abs(alt_beg - alt_end) > 1.0e-6:
        raise MissionProfileError(
            "CruiseBRE initial and final altitudes must be the same."
        )

    if npoint < 2:
        npoint = 100
        aircraft["Settings"]["CrsPoints"] = npoint
        profile["SegPts"][seg_id] = npoint
        profile["SegEnd"][seg_id] = seg_beg + npoint
        seg_end = seg_beg + npoint

    _, velocity, _, _, _, _, _ = compute_flight_conditions(
        alt_beg,
        0,
        type_beg,
        vel_beg,
    )
    velocity = float(velocity)
    altitude = float(alt_beg)
    gravity = gravity_at_altitude(altitude)
    arch = cruise_breguet_architecture(specs)
    efuel = specs["Power"]["SpecEnergy"]["Fuel"]
    ebatt = specs["Power"]["SpecEnergy"]["Batt"] * 3600
    lift_drag = specs["Aero"]["L_D"]["Crs"]
    eta_prop = cruise_breguet_propulsive_efficiency(specs)
    eta_em = specs["Power"]["Eta"].get("EM", 0)
    eta_eg = specs["Power"]["Eta"].get("EG", 0)
    phi = cruise_breguet_power_split(specs)
    tsfc = specs["Propulsion"]["TSFC"]
    eta_overall = velocity / (tsfc * efuel)
    eta_gt = eta_overall / eta_prop
    dist = np.zeros(npoint)
    time = np.zeros(npoint)
    fburn = np.zeros(npoint)
    fuel_energy = np.zeros(npoint)
    battery_energy = np.zeros(npoint)
    soc = np.zeros(npoint)
    mass = np.zeros(npoint)

    if seg_beg > 0:
        dist[0] = history["Performance"]["Dist"][seg_beg]
        time[0] = history["Performance"]["Time"][seg_beg]
        mass[0] = history["Weight"]["CurWeight"][seg_beg]
        fburn[0] = history["Weight"]["Fburn"][seg_beg]
        fuel_energy[0] = history["Energy"].get("Fuel", [0 for _ in range(seg_end)])[seg_beg]
        battery_energy[0] = history["Energy"].get("Batt", [0 for _ in range(seg_end)])[seg_beg]

        if is_detailed_battery(specs):
            soc[0] = np.asarray(history["Power"]["SOC"][seg_beg], dtype=float).reshape(-1)[0]
    else:
        mass[0] = specs["Weight"]["MTOW"]

        if is_detailed_battery(specs):
            soc[0] = specs["Power"]["Battery"].get("BegSOC", 100)

    dist = np.linspace(dist[0], target, npoint)
    ddist = np.diff(dist)
    dtime = ddist / velocity

    (
        pfuel,
        pbatt,
        pprop,
        pem,
        peg,
        preq,
        dfburn,
        dfuel_energy,
        dbattery_energy,
        phi_history,
        soc,
    ) = cruise_breguet_power_history(
        aircraft,
        arch,
        mass,
        ddist,
        dtime,
        velocity,
        gravity,
        efuel,
        ebatt,
        lift_drag,
        eta_prop,
        eta_em,
        eta_eg,
        eta_gt,
        phi,
        soc,
    )

    time[1:] = time[0] + np.cumsum(dtime)
    fburn[1:] = fburn[0] + np.cumsum(dfburn)
    fuel_energy[1:] = fuel_energy[0] + np.cumsum(dfuel_energy)
    battery_energy[1:] = battery_energy[0] + np.cumsum(dbattery_energy)
    alt = np.ones(npoint) * altitude
    tas = np.ones(npoint) * velocity
    dh_dt = np.zeros(npoint)
    dvelocity_dt = np.zeros(npoint)
    fpa = np.zeros(npoint)
    eas, _, mach, _, _, rho, _ = compute_flight_conditions(
        alt.tolist(),
        0,
        "TAS",
        tas.tolist(),
    )
    eas = np.asarray(eas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    rho = np.asarray(rho, dtype=float)
    pe = mass * gravity * alt
    ke = 0.5 * mass * tas ** 2
    treq = preq / tas
    eta_history = np.ones(npoint) * eta_overall
    tsfc_history = np.ones(npoint) * tsfc
    source_energy, source_energy_left = cruise_breguet_source_energy(
        specs,
        history,
        seg_beg,
        npoint,
        fuel_energy,
        battery_energy,
    )
    ntrn = len(specs["Propulsion"]["PropArch"]["TrnType"])
    tsfc_matrix = np.tile(tsfc_history.reshape(-1, 1), (1, ntrn))
    mdot_matrix = np.zeros((npoint, ntrn))
    engine_cols = [
        index
        for index, kind in enumerate(specs["Propulsion"]["PropArch"]["TrnType"])
        if kind == 1
    ]

    if engine_cols:
        mdot = np.zeros(npoint)
        mdot[:-1] = dfburn / dtime
        for index in engine_cols:
            mdot_matrix[:, index] = mdot / len(engine_cols)

    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", dvelocity_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Rho", rho, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "Fburn", fburn, seg_beg, seg_end)
    assign_history_vector(history["Propulsion"], "Treq", treq, seg_beg, seg_end)
    assign_history_vector(history["Propulsion"], "Eta", eta_history, seg_beg, seg_end)
    assign_history_matrix(history["Propulsion"], "TSFC", tsfc_matrix, seg_beg, seg_end)
    assign_history_matrix(history["Propulsion"], "MDotFuel", mdot_matrix, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Out", preq, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Fuel", pfuel, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Batt", pbatt, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Prop", pprop, seg_beg, seg_end)
    assign_history_vector(history["Power"], "EM", pem, seg_beg, seg_end)
    assign_history_vector(history["Power"], "EG", peg, seg_beg, seg_end)
    assign_history_vector(history["Power"], "Phi", phi_history, seg_beg, seg_end)
    assign_history_matrix(history["Power"], "SOC", soc.reshape(-1, 1), seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "Fuel", fuel_energy, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "Batt", battery_energy, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "E_ES", source_energy, seg_beg, seg_end)
    assign_history_matrix(
        history["Energy"],
        "Eleft_ES",
        source_energy_left,
        seg_beg,
        seg_end,
    )
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Cruise" for _ in range(npoint)
    ]
    return aircraft


def cruise_breguet_power_history(
    aircraft,
    arch,
    mass,
    ddist,
    dtime,
    velocity,
    gravity,
    efuel,
    ebatt,
    lift_drag,
    eta_prop,
    eta_em,
    eta_eg,
    eta_gt,
    phi,
    soc,
):
    """Return Breguet power, energy, and mass histories for one cruise.

    Inputs:
        aircraft: Aircraft dictionary, used for detailed battery settings.
        arch: FAST propulsion architecture label such as AC, E, PHE, SHE, TE,
            or PE.
        mass: Segment mass history in kg; updated in place for fuel-burning
            Breguet branches.
        ddist: Distance increments in meters.
        dtime: Time increments in seconds.
        velocity: Cruise true airspeed history in m/s.
        gravity: Local gravitational acceleration in m/s^2.
        efuel, ebatt: Fuel and battery specific energy values in FAST mission
            energy units per kg.
        lift_drag: Segment L/D ratio.
        eta_prop, eta_em, eta_eg, eta_gt: Propulsive, motor, generator, and
            gas-turbine efficiencies.
        phi: Electric/fuel power split fraction.
        soc: Battery SOC history in percent; updated when detailed cells run.

    Outputs:
        Tuple of fuel power, battery power, propulsor power, motor power,
        generator power, required power, fuel-burn increments, fuel-energy
        increments, battery-energy increments, phi history, and SOC history.

    Assumptions:
        The branch equations follow EvalCruiseBRE. Detailed-battery branches
        may reduce battery contribution after SOC depletion in hybrid missions.
    """

    npoint = len(mass)
    pfuel = np.zeros(npoint)
    pbatt = np.zeros(npoint)
    pprop = np.zeros(npoint)
    pem = np.zeros(npoint)
    peg = np.zeros(npoint)
    preq = np.zeros(npoint)
    dfburn = np.zeros(npoint - 1)
    dfuel_energy = np.zeros(npoint - 1)
    dbattery_energy = np.zeros(npoint - 1)
    phi_history = np.ones(npoint) * phi

    if arch in ("AC", "PHE", "SHE", "TE"):
        eta1, eta2, eta3 = cruise_breguet_efficiency_triplet(
            arch,
            eta_prop,
            eta_em,
            eta_eg,
            eta_gt,
        )
        split_ratio = phi / (1 - phi)
        denominator = eta3 * (efuel / gravity) * lift_drag * (
            eta1 + eta2 * split_ratio
        )

        for index in range(npoint - 1):
            mass[index + 1] = mass[index] / np.exp(ddist[index] / denominator)

        dfburn = -np.diff(mass)
        dfuel_energy = dfburn * efuel
        pfuel[:-1] = dfuel_energy / dtime
        pbatt = pfuel * split_ratio

        if is_detailed_battery(aircraft["Specs"]) and len(dtime) > 0:
            pbatt, soc, phi_history = cruise_breguet_discharge_battery(
                aircraft,
                pbatt,
                dtime,
                soc,
                phi_history,
                arch,
            )

        dbattery_energy = pbatt[:-1] * dtime
        preq = eta3 * (eta1 * pfuel + eta2 * pbatt)
        pprop = preq / eta_prop

        if arch == "AC":
            pem = np.zeros(npoint)
            peg = np.zeros(npoint)
        elif arch == "PHE":
            pem = pbatt * eta2
            peg = np.zeros(npoint)
        else:
            pem = pprop
            peg = pfuel * eta_gt * eta_eg

        return (
            pfuel,
            pbatt,
            pprop,
            pem,
            peg,
            preq,
            dfburn,
            dfuel_energy,
            dbattery_energy,
            phi_history,
            soc,
        )

    if arch == "PE":
        zeta = (24 * eta_em * phi) / (
            25 * eta_gt + 24 * eta_em * phi - 25 * eta_gt * phi
        )
        eta0 = (eta_gt * eta_prop * eta_em * eta_eg) / (
            (1 - zeta) * eta_em * eta_eg + zeta
        )

        for index in range(npoint - 1):
            mass[index + 1] = mass[index] * np.exp(
                -ddist[index] / (lift_drag * eta0 * efuel / gravity)
            )

        dfburn = -np.diff(mass)
        dfuel_energy = dfburn * efuel
        pfuel[:-1] = dfuel_energy / dtime
        pem = pfuel * (zeta / (1 - zeta))
        peg = pem / eta_em
        preq = pem * eta_prop / zeta
        pprop = pfuel * eta_gt
        return (
            pfuel,
            pbatt,
            pprop,
            pem,
            peg,
            preq,
            dfburn,
            dfuel_energy,
            dbattery_energy,
            phi_history,
            soc,
        )

    if arch == "E":
        battery_weight = mass[0] * (ddist.sum()) / (
            ebatt / gravity * lift_drag * eta_em * eta_prop
        )
        mass[1:] = mass[0]
        dbattery_energy = np.ones(npoint - 1) * (ebatt / battery_weight)
        pbatt[:-1] = dbattery_energy / dtime

        if is_detailed_battery(aircraft["Specs"]) and len(dtime) > 0:
            pbatt, soc, phi_history = cruise_breguet_discharge_battery(
                aircraft,
                pbatt,
                dtime,
                soc,
                phi_history,
                arch,
            )

        pem = pbatt * eta_em
        preq = pem * eta_prop
        pprop = pem
        return (
            pfuel,
            pbatt,
            pprop,
            pem,
            peg,
            preq,
            dfburn,
            dfuel_energy,
            dbattery_energy,
            phi_history,
            soc,
        )

    raise MissionProfileError("CruiseBRE received an invalid propulsion architecture.")


def cruise_breguet_discharge_battery(aircraft, pbatt, dtime, soc, phi_history, arch):
    """Update CruiseBRE battery power and SOC with FAST's cell model.

    Inputs:
        aircraft: Aircraft dictionary with detailed battery cell counts.
        pbatt: Battery power history in W, mutated in place.
        dtime: Time-step vector in seconds.
        soc: SOC history in percent, mutated in place.
        phi_history: Electric split history, set to zero after depletion for
            hybrids.
        arch: CruiseBRE architecture label.

    Outputs:
        Updated pbatt, soc, and phi_history arrays.

    Side effects:
        Sets Mission.History.Flags.SOCOff for depleted hybrid missions when
        the flag structure exists.
    """

    battery = aircraft["Specs"]["Power"]["Battery"]
    voltage, current, power_out, capacity, soc_values, c_rate = discharging(
        aircraft,
        pbatt[:-1],
        dtime,
        soc[0],
        battery["ParCells"],
        battery["SerCells"],
    )
    _ = voltage, current, capacity, c_rate
    pbatt[:-1] = power_out
    soc[:] = soc_values
    depleted = np.where(soc < 20)[0]

    if len(depleted) > 0 and arch != "E":
        stop = depleted[0]
        pbatt[stop:] = 0
        phi_history[stop:] = 0
        soc[stop:] = soc[max(0, stop - 1)]

        if "Flags" in aircraft["Mission"]["History"]:
            miss_id = int(aircraft["Mission"]["Profile"].get("MissID", 1)) - 1
            aircraft["Mission"]["History"]["Flags"]["SOCOff"][miss_id] = 1

    return pbatt, soc, phi_history


def cruise_breguet_efficiency_triplet(arch, eta_prop, eta_em, eta_eg, eta_gt):
    """Return the eta1/eta2/eta3 coefficients used by EvalCruiseBRE.

    Inputs:
        arch: CruiseBRE architecture label.
        eta_prop, eta_em, eta_eg, eta_gt: Propulsive, motor, generator, and
            gas-turbine efficiencies.

    Outputs:
        Three coefficients that map fuel and battery power into required
        propulsive power for the Breguet cruise equations.
    """

    if arch == "AC":
        return eta_gt, 0, eta_prop

    if arch == "PHE":
        return eta_gt, eta_em, eta_prop

    if arch == "SHE":
        return eta_gt * eta_eg, 1, eta_em * eta_prop

    if arch == "TE":
        return eta_gt * eta_eg, 0, eta_em * eta_prop

    raise MissionProfileError("Invalid CruiseBRE efficiency architecture.")


def cruise_breguet_source_energy(
    specs,
    history,
    seg_beg,
    npoint,
    fuel_energy,
    battery_energy,
):
    """Map aggregate CruiseBRE fuel/battery energy onto source columns.

    Inputs:
        specs: Aircraft Specs dictionary with PropArch.SrcType.
        history: Mission.History.SI dictionary.
        seg_beg: Zero-based segment beginning row.
        npoint: Number of segment control points.
        fuel_energy: Aggregate fuel energy history.
        battery_energy: Aggregate battery energy history.

    Outputs:
        Source energy used and remaining-energy matrices with one column per
        propulsion source.

    Assumptions:
        Aggregate fuel energy is assigned to fuel sources and aggregate battery
        energy to battery sources using FAST's source-type convention.
    """

    src_type = np.asarray(specs["Propulsion"]["PropArch"]["SrcType"], dtype=float).reshape(-1)
    nsrc = len(src_type)
    source_energy = np.zeros((npoint, nsrc))
    source_energy_left = initial_energy_remaining(specs, npoint)

    if seg_beg > 0:
        source_energy[0, :] = np.asarray(
            history["Energy"]["E_ES"][seg_beg],
            dtype=float,
        ).reshape(-1)
        source_energy_left[0, :] = np.asarray(
            history["Energy"]["Eleft_ES"][seg_beg],
            dtype=float,
        ).reshape(-1)

    source_energy[:] = source_energy[0, :]
    source_energy_left[:] = source_energy_left[0, :]
    fuel_cols = np.where(src_type == 1)[0]
    batt_cols = np.where(src_type == 0)[0]
    cruise_breguet_apply_source_delta(
        source_energy,
        source_energy_left,
        fuel_cols,
        fuel_energy - fuel_energy[0],
    )
    cruise_breguet_apply_source_delta(
        source_energy,
        source_energy_left,
        batt_cols,
        battery_energy - battery_energy[0],
    )
    return source_energy, source_energy_left


def cruise_breguet_apply_source_delta(source_energy, source_energy_left, columns, delta):
    """Apply aggregate source-energy deltas across matching source columns."""

    if len(columns) == 0:
        return

    share = delta / len(columns)

    for column in columns:
        source_energy[:, column] = source_energy[0, column] + share
        source_energy_left[:, column] = source_energy_left[0, column] - share


def cruise_breguet_architecture(specs):
    """Return EvalCruiseBRE's architecture label from modern FAST fields."""

    prop = specs["Propulsion"]
    arch = prop.get("Arch")

    if not isinstance(arch, str):
        arch = prop.get("PropArch", {}).get("Type")

    arch = str(arch).upper()

    if arch == "C":
        return "AC"

    if arch in ("AC", "E", "PHE", "SHE", "TE", "PE"):
        return arch

    raise MissionProfileError("Invalid CruiseBRE propulsion architecture.")


def cruise_breguet_propulsive_efficiency(specs):
    """Return propulsive efficiency from FAST's old or new storage path."""

    prop_eta = specs["Propulsion"].get("Eta", {}).get("Prop")

    if is_finite_number(prop_eta):
        return prop_eta

    power_eta = specs["Power"].get("Eta", {}).get("Propeller")

    if is_finite_number(power_eta):
        return power_eta

    raise MissionProfileError("CruiseBRE requires propulsive efficiency.")


def cruise_breguet_power_split(specs):
    """Return the CruiseBRE power split ratio."""

    power = specs["Power"]
    phi = power.get("Phi", {}).get("Crs")

    if is_finite_number(phi):
        return phi

    lam_dwn = power.get("LamDwn", {}).get("Crs", 0)

    if is_finite_number(lam_dwn):
        return lam_dwn

    return 0


def is_finite_number(value):
    """Return True when a value is a finite numeric scalar."""

    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def eval_climb(aircraft):
    """Evaluate a climb mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history and active
            Mission.Profile.SegsID.

    Outputs:
        A deep-copied aircraft dictionary with climb trajectory, power
        required, and energy histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalClimb. When a climb rate is not
        prescribed, segment timing comes from available specific excess power;
        when it is prescribed, acceleration is limited by the same excess-power
        balance used in FAST.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    dh_dt_max = specs["Performance"]["RCMax"]
    lift_drag = specs["Aero"]["L_D"]["Clb"]
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_beg = profile["AltBeg"][seg_id]
    alt_end = profile["AltEnd"][seg_id]
    vel_beg = profile["VelBeg"][seg_id]
    vel_end = profile["VelEnd"][seg_id]
    type_beg = profile["TypeBeg"][seg_id]
    type_end = profile["TypeEnd"][seg_id]
    dh_dt_req = profile["ClbRate"][seg_id]
    gravity = 9.81
    disa = 0
    tolerance = 1.0e-6
    max_iter = 10
    alt = np.linspace(alt_beg, alt_end, npoint)
    eas_end, tas_end, mach_end, _, _, _, _ = compute_flight_conditions(
        alt_end,
        disa,
        type_end,
        vel_end,
    )
    converted_end = {
        "EAS": eas_end,
        "TAS": tas_end,
        "Mach": mach_end,
    }[type_beg]
    vel_seg = np.linspace(vel_beg, converted_end, npoint)
    eas, tas, mach, _, _, rho, _ = compute_flight_conditions(
        alt.tolist(),
        disa,
        type_beg,
        vel_seg.tolist(),
    )
    eas = np.asarray(eas, dtype=float)
    tas = np.asarray(tas, dtype=float)
    mach = np.asarray(mach, dtype=float)
    rho = np.asarray(rho, dtype=float)
    dist = np.zeros(npoint)
    time = np.zeros(npoint)

    if seg_beg > 0:
        mass = np.ones(npoint) * history["Weight"]["CurWeight"][seg_beg]
        dist[0] = history["Performance"]["Dist"][seg_beg]
        time[0] = history["Performance"]["Time"][seg_beg]
        eleft_es = np.tile(
            np.asarray(history["Energy"]["Eleft_ES"][seg_beg], dtype=float),
            (npoint, 1),
        )
    else:
        mass = np.ones(npoint) * specs["Weight"]["MTOW"]
        eleft_es = initial_energy_remaining(specs, npoint)

    set_split_history(history, specs, "Clb", seg_beg, seg_end)
    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    preq_old = np.zeros(npoint)

    if not is_nan(dh_dt_req):
        dh_dt = np.concatenate([np.ones(npoint - 1) * dh_dt_req, [0]])
    else:
        dh_dt = np.zeros(npoint)

    dtime = np.zeros(npoint - 1)
    dvelocity_dt = np.zeros(npoint)
    fpa = np.zeros(npoint)
    ps = np.zeros(npoint)
    pe = np.zeros(npoint)
    ke = np.zeros(npoint)
    preq = np.zeros(npoint)

    for _ in range(max_iter):
        energy_height = alt + tas ** 2 / (2 * gravity)
        denergy_height = np.diff(energy_height)
        eas, _, mach, _, _, rho, _ = compute_flight_conditions(
            alt.tolist(),
            disa,
            "TAS",
            tas.tolist(),
        )
        eas = np.asarray(eas, dtype=float)
        mach = np.asarray(mach, dtype=float)
        rho = np.asarray(rho, dtype=float)
        assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Rho", rho, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
        aircraft = power_available(aircraft)
        aircraft = recompute_splits(aircraft, seg_beg + 1, seg_end)
        history = aircraft["Mission"]["History"]["SI"]
        pav = np.asarray(history["Power"]["TV"][seg_beg:seg_end], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            fpa = np.degrees(np.arcsin(dh_dt / tas))

        lift = mass * gravity * np.cos(np.radians(fpa))
        drag = lift / lift_drag
        drag_power = drag * tas
        ps = (pav - drag_power) / (mass * gravity)

        if is_nan(dh_dt_req):
            dtime = denergy_height / ps[:-1]
            dh_dt = np.concatenate([np.diff(alt) / dtime, [0]])
            high_climb = dh_dt - dh_dt_max > tolerance

            if np.any(high_climb):
                dh_dt[high_climb] = dh_dt_max
                dtime = np.diff(alt) / dh_dt[:-1]

            dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
        else:
            dtime = np.diff(alt) / dh_dt[:-1]
            dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
            dvelocity_dt_max = (ps - dh_dt) * gravity / tas
            high_accel = dvelocity_dt - dvelocity_dt_max > tolerance

            if np.any(high_accel):
                dvelocity_dt = dvelocity_dt_max
                tas[1:] = tas[0] + np.cumsum(dvelocity_dt[:-1] * dtime)
                tas[tas > tas_end] = tas_end
                dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])

        time[1:] = time[0] + np.cumsum(dtime)
        pe = mass * gravity * alt
        ke = 0.5 * mass * tas ** 2
        dpe_dt = mass * gravity * dh_dt
        dke_dt = mass * tas * dvelocity_dt
        preq = dpe_dt + dke_dt + drag_power
        mass_old = mass.copy()
        assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
        assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
        aircraft = prop_analysis(aircraft)
        history = aircraft["Mission"]["History"]["SI"]
        mass = np.asarray(history["Weight"]["CurWeight"][seg_beg:seg_end], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            preq_check = np.abs(preq - preq_old) / preq

        if not np.any(preq_check > tolerance):
            break

        preq_old = preq.copy()

        if not np.any(np.abs(mass - mass_old) > 0):
            continue

    ground_speed = tas * np.cos(np.radians(fpa))
    ddist = ground_speed[:-1] * dtime
    dist[1:] = dist[0] + np.cumsum(ddist)
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", dvelocity_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Climb" for _ in range(npoint)
    ]
    return aircraft


def eval_descent(aircraft):
    """Evaluate a descent mission segment.

    Inputs:
        aircraft: Dictionary with processed mission history and active
            Mission.Profile.SegsID.

    Outputs:
        A deep-copied aircraft dictionary with descent trajectory, power
        required, and energy histories populated.

    Assumptions:
        This follows MissionSegsPkg.EvalDescent. Negative power required is
        clipped to a small idle value, matching the MATLAB model's descent
        bookkeeping.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    profile = aircraft["Mission"]["Profile"]
    history = aircraft["Mission"]["History"]["SI"]
    rate_descent_max = -0.8 * specs["Performance"]["RCMax"]
    lift_drag = specs["Aero"]["L_D"]["Des"]
    seg_id = int(profile["SegsID"]) - 1
    npoint = int(profile["SegPts"][seg_id])
    seg_beg = int(profile["SegBeg"][seg_id]) - 1
    seg_end = int(profile["SegEnd"][seg_id])
    alt_beg = profile["AltBeg"][seg_id]
    alt_end = profile["AltEnd"][seg_id]
    vel_beg = profile["VelBeg"][seg_id]
    vel_end = profile["VelEnd"][seg_id]
    type_beg = profile["TypeBeg"][seg_id]
    type_end = profile["TypeEnd"][seg_id]
    dh_dt_req = profile["ClbRate"][seg_id]
    gravity = 9.81
    disa = 0
    tolerance = 1.0e-6
    max_iter = 10
    _, tas_beg, _, _, _, _, _ = compute_flight_conditions(
        alt_beg,
        disa,
        type_beg,
        vel_beg,
    )
    _, tas_end, _, _, _, _, _ = compute_flight_conditions(
        alt_end,
        disa,
        type_end,
        vel_end,
    )
    alt = np.linspace(alt_beg, alt_end, npoint)
    tas = np.linspace(tas_beg, tas_end, npoint)
    dist = np.zeros(npoint)
    time = np.zeros(npoint)
    dvelocity_dt = np.zeros(npoint)
    dh_dt_old = np.zeros(npoint)
    dvelocity_dt_old = np.zeros(npoint)

    if seg_beg > 0:
        mass = np.ones(npoint) * history["Weight"]["CurWeight"][seg_beg]
        dist[0] = history["Performance"]["Dist"][seg_beg]
        time[0] = history["Performance"]["Time"][seg_beg]
        eleft_es = np.tile(
            np.asarray(history["Energy"]["Eleft_ES"][seg_beg], dtype=float),
            (npoint, 1),
        )
    else:
        mass = np.ones(npoint) * specs["Weight"]["MTOW"]
        eleft_es = initial_energy_remaining(specs, npoint)

    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
    assign_history_matrix(history["Energy"], "Eleft_ES", eleft_es, seg_beg, seg_end)
    set_split_history(history, specs, "Des", seg_beg, seg_end)

    if not is_nan(dh_dt_req):
        dh_dt = np.concatenate([np.ones(npoint - 1) * dh_dt_req, [0]])
    else:
        dh_dt = np.zeros(npoint)

    dtime = np.zeros(npoint - 1)
    fpa = np.zeros(npoint)
    ps = np.zeros(npoint)
    pe = np.zeros(npoint)
    ke = np.zeros(npoint)
    preq = np.zeros(npoint)
    eas = np.zeros(npoint)
    mach = np.zeros(npoint)

    for _ in range(max_iter):
        eas, _, mach, _, _, rho, _ = compute_flight_conditions(
            alt.tolist(),
            disa,
            "TAS",
            tas.tolist(),
        )
        eas = np.asarray(eas, dtype=float)
        mach = np.asarray(mach, dtype=float)
        rho = np.asarray(rho, dtype=float)
        assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Rho", rho, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Mach", mach, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Alt", alt, seg_beg, seg_end)
        aircraft = power_available(aircraft)
        history = aircraft["Mission"]["History"]["SI"]
        pav = np.asarray(history["Power"]["TV"][seg_beg:seg_end], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            fpa = np.degrees(np.arcsin(dh_dt / tas))

        lift = mass * gravity * np.cos(np.radians(fpa))
        drag = lift / lift_drag
        energy_height = alt + tas ** 2 / (2 * gravity)
        denergy_height = np.diff(energy_height)
        drag_power = drag * tas
        ps = (pav - drag_power) / (mass * gravity)

        if is_nan(dh_dt_req):
            dtime = np.abs(denergy_height / ps[:-1])
            dh_dt = np.concatenate([np.diff(alt) / dtime, [0]])
            high_descent = dh_dt < rate_descent_max

            if np.any(high_descent):
                dh_dt[high_descent] = rate_descent_max
                dtime = np.diff(alt) / dh_dt[:-1]

            dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
        else:
            dtime = np.diff(alt) / dh_dt[:-1]
            dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])
            dvelocity_dt_max = (ps - dh_dt) * gravity / tas

            if np.any(dvelocity_dt > dvelocity_dt_max):
                dvelocity_dt = dvelocity_dt_max
                tas[1:] = tas[0] + np.cumsum(dvelocity_dt[:-1] * dtime)
                tas[tas > tas_end] = tas_end
                dvelocity_dt = np.concatenate([np.diff(tas) / dtime, [0]])

        time[1:] = time[0] + np.cumsum(dtime)
        pe = mass * gravity * alt
        ke = 0.5 * mass * tas ** 2
        dpe_dt = mass * gravity * dh_dt
        dke_dt = mass * tas * dvelocity_dt
        preq = drag_power + dpe_dt + dke_dt
        preq[preq < 0] = 0.0001
        assign_history_vector(history["Power"], "Req", preq, seg_beg, seg_end)
        assign_history_vector(history["Weight"], "CurWeight", mass, seg_beg, seg_end)
        assign_history_vector(history["Performance"], "Time", time, seg_beg, seg_end)
        aircraft = prop_analysis(aircraft)
        history = aircraft["Mission"]["History"]["SI"]
        mass = np.asarray(history["Weight"]["CurWeight"][seg_beg:seg_end], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            dh_dt_err = np.abs(dh_dt - dh_dt_old) / dh_dt
            dvelocity_dt_err = np.abs(dvelocity_dt - dvelocity_dt_old) / dvelocity_dt

        if not np.any(np.abs(dh_dt_err) > tolerance) and not np.any(
            np.abs(dvelocity_dt_err) > tolerance
        ):
            break

        dh_dt_old = dh_dt.copy()
        dvelocity_dt_old = dvelocity_dt.copy()

    ground_speed = tas * np.cos(np.radians(fpa))
    ddist = ground_speed[:-1] * dtime
    dist[1:] = dist[0] + np.cumsum(ddist)
    assign_history_vector(history["Performance"], "Dist", dist, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "TAS", tas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "EAS", eas, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "RC", dh_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Acc", dvelocity_dt, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "FPA", fpa, seg_beg, seg_end)
    assign_history_vector(history["Performance"], "Ps", ps, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "PE", pe, seg_beg, seg_end)
    assign_history_vector(history["Energy"], "KE", ke, seg_beg, seg_end)
    aircraft["Mission"]["History"]["Segment"][seg_beg:seg_end] = [
        "Descent" for _ in range(npoint)
    ]
    return aircraft


def fly_mission(aircraft):
    """Evaluate all missions in a processed FAST mission profile.

    Inputs:
        aircraft: Dictionary with Mission.Profile, initialized
            Mission.History, and segment evaluators available for each
            segment name.

    Outputs:
        A deep-copied aircraft dictionary with Mission.History.SI populated
        for each flown mission.

    Assumptions:
        This ports MissionSegsPkg.FlyMission for the currently native segment
        evaluators. Distance and time targets are converted into a total
        cruise-distance target, then iterated until the final mission distance
        matches the requested target.
    """

    aircraft = deepcopy(aircraft)
    tolerance = 1.0e-6
    max_iter = int(aircraft["Settings"]["Analysis"]["MaxIter"])
    target_old = 0
    ielem = 0
    mission = deepcopy(aircraft["Mission"]["Profile"])
    nmiss = len(mission["Target"]["Valu"])
    history = aircraft["Mission"]["History"]
    history.setdefault("Flags", {})
    history["Flags"]["SOCOff"] = [0 for _ in range(nmiss)]

    for imiss in range(nmiss):
        miss_id = imiss + 1
        mission["MissID"] = miss_id
        miss_segs = [
            index
            for index, mission_id in enumerate(mission["ID"])
            if int(mission_id) == miss_id
        ]

        if not miss_segs:
            continue

        target = mission["Target"]["Valu"][imiss]
        icrs = None

        if not is_nan(target):
            cruise_segments = [
                index
                for index in miss_segs
                if "Cruise" in mission["Segs"][index]
            ]

            if cruise_segments:
                icrs = cruise_segments[0]
            else:
                warnings.warn(
                    (
                        f"Mission {miss_id} has no cruise segment; target "
                        "convergence cannot be confirmed."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

            if icrs is not None and mission["Target"]["Type"][imiss] == "Time":
                target = cruise_time_target_to_distance(mission, icrs, target)

        iteration = 0
        mission["CrsTarget"] = target_old + target
        history["Flags"]["SOCOff"][imiss] = 0
        seg_end = mission["SegEnd"][miss_segs[-1]]

        while iteration < max_iter:
            aircraft = clear_mission(aircraft, ielem)

            for iseg in range(miss_segs[0], miss_segs[-1] + 1):
                if iteration > 0 and icrs is not None and iseg < icrs:
                    continue

                mission["SegsID"] = iseg + 1
                aircraft["Mission"]["Profile"] = deepcopy(mission)
                segment_name = mission["Segs"][iseg]
                seg_end = mission["SegEnd"][iseg]
                evaluator = mission_segment_evaluator(segment_name)
                aircraft = evaluator(aircraft)

                if iteration < 1 and icrs is not None and iseg < icrs:
                    ielem = seg_end + 1

            if is_nan(target):
                break

            si_history = aircraft["Mission"]["History"]["SI"]
            dist = si_history["Performance"]["Dist"][seg_end - 1] - target_old
            ddist = target - dist
            rel_err = abs(ddist) / target

            if rel_err < tolerance:
                break

            mission = deepcopy(aircraft["Mission"]["Profile"])
            mission["CrsTarget"] = mission["CrsTarget"] + ddist
            iteration += 1

        if imiss < nmiss - 1:
            ielem = seg_end + 1
            target_old = aircraft["Mission"]["History"]["SI"]["Performance"]["Dist"][seg_end - 1]

    return aircraft


def cruise_time_target_to_distance(mission, icrs, target_minutes):
    """Convert a FAST cruise time target in minutes to distance in meters."""

    alt_beg = mission["AltBeg"][icrs]
    alt_end = mission["AltEnd"][icrs]
    vel_beg = mission["VelBeg"][icrs]
    vel_end = mission["VelEnd"][icrs]
    type_beg = mission["TypeBeg"][icrs]
    type_end = mission["TypeEnd"][icrs]
    _, tas_beg, _, _, _, _, _ = compute_flight_conditions(
        alt_beg,
        0,
        type_beg,
        vel_beg,
    )
    _, tas_end, _, _, _, _, _ = compute_flight_conditions(
        alt_end,
        0,
        type_end,
        vel_end,
    )
    dtime = 60 * target_minutes
    dvelocity_dt = (tas_end - tas_beg) / dtime
    return tas_beg * dtime + 0.5 * dvelocity_dt * dtime * dtime


def mission_segment_evaluator(segment_name):
    """Return the native evaluator for a FAST mission segment name."""

    evaluators = {
        "Takeoff": eval_takeoff,
        "DetailedTakeoff": eval_detailed_takeoff,
        "Climb": eval_climb,
        "Cruise": eval_cruise,
        "CruiseBRE": eval_cruise_breguet,
        "Descent": eval_descent,
        "Landing": eval_landing,
    }

    if segment_name not in evaluators:
        raise MissionProfileError(
            f"Mission segment {segment_name} is not ported to native Python yet."
        )

    return evaluators[segment_name]


def require_non_negative(values, label, tolerance):
    """Validate that numeric values are not negative below tolerance."""

    if any((not is_nan(item)) and item < -tolerance for item in values):
        raise MissionProfileError(f"Mission {label} must be non-negative.")


def is_nan(value):
    """Return True for NaN floats."""

    return isinstance(value, float) and value != value


def to_python_shape(value):
    """Return a scalar or list matching a NumPy value's shape."""

    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    return array.tolist()


def restore_scalar_or_list(value):
    """Return a scalar for scalar arrays, otherwise a Python list."""

    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    if array.size == 1:
        return float(array.reshape(-1)[0])

    return array.tolist()


def list_or_scalar_to_list(value):
    """Return FAST scalar/list fields as a plain Python list."""

    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, list):
        return value

    return [value]


def initial_energy_remaining(specs, npoint):
    """Return initial source energy matrix for a first segment."""

    src_type = np.asarray(specs["Propulsion"]["PropArch"]["SrcType"], dtype=float).reshape(-1)
    eleft_es = np.zeros((npoint, len(src_type)))
    fuel = src_type == 1
    batt = src_type == 0

    if np.any(fuel):
        eleft_es[:, fuel] = (
            specs["Power"]["SpecEnergy"]["Fuel"] * np.asarray(specs["Weight"]["Fuel"], dtype=float)
        )

    if np.any(batt):
        eleft_es[:, batt] = (
            specs["Power"]["SpecEnergy"]["Batt"] * np.asarray(specs["Weight"]["Batt"], dtype=float)
        )

    return eleft_es


def set_split_history(history, specs, segment_key, start, stop):
    """Write mission-history split values for one segment."""

    rows = stop - start
    lam_dwn = row_matrix(specs["Power"]["LamDwn"][segment_key], rows)
    lam_ups = row_matrix(specs["Power"]["LamUps"][segment_key], rows)
    assign_history_matrix(history["Power"], "LamDwn", lam_dwn, start, stop)
    assign_history_matrix(history["Power"], "LamUps", lam_ups, start, stop)


def row_matrix(value, rows):
    """Return one split value or row repeated for each history row."""

    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        array = array.reshape(1)

    return np.tile(array.reshape(1, -1), (rows, 1))


def assign_history_vector(section, name, values, start, stop):
    """Assign a vector slice in a history section."""

    values = np.asarray(values, dtype=float).reshape(-1)

    if name not in section:
        section[name] = [0.0 for _ in range(stop)]

    section[name][start:stop] = values.tolist()


def assign_history_matrix(section, name, values, start, stop):
    """Assign a matrix slice in a history section."""

    values = np.asarray(values, dtype=float)

    if name not in section:
        section[name] = [[0.0 for _ in range(values.shape[1])] for _ in range(stop)]

    section[name][start:stop] = values.tolist()


ComputeFltCon = compute_flight_conditions
ProcessProfile = process_profile
EvalTakeoff = eval_takeoff
EvalDetailedTakeoff = eval_detailed_takeoff
EvalLanding = eval_landing
EvalCruise = eval_cruise
EvalCruiseBRE = eval_cruise_breguet
EvalClimb = eval_climb
EvalDescent = eval_descent
FlyMission = fly_mission
