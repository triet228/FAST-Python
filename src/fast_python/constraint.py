# src/fast_python/constraint.py

"""Constraint diagram helpers ported from FAST ConstraintDiagramPkg."""

import numpy as np

from fast_python.mission import compute_flight_conditions
from fast_python.units import (
    convert_force,
    convert_length,
    convert_mass,
    convert_temperature,
    convert_velocity,
)


class ConstraintError(ValueError):
    """Report invalid constraint-diagram inputs."""


def constraint_diagram(aircraft, npoints=500):
    """Evaluate the constraint-diagram grids without plotting them."""

    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    cfr_part = specs["TLAR"]["CFRPart"]

    if aircraft_class.lower() == "turbofan":
        vertical_center = specs["Propulsion"]["T_W"]["SLS"]
        vertical_range = np.linspace(
            max(0.10, vertical_center - 0.20),
            min(0.80, vertical_center + 0.20),
            npoints,
        )
        vertical_label = "Thrust-Weight Ratio (N/N)"
    elif aircraft_class.lower() in ("turboprop", "piston"):
        vertical_center = specs["Power"]["P_W"]["SLS"] * 1000

        if cfr_part == 25:
            vertical_center = vertical_center / 9.81
            vertical_center = 1 / vertical_center
            vertical_range = np.linspace(
                max(0, vertical_center - 0.25),
                min(0.2, vertical_center + 0.15),
                npoints,
            )
            vertical_label = "Power Loading (N/W)"
        elif cfr_part == 23:
            vertical_range = np.linspace(
                max(10, vertical_center - 150),
                min(1000, vertical_center + 150),
                npoints,
            )
            vertical_label = "Power-Weight Ratio (W/kg)"
        else:
            raise ConstraintError(
                "ConstraintDiagram only supports 14 CFR Part 23 or 25."
            )
    else:
        raise ConstraintError("ConstraintDiagram received an invalid aircraft class.")

    horizontal_center = specs["Aero"]["W_S"]["SLS"]
    horizontal_range = np.linspace(
        max(0, horizontal_center - 1000),
        max(1000, horizontal_center + 1000),
        npoints,
    )
    horizontal_label = "Wing Loading (kg/m^2)"
    horizontal_grid, vertical_grid = np.meshgrid(horizontal_range, vertical_range)
    constraint_names = listify_constraint_values(
        specs["Performance"].get("ConstraintFuns", [])
    )
    constraint_labels = listify_constraint_values(
        specs["Performance"].get("ConstraintLabs", constraint_names)
    )
    constraints = []

    for name in constraint_names:
        constraints.append(
            constraint_function(name)(horizontal_grid, vertical_grid, aircraft)
        )

    if constraints:
        constraint_values = np.stack(constraints, axis=2)
        feasible = np.all(constraint_values <= 0, axis=2)
    else:
        constraint_values = np.zeros(
            (len(vertical_range), len(horizontal_range), 0)
        )
        feasible = np.ones_like(horizontal_grid, dtype=bool)

    if aircraft_class.lower() in ("turboprop", "piston") and cfr_part == 25:
        horizontal_grid = horizontal_grid * 9.81 / 1000
        horizontal_range = horizontal_range * 9.81 / 1000
        horizontal_label = "Wing Loading (kN/m^2)"

    return {
        "horizontal_grid": horizontal_grid,
        "vertical_grid": vertical_grid,
        "constraints": constraint_values,
        "feasible": feasible,
        "constraint_names": constraint_names,
        "constraint_labels": constraint_labels,
        "horizontal_range": horizontal_range,
        "vertical_range": vertical_range,
        "horizontal_label": horizontal_label,
        "vertical_label": vertical_label,
    }


def listify_constraint_values(value):
    """Return MATLAB-style string arrays as a flat Python list."""

    if isinstance(value, str):
        return [value]

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.reshape(-1).tolist()]

    return [str(item) for item in value]


def constraint_function(name):
    """Resolve a MATLAB constraint-function name to a Python callable."""

    key = str(name).split(".")[-1].lower()
    functions = {
        "jet25_111": jet25_111,
        "jet25_119": jet25_119,
        "jet25_121a": jet25_121a,
        "jet25_121b": jet25_121b,
        "jet25_121c": jet25_121c,
        "jet25_121d": jet25_121d,
        "jetaeoclimb": jet_aeo_climb,
        "jetapp": jet_app,
        "jetceil": jet_ceil,
        "jetcrs": jet_crs,
        "jetdiv": jet_div,
        "jetlfl": jet_lfl,
        "jettofl": jet_tofl,
    }

    if key not in functions:
        raise ConstraintError(f"Unknown constraint function: {name}")

    return functions[key]


def sigmoid(aircraft, a, b, c, d):
    """Evaluate FAST's PsLoss sigmoid climb-gradient helper."""

    ps_loss = aircraft["Specs"]["Performance"]["PsLoss"]
    return a / (1 + np.exp(-b * (ps_loss - c))) / 100 + d / 100


def oei_multiplier(aircraft):
    """Return the engine-inoperative multiplier for a constraint analysis."""

    constraint_type = aircraft["Settings"]["ConstraintType"]

    if constraint_type == 0:
        neng = aircraft["Specs"]["Propulsion"]["NumEngines"]
        return neng / (neng - 1)

    if constraint_type == 1:
        ps_loss = aircraft["Specs"]["Performance"]["PsLoss"]
        return 1.045 * ps_loss ** 2 + 1

    raise ConstraintError("OEIMultiplier Type must be 0 or 1.")


def jet_app(w_s, t_w, aircraft):
    """Return the approach-speed constraint residual."""

    _ = t_w
    w_s = np.asarray(w_s, dtype=float)
    specs = aircraft["Specs"]
    req_type = specs["TLAR"]["ReqType"]
    w_s = w_s * convert_mass(1, "kg", "lbm") / convert_length(1, "m", "ft") ** 2

    if req_type == 0:
        return np.zeros_like(w_s)

    cl = specs["Aero"]["CL"]["Lnd"]
    wland_mtow = specs["Performance"]["Wland_MTOW"]
    vapp = convert_velocity(specs["Performance"]["Vels"]["App"], "m/s", "ft/s")
    vstall = vapp / 1.3
    w_s_required = 0.5 * 0.002377 * vstall ** 2 * cl / wland_mtow
    return w_s - w_s_required


def jet_tofl(w_s, t_w, aircraft):
    """Return the takeoff-field-length constraint residual."""

    w_s = np.asarray(w_s, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    cl = specs["Aero"]["CL"]["Tko"]
    balanced_field = specs["Performance"]["TOFL"]
    vstall = specs["Performance"]["Vels"]["Stl"]
    balanced_field = balanced_field * convert_length(1, "m", "ft")
    rho_rwy = 1
    top25 = balanced_field / 37.5
    w_s = w_s * 9.81 * convert_force(1, "N", "lbf")
    w_s = w_s / convert_length(1, "m", "ft") ** 2

    if aircraft_class.lower() in ("turboprop", "piston"):
        t_w = 1 / (1.1 * vstall * t_w)

    return w_s / (rho_rwy * cl * top25) - t_w


def jet_lfl(w_s, t_w, aircraft):
    """Return the landing-field-length constraint residual."""

    _ = t_w
    w_s = np.asarray(w_s, dtype=float)
    specs = aircraft["Specs"]
    req_type = specs["TLAR"]["ReqType"]
    cl = specs["Aero"]["CL"]["Lnd"]
    landing_field = specs["Performance"]["LFL"] * convert_length(1, "m", "ft")
    obstacle = specs["Performance"]["ObstLen"] * convert_length(1, "m", "ft")
    wland_mtow = specs["Performance"]["Wland_MTOW"]
    rho_rwy = 0.95
    w_s = w_s * 9.81 * convert_force(1, "N", "lbf")
    w_s = w_s / convert_length(1, "m", "ft") ** 2

    if req_type == 0:
        distance = 0.6 * landing_field - obstacle
        return w_s - rho_rwy * cl * distance / 80 / wland_mtow

    vapp = np.sqrt(landing_field / 0.3)
    vstall = vapp / 1.3
    vstall = vstall * convert_velocity(1, "kts", "ft/s")
    w_s_required = 0.5 * 0.002377 * vstall ** 2 * cl / wland_mtow
    return w_s - w_s_required


def jet25_111(w_s, t_w, aircraft):
    """Return FAR 25.111 takeoff-climb residual."""

    specs = aircraft["Specs"]
    cd0 = specs["Aero"]["CD0"]["Tko"] - 0.025
    gradient = far25_engine_gradient(aircraft, 0.012, 0.015, 0.017)

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0.5026, -42.54, 0.7925, 1.198)

    correction = specs["Performance"]["TempInc"] * oei_multiplier(aircraft)
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Tko"],
        cd0,
        specs["Aero"]["e"]["Tko"],
        correction,
        gradient,
        1.2,
        "Jet25_111",
    )


def jet25_119(w_s, t_w, aircraft):
    """Return FAR 25.119 balked-landing climb residual."""

    specs = aircraft["Specs"]
    gradient = 0.032

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0, 0, 0, 3.2)

    correction = specs["Performance"]["TempInc"]
    correction *= specs["Performance"]["Wland_MTOW"]
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Lnd"],
        specs["Aero"]["CD0"]["Lnd"],
        specs["Aero"]["e"]["Lnd"],
        correction,
        gradient,
        1.3,
        "Jet25_119",
    )


def jet25_121a(w_s, t_w, aircraft):
    """Return FAR 25.121a transition-segment climb residual."""

    specs = aircraft["Specs"]
    gradient = far25_engine_gradient(aircraft, 0, 0.003, 0.005)

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0.4999, -151.22, 0.7855, 0.001)

    correction = specs["Performance"]["TempInc"] * oei_multiplier(aircraft)
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Tko"],
        specs["Aero"]["CD0"]["Tko"],
        specs["Aero"]["e"]["Tko"],
        correction,
        gradient,
        1.15,
        "Jet25_121a",
    )


def jet25_121b(w_s, t_w, aircraft):
    """Return FAR 25.121b second-segment climb residual."""

    specs = aircraft["Specs"]
    cd0 = specs["Aero"]["CD0"]["Tko"] - 0.025
    gradient = far25_engine_gradient(aircraft, 0.024, 0.027, 0.030)

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0.6024, -42.00, 0.7829, 2.398)

    correction = specs["Performance"]["TempInc"] * oei_multiplier(aircraft)
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Tko"],
        cd0,
        specs["Aero"]["e"]["Tko"],
        correction,
        gradient,
        1.2,
        "Jet25_121b",
    )


def jet25_121c(w_s, t_w, aircraft):
    """Return FAR 25.121c enroute-climb residual."""

    specs = aircraft["Specs"]
    gradient = far25_engine_gradient(aircraft, 0.012, 0.015, 0.017)

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0.5026, -42.54, 0.7925, 1.198)

    correction = specs["Performance"]["TempInc"] * oei_multiplier(aircraft)
    correction *= specs["Performance"]["MaxCont"]
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Crs"],
        specs["Aero"]["CD0"]["Crs"],
        specs["Aero"]["e"]["Crs"],
        correction,
        gradient,
        1.25,
        "Jet25_121c",
    )


def jet25_121d(w_s, t_w, aircraft):
    """Return FAR 25.121d landing-climb residual."""

    specs = aircraft["Specs"]
    cd0 = (specs["Aero"]["CD0"]["Lnd"] + specs["Aero"]["CD0"]["Tko"]) / 2
    gradient = far25_engine_gradient(aircraft, 0.021, 0.024, 0.027)

    if aircraft["Settings"]["ConstraintType"] == 1:
        gradient = sigmoid(aircraft, 0.6025, -41.75, 0.7830, 2.098)

    correction = specs["Performance"]["TempInc"] * oei_multiplier(aircraft)
    correction *= specs["Performance"]["Wland_MTOW"]
    return far25_climb(
        w_s,
        t_w,
        aircraft,
        specs["Aero"]["CL"]["Lnd"],
        cd0,
        specs["Aero"]["e"]["Lnd"],
        correction,
        gradient,
        1.5,
        "Jet25_121d",
    )


def jet_aeo_climb(w_s, t_w, aircraft):
    """Return the all-engines-operative climb constraint residual."""

    w_s = np.asarray(w_s, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    req_type = specs["TLAR"]["ReqType"]
    cl = specs["Aero"]["CL"]["Crs"]
    cd0 = specs["Aero"]["CD0"]["Crs"]
    aspect_ratio = specs["Aero"]["AR"]
    oswald = specs["Aero"]["e"]["Tko"]
    vstall = specs["Performance"]["Vels"]["Stl"]
    gradient = specs["Performance"]["ExtraGrad"]
    correction = oei_multiplier(aircraft)
    ks = 1.2

    if req_type == 0:
        if aircraft_class.lower() in ("turboprop", "piston"):
            t_w = 1 / (ks * vstall * t_w)

        residual = ks ** 2 * cd0 / cl
        residual += cl / ks ** 2 / np.pi / aspect_ratio / oswald
        return correction * (residual + gradient) - t_w

    w_s = w_s * 9.81 * convert_force(1, "N", "lbf")
    w_s = w_s / convert_length(1, "m", "ft") ** 2
    rho_sls = sea_level_density_english()

    if req_type == 1:
        vstall_english = vstall * convert_velocity(1, "m/s", "ft/s")
        q = 0.5 * rho_sls * (vstall_english * ks) ** 2

        if aircraft_class.lower() in ("turboprop", "piston"):
            velocity = convert_velocity(vstall_english * ks, "ft/s", "m/s")
            t_w = 1 / (velocity * t_w)

        residual = q * cd0 / w_s + w_s / q / (np.pi * aspect_ratio * oswald)
        return correction * (residual + gradient) - t_w

    if req_type == 2:
        cl = cl / ks ** 2
        qinf = w_s / cl
        vinf = np.sqrt(2 * qinf / rho_sls)

        if aircraft_class.lower() in ("turboprop", "piston"):
            velocity = convert_velocity(vinf, "ft/s", "m/s")
            t_w = 1 / (velocity * t_w)

        residual = qinf / w_s * (cd0 + cl ** 2 / (np.pi * aspect_ratio * oswald))
        return residual + gradient - t_w

    raise ConstraintError(
        "JetAEOClimb ReqType must be 0, 1, or 2."
    )


def far25_climb(w_s, t_w, aircraft, cl, cd0, oswald, correction, gradient, ks, label):
    """Shared FAR 25 climb residual implementation."""

    w_s = np.asarray(w_s, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    req_type = specs["TLAR"]["ReqType"]
    aspect_ratio = specs["Aero"]["AR"]
    vstall = specs["Performance"]["Vels"]["Stl"]

    if req_type == 0:
        if aircraft_class.lower() in ("turboprop", "piston"):
            t_w = 1 / (ks * vstall * t_w)

        residual = ks ** 2 * cd0 / cl + cl / ks ** 2 / np.pi / aspect_ratio / oswald
        return correction * (residual + gradient) - t_w

    w_s = w_s * 9.81 * convert_force(1, "N", "lbf")
    w_s = w_s / convert_length(1, "m", "ft") ** 2
    rho_sls = sea_level_density_english()

    if req_type == 1:
        vstall_english = vstall * convert_velocity(1, "m/s", "ft/s")
        q = 0.5 * rho_sls * (vstall_english * ks) ** 2

        if aircraft_class.lower() in ("turboprop", "piston"):
            velocity = convert_velocity(vstall_english * ks, "ft/s", "m/s")
            t_w = 1 / (velocity * t_w)

        residual = q * cd0 / w_s + w_s / q / (np.pi * aspect_ratio * oswald)
        return correction * (residual + gradient) - t_w

    if req_type == 2:
        cl = cl / ks ** 2
        qinf = w_s / cl
        vinf = np.sqrt(2 * qinf / rho_sls)

        if aircraft_class.lower() in ("turboprop", "piston"):
            velocity = convert_velocity(vinf, "ft/s", "m/s")
            t_w = 1 / (velocity * t_w)

        residual = qinf / w_s * (cd0 + cl ** 2 / (np.pi * aspect_ratio * oswald))
        return correction * (residual + gradient) - t_w

    raise ConstraintError(f"{label} ReqType must be 0, 1, or 2.")


def far25_engine_gradient(aircraft, two_engine, three_engine, four_engine):
    """Return FAR 25 climb gradient for a discrete engine count."""

    constraint_type = aircraft["Settings"]["ConstraintType"]

    if constraint_type != 0:
        if constraint_type == 1:
            return None

        raise ConstraintError("FAR 25 constraint Type must be 0 or 1.")

    neng = aircraft["Specs"]["Propulsion"]["NumEngines"]

    if neng == 2:
        return two_engine

    if neng == 3:
        return three_engine

    return four_engine


def sea_level_density_english():
    """Return sea-level density in slug/ft^3."""

    _, _, _, _, _, rho_sls, _ = compute_flight_conditions(0, 0, "Mach", 0)
    rho_sls = rho_sls * convert_mass(1, "kg", "slug")
    return rho_sls / convert_length(1, "m", "ft") ** 3


def jet_crs(w_s, t_w, aircraft):
    """Return the cruise-performance constraint residual."""

    return jet_cruise_like(w_s, t_w, aircraft, "Crs", 0.6, 0.1)


def jet_div(w_s, t_w, aircraft):
    """Return the diversion cruise-performance constraint residual."""

    return jet_cruise_like(w_s, t_w, aircraft, "Div", 0.6, 0.2)


def jet_cruise_like(w_s, t_w, aircraft, segment, lapse_exp, devries_exp):
    """Shared cruise/diversion residual implementation."""

    w_s = np.asarray(w_s, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    req_type = specs["TLAR"]["ReqType"]
    cd0 = specs["Aero"]["CD0"]["Crs"]
    aspect_ratio = specs["Aero"]["AR"]
    oswald = specs["Aero"]["e"]["Crs"]
    mach = specs["Performance"]["Vels"][segment]
    altitude = specs["Performance"]["Alts"][segment]
    q, rho_ratio, velocity = cruise_dynamic_pressure(altitude, mach)
    w_s = w_s * convert_mass(1, "kg", "lbm")
    w_s = w_s / convert_length(1, "m", "ft") ** 2

    if aircraft_class.lower() in ("turboprop", "piston"):
        velocity_mps = convert_velocity(velocity, "ft/s", "m/s")
        t_w = 1 / (velocity_mps * t_w)

    if req_type in (0, 1):
        residual = q * cd0 / w_s + w_s / (q * np.pi * aspect_ratio * oswald)

        if aircraft_class.lower() in ("turboprop", "piston"):
            return residual - t_w

        return residual / rho_ratio ** lapse_exp - t_w

    if req_type == 2:
        cl = w_s / q
        residual = q / w_s * (cd0 + cl ** 2 / (np.pi * aspect_ratio * oswald))
        return residual / rho_ratio ** devries_exp - t_w

    raise ConstraintError(
        "Jet cruise constraints require ReqType 0, 1, or 2."
    )


def jet_ceil(w_s, t_w, aircraft):
    """Return the service-ceiling constraint residual."""

    w_s = np.asarray(w_s, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    specs = aircraft["Specs"]
    aircraft_class = specs["TLAR"]["Class"]
    req_type = specs["TLAR"]["ReqType"]
    cd0 = specs["Aero"]["CD0"]["Crs"]
    aspect_ratio = specs["Aero"]["AR"]
    oswald = specs["Aero"]["e"]["Crs"]
    altitude = specs["Performance"]["Alts"]["Srv"]
    mach = specs["Performance"]["Vels"]["Crs"]
    _, _, _, _, _, rho_sls, _ = compute_flight_conditions(0, 0, "Mach", mach)
    _, velocity, _, _, _, rho_srv, _ = compute_flight_conditions(
        altitude,
        0,
        "Mach",
        mach,
    )
    rho_ratio = rho_srv / rho_sls
    gradient = 0.001

    if req_type == 0:
        residual = 2 * np.sqrt(cd0 / np.pi / oswald / aspect_ratio) + gradient
        return residual / rho_ratio ** 0.6 - t_w

    if req_type == 1:
        w_s = w_s * 9.81 * convert_force(1, "N", "lbf")
        w_s = w_s / convert_length(1, "m", "ft") ** 2
        rho_sls = rho_sls * convert_mass(1, "kg", "slug")
        rho_sls = rho_sls / convert_length(1, "m", "ft") ** 3
        velocity = velocity * convert_velocity(1, "m/s", "ft/s")
        q = 0.5 * rho_sls * velocity ** 2

        if aircraft_class.lower() in ("turboprop", "piston"):
            velocity_mps = convert_velocity(velocity, "ft/s", "m/s")
            t_w = 1 / (velocity_mps * t_w)

        residual = q * cd0 / w_s + w_s / q / (np.pi * aspect_ratio * oswald)
        residual = residual + gradient
        return residual / rho_ratio ** 0.6 - t_w

    raise ConstraintError("JetCeil ReqType must be 0 or 1.")


def cruise_dynamic_pressure(altitude, mach):
    """Return English-unit dynamic pressure, density ratio, and velocity."""

    _, _, _, _, _, rho_sls, _ = compute_flight_conditions(0, 0, "Mach", 0)
    _, _, _, temp, _, rho, _ = compute_flight_conditions(altitude, 0, "Mach", 0)
    sound_speed = np.sqrt(1.4 * 1716 * convert_temperature(temp, "K", "R"))
    velocity = sound_speed * mach
    rho_slug = rho * convert_mass(1, "kg", "slug")
    rho_slug = rho_slug / convert_length(1, "m", "ft") ** 3
    q = 0.5 * rho_slug * velocity ** 2
    return q, rho / rho_sls, velocity


Sigmoid = sigmoid
ConstraintDiagram = constraint_diagram
OEIMultiplier = oei_multiplier
JetApp = jet_app
JetCeil = jet_ceil
JetCrs = jet_crs
JetDiv = jet_div
JetLFL = jet_lfl
Jet25_111 = jet25_111
Jet25_119 = jet25_119
Jet25_121a = jet25_121a
Jet25_121b = jet25_121b
Jet25_121c = jet25_121c
Jet25_121d = jet25_121d
JetAEOClimb = jet_aeo_climb
JetTOFL = jet_tofl
