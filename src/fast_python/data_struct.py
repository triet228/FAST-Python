# src/fast_python/data_struct.py

"""Data-structure helpers ported from FAST DataStructPkg.

The preprocessing functions normalize user aircraft dictionaries into the
nested FAST structure expected by the sizing, mission, and propulsion modules.
Most functions return deep-copied aircraft dictionaries; small helper functions
that update preallocated mission-history arrays document their in-place effects
locally.
"""

import math
from copy import deepcopy

import numpy as np

from fast_python.database import load_ideas_database
from fast_python.projection import kpp_projection
from fast_python.regression import (
    nlgpr,
    prior_calculation,
    reg_processing,
    search_db,
    vary_user_inputs,
)
from fast_python.units import convert_velocity


NAN = math.nan


DEFAULT_AIRCRAFT = {
    "Specs": {
        "TLAR": {
            "EIS": NAN,
            "Class": NAN,
            "MaxPax": NAN,
        },
        "Performance": {
            "Vels": {
                "Tko": NAN,
                "Crs": NAN,
            },
            "Alts": {
                "Tko": NAN,
                "Crs": NAN,
            },
            "RCMax": NAN,
            "Range": NAN,
        },
        "Aero": {
            "L_D": {
                "Clb": NAN,
                "Crs": NAN,
                "Des": NAN,
            },
            "W_S": {
                "SLS": NAN,
            },
        },
        "Weight": {
            "MTOW": NAN,
            "EG": NAN,
            "EM": NAN,
            "Fuel": NAN,
            "Batt": NAN,
            "WairfCF": NAN,
            "EAP": NAN,
        },
        "Propulsion": {
            "Engine": NAN,
            "NumEngines": NAN,
            "T_W": {
                "SLS": NAN,
            },
            "Thrust": {
                "SLS": NAN,
            },
            "Eta": {
                "Prop": NAN,
            },
            "MDotCF": NAN,
            "PropArch": {
                "Type": NAN,
            },
        },
        "Power": {
            "SLS": NAN,
            "SpecEnergy": {
                "Fuel": NAN,
                "Batt": NAN,
            },
            "Eta": {
                "EM": NAN,
                "EG": NAN,
                "Propeller": NAN,
            },
            "P_W": {
                "SLS": NAN,
                "EM": NAN,
                "EG": NAN,
            },
            "LamDwn": {
                "SLS": NAN,
                "Tko": NAN,
                "Clb": NAN,
                "Crs": NAN,
                "Des": NAN,
                "Lnd": NAN,
            },
            "LamUps": {
                "SLS": NAN,
                "Tko": NAN,
                "Clb": NAN,
                "Crs": NAN,
                "Des": NAN,
                "Lnd": NAN,
            },
            "Battery": {
                "ParCells": NAN,
                "SerCells": NAN,
                "BegSOC": NAN,
            },
        },
        "Battery": {
            "NomVolCell": NAN,
            "MaxExtVolCell": NAN,
            "CapCell": NAN,
            "IntResist": NAN,
            "ExpVol": NAN,
            "ExpCap": NAN,
            "MinSOC": NAN,
            "MaxAllowCRate": NAN,
            "Chem": NAN,
            "GroundT": NAN,
            "Cpower": NAN,
            "FEC": NAN,
            "SOH": NAN,
            "OpTemp": NAN,
            "Degradation": NAN,
        },
    },
    "Settings": {
        "TkoPoints": NAN,
        "ClbPoints": NAN,
        "CrsPoints": NAN,
        "DesPoints": NAN,
        "OEW": {
            "MaxIter": NAN,
            "Tol": NAN,
        },
        "Analysis": {
            "MaxIter": NAN,
            "Type": NAN,
        },
        "Plotting": NAN,
        "PrintOut": NAN,
        "Table": NAN,
        "VisualizeAircraft": NAN,
        "Degradation": NAN,
        "Dir": {
            "Size": NAN,
            "Oper": NAN,
        },
    },
    "Geometry": {
        "LengthSet": NAN,
        "Preset": NAN,
    },
}


class DataStructError(ValueError):
    """Report invalid aircraft data-structure inputs."""


def pre_spec_processing(aircraft):
    """Fill omitted aircraft specification fields with NaN defaults.

    Inputs:
        aircraft: User aircraft dictionary, usually loaded from wrapper JSON.

    Outputs:
        A deep-copied aircraft dictionary with the FAST-required nested Specs,
        Settings, and Geometry fields present.

    Assumptions:
        MATLAB's PreSpecProcessing is field-presence oriented. The Python port
        uses one recursive default tree so future processing can rely on stable
        paths without mutating the caller's object.
    """

    processed = deepcopy(aircraft)
    merge_defaults(processed, DEFAULT_AIRCRAFT)
    return processed


def init_mission_history(aircraft):
    """Initialize zero-filled mission history arrays.

    Inputs:
        aircraft: Dictionary with a processed Mission.Profile and PropArch.

    Outputs:
        A deep-copied aircraft dictionary whose Mission.History mirrors
        DataStructPkg.InitMissionHistory.

    Assumptions:
        Profile segment indices are one-based, matching FAST/Matlab. Python
        stores the allocated arrays as nested lists to remain JSON-friendly.
    """

    aircraft = deepcopy(aircraft)
    profile = aircraft["Mission"]["Profile"]
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    settings = aircraft.get("Settings", {})
    npnt = int(profile["SegEnd"][-1])
    ncomp = len(prop_arch["Arch"])
    nsrc = len(prop_arch_vector(prop_arch["SrcType"]))
    ntrn = len(prop_arch_vector(prop_arch["TrnType"]))
    narg_ups = max(1, int(settings.get("nargOperUps", 0)))
    narg_dwn = max(1, int(settings.get("nargOperDwn", 0)))

    performance = {
        "Time": zero_vector(npnt),
        "Dist": zero_vector(npnt),
        "TAS": zero_vector(npnt),
        "EAS": zero_vector(npnt),
        "RC": zero_vector(npnt),
        "Alt": zero_vector(npnt),
        "Acc": zero_vector(npnt),
        "FPA": zero_vector(npnt),
        "Mach": zero_vector(npnt),
        "Rho": zero_vector(npnt),
        "Ps": zero_vector(npnt),
    }
    propulsion = {
        "TSFC": zero_matrix(npnt, ntrn),
        "ExitMach": zero_matrix(npnt, ntrn),
        "FanDiam": zero_matrix(npnt, ntrn),
        "MDotAir": zero_matrix(npnt, ntrn),
        "MDotFuel": zero_matrix(npnt, ntrn),
    }
    weight = {
        "CurWeight": zero_vector(npnt),
        "Fburn": zero_vector(npnt),
    }
    power = {
        "TV": zero_vector(npnt),
        "Req": zero_vector(npnt),
        "LamUps": zero_matrix(npnt, narg_ups),
        "LamDwn": zero_matrix(npnt, narg_dwn),
        "SOC": zero_matrix(npnt, nsrc),
        "Pav": zero_matrix(npnt, ncomp),
        "Preq": zero_matrix(npnt, ncomp),
        "Tav": zero_matrix(npnt, ncomp),
        "Treq": zero_matrix(npnt, ncomp),
        "Pout": zero_matrix(npnt, ncomp),
        "Tout": zero_matrix(npnt, ncomp),
        "Voltage": zero_matrix(npnt, nsrc),
        "Current": zero_matrix(npnt, nsrc),
        "Capacity": zero_matrix(npnt, nsrc),
    }
    energy = {
        "KE": zero_vector(npnt),
        "PE": zero_vector(npnt),
        "E_ES": zero_matrix(npnt, nsrc),
        "Eleft_ES": zero_matrix(npnt, nsrc),
    }
    mission_vars = {
        "Performance": performance,
        "Propulsion": propulsion,
        "Weight": weight,
        "Power": power,
        "Energy": energy,
    }
    aircraft["Mission"]["History"] = {
        "SI": mission_vars,
        "EE": deepcopy(mission_vars),
        "Segment": ["" for _ in range(npnt)],
    }
    return aircraft


def clear_mission(aircraft, ielem=0):
    """Reset mission-history values from a one-based index onward.

    Inputs:
        aircraft: Dictionary with Mission.History already initialized.
        ielem: Optional one-based starting index. Use 0 to reset all rows.

    Outputs:
        A deep-copied aircraft dictionary with SI history arrays and segment
        labels cleared.

    Assumptions:
        This follows DataStructPkg.ClearMission: English-unit mirrors are left
        untouched, while the SI working arrays are reset before a mission is
        re-flown.
    """

    aircraft = deepcopy(aircraft)
    start = 0 if int(ielem) == 0 else int(ielem) - 1
    history = aircraft["Mission"]["History"]

    for section_name in ("Performance", "Propulsion", "Weight", "Power", "Energy"):
        section = history["SI"][section_name]

        for field_name, value in section.items():
            section[field_name] = reset_sequence(value, start, 0)

    history["Segment"] = reset_sequence(history["Segment"], start, "")
    return aircraft


def spec_processing(aircraft, database=None):
    """Fill unknown aircraft specification values before FAST sizing.

    Inputs:
        aircraft: Aircraft dictionary after pre_spec_processing().
        database: Optional dictionary returned by load_ideas_database().

    Outputs:
        A deep-copied aircraft dictionary with regression/default values,
        converted energy and specific-power units, payload/crew weights, and
        default engine information populated.

    Assumptions:
        This ports the sizing-preparation behavior of DataStructPkg.SpecProcessing
        for dictionary data. Visualization presets are stored as descriptive
        strings because Python has no MATLAB function-handle equivalent here.
    """

    aircraft = deepcopy(aircraft)
    database = database or load_ideas_database()
    specs = aircraft["Specs"]
    settings = aircraft["Settings"]
    geometry = aircraft["Geometry"]
    tlar = specs["TLAR"]
    performance = specs["Performance"]
    aero = specs["Aero"]
    weight = specs["Weight"]
    propulsion = specs["Propulsion"]
    power = specs["Power"]

    normalize_prop_arch_type(propulsion)
    validate_required_specs(tlar, performance, propulsion)

    aircraft_class = tlar["Class"]

    if is_missing(tlar["EIS"]):
        tlar["EIS"] = 2021

    if aircraft_class == "Turbofan":
        data_ac = database["TurbofanAC"]
        data_engine = database["TurbofanEngines"]
    elif aircraft_class == "Turboprop":
        data_ac = database["TurbopropAC"]
        data_engine = database["TurbopropEngines"]
    else:
        raise DataStructError(f"Unsupported aircraft class: {aircraft_class}")

    default_performance = {
        "Vels": {
            "Tko": performance["Vels"]["Tko"],
        },
        "Alts": {},
    }
    default_aero = {
        "TipChord": NAN,
        "RootChord": NAN,
        "TaperRatio": NAN,
        "Sweep": NAN,
        "WingtipDevice": NAN,
        "MAC": NAN,
        "S": NAN,
        "AR": NAN,
        "L_D": {
            "CrsMAC": NAN,
        },
    }
    default_weight = {
        "Cargo": NAN,
        "OEW": NAN,
        "OEW_MTOW": NAN,
        "Fuel": NAN,
        "FuelFrac": NAN,
        "MTOW": NAN,
    }
    default_propulsion = {
        "Fuel": {
            "Type": NAN,
            "Density": NAN,
            "CapUsable": NAN,
            "CapUnusable": NAN,
        },
        "EngineDesignation": NAN,
        "AlternateEngines": NAN,
        "NumEngines": 2,
        "MDotCF": 1,
        "Eta": {
            "Therm": 0.3,
            "Prop": 0.85,
        },
        "T_W": {
            "SLS": get_nested_or_nan(propulsion, ["T_W", "SLS"]),
        },
        "Thrust": {
            "Crs": NAN,
            "SLS": get_nested_or_nan(propulsion, ["Thrust", "SLS"]),
        },
    }
    default_power = {
        "SLS": get_nested_or_nan(power, ["SLS"]),
        "SpecEnergy": {
            "Fuel": default_fuel_specific_energy(aircraft_class),
            "Batt": kpp_projection(
                aircraft_class,
                tlar["EIS"],
                "Battery Specific Energy",
            ) / 1.0e3,
        },
        "Eta": {
            "EM": 0.96,
            "EG": 0.96,
            "Propeller": 0.8,
        },
        "P_W": {
            "SLS": get_nested_or_nan(power, ["P_W", "SLS"]),
            "EG": 5,
            "EM": kpp_projection(
                aircraft_class,
                tlar["EIS"],
                "Electric Motor Specific Power",
            ),
        },
        "LamDwn": zero_split_defaults(),
        "LamUps": zero_split_defaults(),
        "Battery": {
            "ParCells": NAN,
            "SerCells": NAN,
            "BegSOC": NAN,
        },
    }
    default_settings = {
        "TkoPoints": 10,
        "ClbPoints": 10,
        "CrsPoints": 10,
        "DesPoints": 10,
        "OEW": {
            "MaxIter": 20,
            "Tol": 1.0e-6,
        },
        "Analysis": {
            "MaxIter": 50,
            "Type": 1,
        },
        "Plotting": 0,
        "Table": 0,
        "VisualizeAircraft": 0,
        "PrintOut": 1,
        "Degradation": 0,
        "Dir": {
            "Size": ".",
            "Oper": "EAP-CNAP",
        },
    }
    default_geometry = {
        "LengthSet": NAN,
        "Preset": geometry_preset_name(aircraft_class, tlar["MaxPax"]),
    }

    fill_regression_defaults(
        aircraft,
        data_ac,
        aircraft_class,
        default_performance,
        default_aero,
        default_weight,
        default_propulsion,
        default_power,
    )
    fill_dependent_defaults(
        aircraft_class,
        performance,
        aero,
        weight,
        propulsion,
        default_performance,
        default_aero,
        default_weight,
        default_propulsion,
        default_power,
    )
    default_geometry["LengthSet"] = regression_scalar(
        data_ac,
        [["Specs", "TLAR", "MaxPax"], ["Specs", "Aero", "Length"]],
        [tlar["MaxPax"]],
        [1],
    )

    fill_nan_defaults(performance, default_performance)
    fill_nan_defaults(aero, default_aero)
    fill_nan_defaults(weight, default_weight)
    fill_nan_defaults(propulsion, default_propulsion)
    fill_nan_defaults(power, default_power)
    fill_nan_defaults(settings, default_settings)
    fill_nan_defaults(geometry, default_geometry)
    remove_class_incompatible_fields(aircraft_class, propulsion, power)
    convert_spec_units(settings, power)
    weight["Payload"] = tlar["MaxPax"] * 95

    if settings["Analysis"]["Type"] > -2:
        weight["Crew"] = weight["Payload"] / 26.1

    aircraft["HistData"] = {
        "AC": data_ac,
        "Eng": data_engine,
    }
    aircraft["RegressionParams"] = build_regression_params(
        aircraft_class,
        data_ac,
        data_engine,
    )
    aircraft = engine_spec_processing(aircraft, database)
    return aircraft


def engine_spec_processing(aircraft, database=None):
    """Create default engine specs when an aircraft omits a valid engine."""

    aircraft = deepcopy(aircraft)
    database = database or load_ideas_database()
    specs = aircraft["Specs"]
    propulsion = specs["Propulsion"]

    if isinstance(propulsion.get("Engine"), dict):
        return aircraft

    aircraft_class = specs["TLAR"]["Class"]

    if aircraft_class == "Turbofan":
        data = database["TurbofanEngines"]
        design_thrust = propulsion["Thrust"]["SLS"] / propulsion["NumEngines"]
        two_spool_data, _ = search_db(data, "LPCStages", 0)
        lp_rpm = regression_scalar(two_spool_data, [["Thrust_SLS"], ["LP100"]], [design_thrust], [1])
        hp_rpm = regression_scalar(two_spool_data, [["Thrust_SLS"], ["HP100"]], [design_thrust], [1])
        engine = {
            "Mach": 0.05,
            "Alt": 0,
            "DesignThrust": design_thrust,
            "Tt4Max": 1800,
            "NoSpools": 2,
            "FanGearRatio": NAN,
            "FanBoosters": False,
            "MaxIter": 300,
            "CoreFlow": {
                "PaxBleed": 0.03,
                "Leakage": 0.01,
                "Cooling": 0.2,
            },
            "EtaPoly": {
                "Inlet": 0.99,
                "Diffusers": 0.99,
                "Fan": 0.95,
                "Compressors": 0.95,
                "BypassNozzle": 0.99,
                "Combustor": 0.995,
                "Turbines": 0.95,
                "CoreNozzle": 0.99,
                "Nozzles": 0.99,
                "Mixing": 0,
            },
            "RPMs": [
                lp_rpm,
                hp_rpm,
            ],
            "OPR": regression_scalar(data, [["Thrust_SLS"], ["OPR_SLS"]], [design_thrust], [1]),
            "BPR": regression_scalar(data, [["Thrust_SLS"], ["BPR"]], [design_thrust], [1]),
            "FPR": regression_scalar(data, [["Thrust_SLS"], ["FPR"]], [design_thrust], [1]),
        }
    elif aircraft_class == "Turboprop":
        data = database["TurbopropEngines"]
        req_power = specs["Power"]["SLS"] / propulsion["NumEngines"]
        ip_rpm = regression_scalar(data, [["Power_SLS"], ["IPMaxTO"]], [req_power], [1])
        hp_rpm = regression_scalar(data, [["Power_SLS"], ["HPMaxTO"]], [req_power], [1])
        engine = {
            "Mach": 0.05,
            "Alt": 0,
            "ReqPower": req_power,
            "Tt4Max": 1200,
            "NPR": 1.3,
            "NoSpools": 2,
            "EtaPoly": {
                "Inlet": 0.99,
                "Diffusers": 0.99,
                "Compressors": 0.9,
                "Combustor": 0.995,
                "Turbines": 0.9,
                "Nozzles": 0.985,
            },
            "OPR": regression_scalar(data, [["Power_SLS"], ["OPR_SLS"]], [req_power], [1]),
            "RPMs": [
                hp_rpm,
                ip_rpm,
            ],
        }
    else:
        raise DataStructError(f"Unsupported aircraft class: {aircraft_class}")

    propulsion["Engine"] = engine
    return aircraft


def merge_defaults(target, defaults):
    """Recursively insert missing dictionary defaults."""

    for key, default in defaults.items():
        if key not in target or target[key] is None:
            target[key] = deepcopy(default)
        elif isinstance(target[key], dict) and isinstance(default, dict):
            merge_defaults(target[key], default)


def zero_vector(rows):
    """Return a zero column represented as a list."""

    return [0.0 for _ in range(rows)]


def zero_matrix(rows, columns):
    """Return a zero matrix represented as nested lists."""

    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def prop_arch_vector(value):
    """Return source/transmitter metadata as a one-dimensional array."""

    if hasattr(value, "value"):
        value = value.value

    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        return array.reshape(1)

    return array.reshape(-1)


def reset_sequence(value, start, replacement):
    """Return value with rows from start onward reset."""

    if not isinstance(value, list):
        return deepcopy(replacement)

    result = deepcopy(value)

    for index in range(start, len(result)):
        result[index] = reset_like(result[index], replacement)

    return result


def reset_like(value, replacement):
    """Return replacement with the same list nesting as value."""

    if isinstance(value, list):
        return [reset_like(item, replacement) for item in value]

    return deepcopy(replacement)


def normalize_prop_arch_type(propulsion):
    """Normalize PropArch string shortcuts into dictionaries."""

    prop_arch = propulsion.get("PropArch")

    if isinstance(prop_arch, str):
        propulsion["PropArch"] = {
            "Type": prop_arch,
        }


def validate_required_specs(tlar, performance, propulsion):
    """Validate required user inputs before sizing preprocessing."""

    if not is_text(tlar.get("Class")):
        raise DataStructError("Aircraft Class (Aircraft.TLAR.Class) not specified.")

    if is_missing(tlar.get("MaxPax")):
        raise DataStructError("Number of Passengers (Aircraft.TLAR.MaxPax) not specified.")

    if is_missing(performance.get("Range")):
        raise DataStructError("Design Range (Aircraft.Performance.Range) not specified.")

    prop_arch = propulsion.get("PropArch", {})

    if not isinstance(prop_arch, dict) or not is_text(prop_arch.get("Type")):
        raise DataStructError("Propulsion Architecture (Aircraft.Propulsion.Arch) not specified.")


def fill_regression_defaults(
    aircraft,
    data_ac,
    aircraft_class,
    default_performance,
    default_aero,
    default_weight,
    default_propulsion,
    default_power,
):
    """Run SpecProcessing regressions for user-omitted values."""

    specs = aircraft["Specs"]
    tlar = specs["TLAR"]
    knowns, unknowns = vary_user_inputs(aircraft, aircraft_class)
    input_paths = [
        ["Specs", "TLAR", "EIS"],
        ["Specs", "Performance", "Range"],
        ["Specs", "TLAR", "MaxPax"],
    ] + knowns["names"]
    target = [
        tlar["EIS"],
        specs["Performance"]["Range"],
        tlar["MaxPax"],
    ] + knowns["values"]
    weights = [1.0 for _ in range(len(target))]
    weights[0] = 0.2

    for output in unknowns:
        io_space = input_paths + [output]

        if output == ["Specs", "Performance", "Vels", "Crs"]:
            default_performance.setdefault("Vels", {})["Crs"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )
        elif output == ["Specs", "Performance", "Alts", "Crs"]:
            default_performance.setdefault("Alts", {})["Crs"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )
        elif output == ["Specs", "Aero", "L_D", "Crs"]:
            if aircraft_class == "Turbofan":
                io_space[-1] = ["Specs", "Aero", "L_D", "CrsMAC"]
                default_aero.setdefault("L_D", {})["Crs"] = regression_scalar(
                    data_ac,
                    io_space,
                    target,
                    weights,
                )
            elif aircraft_class == "Turboprop":
                default_aero.setdefault("L_D", {})["Crs"] = 16
        elif output == ["Specs", "Weight", "MTOW"]:
            default_weight["MTOW"] = regression_scalar(data_ac, io_space, target, weights)
        elif output == ["Specs", "Propulsion", "T_W", "SLS"]:
            default_propulsion.setdefault("T_W", {})["SLS"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )
        elif output == ["Specs", "Propulsion", "Thrust", "SLS"]:
            default_propulsion.setdefault("Thrust", {})["SLS"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )
        elif output == ["Specs", "Power", "P_W", "SLS"]:
            default_power.setdefault("P_W", {})["SLS"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )
        elif output == ["Specs", "Power", "SLS"]:
            default_power["SLS"] = regression_scalar(data_ac, io_space, target, weights)
        elif output == ["Specs", "Weight", "Fuel"]:
            default_weight["Fuel"] = regression_scalar(data_ac, io_space, target, weights)
        elif output == ["Specs", "Aero", "W_S", "SLS"]:
            default_aero.setdefault("W_S", {})["SLS"] = regression_scalar(
                data_ac,
                io_space,
                target,
                weights,
            )


def fill_dependent_defaults(
    aircraft_class,
    performance,
    aero,
    weight,
    propulsion,
    default_performance,
    default_aero,
    default_weight,
    default_propulsion,
    default_power,
):
    """Populate defaults that depend on regression or user values."""

    mtow = first_available(default_weight.get("MTOW"), weight.get("MTOW"))
    t_w_sls = first_available(
        get_nested_or_nan(default_propulsion, ["T_W", "SLS"]),
        get_nested_or_nan(propulsion, ["T_W", "SLS"]),
    )
    p_w_sls = first_available(
        get_nested_or_nan(default_power, ["P_W", "SLS"]),
        get_nested_or_nan(default_power, ["P_W", "SLS"]),
    )

    if aircraft_class == "Turbofan":
        default_propulsion.setdefault("Thrust", {})["SLS"] = t_w_sls * mtow
    elif aircraft_class == "Turboprop":
        default_power["SLS"] = p_w_sls * mtow * 1000

    cruise_ld = first_available(
        get_nested_or_nan(aero, ["L_D", "Crs"]),
        get_nested_or_nan(default_aero, ["L_D", "Crs"]),
    )
    default_aero.setdefault("L_D", {})["Clb"] = cruise_ld * 0.6
    default_aero.setdefault("L_D", {})["Des"] = cruise_ld * 0.6
    default_performance.setdefault("Alts", {})["Tko"] = 0
    default_performance["RCMax"] = 10.5

    if aircraft_class == "Turbofan":
        default_performance.setdefault("Vels", {})["Tko"] = convert_velocity(
            135,
            "kts",
            "m/s",
        )
    elif aircraft_class == "Turboprop":
        default_performance.setdefault("Vels", {})["Tko"] = convert_velocity(
            115,
            "kts",
            "m/s",
        )

    default_weight["MLW"] = 0
    default_weight["Batt"] = 0
    default_weight["EG"] = 0
    default_weight["EM"] = 0
    default_weight["EAP"] = 0
    default_weight["WairfCF"] = 1

    if aircraft_class == "Turbofan":
        default_power.setdefault("P_W", {})["AC"] = t_w_sls * default_performance["Vels"]["Tko"]
    elif aircraft_class == "Turboprop":
        default_propulsion.setdefault("Thrust", {}).setdefault("T_W", {})["SLS"] = (
            p_w_sls / default_performance["Vels"]["Tko"]
        )


def remove_class_incompatible_fields(aircraft_class, propulsion, power):
    """Remove FAST fields MATLAB drops for the active aircraft class."""

    if aircraft_class == "Turbofan":
        power.get("Eta", {}).pop("Propeller", None)
        power.pop("SLS", None)
    elif aircraft_class == "Turboprop":
        propulsion.pop("T_W", None)
        propulsion.pop("Thrust", None)


def build_regression_params(aircraft_class, data_ac, data_engine):
    """Precompute expensive regression matrices used later in sizing."""

    if aircraft_class != "Turbofan":
        return {}

    io_space = [
        ["Specs", "Aero", "S"],
        ["Specs", "Propulsion", "Thrust", "SLS"],
        ["Specs", "TLAR", "EIS"],
        ["Specs", "Weight", "MTOW"],
        ["Specs", "Weight", "Airframe"],
    ]
    prior = prior_calculation(data_ac, io_space)
    data_matrix, hyperparams, inverse_term = reg_processing(
        data_ac,
        io_space,
        prior,
        [1, 1, 0.2, 1],
    )
    engine_io = [["Thrust_Max"], ["DryWeight"]]
    engine_prior = prior_calculation(data_engine, engine_io)
    engine_data_matrix, engine_hyperparams, engine_inverse = reg_processing(
        data_engine,
        engine_io,
        engine_prior,
        [1],
    )
    return {
        "OEW": {
            "DataMatrix": data_matrix,
            "HyperParams": hyperparams,
            "InverseTerm": inverse_term,
        },
        "WEngine": {
            "DataMatrix": engine_data_matrix,
            "HyperParams": engine_hyperparams,
            "InverseTerm": engine_inverse,
        },
    }


def regression_scalar(data_struct, io_space, target, weights):
    """Return the scalar posterior mean from NLGPR."""

    mean, _ = nlgpr(
        data_struct,
        io_space,
        target,
        weights=weights,
    )
    return float(np.asarray(mean).reshape(-1)[0])


def fill_nan_defaults(target, defaults):
    """Recursively replace missing values in target with defaults."""

    for key, default in defaults.items():
        if isinstance(default, dict):
            if key not in target or is_missing(target[key]) or not isinstance(target[key], dict):
                target[key] = {}

            fill_nan_defaults(target[key], default)
        elif key not in target or is_missing(target[key]):
            target[key] = deepcopy(default)


def convert_spec_units(settings, power):
    """Convert SpecProcessing power units to SI when applicable."""

    if settings["Analysis"]["Type"] == -2:
        return

    power["P_W"]["SLS"] *= 1.0e3
    power["P_W"]["EG"] *= 1.0e3
    power["P_W"]["EM"] *= 1.0e3
    power["SpecEnergy"]["Fuel"] *= 3.6e6
    power["SpecEnergy"]["Batt"] *= 3.6e6


def zero_split_defaults():
    """Return zero defaults for all named power split segments."""

    return {
        "SLS": 0,
        "Tko": 0,
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }


def default_fuel_specific_energy(aircraft_class):
    """Return default fuel specific energy in kWh/kg."""

    if aircraft_class == "Piston":
        return 4.465e7 / 3.6e6

    return 4.32e7 / 3.6e6


def geometry_preset_name(aircraft_class, max_pax):
    """Return the visualization preset name chosen by SpecProcessing."""

    if aircraft_class == "Turbofan":
        if max_pax > 200:
            return "LargeTurbofan"

        if max_pax > 100:
            return "SmallDoubleAisleTurbofan"

        return "Transport"

    if aircraft_class == "Turboprop":
        if max_pax > 19:
            return "LargeTurboprop"

        return "SmallTurboprop"

    return NAN


def first_available(primary, fallback):
    """Return the first value that is not FAST-missing."""

    if is_missing(primary):
        return fallback

    return primary


def get_nested_or_nan(value, keys):
    """Read a nested dictionary path, returning NaN when missing."""

    current = value

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return NAN

        current = current[key]

    return current


def is_text(value):
    """Return True when value is a non-NaN text scalar."""

    return isinstance(value, str) and value.lower() != "nan"


def is_missing(value):
    """Return True for FAST missing values."""

    if value is None:
        return True

    if isinstance(value, str):
        return value.lower() == "nan"

    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


PreSpecProcessing = pre_spec_processing
SpecProcessing = spec_processing
EngineSpecProcessing = engine_spec_processing
InitMissionHistory = init_mission_history
ClearMission = clear_mission
