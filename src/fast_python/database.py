# src/fast_python/database.py

"""Historical FAST database loading helpers."""

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from fast_python.atmosphere import standard_atmosphere
from fast_python.units import convert_length


DATABASE_NAMES = (
    "TurbofanAC",
    "TurbopropAC",
    "FanUnitsReference",
    "PropUnitsReference",
    "TurbofanEngines",
    "TurbopropEngines",
)


class DatabaseError(RuntimeError):
    """Report missing or invalid FAST database files."""


def load_ideas_database(fast_path=None):
    """Load FAST's IDEAS_DB.mat as nested Python dictionaries.

    Inputs:
        fast_path: Optional path to the MATLAB FAST checkout. When omitted,
            FAST_MATLAB_PATH, FAST_PATH, and ~/Projects/FAST are checked.

    Outputs:
        Dictionary containing the FAST historical aircraft and engine
        databases used by RegressionPkg.

    Assumptions:
        The database is distributed with the MATLAB FAST repository under
        +DatabasePkg/IDEAS_DB.mat. SciPy handles the MAT-file conversion while
        this module normalizes arrays/scalars for Python dictionary traversal.
    """

    database_path = resolve_database_path(fast_path)
    return load_ideas_database_file(database_path)


@lru_cache(maxsize=4)
def load_ideas_database_file(database_path):
    """Load and cache a specific IDEAS_DB.mat file."""

    database_path = Path(database_path)

    if not database_path.exists():
        raise DatabaseError(f"FAST database file was not found: {database_path}")

    raw = loadmat(database_path, simplify_cells=True)
    database = {}

    for name in DATABASE_NAMES:
        if name not in raw:
            raise DatabaseError(f"FAST database is missing {name}.")

        database[name] = sanitize_mat_value(raw[name])

    return database


def resolve_database_path(fast_path=None):
    """Return the local path to +DatabasePkg/IDEAS_DB.mat."""

    if fast_path is not None:
        return Path(fast_path) / "+DatabasePkg" / "IDEAS_DB.mat"

    candidates = []

    for env_name in ("FAST_MATLAB_PATH", "FAST_PATH"):
        value = os.environ.get(env_name)

        if value:
            candidates.append(Path(value))

    candidates.append(Path.home() / "Projects" / "FAST")

    for candidate in candidates:
        database_path = candidate / "+DatabasePkg" / "IDEAS_DB.mat"

        if database_path.exists():
            return database_path

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise DatabaseError(
        "FAST database file was not found. Set FAST_MATLAB_PATH to the MATLAB "
        f"FAST checkout. Checked: {checked}"
    )


def sanitize_mat_value(value):
    """Convert SciPy MAT values into plain Python containers where practical."""

    if isinstance(value, dict):
        return {
            str(key): sanitize_mat_value(item)
            for key, item in value.items()
        }

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return sanitize_mat_value(value.item())

        if value.dtype == object:
            return sanitize_mat_value(value.tolist())

        return value.tolist()

    if isinstance(value, list):
        return [sanitize_mat_value(item) for item in value]

    if isinstance(value, tuple):
        return [sanitize_mat_value(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def tree_branch(trunk, trunk_name):
    """Return nested-dict branches and scalar leaves for one tree level."""

    branches = []
    leaves = []

    for name, value in trunk.items():
        if isinstance(value, dict):
            branches.append((name, value))
        else:
            leaves.append((f"{trunk_name}_{name}", value))

    return branches, leaves


def struct_tree_search(trunk):
    """Return all lowest-level leaves in a nested database structure.

    Inputs:
        trunk: Nested dictionary equivalent to a MATLAB struct.

    Outputs:
        Two-row list `[names, values]`, matching DatabasePkg.StructTreeSearch's
        transposed cell-array convention.
    """

    branches = [("Plane", trunk)]
    leaves = []
    index = 0

    while index < len(branches):
        branch_name, branch = branches[index]
        index += 1
        new_branches, new_leaves = tree_branch(branch, branch_name)
        branches.extend(new_branches)
        leaves.extend(new_leaves)

    return [
        [name for name, _ in leaves],
        [value for _, value in leaves],
    ]


def randomize_db(database, percent, rng=None):
    """Randomize Settings.DataTypeValidation fields in a database tree.

    Inputs:
        database: Dict of aircraft or engine entries.
        percent: MATLAB threshold used by RandomizeDB. Entries with a random
            integer greater than this value become Validation; others become
            Training.
        rng: Optional integer seed or NumPy random generator for deterministic
            tests.

    Outputs:
        Deep-copied database with randomized Settings.DataTypeValidation
        values.
    """

    randomized = deepcopy(database)

    if rng is None:
        generator = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        generator = np.random.default_rng(int(rng))
    else:
        generator = rng

    for name in randomized:
        draw = int(generator.integers(1, 101))
        settings = randomized[name].setdefault("Settings", {})

        if draw > percent:
            settings["DataTypeValidation"] = "Validation"
        else:
            settings["DataTypeValidation"] = "Training"

    return randomized


def calc_fan_vals(plane, unitsflag):
    """Assign CalcFanVals unit metadata for turbofan database aircraft."""

    result = deepcopy(plane)

    if unitsflag == "Vals":
        return calc_fan_values(result)

    if unitsflag != "Units":
        raise DatabaseError('Enter either "Units" or "Vals" for CalcFanVals.')

    set_nested(result, ["Specs", "Performance", "Range"], "m")
    set_nested(result, ["Specs", "Aero", "L_D", "CrsBRE"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "Crs"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "Clb"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "Des"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "CrsMAC"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "CrsMAC2"], "ratio")
    set_nested(result, ["Specs", "Weight", "Airframe"], "kg")
    set_nested(result, ["Specs", "Weight", "OEW_MTOW"], "ratio")
    set_nested(result, ["Specs", "Weight", "MZFW_MTOW"], "ratio")
    set_nested(result, ["Specs", "Propulsion", "T_W", "SLS"], "ratio")
    set_nested(result, ["Specs", "Propulsion", "Thrust", "Crs"], "N")
    set_nested(result, ["Specs", "Aero", "W_S", "SLS"], "kg/m2")
    set_nested(result, ["Specs", "Weight", "EngFrac"], "ratio")
    set_nested(result, ["Specs", "Weight", "FuelFrac"], "ratio")
    set_nested(result, ["Overview", "Aisle"], "type")
    set_nested(result, ["Overview", "RangeCapability"], "type")
    set_nested(result, ["Overview", "ModelType"], "type")
    set_nested(result, ["Overview", "PayloadType"], "type")
    set_nested(result, ["Overview", "PassengerType"], "type")
    set_nested(result, ["Specs", "Aero", "TaperRatio"], "ratio")
    set_nested(result, ["Specs", "Aero", "AR"], "ratio")
    set_nested(result, ["Specs", "Performance", "Vels", "Tko"], "m/s")
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Fuel"], "kWh/kg")
    set_nested(result, ["Specs", "TLAR", "Class"], "flag")
    set_nested(result, ["Specs", "TLAR", "ADG"], "Category I through VI")
    set_nested(result, ["Specs", "TLAR", "ADGPercent"], "Percent")
    set_nested(result, ["Specs", "TLAR", "ADGLimitor"], "Type")
    set_nested(result, ["Specs", "Propulsion", "Arch", "Type"], "flag")
    set_nested(result, ["Specs", "Propulsion", "Eta", "Prop"], "efficiency")
    set_nested(result, ["Specs", "Performance", "Alts", "Tko"], "m")
    set_nested(result, ["Specs", "Performance", "RCMax"], "m/s")
    set_nested(result, ["Specs", "Weight", "Batt"], "kg")
    set_nested(result, ["Specs", "Weight", "EM"], "kg")
    set_nested(result, ["Specs", "Weight", "EG"], "kg")
    set_nested(result, ["Specs", "Weight", "Payload"], "kg")
    set_nested(result, ["Specs", "Weight", "Burden"], "kg")
    set_nested(result, ["Specs", "Weight", "Structure_Burden"], "ratio")
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Batt"], "kWh/kg")
    set_nested(result, ["Specs", "Power", "Eta", "EM"], "efficiency")
    set_nested(result, ["Specs", "Power", "Eta", "EG"], "efficiency")
    set_power_split_units(result)
    set_nested(result, ["Specs", "Power", "P_W", "AC"], "W/kg")
    set_nested(result, ["Specs", "Power", "P_W", "EM"], "kW/kg")
    set_nested(result, ["Specs", "Power", "P_W", "EG"], "kW/kg")
    set_settings_units(result)
    set_nested(result, ["Settings", "DataTypeValidation"], "data type")
    set_nested(
        result,
        ["Specs", "Propulsion", "Engine", "DataTypeValidation"],
        "data type",
    )
    result.setdefault("Overview", {}).pop("KeyWords", None)
    return result


def calc_fan_values(result):
    """Compute CalcFanVals derived values for turbofan aircraft."""

    specs = result["Specs"]
    weight = specs["Weight"]
    aero = specs["Aero"]
    propulsion = specs["Propulsion"]
    engine = propulsion["Engine"]
    performance = specs["Performance"]
    overview = result["Overview"]
    num_engines = propulsion["NumEngines"]
    c = engine["TSFC_Crs"] * 0.453592 / 4.44822 / 3600 * 9.81
    range_m = performance["Range"] * 1e3
    weight_ratio = np.log(weight["MTOW"] / (weight["MTOW"] - weight["Fuel"]))
    temp, _, _ = standard_atmosphere(performance["Alts"]["Crs"])
    mach = performance["Vels"]["Crs"]
    velocity = mach * np.sqrt(1.4 * 287 * temp)
    aero.setdefault("L_D", {})["CrsBRE"] = range_m * c / velocity / weight_ratio
    aircraft_weight = weight["MTOW"] * 0.995 * 0.985 * 9.81
    thrust = engine["Thrust_Crs"] * num_engines
    aero["L_D"]["Crs"] = aircraft_weight / thrust
    aero["L_D"]["Clb"] = np.nan
    aero["L_D"]["Des"] = np.nan
    aspect_ratio = aero["Span"] ** 2 / aero["S"]
    cmac = convert_length(aero["MAC"], "m", "ft")
    altitude = convert_length(performance["Alts"]["Crs"], "m", "ft")

    if is_nan_value(mach):
        mach = 0.8

    reynolds = 7.093e6 * cmac * mach * (1 - 0.5 * (altitude / 23500) ** 0.7)
    aero["L_D"]["CrsMAC"] = mac_ld(aspect_ratio, reynolds)

    if is_nan_value(aero["WingtipDevice"]) or aero["WingtipDevice"] == "No Wingtip Device":
        effective_ar = aspect_ratio
    else:
        effective_ar = 1.2 * aspect_ratio

    aero["L_D"]["CrsMAC2"] = mac_ld(effective_ar, reynolds)
    weight["Airframe"] = weight["OEW"] - engine["DryWeight"] * num_engines
    weight["OEW_MTOW"] = weight["OEW"] / weight["MTOW"]
    weight["MZFW_MTOW"] = weight["MZFW"] / weight["MTOW"]
    thrust_specs = propulsion.setdefault("Thrust", {})

    if is_nan_value(thrust_specs["SLS"]):
        propulsion.setdefault("T_W", {})["SLS"] = (
            engine["Thrust_Max"] * num_engines / weight["MTOW"] / 9.81
        )
    else:
        propulsion.setdefault("T_W", {})["SLS"] = (
            thrust_specs["Max"] * num_engines / weight["MTOW"] / 9.81
        )

    if is_nan_value(thrust_specs["SLS"]):
        thrust_specs["SLS"] = engine["Thrust_SLS"] * num_engines
    else:
        thrust_specs["SLS"] = thrust_specs["SLS"] * num_engines

    if is_nan_value(thrust_specs["Max"]):
        thrust_specs["Max"] = engine["Thrust_Max"] * num_engines
    else:
        thrust_specs["Max"] = thrust_specs["Max"] * num_engines

    thrust_specs["Crs"] = engine["Thrust_Crs"] * num_engines
    aero.setdefault("W_S", {})["SLS"] = weight["MTOW"] / aero["S"]
    weight["EngineFrac"] = engine["DryWeight"] * num_engines / weight["MTOW"]
    weight["FuelFrac"] = weight["Fuel"] / weight["MTOW"]
    process_fan_keywords(overview)
    overview.pop("KeyWords", None)

    if is_nan_value(overview["Monikers"]):
        overview["Monikers"] = "N/A"

    if is_nan_value(overview["AlternateDesignation"]):
        overview["AlternateDesignation"] = "N/A"

    aero["TaperRatio"] = aero["TipChord"] / aero["RootChord"]
    aero["AR"] = aspect_ratio
    performance["Vels"]["Tko"] = convert_length(
        performance["Vels"]["Tko"],
        "naut mi",
        "m",
    ) / 3600

    if is_nan_value(weight["Cargo"]):
        weight["Cargo"] = 0

    weight["Payload"] = weight["Cargo"] + specs["TLAR"]["MaxPax"] * 95
    performance["Range"] = range_m
    set_fan_fast_defaults(result)
    set_aircraft_design_group(specs)
    weight["Burden"] = (
        weight["Payload"]
        + weight["Fuel"]
        + engine["DryWeight"] * num_engines
    )
    weight["Structure_Burden"] = (
        weight["MTOW"] - weight["Burden"]
    ) / weight["Burden"]
    return result


def mac_ld(aspect_ratio, reynolds):
    """Return the Korn-style cruise L/D estimate used by CalcFanVals."""

    return (
        0.321
        * (aspect_ratio ** 2 * reynolds) ** (3 / 16)
        * (1 + 3.6 * aspect_ratio ** (-9 / 4)) ** (-0.5)
    )


def process_fan_keywords(overview):
    """Fill turbofan keyword-derived overview fields."""

    keywords = overview["KeyWords"]

    if is_nan_value(keywords):
        overview["Aisle"] = ""
        overview["RangeCapability"] = ""
        overview["ModelType"] = "Variant"
        overview["PassengerType"] = ""
    else:
        if contains_ignore_case(keywords, "single"):
            overview["Aisle"] = "Single"
        elif contains_ignore_case(keywords, "double"):
            overview["Aisle"] = "Double"
        else:
            overview["Aisle"] = ""

        if contains_ignore_case(keywords, "extended"):
            overview["RangeCapability"] = "Extended Range"
        elif contains_ignore_case(keywords, "long"):
            overview["RangeCapability"] = "Long Range"
        elif contains_ignore_case(keywords, "medium"):
            overview["RangeCapability"] = "Medium Range"
        elif contains_ignore_case(keywords, "short"):
            overview["RangeCapability"] = "Short Range"
        else:
            overview["RangeCapability"] = ""

        if contains_ignore_case(keywords, "baseline"):
            overview["ModelType"] = "Baseline"
        else:
            overview["ModelType"] = "Variant"

        if contains_ignore_case(keywords, "business"):
            overview["PayloadType"] = "Passenger"
            overview["PassengerType"] = "Business"

    payload_type = overview["PayloadType"]

    if payload_type == "P":
        overview["PayloadType"] = "Passenger"
        overview["PassengerType"] = "Commercial"
    elif payload_type == "C":
        overview["PayloadType"] = "Cargo"
        overview["PassengerType"] = "N/A"
        overview["Aisle"] = np.nan
    elif payload_type == "M":
        overview["PayloadType"] = "Mixed Pax and Cargo"
        overview["PassengerType"] = "Commercial"


def set_fan_fast_defaults(result):
    """Assign FAST default fields produced by CalcFanVals."""

    specs = result["Specs"]
    specs["TLAR"]["Class"] = "Turbofan"
    set_nested(result, ["Specs", "Propulsion", "Arch", "Type"], "C")
    set_nested(result, ["Specs", "Propulsion", "Eta", "Prop"], np.nan)
    set_nested(result, ["Specs", "Performance", "Alts", "Tko"], 0)
    set_nested(result, ["Specs", "Performance", "RCMax"], np.nan)
    set_nested(result, ["Specs", "Weight", "Batt"], np.nan)
    set_nested(result, ["Specs", "Weight", "EM"], np.nan)
    set_nested(result, ["Specs", "Weight", "EG"], np.nan)
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Fuel"], 43.2e6 / 3.6e6)
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Batt"], np.nan)
    set_nested(result, ["Specs", "Power", "Eta", "EM"], np.nan)
    set_nested(result, ["Specs", "Power", "Eta", "EG"], np.nan)

    for name in ("LamDwn", "LamUps"):
        for segment in ("SLS", "Tko", "Clb", "Crs", "Des", "Lnd"):
            set_nested(result, ["Specs", "Power", name, segment], 0)

    t_w_sls = specs["Propulsion"]["T_W"]["SLS"]
    v_tko = specs["Performance"]["Vels"]["Tko"]
    set_nested(result, ["Specs", "Power", "P_W", "SLS"], t_w_sls * v_tko)
    set_nested(result, ["Specs", "Power", "P_W", "EM"], np.nan)
    set_nested(result, ["Specs", "Power", "P_W", "EG"], np.nan)
    set_nested(result, ["Settings", "TkoPoints"], np.nan)
    set_nested(result, ["Settings", "ClbPoints"], np.nan)
    set_nested(result, ["Settings", "CrsPoints"], np.nan)
    set_nested(result, ["Settings", "DesPoints"], np.nan)
    set_nested(result, ["Settings", "OEW", "MaxIter"], np.nan)
    set_nested(result, ["Settings", "OEW", "Tol"], np.nan)
    set_nested(result, ["Settings", "Analysis", "MaxIter"], np.nan)
    set_nested(result, ["Settings", "Analysis", "Type"], 1)
    set_nested(result, ["Settings", "Plotting"], 0)

    draw = int(np.random.randint(1, 101))

    if draw > 90:
        set_nested(result, ["Settings", "DataTypeValidation"], "Validation")
    else:
        set_nested(result, ["Settings", "DataTypeValidation"], "Training")


def calc_prop_vals(plane, units):
    """Assign CalcPropVals unit metadata for turboprop database aircraft."""

    result = deepcopy(plane)

    if units == "Vals":
        return calc_prop_values(result)

    if units != "Units":
        raise DatabaseError('Enter either "Units" or "Vals" for CalcPropVals.')

    set_nested(result, ["Specs", "Performance", "Vels", "Crs"], "Mach")
    set_nested(result, ["Specs", "Weight", "Airframe"], "kg")
    set_nested(result, ["Specs", "Weight", "OEW_MTOW"], "ratio")
    set_nested(result, ["Specs", "Power", "P_W", "SLS"], "kW/kg")
    set_nested(result, ["Specs", "Power", "SLS"], "W")
    set_nested(result, ["Specs", "Power", "Cont"], "W")
    set_nested(result, ["Specs", "Power", "Clb"], "W")
    set_nested(result, ["Specs", "Power", "Crs"], "W")
    set_nested(result, ["Specs", "Aero", "W_S", "SLS"], "kg/m2")
    set_nested(result, ["Specs", "Weight", "EngFrac"], "ratio")
    set_nested(result, ["Specs", "Weight", "FuelFrac"], "ratio")
    set_nested(result, ["Overview", "ModelType"], "type")
    set_nested(result, ["Overview", "PayloadType"], "type")
    set_nested(result, ["Overview", "PassengerType"], "type")
    set_nested(result, ["Specs", "Aero", "TaperRatio"], "ratio")
    set_nested(result, ["Specs", "Aero", "L_D", "Crs"], "ratio")
    set_nested(result, ["Specs", "Aero", "AR"], "ratio")
    set_nested(result, ["Specs", "TLAR", "Class"], "flag")
    set_nested(result, ["Specs", "TLAR", "ADG"], "Category I through VI")
    set_nested(result, ["Specs", "TLAR", "ADGPercent"], "Percent")
    set_nested(result, ["Specs", "TLAR", "ADGLimitor"], "Type")
    set_nested(result, ["Specs", "Propulsion", "Arch", "Type"], "flag")
    set_nested(result, ["Specs", "Propulsion", "Eta", "Prop"], "efficiency")
    set_nested(result, ["Specs", "Performance", "Alts", "Tko"], "m")
    set_nested(result, ["Specs", "Performance", "RCMax"], "m/s")
    set_nested(result, ["Specs", "Weight", "Batt"], "kg")
    set_nested(result, ["Specs", "Weight", "EM"], "kg")
    set_nested(result, ["Specs", "Weight", "EG"], "kg")
    set_nested(result, ["Specs", "Weight", "Payload"], "kg")
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Batt"], "kWh/kg")
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Fuel"], "kWh/kg")
    set_nested(result, ["Specs", "Power", "Eta", "EM"], "efficiency")
    set_nested(result, ["Specs", "Power", "Eta", "EG"], "efficiency")
    set_power_split_units(result)
    set_nested(result, ["Specs", "Propulsion", "T_W", "SLS"], "ratio")
    set_nested(result, ["Specs", "Power", "P_W", "EM"], "kW/kg")
    set_nested(result, ["Specs", "Power", "P_W", "EG"], "kW/kg")
    set_settings_units(result)
    set_nested(
        result,
        ["Specs", "Propulsion", "Engine", "DataTypeValidation"],
        "data type",
    )
    result.setdefault("Overview", {}).pop("KeyWords", None)
    result.setdefault("Specs", {}).setdefault("Weight", {}).pop("MaxPayload", None)
    return result


def calc_prop_values(result):
    """Compute CalcPropVals derived values for turboprop aircraft."""

    specs = result["Specs"]
    weight = specs["Weight"]
    propulsion = specs["Propulsion"]
    engine = propulsion["Engine"]
    power = specs["Power"]
    aero = specs["Aero"]
    performance = specs["Performance"]
    overview = result["Overview"]
    num_engines = propulsion["NumEngines"]
    weight["Airframe"] = weight["OEW"] - engine["DryWeight"] * num_engines
    weight["OEW_MTOW"] = weight["OEW"] / weight["MTOW"]

    if is_nan_value(power["SLS"]):
        if is_nan_value(engine["Power_SLS_Eq"]):
            power["SLS"] = 1000 * engine["Power_SLS"] * num_engines
        else:
            power["SLS"] = 1000 * engine["Power_SLS_Eq"] * num_engines
    else:
        power["SLS"] = 1000 * power["SLS"] * num_engines

    if is_nan_value(power["Cont"]):
        power["Cont"] = 1000 * engine["Power_Cont_Eq"] * num_engines
    else:
        power["Cont"] = 1000 * power["Cont"] * num_engines

    power["Clb"] = 1000 * power["Clb"] * num_engines
    power["Crs"] = 1000 * power["Crs"] * num_engines
    power.setdefault("P_W", {})["SLS"] = power["SLS"] / weight["MTOW"] / 1000
    aero.setdefault("W_S", {})["SLS"] = weight["MTOW"] / aero["S"]
    weight["EngineFrac"] = engine["DryWeight"] * num_engines / weight["MTOW"]
    weight["FuelFrac"] = weight["Fuel"] / weight["MTOW"]
    process_prop_keywords(overview)
    overview.pop("KeyWords", None)

    if is_nan_value(overview["Monikers"]):
        overview["Monikers"] = "N/A"

    if is_nan_value(overview["AlternateDesignation"]):
        overview["AlternateDesignation"] = "N/A"

    aero["TaperRatio"] = aero["TipChord"] / aero["RootChord"]
    aero["AR"] = aero["Span"] ** 2 / aero["S"]
    temp, _, _ = standard_atmosphere(7500)
    mach = 0.4 if is_nan_value(performance["Vels"]["Crs"]) else performance["Vels"]["Crs"]
    thrust = power["Crs"] / (mach * np.sqrt(1.4 * 287 * temp))
    aero.setdefault("L_D", {})["Crs"] = weight["MTOW"] * 0.995 * 0.985 * 9.81 / thrust
    performance["Vels"]["Tko"] = convert_length(
        performance["Vels"]["Tko"],
        "naut mi",
        "m",
    ) / 3600

    if is_nan_value(weight["Cargo"]):
        weight["Cargo"] = 0

    if is_nan_value(weight["MaxPayload"]):
        weight["Payload"] = weight["Cargo"] + specs["TLAR"]["MaxPax"] * 95
    else:
        weight["Payload"] = weight["MaxPayload"]

    set_aircraft_design_group(specs)
    set_prop_fast_defaults(result)
    weight.pop("MaxPayload", None)
    return result


def process_prop_keywords(overview):
    """Fill turboprop keyword-derived overview fields."""

    keywords = overview["KeyWords"]

    if is_nan_value(keywords):
        overview["ModelType"] = "Variant"
        overview["PayloadType"] = ""
        overview["PassengerType"] = ""
        return

    if contains_ignore_case(keywords, "baseline"):
        overview["ModelType"] = "Baseline"
    else:
        overview["ModelType"] = "Variant"

    payload_type = overview["PayloadType"]

    if payload_type == "P":
        overview["PayloadType"] = "Passenger"
        overview["PassengerType"] = "Commercial"
    elif payload_type == "C":
        overview["PayloadType"] = "Cargo"
        overview["PassengerType"] = "N/A"
    elif payload_type == "M":
        overview["PayloadType"] = "Mixed Pax and Cargo"
        overview["PassengerType"] = "Commercial"
    else:
        overview["PayloadType"] = "Unspecified"
        overview["PassengerType"] = "Unspecified"

    if contains_ignore_case(keywords, "business"):
        overview["PayloadType"] = "Passenger"
        overview["PassengerType"] = "Business"


def set_aircraft_design_group(specs):
    """Assign FAA airplane design group fields used by FAST databases."""

    span = convert_length(specs["Aero"]["Span"], "m", "ft")
    height = convert_length(specs["Aero"]["Height"], "m", "ft")
    adg = np.nan
    bpercent = np.nan
    thpercent = np.nan

    if span < 262 and height < 80:
        adg = "VI"
        bpercent = span / 262
        thpercent = height / 80

    if span < 214 and height < 66:
        adg = "V"
        bpercent = span / 214
        thpercent = height / 66

    if span < 171 and height < 60:
        adg = "IV"
        bpercent = span / 171
        thpercent = height / 60

    if span < 118 and height < 45:
        adg = "III"
        bpercent = span / 118
        thpercent = height / 45

    if span < 79 and height < 30:
        adg = "II"
        bpercent = span / 79
        thpercent = height / 30

    if span < 49 and height < 20:
        adg = "I"
        bpercent = span / 49
        thpercent = height / 20

    if is_nan_value(span) or is_nan_value(height):
        bpercent = np.nan
        thpercent = np.nan

    if bpercent > thpercent:
        adg_percent = bpercent
        adg_limitor = "Wingspan"
    elif bpercent == thpercent:
        adg_percent = bpercent
        adg_limitor = "Both"
    elif bpercent < thpercent:
        adg_percent = thpercent
        adg_limitor = "TailHeight"
    else:
        adg_percent = np.nan
        adg_limitor = np.nan

    specs["TLAR"]["ADG"] = adg
    specs["TLAR"]["ADGPercent"] = adg_percent
    specs["TLAR"]["ADGLimitor"] = adg_limitor


def set_prop_fast_defaults(result):
    """Assign FAST default fields produced by CalcPropVals."""

    specs = result["Specs"]
    specs["TLAR"]["Class"] = "Turboprop"
    set_nested(result, ["Specs", "Propulsion", "Arch", "Type"], "C")
    set_nested(result, ["Specs", "Propulsion", "Eta", "Prop"], np.nan)
    set_nested(result, ["Specs", "Performance", "Alts", "Tko"], 0)
    set_nested(result, ["Specs", "Performance", "RCMax"], np.nan)
    set_nested(result, ["Specs", "Weight", "Batt"], np.nan)
    set_nested(result, ["Specs", "Weight", "EM"], np.nan)
    set_nested(result, ["Specs", "Weight", "EG"], np.nan)
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Fuel"], 43.2e6 / 3.6e6)
    set_nested(result, ["Specs", "Power", "SpecEnergy", "Batt"], np.nan)
    set_nested(result, ["Specs", "Power", "Eta", "EM"], np.nan)
    set_nested(result, ["Specs", "Power", "Eta", "EG"], np.nan)

    for name in ("LamDwn", "LamUps"):
        for segment in ("SLS", "Tko", "Clb", "Crs", "Des", "Lnd"):
            set_nested(result, ["Specs", "Power", name, segment], 0)

    p_w_sls = specs["Power"]["P_W"]["SLS"]
    v_tko = specs["Performance"]["Vels"]["Tko"]
    set_nested(result, ["Specs", "Propulsion", "T_W", "SLS"], p_w_sls / v_tko / 9.81)
    set_nested(result, ["Specs", "Power", "P_W", "EM"], np.nan)
    set_nested(result, ["Specs", "Power", "P_W", "EG"], np.nan)
    set_nested(result, ["Settings", "TkoPoints"], np.nan)
    set_nested(result, ["Settings", "ClbPoints"], np.nan)
    set_nested(result, ["Settings", "CrsPoints"], np.nan)
    set_nested(result, ["Settings", "DesPoints"], np.nan)
    set_nested(result, ["Settings", "OEW", "MaxIter"], np.nan)
    set_nested(result, ["Settings", "OEW", "Tol"], np.nan)
    set_nested(result, ["Settings", "Analysis", "MaxIter"], np.nan)
    set_nested(result, ["Settings", "Analysis", "Type"], 1)
    set_nested(result, ["Settings", "Plotting"], 0)


def contains_ignore_case(text, needle):
    """Return whether text contains a substring, ignoring case."""

    return needle.lower() in str(text).lower()


def is_nan_value(value):
    """Return True for numeric NaN values and False for strings/containers."""

    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def set_power_split_units(target):
    """Assign common power split unit metadata."""

    for name in ("LamDwn", "LamUps"):
        for segment in ("SLS", "Tko", "Clb", "Crs", "Des", "Lnd"):
            set_nested(target, ["Specs", "Power", name, segment], "ratio")


def set_settings_units(target):
    """Assign common FAST sizing setting unit metadata."""

    set_nested(target, ["Settings", "TkoPoints"], "count")
    set_nested(target, ["Settings", "ClbPoints"], "count")
    set_nested(target, ["Settings", "CrsPoints"], "count")
    set_nested(target, ["Settings", "DesPoints"], "count")
    set_nested(target, ["Settings", "OEW", "MaxIter"], "count")
    set_nested(target, ["Settings", "OEW", "Tol"], "ratio")
    set_nested(target, ["Settings", "Analysis", "MaxIter"], "count")
    set_nested(target, ["Settings", "Analysis", "Type"], "flag")
    set_nested(target, ["Settings", "Plotting"], "flag")


def set_nested(target, path, value):
    """Set a nested dictionary path, creating dictionaries as needed."""

    current = target

    for key in path[:-1]:
        current = current.setdefault(key, {})

    current[path[-1]] = value


CalcFanVals = calc_fan_vals
CalcPropVals = calc_prop_vals
LoadIDEASDatabase = load_ideas_database
RandomizeDB = randomize_db
StructTreeSearch = struct_tree_search
TreeBranch = tree_branch
