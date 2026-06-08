# src/fast_python/history.py

"""Mission history table helpers ported from FAST MissionHistTable."""

import numpy as np


MISSION_HISTORY_COLUMNS = [
    "Time (min)",
    "Segment",
    "Distance (km)",
    "TAS (m/s)",
    "EAS (m/s)",
    "R/C (m/s)",
    "Altitude (m)",
    "Acceleration (m/s^2)",
    "FPA (deg.)",
    "Mach",
    "Density (kg/m^3)",
    "Specific Excess Power (m/s)",
    "SFC (kg/Ns)",
    "Fuel Flow (kg/s)",
    "Weight (kg)",
    "Fuel Burn (kg)",
    "Thrust Available (kN)",
    "Thrust Required (kN)",
    "Thrust Power (MW)",
    "Required Power (MW)",
    "Power Available (MW)",
    "Power Required (MW)",
    "State of Charge (%)",
    "Kinetic Energy (MJ)",
    "Potential Energy (MJ)",
    "Energy Expended (MJ)",
    "Energy Remaining (MJ)",
]


def mission_history_table(aircraft):
    """Return FAST mission history as rows with MATLAB table column names."""

    history = aircraft["Mission"]["History"]
    si = history["SI"]
    performance = si["Performance"]
    propulsion = si["Propulsion"]
    weight = si["Weight"]
    power = si["Power"]
    energy = si["Energy"]
    segments = history["Segment"]
    nrow = len(performance["Time"])
    rows = []

    for index in range(nrow):
        rows.append(
            {
                "Time (min)": table_value(performance["Time"], index, 1 / 60),
                "Segment": segments[index],
                "Distance (km)": table_value(performance["Dist"], index, 1 / 1000),
                "TAS (m/s)": table_value(performance["TAS"], index),
                "EAS (m/s)": table_value(performance["EAS"], index),
                "R/C (m/s)": table_value(performance["RC"], index),
                "Altitude (m)": table_value(performance["Alt"], index),
                "Acceleration (m/s^2)": table_value(performance["Acc"], index),
                "FPA (deg.)": table_value(performance["FPA"], index),
                "Mach": table_value(performance["Mach"], index),
                "Density (kg/m^3)": table_value(performance["Rho"], index),
                "Specific Excess Power (m/s)": table_value(performance["Ps"], index),
                "SFC (kg/Ns)": table_value(propulsion["TSFC"], index),
                "Fuel Flow (kg/s)": table_value(propulsion["MDotFuel"], index),
                "Weight (kg)": table_value(weight["CurWeight"], index),
                "Fuel Burn (kg)": table_value(weight["Fburn"], index),
                "Thrust Available (kN)": table_value(power["Tav"], index, 1 / 1.0e3),
                "Thrust Required (kN)": table_value(power["Treq"], index, 1 / 1.0e3),
                "Thrust Power (MW)": table_value(power["TV"], index, 1 / 1.0e6),
                "Required Power (MW)": table_value(power["Req"], index, 1 / 1.0e6),
                "Power Available (MW)": table_value(power["Pav"], index, 1 / 1.0e6),
                "Power Required (MW)": table_value(power["Preq"], index, 1 / 1.0e6),
                "State of Charge (%)": table_value(power["SOC"], index),
                "Kinetic Energy (MJ)": table_value(energy["KE"], index, 1 / 1.0e6),
                "Potential Energy (MJ)": table_value(energy["PE"], index, 1 / 1.0e6),
                "Energy Expended (MJ)": table_value(energy["E_ES"], index, 1 / 1.0e6),
                "Energy Remaining (MJ)": table_value(energy["Eleft_ES"], index, 1 / 1.0e6),
            }
        )

    return {
        "columns": MISSION_HISTORY_COLUMNS.copy(),
        "rows": rows,
    }


def table_value(values, index, scale=1):
    """Return one row value from a scalar or matrix history variable."""

    array = np.asarray(values, dtype=object)
    value = array[index]

    if isinstance(value, np.ndarray):
        return [scale_table_scalar(item, scale) for item in value.tolist()]

    if isinstance(value, list):
        return [scale_table_scalar(item, scale) for item in value]

    return scale_table_scalar(value, scale)


def scale_table_scalar(value, scale):
    """Scale a scalar table value while preserving non-numeric values."""

    try:
        return value * scale
    except TypeError:
        return value


MissionHistTable = mission_history_table
