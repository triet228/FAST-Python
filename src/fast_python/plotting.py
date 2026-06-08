# src/fast_python/plotting.py

"""PlotPkg data-formatting helpers.

FAST's MATLAB plotting functions mix data extraction, unit conversion, and
figure rendering. The Python port deliberately returns structured plot-ready
data instead of drawing figures so tests and CLI users can inspect the same
quantities without requiring a graphical backend.
"""

import numpy as np

from fast_python.units import (
    convert_length,
    convert_mass,
    convert_tsfc,
    convert_velocity,
)


def plot_perf_param_data(x, y, inst, lx, ly, name):
    """Return PlotPerfParam-ready data and labels without rendering a plot."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)

    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)

    if y_values.ndim == 1:
        y_values = y_values.reshape(-1, 1)

    x_values, y_values = align_plot_arrays(x_values, y_values)

    if inst == 1:
        x_values, y_values = step_like_data(x_values, y_values)

    return {
        "x": restore_plot_array(x_values),
        "y": restore_plot_array(y_values),
        "xlabel": lx,
        "ylabel": ly,
        "title": f"{name} Profile",
    }


def plot_mission_data(aircraft):
    """Return PlotMission's converted data grouped by MATLAB figure intent."""

    values = mission_plot_values(aircraft)
    time = values["time_min"]
    rc_time = time[:-1]
    rc = values["rate_climb_ft_min"][:-1]
    tout = values["thrust_output_kn"][:-1]
    pout = values["power_output_mw"][:-1]

    altitude_plot = plot_perf_param_data(time, values["altitude_ft"], 0, "Flight Time (min)", "Altitude (ft)", "Altitude")
    distance_plot = plot_perf_param_data(time, values["distance_nmi"], 0, "Flight Time (min)", "Distance Flown (nmi)", "Distance")
    airspeed_plot = plot_perf_param_data(time, values["tas_kts"], 0, "Flight Time (min)", "Airspeed (kts)", "Airspeed (TAS)")
    climb_plot = plot_perf_param_data(rc_time, rc, 1, "Flight Time (min)", "Rate of Climb (ft/min)", "Rate of Climb")

    figures = [
        {
            "name": "flight_performance",
            "grid": [2, 2],
            "plots": [
                altitude_plot,
                airspeed_plot,
                distance_plot,
                climb_plot,
            ],
        },
        {
            "name": "weights",
            "grid": [2, 2],
            "plots": [
                altitude_plot,
                plot_perf_param_data(time, values["weight_lbm"], 0, "Flight Time (min)", "Aircraft Weight (lbf)", "Weight"),
                distance_plot,
                plot_perf_param_data(time, values["fuel_burn_lbm"], 0, "Flight Time (min)", "Total Fuel Burned (lbf)", "Fuel Burn"),
            ],
        },
        {
            "name": "propulsion",
            "grid": [2, 2],
            "plots": [
                altitude_plot,
                plot_perf_param_data(time, values["fuel_flow_kg_s"], 1, "Flight Time (min)", "Fuel Flow (kg/s)", "Fuel Flow"),
                plot_perf_param_data(time, values["power_output_mw"], 1, "Flight Time (min)", "Power Output (MW)", "Power Output"),
                plot_perf_param_data(time, values["sfc"], 0, "Flight Time (min)", values["sfc_ylabel"], "SFC"),
            ],
        },
        {
            "name": "power_detail",
            "grid": [3, 3],
            "plots": [
                altitude_plot,
                airspeed_plot,
                climb_plot,
                overlay_plot(
                    "power_available_required",
                    [
                        plot_perf_param_data(time, values["power_available_mw"], 1, "Flight Time (min)", "Power (MW)", "Power"),
                        plot_perf_param_data(time, values["power_required_mw"], 1, "Flight Time (min)", "Power (MW)", "Power"),
                    ],
                ),
                overlay_plot(
                    "thrust_available_required",
                    [
                        plot_perf_param_data(time, values["thrust_available_kn"], 0, "Flight Time (min)", "Thrust (kN)", "Thrust"),
                        plot_perf_param_data(time, values["thrust_required_kn"], 0, "Flight Time (min)", "Thrust (kN)", "Thrust"),
                    ],
                ),
                plot_perf_param_data(rc_time, tout, 1, "Flight Time (min)", "Thrust Output (N)", "Thrust Output"),
                plot_perf_param_data(rc_time, pout, 1, "Flight Time (min)", "Power Output (MW)", "Power Output"),
            ],
        },
        {
            "name": "power_summary",
            "grid": [2, 2],
            "plots": [
                altitude_plot,
                plot_perf_param_data(time, values["total_power_available_mw"], 1, "Flight Time (min)", "Power Available (MW)", "Total Power Available"),
                plot_perf_param_data(time, values["specific_excess_power_m_s"], 1, "Flight Time (min)", "Specific Excess Power (m/s)", "Specific Excess Power"),
                plot_perf_param_data(time, values["total_power_required_mw"], 1, "Flight Time (min)", "Power Required (MW)", "Total Power Required"),
            ],
        },
    ]

    return {
        "values": values,
        "figures": figures,
    }


def mission_plot_values(aircraft):
    """Extract and convert the mission history quantities used by PlotMission."""

    history = aircraft["Mission"]["History"]["SI"]
    performance = history["Performance"]
    weight = history["Weight"]
    power = history["Power"]
    propulsion = history["Propulsion"]
    transmitter_mask = non_propeller_transmitter_mask(aircraft)
    aircraft_class = aircraft["Specs"]["TLAR"]["Class"]

    return {
        "time_min": restore_numeric_array(as_column_matrix(performance["Time"])[:, 0] / 60),
        "altitude_ft": restore_numeric_array(as_column_matrix(convert_length(performance["Alt"], "m", "ft"))[:, 0]),
        "distance_nmi": restore_numeric_array(as_column_matrix(convert_length(performance["Dist"], "m", "naut mi"))[:, 0]),
        "tas_kts": restore_numeric_array(as_column_matrix(convert_velocity(performance["TAS"], "m/s", "kts"))[:, 0]),
        "rate_climb_ft_min": restore_numeric_array(as_column_matrix(convert_velocity(performance["RC"], "m/s", "ft/min"))[:, 0]),
        "weight_lbm": restore_numeric_array(as_column_matrix(convert_mass(weight["CurWeight"], "kg", "lbm"))[:, 0]),
        "fuel_burn_lbm": restore_numeric_array(as_column_matrix(convert_mass(weight["Fburn"], "kg", "lbm"))[:, 0]),
        "total_power_required_mw": restore_numeric_array(as_column_matrix(power["Req"])[:, 0] / 1.0e6),
        "total_power_available_mw": restore_numeric_array(as_column_matrix(power["TV"])[:, 0] / 1.0e6),
        "specific_excess_power_m_s": restore_numeric_array(as_column_matrix(performance["Ps"])[:, 0]),
        "power_required_mw": restore_numeric_array(select_plot_columns(power["Preq"], transmitter_mask) / 1.0e6),
        "power_available_mw": restore_numeric_array(select_plot_columns(power["Pav"], transmitter_mask) / 1.0e6),
        "power_output_mw": restore_numeric_array(select_plot_columns(power["Pout"], transmitter_mask) / 1.0e6),
        "thrust_required_kn": restore_numeric_array(select_plot_columns(power["Treq"], transmitter_mask) / 1000),
        "thrust_available_kn": restore_numeric_array(select_plot_columns(power["Tav"], transmitter_mask) / 1000),
        "thrust_output_kn": restore_numeric_array(select_plot_columns(power["Tout"], transmitter_mask) / 1000),
        "sfc": restore_numeric_array(plot_mission_sfc(propulsion["TSFC"], aircraft_class)),
        "sfc_ylabel": plot_mission_sfc_ylabel(aircraft_class),
        "fuel_flow_kg_s": restore_numeric_array(as_column_matrix(propulsion["MDotFuel"])[:, 0]),
        "transmitter_mask": transmitter_mask.tolist(),
    }


def plot_mission_sfc(tsfc, aircraft_class):
    """Return PlotMission's class-specific SFC conversion."""

    class_name = aircraft_class.lower()

    if class_name == "turbofan":
        return as_column_matrix(convert_tsfc(tsfc, "SI", "Imp"))[:, 0]

    if class_name in ("turboprop", "piston"):
        return as_column_matrix(tsfc)[:, 0] * 3.6e3 / 0.00134102 * 2.20462

    raise ValueError("invalid aircraft class for PlotMission")


def plot_mission_sfc_ylabel(aircraft_class):
    """Return the SFC y-axis label matching FAST PlotMission branches."""

    if aircraft_class.lower() == "turbofan":
        return "SFC (lbm/lbf/hr)"

    return "SFC (lbm/hp/hr)"


def non_propeller_transmitter_mask(aircraft):
    """Return PlotMission's component mask for transmitters that are not props."""

    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]

    if "Arch" not in prop_arch:
        raise ValueError("PropArch.Arch is required for PlotMission data.")

    arch = np.asarray(prop_arch["Arch"])
    src_type = as_flat_array(prop_arch["SrcType"])
    trn_type = as_flat_array(prop_arch["TrnType"])
    ncomp = arch.shape[0] if arch.ndim > 1 else len(arch)
    nsink = ncomp - len(src_type) - len(trn_type)

    if nsink < 0:
        raise ValueError("PropArch component counts are inconsistent.")

    return np.asarray(
        [False] * len(src_type)
        + [item != 2 for item in trn_type]
        + [False] * nsink,
        dtype=bool,
    )


def select_plot_columns(values, mask):
    """Select PlotMission component columns from a history matrix."""

    array = as_column_matrix(values)

    if array.shape[1] == len(mask):
        return array[:, mask]

    if array.shape[1] == int(np.count_nonzero(mask)):
        return array

    raise ValueError("mission history matrix width does not match PropArch components.")


def overlay_plot(name, traces):
    """Return a non-rendered description of a MATLAB hold-on plot."""

    return {
        "name": name,
        "traces": traces,
    }


def align_plot_arrays(x_values, y_values):
    """Broadcast one-column x/y data across multiple plotted columns."""

    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x and y must have the same row count.")

    if x_values.shape[1] == y_values.shape[1]:
        return x_values, y_values

    if x_values.shape[1] == 1:
        return np.repeat(x_values, y_values.shape[1], axis=1), y_values

    if y_values.shape[1] == 1:
        return x_values, np.repeat(y_values, x_values.shape[1], axis=1)

    raise ValueError("x and y column counts must match, or one side must be one column.")


def step_like_data(x_values, y_values):
    """Return FAST's step-like instantaneous plotting arrays."""

    nrow, ncol = x_values.shape
    xnew = np.zeros((2 * (nrow - 1) + 1, ncol))
    ynew = np.zeros((2 * (nrow - 1) + 1, ncol))

    for icol in range(ncol):
        x_pairs = np.vstack(
            [
                x_values[:-1, icol],
                x_values[1:, icol],
            ]
        )
        y_pairs = np.vstack(
            [
                y_values[:-1, icol],
                y_values[:-1, icol],
            ]
        )
        xnew[:, icol] = np.concatenate([x_pairs.T.reshape(-1), [x_values[-1, icol]]])
        ynew[:, icol] = np.concatenate([y_pairs.T.reshape(-1), [y_values[-1, icol]]])

    return xnew, ynew


def restore_plot_array(values):
    """Return plot arrays as flat lists for one column or nested lists."""

    if values.shape[1] == 1:
        return values[:, 0].tolist()

    return values.tolist()


def as_flat_array(values):
    """Return scalar or vector values as a flat numeric array."""

    return np.asarray(values, dtype=float).reshape(-1)


def as_column_matrix(values):
    """Return scalar, vector, or matrix data as a two-dimensional array."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return array.reshape(1, 1)

    if array.ndim == 1:
        return array.reshape(-1, 1)

    return array


def restore_numeric_array(values):
    """Return NumPy arrays as plain Python lists for JSON-friendly results."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        return array.tolist()

    if array.shape[1] == 1:
        return array[:, 0].tolist()

    return array.tolist()


PlotPerfParam = plot_perf_param_data
PlotMission = plot_mission_data
