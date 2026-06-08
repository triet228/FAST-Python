# src/fast_python/battery.py

"""Battery sizing helpers ported from BatteryPkg."""

from copy import deepcopy

import numpy as np


class BatteryError(ValueError):
    """Report invalid battery model inputs."""


def discharging(aircraft, preq, time, soc_beg=None, parallel=1, series=1):
    """Model FAST's Lithium-ion battery discharge dynamics.

    Inputs:
        aircraft: Aircraft dictionary with Specs.Battery cell parameters.
        preq: Required battery power in watts for each time interval.
        time: Interval duration in seconds.
        soc_beg: Initial state of charge in percent. Defaults to 100.
        parallel: Number of cells in parallel.
        series: Number of cells in series.

    Outputs:
        Tuple of pack voltage, current, output power, charge capacity, SOC,
        and C-rate arrays. SOC includes the initial value and each updated
        control-point value.

    Assumptions:
        Mirrors BatteryPkg.Discharging's equivalent circuit and root-selection
        logic. Positive power discharges the battery; negative power charges it.
    """

    return battery_power_dynamics(
        aircraft,
        preq,
        time,
        soc_beg,
        parallel,
        series,
        False,
        "Discharging",
    )


def charging(aircraft, preq, time, soc_beg=None, parallel=1, series=1):
    """Model FAST's Lithium-ion battery charging dynamics."""

    return battery_power_dynamics(
        aircraft,
        preq,
        time,
        soc_beg,
        parallel,
        series,
        True,
        "Charging",
    )


def ground_charge(aircraft, chrg_time, power_strategy=None):
    """Simulate FAST's one-second ground charging process.

    Inputs:
        aircraft: Aircraft dictionary with mission SOC history and detailed
            battery cell parameters.
        chrg_time: Available charging time in seconds.
        power_strategy: Scalar charging power or a per-second power sequence.
            When omitted, Specs.Battery.Charging is used.

    Outputs:
        A deep-copied aircraft dictionary with
        Mission.History.SI.Power.ChargedAC populated.

    Assumptions:
        This mirrors BatteryPkg.GroundCharge's CC/CV strategy and uses the
        ported Charging equivalent-circuit model for every one-second step.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]

    if power_strategy is None:
        power_strategy = specs["Battery"]["Charging"]

    time_step = 1
    max_steps = int(np.ceil(chrg_time / time_step))
    power = aircraft["Mission"]["History"]["SI"]["Power"]
    charged = {
        "CtrlPtsTimeStep": time_step,
        "SOCEnd": as_vector(power["SOC"])[-1],
        "SOC": [as_vector(power["SOC"])[-1]],
        "Voltage": [],
        "VolCell": [],
        "Current": [],
        "P_in": [],
        "Capacity": [0],
        "C_rate": [],
    }

    if max_steps <= 0:
        power["ChargedAC"] = charged
        return aircraft

    soc_series = np.zeros(max_steps + 1)
    voltage_series = np.zeros(max_steps)
    current_series = np.zeros(max_steps)
    pout_series = np.zeros(max_steps)
    capacity_series = np.zeros(max_steps + 1)
    c_rate_series = np.zeros(max_steps)
    soc_series[0] = as_vector(power["SOC"])[-1]
    series = specs["Power"]["Battery"]["SerCells"]
    parallel = specs["Power"]["Battery"]["ParCells"]
    strategy = as_vector(power_strategy)
    dynamic_array = len(strategy) > 1
    base_power = strategy[0]
    q_cell = available_cell_capacity(aircraft)
    v_cell_cutoff = specs["Battery"]["MaxExtVolCell"]
    in_cv = False
    c_cutoff = 0
    v_pack_cutoff = 0
    soc_cutoff = 80
    valid_len = max_steps

    for step in range(max_steps):
        soc_now = soc_series[step]

        if not in_cv:
            if soc_now < 80:
                if dynamic_array:
                    desired = strategy[min(step, len(strategy) - 1)]
                else:
                    desired = base_power

                current_power = -abs(desired)
            else:
                v_cell_ocv = estimate_charge_ocv(aircraft, soc_now, parallel, series)
                c_rate_prev = abs(c_rate_series[step - 1]) if step > 0 else 0
                current_prev = current_series[step - 1] if step > 0 else 0
                voltage_prev = (
                    voltage_series[step - 1]
                    if step > 0
                    else v_cell_ocv * series
                )

                if v_cell_ocv < v_cell_cutoff:
                    if c_rate_prev < 1:
                        current_power = voltage_prev * current_prev
                    else:
                        current_target = q_cell * parallel
                        voltage_estimate = v_cell_ocv * series
                        current_power = -abs(current_target * voltage_estimate)
                else:
                    in_cv = True
                    soc_cutoff = soc_now
                    c_cutoff = c_rate_prev
                    v_pack_cutoff = voltage_prev
                    taper = ((100 - soc_now) / (100 - soc_cutoff)) ** 2
                    current_power = v_pack_cutoff * current_prev * taper
        else:
            taper = ((100 - soc_now) / (100 - soc_cutoff)) ** 2
            taper = max(min(taper, 1), 0)
            current_target = c_cutoff * q_cell * parallel * taper
            current_power = -abs(current_target * v_pack_cutoff)

        voltage, current, pout, capacity, soc_next, c_rate = charging(
            aircraft,
            current_power,
            time_step,
            soc_now,
            parallel,
            series,
        )
        voltage_series[step] = voltage[0]
        current_series[step] = current[0]
        pout_series[step] = pout[0]
        capacity_series[step] = capacity[0]
        c_rate_series[step] = c_rate[0]
        soc_next = max(min(as_vector(soc_next)[0], 100), 0)
        soc_series[step + 1] = soc_next
        valid_len = step + 1

        if abs(c_rate_series[step]) <= 0.02:
            soc_series[step] = soc_next
            capacity_series[step + 1] = capacity_series[step]
            break

        if soc_now >= 100:
            soc_series[step] = 100
            capacity_series[step + 1] = capacity_series[step]
            break

    voltage_series = voltage_series[:valid_len]
    current_series = current_series[:valid_len]
    pout_series = pout_series[:valid_len]
    capacity_series = capacity_series[: valid_len + 1]
    c_rate_series = c_rate_series[:valid_len]
    soc_series = soc_series[: valid_len + 1]
    power["ChargedAC"] = {
        "CtrlPtsTimeStep": time_step,
        "SOCEnd": float(soc_series[-1]),
        "SOC": soc_series.tolist(),
        "Voltage": voltage_series.tolist(),
        "VolCell": (voltage_series / series).tolist(),
        "Current": (-current_series).tolist(),
        "P_in": pout_series.tolist(),
        "Capacity": capacity_series.tolist(),
        "C_rate": (-c_rate_series).tolist(),
    }
    return aircraft


def cycling_aging(aircraft, chem_type, cumul_fecs, charging_time, chrg_rate):
    """Predict lithium-ion cycling-aging SOH and FEC.

    Inputs:
        aircraft: Aircraft dictionary with mission discharge history and
            detailed battery parameters.
        chem_type: 1 for NMC chemistry or 2 for LFP chemistry.
        cumul_fecs: Full equivalent cycles accumulated before this mission.
        charging_time: Available ground-charging time in seconds.
        chrg_rate: Ground charging power in watts.

    Outputs:
        Tuple of SOH percent, full equivalent cycles, and the aircraft updated
        with ground-charge history.

    Assumptions:
        This mirrors BatteryPkg.CyclAging's empirical NMC/LFP degradation
        model and calls the ported GroundCharge helper before evaluating
        charge-side C-rate and capacity deltas.
    """

    params = cycling_aging_parameters(chem_type)
    power = aircraft["Mission"]["History"]["SI"]["Power"]
    capacity = as_history_matrix(power["Capacity"])
    valid_columns = np.where(np.any(capacity != 0, axis=0))[0]

    if len(valid_columns) == 0:
        raise BatteryError("CyclAging requires nonzero battery capacity history.")

    column = valid_columns[0]
    soc = as_history_matrix(power["SOC"])[:, column]
    depth_of_discharge = float(np.max(soc) - np.min(soc))
    discharge_c_rate = nonzero_mean(as_vector(power["C_rate"]))
    charged_aircraft = ground_charge(aircraft, charging_time, chrg_rate)
    charge = charged_aircraft["Mission"]["History"]["SI"]["Power"]["ChargedAC"]
    charge_c_rate = nonzero_mean(as_vector(charge["C_rate"]))
    active_soc = soc[np.concatenate([[True], np.diff(soc) != 0])]
    mean_soc = float(np.mean(active_soc))
    q_max = aircraft["Specs"]["Battery"]["CapCell"]
    cap_col = capacity[:, column]
    cap_nonzero = cap_col[cap_col != 0]
    discharge_delta = float(np.max(cap_col) - np.min(cap_nonzero))
    charge_capacity = as_vector(charge["Capacity"])
    charge_nonzero = charge_capacity[charge_capacity != 0]
    charge_delta = float(np.max(charge_capacity) - np.min(charge_nonzero))
    parallel = aircraft["Specs"]["Power"]["Battery"]["ParCells"]
    fec = (discharge_delta + charge_delta) / (2 * q_max * parallel) + cumul_fecs
    temp_actual = aircraft["Specs"]["Battery"]["OpTemp"] + 273.15
    theta_temp = params["coeff_T"] * (
        (temp_actual - params["temp_ref"]) / temp_actual
    )
    theta_dod = params["coeff_DOD"] * depth_of_discharge
    theta_c = params["coeff_Cch"] * charge_c_rate
    theta_c += params["coeff_Cdch"] * discharge_c_rate
    soc_shape = 1 + params["coeff_mSOC"] * mean_soc * (
        1 - (mean_soc / (2 * params["mSOC_ref"]))
    )
    degradation = params["beta"] * np.exp(theta_temp + theta_dod + theta_c)
    degradation *= soc_shape * fec ** params["alpha"]
    soh = 100 - degradation
    return float(soh), float(fec), charged_aircraft


def estimate_charge_ocv(aircraft, soc_percent, parallel, series):
    """Estimate open-circuit cell voltage for GroundCharge."""

    voltage, _, _, _, _, _ = charging(
        aircraft,
        0,
        1,
        soc_percent,
        parallel,
        series,
    )
    return voltage[0] / series


def cycling_aging_parameters(chem_type):
    """Return empirical cycling-aging constants for NMC or LFP chemistry."""

    if chem_type == 1:
        return {
            "beta": 0.001673,
            "coeff_T": 21.6745,
            "coeff_DOD": 0.022,
            "coeff_Cch": 0.2553,
            "coeff_Cdch": 0.1571,
            "coeff_mSOC": -0.0212,
            "alpha": 0.915,
            "temp_ref": 293.15,
            "mSOC_ref": 42,
        }

    if chem_type == 2:
        return {
            "beta": 0.003414,
            "coeff_T": 5.8755,
            "coeff_DOD": -0.0045,
            "coeff_Cch": 0.1038,
            "coeff_Cdch": 0.296,
            "coeff_mSOC": 0.0513,
            "alpha": 0.869,
            "temp_ref": 293.15,
            "mSOC_ref": 42,
        }

    raise BatteryError('Invalid ChemType input. Use "1" for NMC or "2" for LFP.')


def battery_power_dynamics(
    aircraft,
    preq,
    time,
    soc_beg,
    parallel,
    series,
    drop_initial_soc,
    label,
):
    """Run the shared FAST battery equivalent-circuit model."""

    preq, time = prepare_power_time(preq, time, label)
    soc_beg = prepare_initial_soc(soc_beg, label)
    time = time / 3600
    specs = aircraft["Specs"]["Battery"]
    voltage_open = specs["MaxExtVolCell"]
    resistance = specs["IntResist"]
    ncell = series * parallel
    exp_voltage = specs["ExpVol"]
    exp_capacity = specs["ExpCap"]
    q_cell = available_cell_capacity(aircraft)
    discharge_curve_slope = 0.29732
    polarized_voltage = 0.0011
    ntime = len(time)
    soc = np.zeros(ntime + 1)
    soc[0] = soc_beg
    current = np.zeros(ntime)
    capacity = np.zeros(ntime)
    voltage = np.zeros(ntime)
    pout = np.zeros(ntime)
    c_rate = np.zeros(ntime)

    for index in range(ntime):
        preq_cell = preq[index] / ncell
        soc_fraction = soc[index] / 100
        discharged_start = (1 - soc_fraction) * q_cell
        is_discharge = preq[index] >= 0

        with np.errstate(divide="ignore", invalid="ignore"):
            if is_discharge:
                hot_voltage = -(polarized_voltage / soc_fraction + resistance)
            else:
                hot_voltage = (
                    polarized_voltage
                    / ((discharged_start + 0.1 * q_cell) / q_cell)
                    + resistance
                )

            cold_voltage = voltage_open + (
                exp_voltage * np.exp(-exp_capacity * discharged_start)
                - polarized_voltage * discharged_start / soc_fraction
                - discharge_curve_slope * discharged_start
            )

        cell_current = solve_battery_current(
            hot_voltage,
            cold_voltage,
            preq_cell,
            is_discharge,
        )
        current[index] = cell_current * parallel
        cell_voltage = cold_voltage + hot_voltage * cell_current
        voltage[index] = cell_voltage * series
        discharged_capacity = cell_current * time[index]
        soc[index + 1] = soc[index] - 100 * discharged_capacity / q_cell
        capacity[index] = q_cell * soc[index] / 100 * parallel
        capacity[index] = min(capacity[index], q_cell * parallel)
        pout[index] = voltage[index] * current[index]
        c_rate[index] = current[index] / (q_cell * parallel)

    if drop_initial_soc:
        soc = soc[1:]

    return voltage, current, pout, capacity, soc, c_rate


def prepare_power_time(preq, time, label):
    """Broadcast scalar power or time inputs the same way FAST does."""

    preq = as_vector(preq)
    time = as_vector(time)

    if len(preq) == 1 and len(time) > 1:
        preq = np.repeat(preq, len(time))
    elif len(preq) > 1 and len(time) == 1:
        time = np.repeat(time, len(preq))
    elif len(preq) != len(time):
        raise BatteryError(
            f"{label}: required power and time are different sizes."
        )

    return preq, time


def prepare_initial_soc(soc_beg, label):
    """Return scalar initial SOC, defaulting to a full battery."""

    if soc_beg is None:
        return 100

    soc = as_vector(soc_beg)

    if len(soc) > 1:
        raise BatteryError(f"{label}: initial SOC must be a scalar or empty.")

    if len(soc) == 0:
        return 100

    return float(soc[0])


def available_cell_capacity(aircraft):
    """Return cell capacity after optional degradation adjustment."""

    specs = aircraft["Specs"]["Battery"]
    settings = aircraft.get("Settings", {})
    analysis_type = settings.get("Analysis", {}).get("Type", 0)

    if analysis_type < 0 and specs.get("Degradation") == 1:
        soh = as_vector(specs["SOH"])[-1]
        return specs["CapCell"] * soh / 100

    return specs["CapCell"]


def solve_battery_current(hot_voltage, cold_voltage, preq_cell, is_discharge):
    """Solve the cell current root selected by FAST's battery model."""

    roots = np.roots([hot_voltage, cold_voltage, -preq_cell])

    if not np.any(np.iscomplex(roots)):
        candidates = np.real(roots)

        if is_discharge:
            candidates = candidates[candidates >= 0]

            if len(candidates) == 0:
                return 0

            return float(np.min(candidates))

        candidates = candidates[candidates <= 0]

        if len(candidates) == 0:
            return 0

        return float(np.max(candidates))

    if is_discharge:
        candidates = roots[np.real(roots) >= 0]

        if len(candidates) == 0:
            return 0

        selected = candidates[np.argmin(np.abs(candidates))]
        current_guess = abs(selected)
    else:
        candidates = roots[np.real(roots) <= 0]

        if len(candidates) == 0:
            return 0

        selected = candidates[np.argmax(np.abs(candidates))]
        current_guess = -abs(selected)

    current_range = np.arange(current_guess - 10, current_guess + 10.05, 0.1)
    voltage_range = cold_voltage + hot_voltage * current_range
    index = int(np.argmin(np.abs(preq_cell - voltage_range * current_range)))
    return float(current_range[index])


def resize_battery(aircraft):
    """Resize battery weight and optional cell counts after a mission.

    Inputs:
        aircraft: Dictionary with propulsion source types, battery energy
            history, and battery specification fields.

    Outputs:
        A deep-copied aircraft dictionary with Specs.Weight.Batt updated.

    Assumptions:
        This mirrors BatteryPkg.ResizeBattery. Energy-based sizing is always
        applied when a battery source exists; detailed cell resizing runs only
        when Settings.DetailedBatt is 1.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    prop_arch = specs["Propulsion"]["PropArch"]
    src_type = as_vector(prop_arch["SrcType"])
    battery_sources = src_type == 0

    if not np.any(battery_sources):
        specs.setdefault("Weight", {})["Batt"] = 0
        return aircraft

    ebatt = specs["Power"]["SpecEnergy"]["Batt"]
    history = aircraft["Mission"]["History"]["SI"]
    e_es = history_matrix(history["Energy"]["E_ES"], len(src_type))
    ebatt_used = e_es[:, battery_sources]
    energy = np.sum(ebatt_used)

    if energy == 0:
        specs.setdefault("Weight", {})["Batt"] = 0
        return aircraft

    specs.setdefault("Weight", {})["Batt"] = restore_scalar_or_list(
        ebatt_used[-1, :] / ebatt
    )

    if aircraft["Settings"].get("DetailedBatt") == 1:
        resize_detailed_battery(aircraft, battery_sources, ebatt)

    return aircraft


def resize_detailed_battery(aircraft, battery_sources, ebatt):
    """Resize detailed cell counts and mass for battery sources."""

    specs = aircraft["Specs"]
    battery_specs = specs["Battery"]
    power_battery = specs["Power"]["Battery"]
    prop_arch = specs["Propulsion"]["PropArch"]
    nsrc = len(as_vector(prop_arch["SrcType"]))
    v_nom = battery_specs["NomVolCell"]
    q_max = battery_specs["CapCell"]
    min_soc = battery_specs["MinSOC"]
    max_c_rate = battery_specs["MaxAllowCRate"]
    nser = power_battery["SerCells"]
    npar = power_battery["ParCells"]
    history = aircraft["Mission"]["History"]["SI"]
    soc = history_matrix(history["Power"]["SOC"], nsrc)[:, battery_sources]
    current = history_matrix(history["Power"]["Current"], nsrc)[:, battery_sources]
    delta_soc = np.max(min_soc - soc, axis=0)
    size_to = delta_soc / 100
    existing_capacity = q_max * npar
    npar_soc = np.ceil(
        np.ceil((existing_capacity + size_to * q_max * npar) / q_max)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        c_rate = current / existing_capacity

    c_rate[~np.isfinite(c_rate)] = 0
    exceed_c_rate = np.abs(c_rate) > max_c_rate

    if np.any(exceed_c_rate):
        max_crate = np.max(np.abs(c_rate), axis=0)
        npar_crate = np.ceil(max_crate / max_c_rate) * npar
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            c_rate_soc = current / (npar_soc * q_max)

        exceed_c_rate_soc = np.abs(c_rate_soc) > max_c_rate

        if np.any(exceed_c_rate_soc):
            max_crate_soc = np.max(np.abs(c_rate_soc), axis=0)
            npar_crate = np.ceil(max_crate_soc / max_c_rate) * npar_soc
        else:
            npar_crate = 0

    npar = np.maximum(npar_soc, npar_crate)
    wbatt = q_max * npar * v_nom * nser * 3600 / ebatt
    specs["Weight"]["Batt"] = restore_scalar_or_list(wbatt)
    power_battery["ParCells"] = restore_scalar_or_list(npar)

    with np.errstate(divide="ignore", invalid="ignore"):
        history["Power"]["C_rate"] = (current / (q_max * npar)).tolist()


def as_vector(value):
    """Return value as a one-dimensional numeric array."""

    if hasattr(value, "value"):
        value = value.value

    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        return array.reshape(1)

    return array.reshape(-1)


def as_history_matrix(value):
    """Return a battery history field as rows by battery columns."""

    array = np.asarray(value, dtype=float)

    if array.ndim == 1:
        return array.reshape(-1, 1)

    return array


def nonzero_mean(value):
    """Return the mean of nonzero values, or zero when none exist."""

    values = as_vector(value)
    values = values[values != 0]

    if len(values) == 0:
        return 0

    return float(np.mean(values))


def history_matrix(value, columns):
    """Return mission history as a two-dimensional array."""

    array = np.asarray(value, dtype=float)

    if array.ndim == 1 and columns == 1:
        return array.reshape(-1, 1)

    if array.ndim == 1:
        return array.reshape(1, -1)

    return array


def restore_scalar_or_list(value):
    """Return scalar for one value, otherwise a list."""

    array = np.asarray(value)

    if array.size == 1:
        return float(array.reshape(-1)[0])

    return array.tolist()


ResizeBattery = resize_battery
GroundCharge = ground_charge
CyclAging = cycling_aging
