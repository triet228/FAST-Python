# src/fast_python/specs.py

"""Aircraft and engine specification presets ported from FAST specs packages.

Preset functions return fresh dictionaries in SI units where the MATLAB package
stored converted values. Shared builder helpers centralize repeated engine
fields so individual presets stay close to their source data.
"""

import numpy as np

from fast_python.units import (
    convert_force,
    convert_length,
    convert_mass,
    convert_velocity,
)


def leap_1a26():
    """Return EngineSpecsPkg.LEAP_1A26's engine specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 50,
        "FPR": 1.4,
        "BPR": 11,
        "Tt4Max": 1593,
        "TempLimit": {
            "Val": np.nan,
            "Type": np.nan,
        },
        "DesignThrust": 120640,
        "NoSpools": 2,
        "RPMs": [3894, 19391],
        "FanGearRatio": np.nan,
        "FanBoosters": True,
        "CoreFlow": {
            "PaxBleed": 0.03,
            "Leakage": 0.01,
            "Cooling": 0.0,
        },
        "MaxIter": 300,
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Fan": 0.99,
            "Compressors": 0.96,
            "BypassNozzle": 0.99,
            "Combustor": 0.995,
            "Turbines": 0.96,
            "CoreNozzle": 0.99,
            "Nozzles": 0.99,
            "Mixing": 0.0,
        },
        "Cff3": 0.4006,
        "Cff2": -0.4323,
        "Cff1": 0.9946,
        "Cffch": 6.1e-7,
        "HEcoeff": 1,
    }


def ceras_engine():
    """Return EngineSpecsPkg.CeRAS's engine specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 24.5,
        "FPR": 1.6,
        "BPR": 4,
        "Tt4Max": 1711,
        "TempLimit": {
            "Val": np.nan,
            "Type": np.nan,
        },
        "DesignThrust": convert_force(14510, "lbf", "N"),
        "NoSpools": 2,
        "RPMs": [7400, 17820],
        "FanGearRatio": np.nan,
        "FanBoosters": False,
        "CoreFlow": {
            "PaxBleed": 0.03,
            "Leakage": 0.01,
            "Cooling": 0.0,
        },
        "MaxIter": 300,
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Fan": 0.99,
            "Compressors": 0.94,
            "BypassNozzle": 0.99,
            "Combustor": 0.995,
            "Turbines": 0.94,
            "CoreNozzle": 0.99,
            "Nozzles": 0.99,
            "Mixing": 0.0,
        },
        "PerElec": 0,
        "Cff3": 0.299,
        "Cff2": -0.346,
        "Cff1": 0.701,
        "Cffch": 8.0e-7,
        "HEcoeff": 1,
    }


def pw_127m():
    """Return EngineSpecsPkg.PW_127M's engine specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 14.7,
        "Tt4Max": 1110,
        "ReqPower": 2051e3,
        "NPR": 1.3,
        "NoSpools": 3,
        "RPMs": [28870, 33300, 1200],
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Compressors": 0.88,
            "Combustor": 0.995,
            "Turbines": 0.88,
            "Nozzles": 0.985,
        },
    }


def cf34_8e5():
    """Return EngineSpecsPkg.CF34_8E5's engine specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 28.5,
        "FPR": 1.6,
        "BPR": 5,
        "Tt4Max": 1511,
        "TempLimit": {
            "Val": np.nan,
            "Type": np.nan,
        },
        "DesignThrust": convert_force(14510, "lbf", "N"),
        "NoSpools": 2,
        "RPMs": [7400, 17820],
        "FanGearRatio": np.nan,
        "FanBoosters": False,
        "CoreFlow": {
            "PaxBleed": 0.03,
            "Leakage": 0.01,
            "Cooling": 0.0,
        },
        "MaxIter": 300,
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Fan": 0.99,
            "Compressors": 0.94,
            "BypassNozzle": 0.99,
            "Combustor": 0.995,
            "Turbines": 0.94,
            "CoreNozzle": 0.99,
            "Nozzles": 0.99,
            "Mixing": 0.0,
        },
        "PerElec": 0,
        "Cff3": 0.299,
        "Cff2": -0.346,
        "Cff1": 0.701,
        "Cffch": 8.0e-7,
        "HEcoeff": 1,
    }


def example_turbofan():
    """Return EngineSpecsPkg.ExampleTF's template engine specification."""

    return {
        "Mach": np.nan,
        "Alt": np.nan,
        "OPR": np.nan,
        "FPR": np.nan,
        "BPR": np.nan,
        "Tt4Max": np.nan,
        "TempLimit": {
            "Val": np.nan,
            "Type": np.nan,
        },
        "DesignThrust": np.nan,
        "NoSpools": np.nan,
        "RPMs": [np.nan, np.nan, np.nan],
        "FanGearRatio": np.nan,
        "FanBoosters": False,
        "CoreFlow": {
            "PaxBleed": np.nan,
            "Leakage": np.nan,
            "Cooling": np.nan,
        },
        "MaxIter": 300,
        "Eta": {
            "Inlet": 1,
            "Fan": 1,
            "Compressor": 1,
            "BypassNozzle": 1,
            "Combustor": 1,
            "HPT": 1,
            "LPT": 1,
            "CoreNozzle": 1,
        },
    }


def example_turboprop():
    """Return EngineSpecsPkg.ExampleTP's template turboprop specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 15,
        "Tt4Max": 1200,
        "ReqPower": 3e6,
        "NPR": 1.3,
        "NoSpools": 2,
        "RPMs": [15000, 12000],
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Compressors": 0.9,
            "Combustor": 0.98,
            "Turbines": 0.9,
            "Nozzles": 0.985,
        },
    }


def ae2100_d3():
    """Return EngineSpecsPkg.AE2100_D3's turboprop engine specification."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 16.6,
        "Tt4Max": 1200,
        "ReqPower": 3458e3,
        "NPR": 1.3,
        "NoSpools": 2,
        "RPMs": [15284, 14267],
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Compressors": 0.86,
            "Combustor": 0.98,
            "Turbines": 0.86,
            "Nozzles": 0.985,
        },
    }


def ae3007a():
    """Return EngineSpecsPkg.AE3007A's turbofan engine specification."""

    engine = turbofan_engine_spec(0.05, 0, 24, 1.5, 6, 1450, 33717, 2, [5000, 20000], np.nan, False, 0.03, 0.01, 0.0, 0.94, 0.94, 0.0)
    engine["PerElec"] = 0
    return engine


def pw_1919g():
    """Return EngineSpecsPkg.PW_1919G's geared turbofan specification."""

    return turbofan_engine_spec(0.05, 0, 40, 1.4, 12, 2000, 100310, 2, [10600, 24470], 3.0625, False, 0.03, 0.01, 0.0, 0.9, 0.98, 0.95)


def pw_2037():
    """Return EngineSpecsPkg.PW_2037's turbofan specification."""

    return turbofan_engine_spec(0.05, 0, 27.4, 1.63, 5.8, 1500, 166941, 2, [4575, 12250], np.nan, False, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, fan_eta=0.9, core_nozzle_eta=0.98, nozzle_eta=0.98)


def cf6_80c2_b7f():
    """Return EngineSpecsPkg.CF6_80C2_B7F's turbofan specification."""

    return turbofan_engine_spec(0.05, 0, 31.8, 1.4, 5.31, 1500, 267000, 2, [3854, 11055], np.nan, True, 0.03, 0.01, 0.0, 0.9, 0.9, 0.0)


def rb211_22b_02():
    """Return EngineSpecsPkg.RB211_22B_02's turbofan specification."""

    return turbofan_engine_spec(0.85, 10668, 24.5, 1.42, 4.8, 2000, 42236, 3, [3900, 7000, 11611], np.nan, False, 0.03, 0.01, 0.25, 0.9, 0.98, None)


def trent_970b_84():
    """Return EngineSpecsPkg.Trent_970B_84's turbofan specification."""

    return turbofan_engine_spec(0.05, 0, 41, 1.5, 8.5, 1870, 348310, 3, [2900, 8300, 12200], np.nan, False, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, fan_eta=0.9, core_nozzle_eta=0.98, nozzle_eta=0.98)


def ae501d_22g():
    """Return EngineSpecsPkg.AE501D_22G's turboprop specification."""

    return turboprop_engine_spec(0.05, 0, 9.6, 1350.2, 3356e3, 1.3, 1, [13820], 0.9, 0.995, 0.9, itt_max=1106.483)


def allison_250_c30g():
    """Return EngineSpecsPkg.Allison_250_C30G's turboshaft specification."""

    return turboprop_engine_spec(0.05, 0, 8.6, 1100, 485e3, 1.3, 2, [50340, 30648], 0.9, 0.995, 0.9, itt_max=1106.483, jet_thrust=0)


def pt6a_114a():
    """Return EngineSpecsPkg.PT6A_114A's turboprop specification."""

    return turboprop_engine_spec(0.05, 0, 7, 1010, 541e3, 1.3, 2, [38100, 32986], 0.83, 0.92, 0.83, jet_thrust=552)


def pw_123():
    """Return EngineSpecsPkg.PW_123's turboprop specification."""

    return turboprop_engine_spec(0.05, 0, 13.8, 1300, 1775e3, 1.3, 3, [28800, 33300, 1200], 0.9, 0.995, 0.9)


def tpe331_14gr_805h():
    """Return EngineSpecsPkg.TPE331_14GR_805H's turboprop specification."""

    return turboprop_engine_spec(0.05, 0, 11.4, 1320, 1230e3, 1.3, 1, [35645], 0.9, 0.995, 0.9, itt_max=1106.483, jet_thrust=3300)


def turbofan_engine_spec(mach, alt, opr, fpr, bpr, tt4_max, design_thrust, no_spools, rpms, fan_gear_ratio, fan_boosters, pax_bleed, leakage, cooling, compressor_eta, turbine_eta, mixing_eta, fan_eta=0.99, core_nozzle_eta=0.99, nozzle_eta=0.99):
    """Return a common EngineSpecsPkg turbofan dictionary.

    Inputs:
        mach: Design Mach number.
        alt: Design altitude in meters.
        opr, fpr, bpr: Overall pressure ratio, fan pressure ratio, and bypass
            ratio.
        tt4_max: Turbine inlet temperature limit in K.
        design_thrust: Sea-level-static or design thrust in N.
        no_spools: Number of engine spools.
        rpms: Spool speeds in RPM.
        fan_gear_ratio: Fan gear ratio, or NaN when ungeared.
        fan_boosters: Whether fan boosters are present.
        pax_bleed, leakage, cooling: Core-flow fractions.
        compressor_eta, turbine_eta, mixing_eta: Polytropic efficiencies.
        fan_eta, core_nozzle_eta, nozzle_eta: Optional component efficiencies.

    Outputs:
        Engine specification dictionary matching EngineSpecsPkg turbofan fields.
    """

    eta_poly = {
        "Inlet": 0.99,
        "Diffusers": 0.99,
        "Fan": fan_eta,
        "Compressors": compressor_eta,
        "BypassNozzle": 0.99,
        "Combustor": 0.995,
        "Turbines": turbine_eta,
        "CoreNozzle": core_nozzle_eta,
        "Nozzles": nozzle_eta,
    }

    if mixing_eta is not None:
        eta_poly["Mixing"] = mixing_eta

    return {
        "Mach": mach,
        "Alt": alt,
        "OPR": opr,
        "FPR": fpr,
        "BPR": bpr,
        "Tt4Max": tt4_max,
        "TempLimit": {
            "Val": np.nan,
            "Type": np.nan,
        },
        "DesignThrust": design_thrust,
        "NoSpools": no_spools,
        "RPMs": rpms,
        "FanGearRatio": fan_gear_ratio,
        "FanBoosters": fan_boosters,
        "CoreFlow": {
            "PaxBleed": pax_bleed,
            "Leakage": leakage,
            "Cooling": cooling,
        },
        "MaxIter": 300,
        "EtaPoly": eta_poly,
    }


def turboprop_engine_spec(mach, alt, opr, tt4_max, req_power, npr, no_spools, rpms, compressor_eta, combustor_eta, turbine_eta, itt_max=None, jet_thrust=None):
    """Return a common EngineSpecsPkg turboprop/turboshaft dictionary.

    Inputs:
        mach: Design Mach number.
        alt: Design altitude in meters.
        opr: Overall pressure ratio.
        tt4_max: Turbine inlet temperature limit in K.
        req_power: Required shaft power in W.
        npr: Nozzle pressure ratio.
        no_spools: Number of spools.
        rpms: Spool speeds in RPM.
        compressor_eta, combustor_eta, turbine_eta: Polytropic efficiencies.
        itt_max: Optional inter-turbine temperature limit in K.
        jet_thrust: Optional residual jet thrust in N.

    Outputs:
        Engine specification dictionary matching EngineSpecsPkg turboprop
        fields.
    """

    engine = {
        "Mach": mach,
        "Alt": alt,
        "OPR": opr,
        "Tt4Max": tt4_max,
        "ReqPower": req_power,
        "NPR": npr,
        "NoSpools": no_spools,
        "RPMs": rpms,
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Compressors": compressor_eta,
            "Combustor": combustor_eta,
            "Turbines": turbine_eta,
            "Nozzles": 0.985,
        },
    }

    if itt_max is not None:
        engine["ITTMax"] = itt_max

    if jet_thrust is not None:
        engine["JetThrust"] = jet_thrust

    return engine


def example_aircraft():
    """Return AircraftSpecsPkg.Example's example aircraft specification."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": np.nan,
                "Class": "Turbofan",
                "MaxPax": 100,
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.8,
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": convert_length(35000, "ft", "m"),
                },
                "Range": convert_length(3350, "naut mi", "m"),
                "RCMax": convert_velocity(2000 / 60, "ft/s", "m/s"),
            },
            "Aero": {
                "L_D": {
                    "Clb": 10.936,
                    "Crs": 18.227,
                    "Des": 10.936,
                },
                "W_S": {
                    "SLS": convert_mass(112.56, "lbm", "kg")
                    / convert_length(1, "ft", "m") ** 2,
                },
            },
            "Weight": {
                "MTOW": convert_mass(124341, "lbm", "kg"),
                "Fuel": convert_mass(27452, "lbm", "kg"),
                "MLW": np.nan,
                "Batt": np.nan,
                "EM": np.nan,
                "EG": np.nan,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "C",
                },
                "T_W": {
                    "SLS": 0.3817,
                },
                "Thrust": {
                    "SLS": convert_force(23814 * 2, "lbf", "N"),
                },
                "Eta": {
                    "Prop": 0.8,
                },
                "NumEngines": 2,
                "Engine": cf34_8e5(),
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43.2e6 / 3.6e6,
                    "Batt": 0.25,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Settings": {
            "TkoPoints": 3,
            "ClbPoints": 5,
            "CrsPoints": 5,
            "DesPoints": 5,
            "OEW": {
                "MaxIter": np.nan,
                "Tol": np.nan,
            },
            "Analysis": {
                "MaxIter": np.nan,
                "Type": 1,
            },
            "Plotting": np.nan,
        },
    }


def lm100j_conventional():
    """Return AircraftSpecsPkg.LM100J_Conventional's aircraft specification."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": 2016,
                "Class": "Turboprop",
                "MaxPax": 4e4 / 209,
            },
            "Performance": {
                "Vels": {
                    "Tko": np.nan,
                    "Crs": 0.59,
                },
                "Alts": {
                    "Tko": np.nan,
                    "Crs": np.nan,
                },
                "Range": convert_length(2390, "naut mi", "m"),
                "RCMax": np.nan,
            },
            "Aero": {
                "L_D": {
                    "Clb": 12.3,
                    "Crs": 14.3,
                    "Des": np.nan,
                },
                "W_S": {
                    "SLS": 74389.1487
                    / convert_length(132 + 7 / 12, "ft", "m")
                    / convert_length(10, "ft", "m"),
                },
            },
            "Weight": {
                "MTOW": 74389.1487,
                "Fuel": np.nan,
                "MLW": np.nan,
                "Batt": np.nan,
                "EM": np.nan,
                "EG": np.nan,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "C",
                },
                "Eta": {
                    "Prop": 0.8,
                },
                "NumEngines": 4,
                "Engine": ae2100_d3(),
            },
            "Power": {
                "SLS": np.nan,
                "SpecEnergy": {
                    "Fuel": 43.2e6 / 3.6e6,
                    "Batt": 0.35,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                    "Propeller": 0.8,
                },
                "P_W": {
                    "SLS": 4 * 3410 / convert_mass(164e3, "lbm", "kg"),
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Geometry": {
            "Preset": "LM100JNominalGeometry",
            "LengthSet": convert_length(100.17, "ft", "m"),
        },
        "Settings": {
            "TkoPoints": 5,
            "ClbPoints": 10,
            "CrsPoints": 10,
            "DesPoints": 5,
            "OEW": {
                "MaxIter": np.nan,
                "Tol": np.nan,
            },
            "Analysis": {
                "MaxIter": 100,
                "Type": 1,
            },
            "Plotting": 1,
            "VisualizeAircraft": 0,
            "Table": 0,
        },
    }


def lm100j_hybrid():
    """Return AircraftSpecsPkg.LM100J_Hybrid's custom-architecture aircraft."""

    aircraft = lm100j_conventional()
    specs = aircraft["Specs"]
    architecture = lm100j_hybrid_architecture()
    specs["Propulsion"]["PropArch"] = architecture
    specs["Weight"]["Batt"] = 0
    specs["Power"]["LamDwn"] = {
        "SLS": 0.05,
        "Tko": 0.03,
        "Clb": 0.01,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }
    specs["Power"]["LamUps"] = {
        "SLS": 1,
        "Tko": 1,
        "Clb": 1,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }
    specs["Power"]["P_W"]["EM"] = 10
    specs["Power"]["P_W"]["EG"] = np.nan
    aircraft.pop("Geometry", None)
    aircraft["Settings"]["Analysis"]["MaxIter"] = np.nan
    aircraft["Settings"]["Plotting"] = 0
    return aircraft


def lm100j_hybrid_architecture():
    """Return LM100J_Hybrid's custom propulsion architecture matrices."""

    arch = [
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    eta_ups = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0.96, 0.96, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0.80, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0.80, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0.80, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.80, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    eta_dwn = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0.96, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0.96, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0.80, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0.80, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0.80, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0.80, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    return {
        "Type": "O",
        "Arch": arch,
        "OperUps": lm100j_hybrid_oper_ups,
        "OperDwn": lm100j_hybrid_oper_dwn,
        "EtaUps": eta_ups,
        "EtaDwn": eta_dwn,
        "SrcType": [1, 0],
        "TrnType": [1, 1, 0, 0, 2, 2, 2, 2],
    }


def lm100j_hybrid_oper_ups(lam):
    """Return LM100J_Hybrid's upstream operation matrix for one split."""

    value = split_scalar(lam)
    return [
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, value, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, value, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def lm100j_hybrid_oper_dwn(lam):
    """Return LM100J_Hybrid's downstream operation matrix for one split."""

    value = split_scalar(lam)
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.5 - value, 0.5 - value, value, value, 0],
    ]


def split_scalar(value):
    """Return the first numeric split value from a scalar or vector."""

    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def erj175lr():
    """Return AircraftSpecsPkg.ERJ175LR's conventional aircraft specification."""

    aircraft = erj175lr_base()
    power = aircraft["Specs"]["Power"]
    aircraft["Specs"]["Propulsion"]["PropArch"]["Type"] = "C"
    aircraft["Specs"]["Weight"]["Batt"] = 0
    aircraft["Specs"]["Power"]["SpecEnergy"]["Batt"] = np.nan
    power["LamDwn"] = zero_segment_splits()
    power["LamUps"] = zero_segment_splits()
    power["Eta"]["EM"] = np.nan
    power["Eta"]["EG"] = np.nan
    power["P_W"]["EM"] = np.nan
    power["P_W"]["EG"] = np.nan
    return aircraft


def erj175lr_elec():
    """Return AircraftSpecsPkg.ERJ175LR_Elec's parallel-hybrid specification."""

    aircraft = erj175lr_base()
    specs = aircraft["Specs"]
    power = specs["Power"]
    specs["Propulsion"]["PropArch"]["Type"] = "PHE"
    specs["Weight"]["Batt"] = 0
    power["SpecEnergy"]["Batt"] = 0.25
    power["LamDwn"] = {
        "SLS": 0.1,
        "Tko": 0.1,
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }
    power["LamUps"] = {
        "SLS": 1,
        "Tko": 1,
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }
    power["Eta"]["EM"] = 0.96
    power["Eta"]["EG"] = 0.96
    power["P_W"]["EM"] = 10
    power["P_W"]["EG"] = np.nan
    power["Battery"] = {
        "ParCells": 100,
        "SerCells": 62,
        "BegSOC": 100,
    }
    specs["Battery"] = {
        "NomVolCell": 3.6,
        "MaxExtVolCell": 4.0880,
        "CapCell": 3,
        "IntResist": 0.0199,
        "ExpVol": 0.0986,
        "ExpCap": 30,
        "MinSOC": 20,
        "BegSOC": 100,
        "MaxAllowCRate": 5,
        "Charging": 500 * 1000,
        "Degradation": 0,
    }
    return aircraft


def erj175lr_base():
    """Return shared ERJ175LR fields before propulsion-electric differences."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": 2005,
                "Class": "Turbofan",
                "MaxPax": 78,
            },
            "Aero": {
                "L_D": {
                    "ClbCF": 1.002,
                    "CrsCF": 1.000,
                    "Clb": 10.9773 * 1.002,
                    "Crs": 15.2000 * 1.000,
                    "Des": 10.9773 * 1.002,
                },
                "W_S": {
                    "SLS": convert_mass(109.25, "lbm", "kg")
                    / convert_length(1, "ft", "m") ** 2,
                },
            },
            "Propulsion": {
                "MDotCF": 1.029,
                "PropArch": {
                    "Type": "C",
                },
                "Engine": cf34_8e5(),
                "NumEngines": 2,
                "T_W": {
                    "SLS": 0.3393,
                },
                "Thrust": {
                    "SLS": convert_force(2 * 14510, "lbf", "N"),
                },
                "Eta": {
                    "Prop": 0.8,
                },
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.78,
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": convert_length(35000, "ft", "m"),
                },
                "Range": convert_length(2150, "naut mi", "m"),
                "RCMax": convert_velocity(2250, "ft/min", "m/s"),
            },
            "Weight": {
                "WairfCF": 1.018,
                "MTOW": convert_mass(85517, "lbm", "kg"),
                "EG": np.nan,
                "EM": 0,
                "Fuel": convert_mass(20785, "lbm", "kg"),
                "Batt": 0,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 12,
                    "Batt": np.nan,
                },
                "Eta": {
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": np.nan,
                    "EG": np.nan,
                },
            },
        },
        "Settings": {
            "TkoPoints": np.nan,
            "ClbPoints": np.nan,
            "CrsPoints": np.nan,
            "DesPoints": np.nan,
            "OEW": {
                "MaxIter": 50,
                "Tol": 0.001,
            },
            "Analysis": {
                "MaxIter": 30,
                "Type": 1,
            },
            "Plotting": 0,
            "Table": 0,
        },
    }


def erj190_e2():
    """Return AircraftSpecsPkg.ERJ190_E2's conventional aircraft specification."""

    aircraft = erj190_base()
    specs = aircraft["Specs"]
    specs["Propulsion"]["PropArch"]["Type"] = "C"
    specs["Propulsion"]["NumEngines"] = 2
    specs["Propulsion"]["Engine"] = cf34_8e5()
    specs["Power"]["LamUps"] = zero_segment_splits()
    specs["Power"]["LamDwn"] = zero_segment_splits()
    specs["Power"]["P_W"]["SLS"] = np.nan
    return aircraft


def erj190_fe():
    """Return AircraftSpecsPkg.ERJ190_FE's fully-electric aircraft specification."""

    aircraft = erj190_base()
    specs = aircraft["Specs"]
    specs["Propulsion"]["PropArch"]["Type"] = "E"
    specs["Power"]["SpecEnergy"]["Batt"] = 0.5
    specs["Power"]["P_W"] = {
        "AC": np.nan,
        "EM": np.nan,
        "EG": np.nan,
    }
    aircraft["Settings"].pop("DesPoints", None)
    return aircraft


def erj190_base():
    """Return shared ERJ190 fields before architecture-specific settings."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": np.nan,
                "Class": "Turbofan",
                "MaxPax": 100,
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.8,
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": 10668,
                },
                "Range": convert_length(3350, "naut mi", "m"),
                "RCMax": convert_velocity(2000 / 60, "ft/s", "m/s"),
            },
            "Aero": {
                "L_D": {
                    "Clb": 10.936,
                    "Crs": 18.227,
                    "Des": 10.936,
                },
                "W_S": {
                    "SLS": convert_mass(112.56, "lbm", "kg")
                    / convert_length(1, "ft", "m") ** 2,
                },
            },
            "Weight": {
                "MTOW": convert_mass(124341, "lbm", "kg"),
                "Fuel": convert_mass(27452, "lbm", "kg"),
                "MLW": np.nan,
                "Batt": np.nan,
                "EM": np.nan,
                "EG": np.nan,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "C",
                },
                "T_W": {
                    "SLS": 0.3817,
                },
                "Thrust": {
                    "SLS": convert_force(23814 * 2, "lbf", "N"),
                },
                "Eta": {
                    "Prop": 0.8,
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43.2e6 / 3.6e6,
                    "Batt": 0.25,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Settings": {
            "TkoPoints": np.nan,
            "ClbPoints": np.nan,
            "CrsPoints": np.nan,
            "DesPoints": np.nan,
            "OEW": {
                "MaxIter": np.nan,
                "Tol": np.nan,
            },
            "Analysis": {
                "MaxIter": np.nan,
                "Type": 1,
            },
            "Plotting": 0,
        },
    }


def a320neo():
    """Return AircraftSpecsPkg.A320Neo's aircraft specification."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": 2016,
                "Class": "Turbofan",
                "MaxPax": 15309 / 95,
            },
            "Aero": {
                "L_D": {
                    "ClbCF": 1,
                    "CrsCF": 1,
                    "Clb": 16,
                    "Crs": 18.23,
                    "Des": 16,
                },
                "W_S": {
                    "SLS": 79000 / 126.5,
                },
            },
            "Propulsion": {
                "MDotCF": 1,
                "PropArch": {
                    "Type": "C",
                },
                "Engine": leap_1a26(),
                "NumEngines": 2,
                "T_W": {
                    "SLS": 2.37e5 / (73500 * 9.81),
                },
                "Thrust": {
                    "SLS": 2.37e5,
                },
                "Eta": {
                    "Prop": 0.8,
                },
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.82,
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": convert_length(35000, "ft", "m"),
                },
                "Range": 4815e3,
                "RCMax": convert_length(2250 / 60, "ft", "m"),
            },
            "Weight": {
                "WairfCF": 1,
                "MTOW": 79000,
                "EG": np.nan,
                "EM": np.nan,
                "Fuel": 19000,
                "Batt": np.nan,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 12,
                    "Batt": np.nan,
                },
                "Eta": {
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "LamUps": zero_segment_splits(),
                "LamDwn": zero_segment_splits(),
                "Battery": {
                    "ParCells": np.nan,
                    "SerCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Settings": base_settings(
            tko_points=4,
            clb_points=5,
            crs_points=5,
            des_points=5,
            plotting=0,
            table=0,
        ),
    }


def atr42():
    """Return AircraftSpecsPkg.ATR42's aircraft specification."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": np.nan,
                "Class": "Turboprop",
                "MaxPax": 48,
            },
            "Performance": {
                "Vels": {
                    "Tko": np.nan,
                    "Crs": 0.4,
                },
                "Alts": {
                    "Tko": np.nan,
                    "Crs": convert_length(25000, "ft", "m"),
                },
                "Range": 1326e3,
                "RCMax": convert_velocity(1475 / 60, "ft/s", "m/s"),
            },
            "Aero": {
                "L_D": {
                    "Clb": 10,
                    "Crs": 12,
                    "Des": np.nan,
                },
                "W_S": {
                    "SLS": 342,
                },
            },
            "Weight": {
                "MTOW": 18600,
                "Fuel": 4500,
                "MLW": np.nan,
                "Batt": np.nan,
                "EM": np.nan,
                "EG": np.nan,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "C",
                },
                "T_W": {
                    "SLS": np.nan,
                },
                "Eta": {
                    "Prop": 0.8,
                },
                "Engine": pw_127m(),
            },
            "Power": {
                "SLS": np.nan,
                "SpecEnergy": {
                    "Fuel": 43.2e6 / 3.6e6,
                    "Batt": 0.35,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                },
                "P_W": {
                    "SLS": 0.1731,
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Settings": {
            "Analysis": {
                "Type": 1,
            },
            "Plotting": 1,
        },
    }


def aea():
    """Return AircraftSpecsPkg.AEA's aircraft specification."""

    architecture = aea_architecture_matrices()
    return {
        "Specs": {
            "TLAR": {
                "EIS": 2016,
                "Class": "Turbofan",
                "MaxPax": 1.7586e4 / 95,
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.747,
                    "Type": "TAS",
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": 7829,
                },
                "Range": convert_length(500, "naut mi", "m"),
                "RCMax": convert_length(2000 / 60, "ft", "m"),
            },
            "Aero": {
                "L_D": {
                    "Clb": 16,
                    "Crs": 18.6,
                    "Des": 16,
                },
                "W_S": {
                    "SLS": 109500 / 125.6,
                },
            },
            "Weight": {
                "MTOW": 109500,
                "EG": np.nan,
                "EM": np.nan,
                "Fuel": 0,
                "Batt": 36e3,
                "WairfCF": 0.87,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "O",
                    "Arch": architecture["Arch"],
                    "OperUps": architecture["OperUps"],
                    "OperDwn": architecture["OperDwn"],
                    "EtaUps": architecture["EtaUps"],
                    "EtaDwn": architecture["EtaDwn"],
                    "SrcType": 0,
                    "TrnType": [0, 0, 0, 0, 2, 2, 2, 2],
                },
                "Engine": np.nan,
                "NumEngines": 4,
                "T_W": {
                    "SLS": 0.3,
                },
                "Thrust": {
                    "SLS": np.nan,
                },
                "Eta": {
                    "Prop": 0.8,
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 12,
                    "Batt": 0.8 * 0.7,
                },
                "Eta": {
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": np.nan,
                    "EG": np.nan,
                },
                "Battery": {
                    "ParCells": np.nan,
                    "SerCells": np.nan,
                    "BegSOC": 100,
                },
            },
        },
        "Settings": {
            "TkoPoints": np.nan,
            "ClbPoints": np.nan,
            "CrsPoints": np.nan,
            "DesPoints": np.nan,
            "OEW": {
                "MaxIter": 50,
                "Tol": 0.001,
            },
            "Analysis": {
                "MaxIter": 50,
                "Type": 1,
            },
            "Plotting": 0,
            "Table": 0,
            "VisualizeAircraft": 0,
            "Offtake": 0,
        },
    }


def aircraft_ceras():
    """Return AircraftSpecsPkg.CeRAS's aircraft specification."""

    return {
        "Specs": {
            "TLAR": {
                "EIS": 2005,
                "Class": "Turbofan",
                "MaxPax": (
                    convert_mass(29762.4 + 7200, "lbm", "kg")
                    + 3394
                    + 634
                ) / 95,
            },
            "Aero": {
                "L_D": {
                    "ClbCF": 1,
                    "CrsCF": 1,
                    "Clb": 13,
                    "Crs": 17,
                    "Des": 13,
                },
                "W_S": {
                    "SLS": convert_mass(199645, "lbm", "kg")
                    / (1317.50 * convert_length(1, "ft", "m") ** 2),
                },
            },
            "Propulsion": {
                "MDotCF": 1.2,
                "PropArch": {
                    "Type": "C",
                },
                "Engine": ceras_engine(),
                "NumEngines": 2,
                "T_W": {
                    "SLS": 0.3,
                },
                "Eta": {
                    "Prop": 0.8,
                },
            },
            "Performance": {
                "Vels": {
                    "Tko": convert_velocity(135, "kts", "m/s"),
                    "Crs": 0.78,
                },
                "Alts": {
                    "Tko": 0,
                    "Crs": convert_length(35000, "ft", "m"),
                },
                "Range": convert_length(2500, "naut mi", "m"),
                "RCMax": convert_velocity(2250, "ft/min", "m/s"),
            },
            "Weight": {
                "WairfCF": 1,
                "MTOW": convert_mass(190000, "lbm", "kg"),
                "EG": np.nan,
                "EM": 0,
                "Fuel": convert_mass(20785, "lbm", "kg"),
                "Batt": 0,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 12,
                    "Batt": 0.25,
                },
                "LamDwn": zero_segment_splits(),
                "LamUps": zero_segment_splits(),
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                },
                "P_W": {
                    "SLS": np.nan,
                    "EM": 10,
                    "EG": np.nan,
                },
                "Battery": {
                    "ParCells": np.nan,
                    "SerCells": np.nan,
                    "BegSOC": np.nan,
                },
            },
        },
        "Settings": base_settings(
            tko_points=np.nan,
            clb_points=np.nan,
            crs_points=np.nan,
            des_points=np.nan,
            plotting=0,
            table=0,
        ),
    }


def aea_architecture_matrices():
    """Return the custom AEA architecture matrices."""

    arch = [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    oper_dwn = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0],
    ]
    eta_ups = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0.661, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0.661, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0.661, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0.661, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    eta_dwn = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0.661, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0.661, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0.661, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0.661, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    return {
        "Arch": arch,
        "OperUps": arch,
        "OperDwn": oper_dwn,
        "EtaUps": eta_ups,
        "EtaDwn": eta_dwn,
    }


def zero_segment_splits():
    """Return zero-valued FAST segment split fields."""

    return {
        "SLS": 0,
        "Tko": 0,
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }


def base_settings(tko_points, clb_points, crs_points, des_points, plotting, table):
    """Return common FAST aircraft settings."""

    return {
        "TkoPoints": tko_points,
        "ClbPoints": clb_points,
        "CrsPoints": crs_points,
        "DesPoints": des_points,
        "OEW": {
            "MaxIter": 50,
            "Tol": 0.001,
        },
        "Analysis": {
            "MaxIter": 50,
            "Type": 1,
        },
        "Plotting": plotting,
        "Table": table,
        "VisualizeAircraft": 0,
    }


LEAP_1A26 = leap_1a26
AE2100_D3 = ae2100_d3
AE3007A = ae3007a
AE501D_22G = ae501d_22g
Allison_250_C30G = allison_250_c30g
CF6_80C2_B7F = cf6_80c2_b7f
CF34_8E5 = cf34_8e5
Example = example_aircraft
ExampleTF = example_turbofan
ExampleTP = example_turboprop
ERJ175LR = erj175lr
ERJ175LR_Elec = erj175lr_elec
ERJ190_E2 = erj190_e2
ERJ190_FE = erj190_fe
LM100J_Conventional = lm100j_conventional
LM100J_Hybrid = lm100j_hybrid
A320Neo = a320neo
PW_127M = pw_127m
PW_123 = pw_123
PW_1919G = pw_1919g
PW_2037 = pw_2037
PT6A_114A = pt6a_114a
ATR42 = atr42
AEA = aea
RB211_22B_02 = rb211_22b_02
TPE331_14GR_805H = tpe331_14gr_805h
Trent_970B_84 = trent_970b_84
