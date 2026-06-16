# tests/test_mission_segments.py

"""Tests for native MissionSegsPkg segment evaluators."""

import numpy as np

from fast_python.atmosphere import gravity
from fast_python.data_struct import init_mission_history
from fast_python.mission import (
    compute_flight_conditions,
    eval_climb,
    eval_cruise,
    eval_cruise_breguet,
    eval_descent,
    eval_detailed_takeoff,
    eval_landing,
    eval_takeoff,
    fly_mission,
    process_profile,
)


def assert_array_close(actual, expected, tolerance=1.0e-9):
    """Assert arrays match within tolerance."""

    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def test_eval_takeoff_fills_trajectory_and_electric_energy():
    """Check EvalTakeoff trajectory and battery bookkeeping."""

    aircraft = init_mission_history(make_takeoff_aircraft())
    result = eval_takeoff(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Performance"]["Time"], [0, 30, 60])
    assert_array_close(history["Performance"]["TAS"], [0, 50, 100])
    assert_array_close(history["Performance"]["Dist"], [0, 750, 3000])
    assert_array_close(history["Performance"]["Acc"], [100 / 60] * 3)
    assert_array_close(history["Power"]["TV"], [1000, 1000, 1000])
    assert_array_close(history["Power"]["Pout"], [[1000, 1000, 1000]] * 3)
    assert_array_close(history["Energy"]["E_ES"], [[0], [30000], [60000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[100000], [70000], [40000]])
    assert result["Mission"]["History"]["Segment"] == ["Takeoff", "Takeoff", "Takeoff"]


def test_eval_takeoff_default_does_not_mutate_input():
    """Check public EvalTakeoff keeps its no-mutation contract."""

    aircraft = init_mission_history(make_takeoff_aircraft())
    result = eval_takeoff(aircraft)

    assert aircraft["Mission"]["History"]["Segment"] == ["", "", ""]
    assert aircraft["Mission"]["History"]["SI"]["Power"]["TV"] == [0.0, 0.0, 0.0]
    assert result["Mission"]["History"]["Segment"] == ["Takeoff", "Takeoff", "Takeoff"]


def test_eval_takeoff_internal_mode_mutates_segment_aircraft():
    """Check FlyMission can reuse its already-copied aircraft."""

    aircraft = init_mission_history(make_takeoff_aircraft())
    result = eval_takeoff(aircraft, copy_aircraft=False)

    assert result is aircraft
    assert aircraft["Mission"]["History"]["Segment"] == ["Takeoff", "Takeoff", "Takeoff"]


def test_eval_detailed_takeoff_computes_acceleration_limited_roll():
    """Check EvalDetailedTakeoff uses available power for roll physics."""

    aircraft = init_mission_history(make_detailed_takeoff_aircraft())
    result = eval_detailed_takeoff(aircraft)
    history = result["Mission"]["History"]["SI"]
    expected = detailed_takeoff_expected_values()

    assert_array_close(history["Performance"]["TAS"], [0, 50, 100])
    assert_array_close(history["Performance"]["Time"], expected["time"], tolerance=1.0e-9)
    assert_array_close(history["Performance"]["Dist"], expected["dist"], tolerance=1.0e-9)
    assert_array_close(history["Performance"]["Acc"], expected["acc"], tolerance=1.0e-9)
    assert history["Performance"]["LD"][0] > 0
    assert_array_close(history["Power"]["Req"], [np.inf, np.inf, np.inf])
    assert result["Mission"]["History"]["Segment"] == ["Takeoff", "Takeoff", "Takeoff"]


def test_eval_landing_fills_trajectory_and_reverse_power():
    """Check EvalLanding trajectory and reverse-power energy demand."""

    aircraft = init_mission_history(make_landing_aircraft())
    result = eval_landing(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Performance"]["Time"], [0, 30])
    assert_array_close(history["Performance"]["TAS"], [100, 0])
    assert_array_close(history["Performance"]["Dist"], [0, 0])
    assert_array_close(history["Performance"]["Acc"], [-100 / 30, 0])
    assert_array_close(history["Power"]["Req"], [300, 0])
    assert_array_close(history["Power"]["Pout"], [[300, 300, 300], [0, 0, 0]])
    assert_array_close(history["Energy"]["E_ES"], [[0], [9000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[100000], [91000]])
    assert result["Mission"]["History"]["Segment"] == ["Landing", "Landing"]


def test_eval_cruise_fills_level_cruise_power_and_energy():
    """Check EvalCruise level-flight drag power and battery energy."""

    aircraft = init_mission_history(make_cruise_aircraft())
    result = eval_cruise(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Performance"]["Dist"], [0, 500, 1000])
    assert_array_close(history["Performance"]["Time"], [0, 5, 10])
    assert_array_close(history["Performance"]["TAS"], [100, 100, 100])
    assert_array_close(history["Performance"]["RC"], [0, 0, 0])
    assert_array_close(history["Performance"]["Acc"], [0, 0, 0])
    assert_array_close(history["Power"]["Req"], [98100, 98100, 98100])
    assert_array_close(history["Power"]["Pout"], [[98100, 98100, 98100]] * 3)
    assert_array_close(history["Energy"]["E_ES"], [[0], [490500], [981000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[2000000], [1509500], [1019000]])
    assert result["Mission"]["History"]["Segment"] == ["Cruise", "Cruise", "Cruise"]


def test_eval_cruise_breguet_fills_conventional_range_solution():
    """Check EvalCruiseBRE fills the Breguet final-weight solution."""

    aircraft = init_mission_history(make_cruise_breguet_aircraft())
    result = eval_cruise_breguet(aircraft)
    history = result["Mission"]["History"]["SI"]
    expected = breguet_conventional_expected()

    assert_array_close(history["Performance"]["Dist"], [0, 500, 1000])
    assert_array_close(history["Performance"]["Time"], [0, 5, 10])
    assert_array_close(history["Performance"]["TAS"], [100, 100, 100])
    assert_array_close(history["Weight"]["CurWeight"], expected["mass"])
    assert_array_close(history["Weight"]["Fburn"], expected["fburn"])
    assert_array_close(history["Energy"]["Fuel"], expected["fuel_energy"])
    assert_array_close(history["Power"]["Fuel"], expected["pfuel"])
    assert_array_close(history["Power"]["Req"], expected["preq"])
    assert_array_close(history["Power"]["Batt"], [0, 0, 0])
    assert_array_close(history["Power"]["Av"], [0, 0, 0])
    assert_array_close(history["Propulsion"]["TSFC"], [2.0e-5, 2.0e-5, 2.0e-5])
    assert_array_close(history["Propulsion"]["MDotFuel"], [0, 0, 0])
    assert_array_close(history["Performance"]["Rho"], [0, 0, 0])
    assert_array_close(history["Energy"]["E_ES"], [[0], [0], [0]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[0], [0], [0]])
    assert result["Mission"]["History"]["Segment"] == ["Cruise", "Cruise", "Cruise"]


def test_fly_mission_runs_cruise_breguet_segment():
    """Check FlyMission dispatches CruiseBRE segments."""

    aircraft = make_cruise_breguet_aircraft()
    aircraft["Mission"]["Profile"] = {
        "Segs": ["CruiseBRE"],
        "ID": [1],
        "Target": {
            "Valu": [1000],
            "Type": ["Dist"],
        },
        "AltBeg": [0],
        "AltEnd": [0],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "ClbRate": [float("nan")],
    }
    aircraft = process_profile(aircraft)
    aircraft = init_mission_history(aircraft)
    result = fly_mission(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Performance"]["Dist"][-1], 1000)
    assert result["Mission"]["Profile"]["CrsTarget"] == 1000
    assert result["Mission"]["History"]["Segment"] == ["Cruise", "Cruise", "Cruise"]


def test_eval_climb_fills_prescribed_climb_power_and_energy():
    """Check EvalClimb prescribed-rate trajectory and battery energy."""

    aircraft = init_mission_history(make_climb_aircraft())
    result = eval_climb(aircraft)
    history = result["Mission"]["History"]["SI"]
    fpa = np.degrees(np.arcsin(10 / 100))
    dist = 100 * np.cos(np.radians(fpa)) * 50
    drag_power = 1000 * 9.81 * np.cos(np.radians(fpa)) / 10 * 100
    level_drag_power = 1000 * 9.81 / 10 * 100
    climb_power = 1000 * 9.81 * 10
    preq = drag_power + climb_power

    assert_array_close(history["Performance"]["Time"], [0, 50, 100])
    assert_array_close(history["Performance"]["Dist"], [0, dist, dist * 2])
    assert_array_close(history["Performance"]["TAS"], [100, 100, 100])
    assert_array_close(history["Performance"]["RC"], [10, 10, 0])
    assert_array_close(history["Performance"]["Acc"], [0, 0, 0])
    assert_array_close(history["Power"]["Req"], [preq, preq, level_drag_power])
    assert_array_close(history["Energy"]["E_ES"], [[0], [preq * 50], [preq * 100]])
    assert result["Mission"]["History"]["Segment"] == ["Climb", "Climb", "Climb"]


def test_eval_descent_fills_prescribed_descent_power_and_energy():
    """Check EvalDescent prescribed-rate trajectory and idle clipping."""

    aircraft = init_mission_history(make_descent_aircraft())
    result = eval_descent(aircraft)
    history = result["Mission"]["History"]["SI"]
    fpa = np.degrees(np.arcsin(-10 / 100))
    dist = 100 * np.cos(np.radians(fpa)) * 50
    level_drag_power = 1000 * 9.81 / 10 * 100

    assert_array_close(history["Performance"]["Time"], [0, 50, 100])
    assert_array_close(history["Performance"]["Dist"], [0, dist, dist * 2])
    assert_array_close(history["Performance"]["TAS"], [100, 100, 100])
    assert_array_close(history["Performance"]["RC"], [-10, -10, 0])
    assert_array_close(history["Performance"]["Acc"], [0, 0, 0])
    assert_array_close(history["Power"]["Req"], [0.0001, 0.0001, level_drag_power])
    assert_array_close(history["Energy"]["E_ES"], [[0], [0.005], [0.01]])
    assert result["Mission"]["History"]["Segment"] == ["Descent", "Descent", "Descent"]


def test_fly_mission_iterates_cruise_to_match_distance_target():
    """Check FlyMission adjusts cruise so the full mission hits target distance."""

    aircraft = process_profile(make_full_mission_aircraft())
    aircraft = init_mission_history(aircraft)
    result = fly_mission(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Performance"]["Dist"][-1], 20000, tolerance=1.0e-5)
    assert result["Mission"]["Profile"]["CrsTarget"] < 20000
    assert result["Mission"]["History"]["Segment"] == [
        "Climb",
        "Climb",
        "Cruise",
        "Cruise",
        "Descent",
        "Descent",
        "Descent",
    ]
    assert result["Mission"]["History"]["Flags"]["SOCOff"] == [0]


def make_takeoff_aircraft():
    """Return a minimal all-electric aircraft for EvalTakeoff."""

    return {
        "Settings": {
            "nargOperUps": 0,
            "nargOperDwn": 0,
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Performance": {
                "Alts": {
                    "Tko": 0,
                }
            },
            "Weight": {
                "MTOW": 1000,
                "Batt": 100,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000,
                },
                "LamDwn": {
                    "Tko": 0,
                },
                "LamUps": {
                    "Tko": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
                "Windmill": {
                    "Tko": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                    "Lnd": 0,
                },
            },
            "Propulsion": {
                "SLSPower": [1000],
                "SLSThrust": [0],
                "PropArch": {
                    "Type": "E",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperUps": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperDwn": [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                    "EtaUps": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "EtaDwn": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "SrcType": [0],
                    "TrnType": [0],
                    "WhichProp": [0],
                    "ParConns": [[]],
                },
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "MissID": 1,
                "SegPts": [3],
                "SegBeg": [1],
                "SegEnd": [3],
                "AltEnd": [0],
                "VelEnd": [100],
                "TypeEnd": ["TAS"],
            }
        },
    }


def make_landing_aircraft():
    """Return a minimal all-electric aircraft for EvalLanding."""

    aircraft = make_takeoff_aircraft()
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [2],
        "SegBeg": [1],
        "SegEnd": [2],
        "AltBeg": [0],
        "VelBeg": [100],
        "TypeBeg": ["TAS"],
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Lnd": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Lnd": 0,
    }
    return aircraft


def make_detailed_takeoff_aircraft():
    """Return a minimal aircraft for EvalDetailedTakeoff."""

    aircraft = make_takeoff_aircraft()
    aircraft["Specs"]["Aero"] = {
        "W_S": {
            "SLS": 100,
        }
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    return aircraft


def detailed_takeoff_expected_values():
    """Return expected detailed-takeoff trajectory values."""

    mtow = 1000
    wing_loading = 100
    wing_area = mtow / wing_loading
    _, _, _, _, _, rho, _ = compute_flight_conditions(0, 0, "TAS", 100)
    gravity = 9.81
    velocity = np.asarray([0, 50, 100], dtype=float)
    cl_max = 2 * mtow * gravity / (rho * (100 / 1.1) ** 2 * wing_area)
    cd0 = 0.0017
    delta_cd0 = wing_loading * 3.16e-5 * mtow ** -0.215
    cd = cd0 + delta_cd0 + (0.02 + 0.6 / (np.pi * 0.9 * 10)) * cl_max ** 2
    drag = np.zeros(3)
    acceleration = np.zeros(3)
    lift = 0.5 * rho * velocity[1:] ** 2 * cl_max * wing_area
    friction = 0.02 * (mtow * gravity - lift)
    friction[friction < 0] = 0
    drag[1:] = 0.5 * rho * velocity[1:] ** 2 * cd * wing_area
    thrust = np.asarray([np.nan, 1000000 / 50, 1000000 / 100])
    acceleration[1:] = (thrust[1:] - drag[1:] - friction) / mtow
    dtime = np.zeros(3)
    ddist = np.zeros(3)
    dtime[1:] = np.diff(velocity) / acceleration[1:]
    ddist[1:] = np.diff(velocity ** 2) / (2 * acceleration[1:])
    return {
        "acc": acceleration.tolist(),
        "time": np.cumsum(dtime).tolist(),
        "dist": np.cumsum(ddist).tolist(),
    }


def make_cruise_aircraft():
    """Return a minimal all-electric aircraft for EvalCruise."""

    aircraft = make_takeoff_aircraft()
    aircraft["Specs"]["Aero"] = {
        "S": 10,
        "L_D": {
            "Method": "ConstantLD",
            "Clb": 10,
            "Crs": 10,
            "Des": 10,
        }
    }
    aircraft["Specs"]["Performance"]["RCMax"] = 1000
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Crs": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Crs": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [200000]
    aircraft["Specs"]["Weight"]["Batt"] = 2000
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltBeg": [0],
        "AltEnd": [0],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "CrsTarget": 1000,
    }
    return aircraft


def make_cruise_breguet_aircraft():
    """Return a minimal conventional aircraft for EvalCruiseBRE."""

    return {
        "Settings": {
            "nargOperUps": 0,
            "nargOperDwn": 0,
            "Analysis": {
                "Type": 1,
                "MaxIter": 10,
            },
            "CrsPoints": 3,
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Aero": {
                "L_D": {
                    "Crs": 10,
                }
            },
            "Weight": {
                "MTOW": 10000,
                "Fuel": 1000,
                "Batt": 0,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                },
                "Phi": {
                    "Crs": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                    "BegSOC": float("nan"),
                },
            },
            "Propulsion": {
                "TSFC": 2.0e-5,
                "Eta": {
                    "Prop": 0.85,
                },
                "PropArch": {
                    "Type": "C",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "SrcType": [1],
                    "TrnType": [1, 2],
                    "WhichProp": [0, 0],
                },
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "MissID": 1,
                "SegPts": [3],
                "SegBeg": [1],
                "SegEnd": [3],
                "AltBeg": [0],
                "AltEnd": [0],
                "VelBeg": [100],
                "VelEnd": [100],
                "TypeBeg": ["TAS"],
                "TypeEnd": ["TAS"],
                "CrsTarget": 1000,
            }
        },
    }


def breguet_conventional_expected():
    """Return expected conventional CruiseBRE values for the test fixture."""

    mass = np.zeros(3)
    mass[0] = 10000
    distance_step = 500
    time_step = 5
    efuel = 43200000
    eta_prop = 0.85
    tsfc = 2.0e-5
    velocity = 100
    lift_drag = 10
    grav = gravity(0)
    eta_overall = velocity / (tsfc * efuel)
    eta_gt = eta_overall / eta_prop
    denominator = eta_prop * (efuel / grav) * lift_drag * eta_gt

    for index in range(2):
        mass[index + 1] = mass[index] / np.exp(distance_step / denominator)

    dfburn = -np.diff(mass)
    fuel_energy_step = dfburn * efuel
    pfuel = np.zeros(3)
    pfuel[:-1] = fuel_energy_step / time_step
    preq = eta_prop * eta_gt * pfuel
    fburn = np.zeros(3)
    fburn[1:] = np.cumsum(dfburn)
    fuel_energy = np.zeros(3)
    fuel_energy[1:] = np.cumsum(fuel_energy_step)
    return {
        "mass": mass.tolist(),
        "fburn": fburn.tolist(),
        "fuel_energy": fuel_energy.tolist(),
        "pfuel": pfuel.tolist(),
        "preq": preq.tolist(),
    }


def make_climb_aircraft():
    """Return a minimal all-electric aircraft for EvalClimb."""

    aircraft = make_cruise_aircraft()
    aircraft["Specs"]["Aero"] = {
        "S": 10,
        "L_D": {
            "Method": "ConstantLD",
            "Clb": 10,
            "Crs": 10,
            "Des": 10,
        }
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Clb": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Clb": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    aircraft["Specs"]["Weight"]["Batt"] = 20000
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltBeg": [0],
        "AltEnd": [1000],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "ClbRate": [10],
    }
    return aircraft


def make_descent_aircraft():
    """Return a minimal all-electric aircraft for EvalDescent."""

    aircraft = make_cruise_aircraft()
    aircraft["Specs"]["Aero"] = {
        "S": 10,
        "L_D": {
            "Method": "ConstantLD",
            "Clb": 10,
            "Crs": 10,
            "Des": 10,
        }
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Des": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Des": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    aircraft["Specs"]["Weight"]["Batt"] = 20000
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltBeg": [1000],
        "AltEnd": [0],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "ClbRate": [-10],
    }
    return aircraft


def make_full_mission_aircraft():
    """Return a minimal all-electric aircraft for FlyMission."""

    aircraft = make_climb_aircraft()
    aircraft["Settings"]["Analysis"]["MaxIter"] = 10
    aircraft["Settings"]["ClbPoints"] = 3
    aircraft["Settings"]["CrsPoints"] = 3
    aircraft["Settings"]["DesPoints"] = 3
    aircraft["Specs"]["Aero"] = {
        "S": 10,
        "L_D": {
            "Method": "ConstantLD",
            "Clb": 10,
            "Crs": 10,
            "Des": 10,
        }
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
    }
    aircraft["Mission"]["Profile"] = {
        "Segs": ["Climb", "Cruise", "Descent"],
        "ID": [1, 1, 1],
        "Target": {
            "Valu": [20000],
            "Type": ["Dist"],
        },
        "AltBeg": [0, 1000, 1000],
        "AltEnd": [1000, 1000, 0],
        "VelBeg": [100, 100, 100],
        "VelEnd": [100, 100, 100],
        "TypeBeg": ["TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS"],
        "ClbRate": [10, float("nan"), -10],
    }
    return aircraft
