# tests/test_battery.py

"""Tests for native BatteryPkg ports."""

import pytest

from fast_python.battery import (
    BatteryError,
    charging,
    cycling_aging,
    discharging,
    ground_charge,
    resize_battery,
)


def assert_array_close(actual, expected, tolerance=1.0e-9):
    """Assert arrays match within tolerance."""

    assert len(actual) == len(expected)

    for actual_item, expected_item in zip(actual, expected):
        assert abs(actual_item - expected_item) < tolerance


def test_discharging_and_charging_match_representative_values():
    """Check BatteryPkg discharge/charge equivalent-circuit outputs."""

    aircraft = make_battery_model_aircraft()
    discharged = discharging(aircraft, [1000, 800], [60, 120], 90, 100, 10)
    charged = charging(aircraft, [-500, -300], [60, 120], 90, 100, 10)

    assert_array_close(discharged[0], [42.1954263786386, 42.161865771118265])
    assert_array_close(discharged[1], [23.699250981055354, 18.974492361009705])
    assert_array_close(discharged[2], [999.9999999999998, 800.0000000000001])
    assert_array_close(discharged[3], [288.0, 287.6050124836491])
    assert_array_close(discharged[4], [90.0, 89.87656640114034, 89.67891543904649])
    assert_array_close(discharged[5], [0.07406015931579799, 0.05929528862815533])
    assert_array_close(charged[0], [42.203658867986604, 42.23054580465345])
    assert_array_close(charged[1], [-11.847314034169505, -7.103862720309491])
    assert_array_close(charged[2], [-500.0, -300.0])
    assert_array_close(charged[3], [288.0, 288.1974552339029])
    assert_array_close(charged[4], [90.06170476059464, 90.13570333059786])
    assert_array_close(charged[5], [-0.0370228563567797, -0.022199571000967158])


def test_ground_charge_populates_charged_aircraft_history():
    """Check GroundCharge records charging history and raises SOC."""

    result = ground_charge(make_cycling_aging_aircraft(1), 10, -1500)
    charged = result["Mission"]["History"]["SI"]["Power"]["ChargedAC"]

    assert charged["CtrlPtsTimeStep"] == 1
    assert charged["SOCEnd"] > 70
    assert len(charged["Voltage"]) == 10
    assert len(charged["SOC"]) == 11
    assert charged["C_rate"][0] > 0


def test_cycling_aging_matches_nmc_baseline():
    """Check CyclAging NMC baseline from FAST TestCyclingAging."""

    soh, fec, result = cycling_aging(
        make_cycling_aging_aircraft(1),
        1,
        0,
        60,
        -1500,
    )

    assert abs(soh - 99.9980155310) < 1.0e-6
    assert abs(fec - 0.1579166798) < 1.0e-6
    assert "ChargedAC" in result["Mission"]["History"]["SI"]["Power"]


def test_cycling_aging_matches_lfp_baseline():
    """Check CyclAging LFP baseline from FAST TestCyclingAging."""

    soh, fec, _ = cycling_aging(
        make_cycling_aging_aircraft(2),
        2,
        0.25,
        90,
        -2000,
    )

    assert abs(soh - 99.9937920058) < 1.0e-6
    assert abs(fec - 0.4012487844) < 1.0e-6


def test_cycling_aging_rejects_invalid_chemistry():
    """Check CyclAging validates chemistry labels."""

    with pytest.raises(BatteryError, match="ChemType"):
        cycling_aging(make_cycling_aging_aircraft(1), 3, 0, 60, -1500)


def test_resize_battery_simple_model_matches_fast_test():
    """Check simple ResizeBattery energy sizing."""

    aircraft = {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": 0,
                    "Arch": 1,
                }
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": 0.4 * 3.6e6,
                }
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Energy": {
                        "E_ES": [0, 100],
                        "Eleft_ES": [0, 0],
                    }
                }
            }
        },
        "Settings": {
            "DetailedBatt": 0,
        },
    }

    result = resize_battery(aircraft)

    assert abs(result["Specs"]["Weight"]["Batt"] - 6.9444e-05) < 1.0e-9


def test_resize_battery_detailed_model_matches_fast_test():
    """Check detailed ResizeBattery cell and weight sizing."""

    aircraft = {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": 0,
                    "Arch": 1,
                }
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": 0.3 * 3.6e6,
                },
                "Battery": {
                    "SerCells": 3,
                    "ParCells": 2,
                },
            },
            "Battery": {
                "NomVolCell": 3.9,
                "CapCell": 2.4,
                "MinSOC": 60,
                "MaxAllowCRate": 2.0,
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Energy": {
                        "E_ES": [0, 123],
                        "Eleft_ES": [0, 0],
                    },
                    "Power": {
                        "SOC": [60, 60],
                        "Pout": [0, 0],
                        "Current": [0, 0],
                    },
                }
            }
        },
        "Settings": {
            "DetailedBatt": 1,
        },
    }

    result = resize_battery(aircraft)

    assert abs(result["Specs"]["Weight"]["Batt"] - 0.1872) < 1.0e-9
    assert result["Specs"]["Power"]["Battery"]["ParCells"] == 2


def make_battery_model_aircraft():
    """Return representative Lithium-ion cell parameters for battery tests."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 1,
            }
        },
        "Specs": {
            "Battery": {
                "MaxExtVolCell": 4.088,
                "IntResist": 0.01,
                "ExpVol": 0.6,
                "ExpCap": 3.0,
                "CapCell": 3.2,
                "Degradation": 0,
            }
        },
    }


def make_cycling_aging_aircraft(chemistry):
    """Return MATLAB TestCyclingAging fixture data."""

    if chemistry == 1:
        op_temp = 20
        cap_cell = 5.0
        parallel = 2
        series = 3
        soc = [70, 60, 50, 60, 70]
        c_rate = [0.3, 0.4, 0.5, 0, 0]
        capacity = [0, 2, 4, 4, 4]
    else:
        op_temp = 25
        cap_cell = 4.0
        parallel = 3
        series = 2
        soc = [80, 60, 50, 55, 65]
        c_rate = [0.2, 0.3, 0.35, 0, 0]
        capacity = [0, 1.5, 3.0, 3.0, 3.0]

    return {
        "Settings": {
            "Analysis": {
                "Type": 0,
            },
            "Degradation": 0,
        },
        "Specs": {
            "Battery": {
                "NomVolCell": 3.7,
                "MaxExtVolCell": 4.0880,
                "ExpVol": 0.12,
                "ExpCap": 0.6,
                "IntResist": 0.01,
                "OpTemp": op_temp,
                "CapCell": cap_cell,
                "Degradation": 0,
            },
            "Power": {
                "Battery": {
                    "ParCells": parallel,
                    "SerCells": series,
                },
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "SOC": soc,
                        "C_rate": c_rate,
                        "Capacity": capacity,
                    },
                },
            },
        },
    }
