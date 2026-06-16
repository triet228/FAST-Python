# tests/test_data_struct.py

"""Tests for native DataStructPkg ports."""

import math
from fast_python.aircraft import prepare_aircraft
from fast_python.data_struct import (
    clear_mission,
    init_mission_history,
    pre_spec_processing,
    spec_processing,
)
from fast_python.markers import MatlabRow
from fast_python.reference import load_bundled_case_inputs


def test_pre_spec_processing_preserves_values_and_fills_defaults():
    """Check aircraft defaults are inserted without replacing user values."""

    aircraft = {
        "Specs": {
            "TLAR": {
                "Class": "Turbofan",
                "MaxPax": 180,
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "C",
                }
            },
        },
        "Settings": {
            "TkoPoints": 8,
        },
    }

    processed = pre_spec_processing(aircraft)

    assert processed["Specs"]["TLAR"]["Class"] == "Turbofan"
    assert processed["Specs"]["TLAR"]["MaxPax"] == 180
    assert processed["Settings"]["TkoPoints"] == 8
    assert math.isnan(processed["Specs"]["Performance"]["Range"])
    assert math.isnan(processed["Specs"]["Power"]["Battery"]["BegSOC"])
    assert math.isnan(processed["Specs"]["Battery"]["Degradation"])
    assert math.isnan(processed["Settings"]["Analysis"]["MaxIter"])
    assert math.isnan(processed["Geometry"]["LengthSet"])
    assert "Performance" not in aircraft["Specs"]


def test_init_mission_history_allocates_fast_shapes():
    """Check mission-history arrays match DataStructPkg dimensions."""

    aircraft = make_history_aircraft()
    initialized = init_mission_history(aircraft)
    history = initialized["Mission"]["History"]
    si = history["SI"]

    assert len(si["Performance"]["Time"]) == 6
    assert len(si["Aero"]["CL"]) == 6
    assert len(si["Propulsion"]["TSFC"]) == 6
    assert len(si["Propulsion"]["TSFC"][0]) == 2
    assert len(si["Power"]["Windmill"][0]) == 1
    assert len(si["Power"]["LamUps"][0]) == 2
    assert len(si["Power"]["LamDwn"][0]) == 1
    assert si["Power"]["DV"] == [0.0] * 6
    assert len(si["Power"]["Pav"][0]) == 4
    assert len(si["Energy"]["E_ES"][0]) == 1
    assert history["Segment"] == ["", "", "", "", "", ""]
    assert history["EE"] is not history["SI"]
    assert history["EE"]["Power"] is not history["SI"]["Power"]


def test_init_mission_history_accepts_scalar_source_and_matlab_row_transmitters():
    """Check wrapper scalar/row propulsion metadata shapes are accepted."""

    aircraft = make_history_aircraft()
    aircraft["Specs"]["Propulsion"]["PropArch"]["SrcType"] = 0
    aircraft["Specs"]["Propulsion"]["PropArch"]["TrnType"] = MatlabRow([0, 2])
    initialized = init_mission_history(aircraft)
    si = initialized["Mission"]["History"]["SI"]

    assert len(si["Energy"]["E_ES"][0]) == 1
    assert len(si["Propulsion"]["TSFC"][0]) == 2


def test_clear_mission_resets_tail_and_preserves_prior_rows():
    """Check one-based tail reset semantics from ClearMission."""

    initialized = init_mission_history(make_history_aircraft())
    history = initialized["Mission"]["History"]["SI"]
    history["Performance"]["Time"] = [1, 2, 3, 4, 5, 6]
    history["Power"]["Pav"] = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5],
        [6, 6, 6, 6],
    ]
    initialized["Mission"]["History"]["Segment"] = [
        "Takeoff",
        "Climb",
        "Cruise",
        "Cruise",
        "Descent",
        "Landing",
    ]

    cleared = clear_mission(initialized, 3)
    cleared_history = cleared["Mission"]["History"]

    assert cleared_history["SI"]["Performance"]["Time"] == [1, 2, 0, 0, 0, 0]
    assert cleared_history["SI"]["Power"]["Pav"][0] == [1, 1, 1, 1]
    assert cleared_history["SI"]["Power"]["Pav"][2] == [0, 0, 0, 0]
    assert cleared_history["Segment"] == ["Takeoff", "Climb", "", "", "", ""]


def test_spec_processing_prepares_turbofan_fixture():
    """Check database-backed preprocessing for a wrapper turbofan case."""

    aircraft = load_wrapper_aircraft("A320")
    processed = spec_processing(pre_spec_processing(prepare_aircraft(aircraft)))
    specs = processed["Specs"]

    assert specs["Weight"]["Payload"] == aircraft["Specs"]["TLAR"]["MaxPax"] * 95
    assert specs["Power"]["SpecEnergy"]["Fuel"] == 43200000
    assert specs["Power"]["P_W"]["EM"] > 1000
    assert processed["Geometry"]["Preset"] == "SmallDoubleAisleTurbofan"
    assert isinstance(specs["Propulsion"]["Engine"], dict)
    assert "OEW" in processed["RegressionParams"]
    assert "TurbofanAC" not in processed["HistData"]
    assert "AC" in processed["HistData"]


def test_spec_processing_prepares_turboprop_fixture_defaults():
    """Check database-backed preprocessing fills ATR42 omitted values."""

    aircraft = load_wrapper_aircraft("ATR42")
    processed = spec_processing(pre_spec_processing(prepare_aircraft(aircraft)))
    specs = processed["Specs"]

    assert specs["TLAR"]["EIS"] == 2021
    assert processed["Settings"]["TkoPoints"] == 10
    assert math.isclose(specs["Performance"]["Vels"]["Tko"], 115 * 0.514444444444445)
    assert math.isclose(specs["Aero"]["L_D"]["Des"], 7.2)
    assert math.isclose(specs["Power"]["SLS"], 0.1731 * 18600 * 1000)
    assert math.isclose(specs["Power"]["P_W"]["SLS"], 173.1)
    assert processed["RegressionParams"] == {}


def make_history_aircraft():
    """Return a minimal aircraft with processed profile and architecture."""

    return {
        "Settings": {
            "nargOperUps": 2,
            "nargOperDwn": 0,
        },
        "Mission": {
            "Profile": {
                "SegEnd": [3, 6],
            }
        },
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0, 1, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    "SrcType": [1],
                    "TrnType": [1, 2],
                }
            }
        },
    }


def load_wrapper_aircraft(case_name):
    """Load one wrapper fixture aircraft JSON file."""

    aircraft, _ = load_bundled_case_inputs(case_name)
    return aircraft
