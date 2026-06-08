# tests/test_profiles.py

"""Tests for MissionProfilesPkg preset ports."""

import numpy as np

from fast_python.profiles import (
    ATRMissionBRE,
    ATRMissionEPASS,
    BRECruise02,
    DiversionMission,
    NotionalMission00,
    TakeoffTestProfile,
    a320,
    aea_profile,
    atr42_600,
    ceras,
    mission_profile_a320,
    mission_profile_aea,
    mission_profile_atr_mission_bre,
    mission_profile_atr_mission_epass,
    mission_profile_atr42_600,
    mission_profile_bre_cruise_00,
    mission_profile_bre_cruise_01,
    mission_profile_bre_cruise_02,
    mission_profile_ceras,
    mission_profile_diversion,
    mission_profile_erj,
    mission_profile_erj_climb_then_accel,
    mission_profile_lm100j,
    mission_profile_lm100j_no_reserve,
    mission_profile_notional_00,
    mission_profile_notional_01,
    mission_profile_notional_02,
    mission_profile_parametric_regional,
    mission_profile_regional_jet_00,
    mission_profile_regional_jet_01,
    mission_profile_regional_jet_02,
    mission_profile_takeoff_test,
    mission_profile_turboprop_00,
    mission_profile_turboprop_01,
    mission_profile_turboprop_02,
)
from fast_python.units import convert_length, convert_velocity


def test_a320_profile_matches_representative_matlab_values():
    """Check the A320 profile target and segment arrays."""

    profile = mission_profile_a320()

    assert len(profile["Segs"]) == 15
    assert profile["Target"]["Type"] == ["Dist", "Dist", "Dist", "Dist", "Time"]
    assert abs(profile["Target"]["Valu"][0] - convert_length(3400 / 3, "naut mi", "m")) < 1.0e-9
    assert profile["Target"]["Valu"][-1] == 30
    assert abs(profile["AltBeg"][2] - convert_length(10000, "ft", "m")) < 1.0e-12
    assert abs(profile["VelBeg"][2] - convert_velocity(250, "kts", "m/s")) < 1.0e-12
    assert np.isnan(profile["ClbRate"][0])


def test_aea_profile_uses_aircraft_takeoff_conditions():
    """Check AEAProfile imports takeoff altitude and speed from aircraft."""

    aircraft = {
        "Specs": {
            "Performance": {
                "Alts": {
                    "Tko": 123,
                },
                "Vels": {
                    "Tko": 45,
                },
            },
        },
    }
    profile = mission_profile_aea(aircraft)
    result = aea_profile(aircraft)

    assert profile["Target"]["Valu"] == [convert_length(500, "naut mi", "m")]
    assert profile["AltBeg"] == [123, 123, 7829, 7829]
    assert profile["VelEnd"][-1] == 54
    assert result["Mission"]["Profile"] == profile
    assert "Mission" not in aircraft


def test_atr42_profile_matches_representative_matlab_values():
    """Check ATR42_600 profile ranges, rates, and speed conversions."""

    profile = mission_profile_atr42_600()
    result = atr42_600({})

    assert len(profile["Segs"]) == 11
    assert profile["Target"]["Valu"][0] == convert_length(703, "naut mi", "m")
    assert profile["Target"]["Valu"][2] == 45
    assert profile["ClbRate"][3] == -6.096
    assert abs(profile["VelBeg"][1] - convert_velocity(160, "kts", "m/s")) < 1.0e-12
    assert result["Mission"]["Profile"]["Segs"][0] == "Takeoff"


def test_ceras_profile_matches_representative_matlab_values():
    """Check CeRAS profile target split and cruise altitudes."""

    profile = mission_profile_ceras()
    result = ceras({})

    assert profile["Segs"] == ["Climb", "Cruise", "Climb", "Cruise", "Descent"]
    assert profile["Target"]["Valu"][0] == convert_length(2500 * 0.35, "naut mi", "m")
    assert profile["Target"]["Valu"][1] == convert_length(2500 * 0.65, "naut mi", "m")
    assert profile["AltBeg"][1] == convert_length(33000, "ft", "m")
    assert profile["AltEnd"][2] == convert_length(35000, "ft", "m")
    assert result["Mission"]["Profile"]["TypeEnd"] == profile["TypeBeg"]


def test_a320_applicator_copies_aircraft():
    """Check mission applicators do not mutate the input aircraft."""

    aircraft = {"Specs": {"Name": "Demo"}}
    result = a320(aircraft)

    assert result["Specs"]["Name"] == "Demo"
    assert result["Mission"]["Profile"]["Segs"][0] == "Takeoff"
    assert "Mission" not in aircraft


def test_notional_mission00_uses_aircraft_performance_values():
    """Check NotionalMission00's aircraft-dependent target, speeds, and altitudes."""

    aircraft = make_notional_aircraft()
    profile = mission_profile_notional_00(aircraft)
    result = NotionalMission00(aircraft)

    assert profile["Target"] == {"Valu": [100000], "Type": ["Dist"]}
    assert profile["Segs"] == ["Takeoff", "Climb", "Cruise", "Descent", "Landing"]
    assert profile["AltBeg"] == [10, 10, 3000, 3000, 10]
    assert profile["VelBeg"] == [0, 80, 0.6, 0.6, 96]
    assert profile["TypeEnd"] == ["TAS", "Mach", "Mach", "TAS", "TAS"]
    assert result["Mission"]["Profile"] == profile
    assert "Mission" not in aircraft


def test_notional_reserve_profiles_match_distance_and_time_targets():
    """Check NotionalMission01/02 reserve target differences."""

    aircraft = make_notional_aircraft()
    distance_profile = mission_profile_notional_01(aircraft)
    time_profile = mission_profile_notional_02(aircraft)

    assert distance_profile["Target"] == {"Valu": [100000, 100], "Type": ["Dist", "Dist"]}
    assert time_profile["Target"] == {"Valu": [100000, 45], "Type": ["Dist", "Time"]}
    assert distance_profile["AltBeg"][4:7] == [300, 600, 600]
    assert distance_profile["AltEnd"][3:7] == [300, 600, 600, 10]
    assert distance_profile["VelEnd"][4:7] == [120, 120, 96]
    assert len(distance_profile["Segs"]) == 8


def test_takeoff_test_profile_matches_matlab_constants():
    """Check TakeoffTestProfile target, segment, altitude, and speed constants."""

    profile = mission_profile_takeoff_test()
    result = TakeoffTestProfile({})

    assert profile["Segs"] == ["DetailedTakeoff", "Climb", "Cruise"]
    assert profile["Target"]["Valu"][0] == convert_length(1.7, "naut mi", "m")
    assert profile["AltEnd"][1] == convert_length(50, "ft", "m")
    assert profile["VelBeg"][1] == convert_velocity(140, "kts", "m/s")
    assert profile["TypeBeg"] == ["TAS", "TAS", "TAS"]
    assert result["Mission"]["Profile"] == profile


def test_regional_jet_profiles_match_matlab_constants():
    """Check RegionalJetMission00/01/02 targets and reserve segment constants."""

    profile00 = mission_profile_regional_jet_00()
    profile01 = mission_profile_regional_jet_01()
    profile02 = mission_profile_regional_jet_02()

    assert profile00["Target"]["Valu"] == [convert_length(1650, "naut mi", "m")]
    assert profile00["AltBeg"][2] == convert_length(35000, "ft", "m")
    assert profile00["VelEnd"][3] == convert_velocity(160, "kts", "m/s")
    assert profile01["Target"]["Valu"][1] == convert_length(100, "naut mi", "m")
    assert profile02["Target"] == {"Valu": [convert_length(2150, "naut mi", "m"), 45], "Type": ["Dist", "Time"]}
    assert profile02["AltBeg"][4] == convert_length(3000, "ft", "m")
    assert len(profile02["Segs"]) == 8


def test_turboprop_profiles_match_matlab_constants():
    """Check TurbopropMission00/01/02 targets, speeds, and type arrays."""

    profile00 = mission_profile_turboprop_00()
    profile01 = mission_profile_turboprop_01()
    profile02 = mission_profile_turboprop_02()

    assert profile00["Target"]["Valu"] == [convert_length(703, "naut mi", "m")]
    assert profile00["AltEnd"][1] == convert_length(22000, "ft", "m")
    assert profile00["TypeBeg"] == ["TAS", "TAS", "EAS", "EAS", "TAS"]
    assert profile01["Target"]["Valu"][1] == convert_length(100, "naut mi", "m")
    assert profile02["Target"]["Type"] == ["Dist", "Time"]
    assert profile02["VelBeg"][5] == convert_velocity(140, "kts", "m/s")
    assert profile02["TypeEnd"][-1] == "TAS"


def test_bre_cruise_profiles_match_matlab_constants():
    """Check BRECruise00/01/02 CruiseBRE-only mission profiles."""

    profile00 = mission_profile_bre_cruise_00()
    profile01 = mission_profile_bre_cruise_01()
    profile02 = mission_profile_bre_cruise_02()
    result = BRECruise02({})

    assert profile00["Segs"] == ["CruiseBRE"]
    assert profile00["ID"] == [1]
    assert profile01["Target"]["Valu"][1] == convert_length(100, "naut mi", "m")
    assert profile01["Segs"] == ["CruiseBRE", "CruiseBRE"]
    assert profile02["Target"] == {"Valu": [convert_length(1650, "naut mi", "m"), 45], "Type": ["Dist", "Time"]}
    assert profile02["AltBeg"] == convert_length([35000, 35000], "ft", "m")
    assert profile02["VelEnd"] == convert_velocity([460, 460], "kts", "m/s")
    assert result["Mission"]["Profile"] == profile02


def test_diversion_mission_uses_takeoff_speed_and_delta_knots():
    """Check DiversionMission default and dV-adjusted transition speed."""

    aircraft = make_diversion_aircraft()
    profile = mission_profile_diversion(aircraft, 10)
    result = DiversionMission(aircraft)

    assert profile["Target"]["Valu"] == [convert_length(200, "naut mi", "m")]
    assert profile["AltBeg"][2] == convert_length(3000, "ft", "m")
    assert profile["AltEnd"][2] == convert_length(10000, "ft", "m")
    assert profile["VelBeg"][2] == 70 + convert_velocity(10, "kts", "m/s")
    assert profile["TypeEnd"][2] == "Mach"
    assert result["Mission"]["Profile"]["VelBeg"][2] == 70


def test_parametric_regional_profile_matches_aircraft_dependent_values():
    """Check ParametricRegional detailed segment arrays."""

    aircraft = make_notional_aircraft()
    profile = mission_profile_parametric_regional(aircraft)

    assert profile["Target"] == {"Valu": [100000, 45], "Type": ["Dist", "Time"]}
    assert len(profile["Segs"]) == 16
    assert profile["AltBeg"][3] == 3000 - convert_length(1000, "ft", "m")
    assert profile["AltBeg"][8] == convert_length(1500, "ft", "m")
    assert profile["VelEnd"][3] == 0.6
    assert profile["VelBeg"][8] == 96
    assert profile["TypeBeg"][4] == "Mach"


def test_lm100j_no_reserve_profile_matches_matlab_constants():
    """Check LM100J_NoRsrv target, climb rates, Mach override, and PowerOpt."""

    aircraft = make_notional_aircraft()
    profile = mission_profile_lm100j_no_reserve(aircraft)

    assert profile["Target"] == {"Valu": [100000], "Type": ["Dist"]}
    assert profile["Segs"] == ["Takeoff", "Climb", "Climb", "Climb", "Cruise", "Descent", "Landing"]
    assert profile["ClbRate"][2] == convert_velocity(2000, "ft/min", "m/s")
    assert profile["ClbRate"][5] == convert_velocity(-1500, "ft/min", "m/s")
    assert profile["VelBeg"][4] == 0.59
    assert profile["VelEnd"][4] == 0.59
    assert profile["PowerOpt"] == [1, 0, 0, 0, 0, 0, 0]


def test_atr_mission_bre_profile_matches_matlab_constants():
    """Check ATRMissionBRE CruiseBRE targets and constants."""

    profile = mission_profile_atr_mission_bre()
    result = ATRMissionBRE({})

    assert profile["Target"] == {
        "Valu": [convert_length(801, "naut mi", "m"), 45, convert_length(87, "naut mi", "m")],
        "Type": ["Dist", "Time", "Dist"],
    }
    assert profile["Segs"] == ["CruiseBRE", "CruiseBRE", "CruiseBRE"]
    assert profile["ID"] == [1, 2, 3]
    assert profile["AltBeg"] == convert_length([25000, 25000, 25000], "ft", "m")
    assert profile["VelBeg"] == convert_velocity([300, 300, 300], "kts", "m/s")
    assert result["Mission"]["Profile"] == profile


def test_atr_mission_epass_profile_matches_matlab_constants():
    """Check ATRMissionEPASS detailed ATR mission arrays."""

    profile = mission_profile_atr_mission_epass()
    result = ATRMissionEPASS({})

    assert len(profile["Segs"]) == 14
    assert profile["Target"]["Valu"][0] == convert_length(801, "naut mi", "m")
    assert profile["Target"]["Valu"][2] == convert_length(87, "naut mi", "m")
    assert profile["AltBeg"][2] == convert_length(24000, "ft", "m")
    assert profile["AltEnd"][4] == convert_length(24000, "ft", "m")
    assert profile["VelBeg"][3] == convert_velocity(300, "kts", "m/s")
    assert profile["TypeBeg"][1] == "EAS"
    assert result["Mission"]["Profile"] == profile


def test_lm100j_profile_matches_reserve_constants():
    """Check LM100J reserve profile targets, Mach overwrite, and PowerOpt."""

    aircraft = make_notional_aircraft()
    profile = mission_profile_lm100j(aircraft)

    assert profile["Target"] == {"Valu": [100000, 45], "Type": ["Dist", "Time"]}
    assert len(profile["Segs"]) == 10
    assert profile["AltBeg"][6] == convert_length(5000, "ft", "m")
    assert profile["ClbRate"][6] == convert_velocity(2000, "ft/min", "m/s")
    assert profile["VelBeg"][4] == 0.59
    assert profile["VelEnd"][4] == 0.59
    assert profile["PowerOpt"] == [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]


def test_erj_profiles_match_parametric_regional_variants():
    """Check ERJ and ERJ_ClimbThenAccel profile differences."""

    aircraft = make_notional_aircraft()
    erj_profile = mission_profile_erj(aircraft)
    climb_then_accel = mission_profile_erj_climb_then_accel(aircraft)

    assert erj_profile["Target"]["Valu"][1] == convert_length(100, "naut mi", "m")
    assert erj_profile["Target"]["Valu"][2] == 45
    assert len(erj_profile["Segs"]) == 17
    assert erj_profile["AltBeg"][3] == 3000 - convert_length(1000, "ft", "m")
    assert climb_then_accel["AltBeg"][3] == 3000
    assert erj_profile["VelBeg"][6] == convert_velocity(200, "kts", "m/s")
    assert climb_then_accel["VelBeg"][6] == convert_velocity(210, "kts", "m/s")
    assert erj_profile["TypeEnd"][3] == "Mach"


def make_notional_aircraft():
    """Return a compact aircraft with performance fields for notional profiles."""

    return {
        "Specs": {
            "Performance": {
                "Range": 100000,
                "Alts": {
                    "Tko": 10,
                    "Crs": 3000,
                },
                "Vels": {
                    "Tko": 80,
                    "Crs": 0.6,
                },
            },
        },
    }


def make_diversion_aircraft():
    """Return a compact aircraft with a takeoff speed for DiversionMission."""

    return {
        "Specs": {
            "Performance": {
                "Vels": {
                    "Tko": 70,
                },
            },
        },
    }
