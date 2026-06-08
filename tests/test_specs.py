# tests/test_specs.py

"""Tests for AircraftSpecsPkg and EngineSpecsPkg preset ports."""

import numpy as np

from fast_python.specs import (
    aea,
    ae2100_d3,
    ae3007a,
    ae501d_22g,
    a320neo,
    aircraft_ceras,
    allison_250_c30g,
    atr42,
    ceras_engine,
    cf34_8e5,
    cf6_80c2_b7f,
    erj175lr,
    erj175lr_elec,
    erj190_e2,
    erj190_fe,
    example_aircraft,
    example_turbofan,
    example_turboprop,
    leap_1a26,
    lm100j_conventional,
    lm100j_hybrid,
    pt6a_114a,
    pw_127m,
    pw_123,
    pw_1919g,
    pw_2037,
    rb211_22b_02,
    tpe331_14gr_805h,
    trent_970b_84,
)
from fast_python.units import convert_force, convert_length, convert_mass, convert_velocity


def test_leap_1a26_engine_matches_matlab_preset_values():
    """Check representative LEAP_1A26 engine fields."""

    engine = leap_1a26()

    assert engine["Mach"] == 0.05
    assert engine["OPR"] == 50
    assert engine["BPR"] == 11
    assert engine["RPMs"] == [3894, 19391]
    assert engine["FanBoosters"] is True
    assert engine["EtaPoly"]["Compressors"] == 0.96
    assert engine["Cffch"] == 6.1e-7
    assert np.isnan(engine["TempLimit"]["Val"])


def test_ceras_engine_matches_matlab_preset_values():
    """Check representative CeRAS engine fields."""

    engine = ceras_engine()

    assert engine["OPR"] == 24.5
    assert engine["FPR"] == 1.6
    assert engine["BPR"] == 4
    assert abs(engine["DesignThrust"] - convert_force(14510, "lbf", "N")) < 1.0e-12
    assert engine["RPMs"] == [7400, 17820]
    assert engine["FanBoosters"] is False
    assert engine["PerElec"] == 0


def test_pw_127m_engine_matches_matlab_preset_values():
    """Check representative PW_127M turboprop engine fields."""

    engine = pw_127m()

    assert engine["Mach"] == 0.05
    assert engine["OPR"] == 14.7
    assert engine["Tt4Max"] == 1110
    assert engine["ReqPower"] == 2051e3
    assert engine["NPR"] == 1.3
    assert engine["NoSpools"] == 3
    assert engine["RPMs"] == [28870, 33300, 1200]
    assert engine["EtaPoly"]["Nozzles"] == 0.985


def test_cf34_8e5_engine_matches_matlab_preset_values():
    """Check representative CF34_8E5 engine fields."""

    engine = cf34_8e5()

    assert engine["Mach"] == 0.05
    assert engine["OPR"] == 28.5
    assert engine["FPR"] == 1.6
    assert engine["BPR"] == 5
    assert abs(engine["DesignThrust"] - convert_force(14510, "lbf", "N")) < 1.0e-12
    assert engine["RPMs"] == [7400, 17820]
    assert engine["EtaPoly"]["Compressors"] == 0.94
    assert engine["Cff3"] == 0.299


def test_ae2100_d3_engine_matches_matlab_preset_values():
    """Check representative AE2100_D3 turboprop engine fields."""

    engine = ae2100_d3()

    assert engine["Mach"] == 0.05
    assert engine["OPR"] == 16.6
    assert engine["Tt4Max"] == 1200
    assert engine["ReqPower"] == 3458e3
    assert engine["RPMs"] == [15284, 14267]
    assert engine["EtaPoly"]["Compressors"] == 0.86


def test_remaining_engine_specs_match_representative_matlab_values():
    """Check representative fields for the remaining EngineSpecsPkg presets."""

    assert ae3007a()["DesignThrust"] == 33717
    assert ae3007a()["PerElec"] == 0
    assert ae501d_22g()["ITTMax"] == 1106.483
    assert ae501d_22g()["RPMs"] == [13820]
    assert allison_250_c30g()["JetThrust"] == 0
    assert allison_250_c30g()["ReqPower"] == 485e3
    assert cf6_80c2_b7f()["FanBoosters"] is True
    assert cf6_80c2_b7f()["DesignThrust"] == 267000
    assert pt6a_114a()["JetThrust"] == 552
    assert pt6a_114a()["EtaPoly"]["Combustor"] == 0.92
    assert pw_123()["NoSpools"] == 3
    assert pw_123()["ReqPower"] == 1775e3
    assert pw_1919g()["FanGearRatio"] == 3.0625
    assert pw_1919g()["EtaPoly"]["Mixing"] == 0.95
    assert pw_2037()["CoreFlow"]["PaxBleed"] == 0.0
    assert pw_2037()["EtaPoly"]["Fan"] == 0.9
    assert rb211_22b_02()["Mach"] == 0.85
    assert rb211_22b_02()["CoreFlow"]["Cooling"] == 0.25
    assert tpe331_14gr_805h()["JetThrust"] == 3300
    assert tpe331_14gr_805h()["RPMs"] == [35645]
    assert trent_970b_84()["BPR"] == 8.5
    assert trent_970b_84()["NoSpools"] == 3


def test_example_engine_templates_match_matlab_values():
    """Check ExampleTF and ExampleTP template engine presets."""

    turbofan = example_turbofan()
    turboprop = example_turboprop()

    assert np.isnan(turbofan["Mach"])
    assert all(np.isnan(value) for value in turbofan["RPMs"])
    assert turbofan["FanBoosters"] is False
    assert turbofan["Eta"]["HPT"] == 1
    assert turboprop["Mach"] == 0.05
    assert turboprop["OPR"] == 15
    assert turboprop["ReqPower"] == 3e6
    assert turboprop["RPMs"] == [15000, 12000]
    assert turboprop["EtaPoly"]["Combustor"] == 0.98


def test_a320neo_aircraft_matches_representative_matlab_values():
    """Check representative A320Neo aircraft preset fields."""

    aircraft = a320neo()
    specs = aircraft["Specs"]

    assert specs["TLAR"]["Class"] == "Turbofan"
    assert abs(specs["TLAR"]["MaxPax"] - 15309 / 95) < 1.0e-12
    assert abs(specs["Performance"]["Vels"]["Tko"] - convert_velocity(135, "kts", "m/s")) < 1.0e-12
    assert specs["Performance"]["Range"] == 4815e3
    assert specs["Aero"]["L_D"]["Crs"] == 18.23
    assert specs["Weight"]["MTOW"] == 79000
    assert specs["Propulsion"]["Engine"]["OPR"] == 50
    assert specs["Propulsion"]["Thrust"]["SLS"] == 2.37e5
    assert specs["Power"]["LamUps"]["Crs"] == 0
    assert aircraft["Settings"]["TkoPoints"] == 4
    assert aircraft["Settings"]["VisualizeAircraft"] == 0


def test_atr42_aircraft_matches_representative_matlab_values():
    """Check representative ATR42 aircraft preset fields."""

    aircraft = atr42()
    specs = aircraft["Specs"]

    assert np.isnan(specs["TLAR"]["EIS"])
    assert specs["TLAR"]["Class"] == "Turboprop"
    assert specs["TLAR"]["MaxPax"] == 48
    assert specs["Performance"]["Vels"]["Crs"] == 0.4
    assert specs["Performance"]["Alts"]["Crs"] == convert_length(25000, "ft", "m")
    assert specs["Performance"]["Range"] == 1326e3
    assert abs(specs["Performance"]["RCMax"] - convert_velocity(1475 / 60, "ft/s", "m/s")) < 1.0e-12
    assert specs["Aero"]["L_D"]["Clb"] == 10
    assert specs["Aero"]["W_S"]["SLS"] == 342
    assert specs["Weight"]["MTOW"] == 18600
    assert specs["Propulsion"]["Engine"]["ReqPower"] == 2051e3
    assert specs["Power"]["P_W"]["SLS"] == 0.1731
    assert aircraft["Settings"]["Plotting"] == 1


def test_aea_aircraft_matches_representative_matlab_values():
    """Check representative AEA aircraft preset and architecture fields."""

    aircraft = aea()
    specs = aircraft["Specs"]
    prop_arch = specs["Propulsion"]["PropArch"]

    assert specs["TLAR"]["EIS"] == 2016
    assert specs["TLAR"]["Class"] == "Turbofan"
    assert abs(specs["TLAR"]["MaxPax"] - 1.7586e4 / 95) < 1.0e-12
    assert specs["Performance"]["Vels"]["Type"] == "TAS"
    assert specs["Performance"]["Range"] == convert_length(500, "naut mi", "m")
    assert specs["Aero"]["W_S"]["SLS"] == 109500 / 125.6
    assert specs["Weight"]["Batt"] == 36e3
    assert prop_arch["Type"] == "O"
    assert len(prop_arch["Arch"]) == 10
    assert prop_arch["Arch"][0] == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert prop_arch["OperDwn"][9] == [0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0]
    assert prop_arch["EtaUps"][1][5] == 0.661
    assert prop_arch["TrnType"] == [0, 0, 0, 0, 2, 2, 2, 2]
    assert specs["Power"]["SpecEnergy"]["Batt"] == 0.8 * 0.7
    assert specs["Power"]["Battery"]["BegSOC"] == 100
    assert aircraft["Settings"]["Offtake"] == 0


def test_ceras_aircraft_matches_representative_matlab_values():
    """Check representative CeRAS aircraft preset fields."""

    aircraft = aircraft_ceras()
    specs = aircraft["Specs"]

    expected_pax = (
        convert_mass(29762.4 + 7200, "lbm", "kg") + 3394 + 634
    ) / 95
    expected_ws = convert_mass(199645, "lbm", "kg") / (
        1317.50 * convert_length(1, "ft", "m") ** 2
    )

    assert specs["TLAR"]["EIS"] == 2005
    assert abs(specs["TLAR"]["MaxPax"] - expected_pax) < 1.0e-12
    assert specs["Performance"]["Vels"]["Crs"] == 0.78
    assert specs["Performance"]["Range"] == convert_length(2500, "naut mi", "m")
    assert abs(specs["Aero"]["W_S"]["SLS"] - expected_ws) < 1.0e-12
    assert specs["Weight"]["MTOW"] == convert_mass(190000, "lbm", "kg")
    assert specs["Weight"]["Batt"] == 0
    assert specs["Propulsion"]["Engine"]["OPR"] == 24.5
    assert specs["Power"]["P_W"]["EM"] == 10
    assert np.isnan(aircraft["Settings"]["TkoPoints"])


def test_example_aircraft_matches_matlab_template_values():
    """Check AircraftSpecsPkg.Example representative fields."""

    aircraft = example_aircraft()
    specs = aircraft["Specs"]

    assert specs["TLAR"]["Class"] == "Turbofan"
    assert specs["TLAR"]["MaxPax"] == 100
    assert specs["Performance"]["Range"] == convert_length(3350, "naut mi", "m")
    assert specs["Performance"]["Vels"]["Crs"] == 0.8
    assert specs["Aero"]["L_D"]["Crs"] == 18.227
    assert abs(
        specs["Aero"]["W_S"]["SLS"]
        - convert_mass(112.56, "lbm", "kg") / convert_length(1, "ft", "m") ** 2
    ) < 1.0e-12
    assert specs["Weight"]["MTOW"] == convert_mass(124341, "lbm", "kg")
    assert specs["Propulsion"]["PropArch"]["Type"] == "C"
    assert specs["Propulsion"]["Engine"]["OPR"] == 28.5
    assert specs["Power"]["SpecEnergy"]["Fuel"] == 12
    assert aircraft["Settings"]["TkoPoints"] == 3
    assert aircraft["Settings"]["Analysis"]["Type"] == 1


def test_erj175lr_aircraft_matches_matlab_conventional_values():
    """Check ERJ175LR conventional preset representative fields."""

    aircraft = erj175lr()
    specs = aircraft["Specs"]

    assert specs["TLAR"]["EIS"] == 2005
    assert specs["TLAR"]["MaxPax"] == 78
    assert specs["Aero"]["L_D"]["Clb"] == 10.9773 * 1.002
    assert specs["Propulsion"]["PropArch"]["Type"] == "C"
    assert specs["Propulsion"]["MDotCF"] == 1.029
    assert specs["Performance"]["Range"] == convert_length(2150, "naut mi", "m")
    assert specs["Weight"]["MTOW"] == convert_mass(85517, "lbm", "kg")
    assert specs["Weight"]["Fuel"] == convert_mass(20785, "lbm", "kg")
    assert specs["Power"]["LamDwn"]["SLS"] == 0
    assert np.isnan(specs["Power"]["SpecEnergy"]["Batt"])
    assert aircraft["Settings"]["Analysis"]["MaxIter"] == 30


def test_erj175lr_elec_aircraft_matches_matlab_electric_values():
    """Check ERJ175LR_Elec parallel-hybrid fields and battery details."""

    aircraft = erj175lr_elec()
    specs = aircraft["Specs"]

    assert specs["Propulsion"]["PropArch"]["Type"] == "PHE"
    assert specs["Power"]["SpecEnergy"]["Batt"] == 0.25
    assert specs["Power"]["LamDwn"]["SLS"] == 0.1
    assert specs["Power"]["LamUps"]["Tko"] == 1
    assert specs["Power"]["Eta"]["EM"] == 0.96
    assert specs["Power"]["P_W"]["EM"] == 10
    assert specs["Power"]["Battery"]["ParCells"] == 100
    assert specs["Battery"]["MaxExtVolCell"] == 4.0880
    assert specs["Battery"]["Charging"] == 500 * 1000
    assert specs["Battery"]["Degradation"] == 0


def test_erj190_e2_aircraft_matches_matlab_values():
    """Check ERJ190_E2 conventional preset representative fields."""

    aircraft = erj190_e2()
    specs = aircraft["Specs"]

    assert specs["TLAR"]["Class"] == "Turbofan"
    assert specs["TLAR"]["MaxPax"] == 100
    assert specs["Performance"]["Alts"]["Crs"] == 10668
    assert specs["Performance"]["Range"] == convert_length(3350, "naut mi", "m")
    assert specs["Aero"]["L_D"]["Crs"] == 18.227
    assert specs["Propulsion"]["PropArch"]["Type"] == "C"
    assert specs["Propulsion"]["NumEngines"] == 2
    assert specs["Propulsion"]["Engine"]["OPR"] == 28.5
    assert specs["Power"]["LamUps"]["SLS"] == 0
    assert np.isnan(specs["Power"]["Battery"]["BegSOC"])


def test_erj190_fe_aircraft_matches_matlab_values():
    """Check ERJ190_FE fully-electric preset representative fields."""

    aircraft = erj190_fe()
    specs = aircraft["Specs"]

    assert specs["Propulsion"]["PropArch"]["Type"] == "E"
    assert specs["Power"]["SpecEnergy"]["Batt"] == 0.5
    assert "AC" in specs["Power"]["P_W"]
    assert np.isnan(specs["Power"]["P_W"]["AC"])
    assert specs["Power"]["Eta"]["EM"] == 0.96
    assert specs["Propulsion"]["Thrust"]["SLS"] == convert_force(23814 * 2, "lbf", "N")
    assert "DesPoints" not in aircraft["Settings"]


def test_lm100j_conventional_aircraft_matches_matlab_values():
    """Check LM100J_Conventional representative fields."""

    aircraft = lm100j_conventional()
    specs = aircraft["Specs"]

    assert specs["TLAR"]["Class"] == "Turboprop"
    assert specs["TLAR"]["MaxPax"] == 4e4 / 209
    assert specs["Performance"]["Range"] == convert_length(2390, "naut mi", "m")
    assert specs["Aero"]["L_D"]["Crs"] == 14.3
    assert specs["Weight"]["MTOW"] == 74389.1487
    assert specs["Propulsion"]["PropArch"]["Type"] == "C"
    assert specs["Propulsion"]["NumEngines"] == 4
    assert specs["Propulsion"]["Engine"]["ReqPower"] == 3458e3
    assert specs["Power"]["P_W"]["SLS"] == 4 * 3410 / convert_mass(164e3, "lbm", "kg")
    assert aircraft["Geometry"]["Preset"] == "LM100JNominalGeometry"
    assert aircraft["Geometry"]["LengthSet"] == convert_length(100.17, "ft", "m")
    assert aircraft["Settings"]["Analysis"]["MaxIter"] == 100


def test_lm100j_hybrid_aircraft_matches_matlab_custom_architecture():
    """Check LM100J_Hybrid custom architecture and split-dependent matrices."""

    aircraft = lm100j_hybrid()
    specs = aircraft["Specs"]
    prop_arch = specs["Propulsion"]["PropArch"]

    assert prop_arch["Type"] == "O"
    assert len(prop_arch["Arch"]) == 11
    assert prop_arch["Arch"][0] == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert prop_arch["SrcType"] == [1, 0]
    assert prop_arch["TrnType"] == [1, 1, 0, 0, 2, 2, 2, 2]
    assert prop_arch["EtaUps"][1][4] == 0.96
    assert prop_arch["EtaDwn"][10] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert prop_arch["OperUps"](0.2)[4][8] == 0.2
    assert prop_arch["OperDwn"](0.2)[10] == [0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.2, 0.2, 0]
    assert specs["Power"]["LamDwn"]["SLS"] == 0.05
    assert specs["Power"]["LamUps"]["Clb"] == 1
    assert specs["Power"]["P_W"]["EM"] == 10
    assert "Geometry" not in aircraft
