# tests/test_engine.py

"""Tests for native EngineModelPkg ports."""

import numpy as np

from fast_python.atmosphere import standard_atmosphere
from fast_python.engine import (
    a_astar,
    astar_a,
    cp_air,
    cp_jeta,
    cv_air,
    burner,
    comp_stage,
    compressor,
    diffuser,
    local_efficiency,
    local_reynolds,
    mass_flow_parameter,
    new_gamma,
    newton_raphson_tt1,
    newton_raphson_tt3,
    off_design_nozzle,
    perf_ex_nozzle,
    ps_pt,
    pt_ps,
    rhos_rhot,
    simple_off_design,
    turb_stage,
    turbine,
    ts_tt,
    tt_ts,
    turbofan_nonlinear_sizing,
    turbofan_on_design_cycle,
    turbofan_linear_sizing,
    turboprop_nonlinear_sizing,
    turboprop_on_design_cycle,
    turboprop_linear_sizing,
)
from fast_python.markers import MatlabRow


def assert_close(actual, expected, tolerance=1.0e-9):
    """Assert scalar values match within tolerance."""

    assert abs(actual - expected) < tolerance


def test_isentropic_and_specific_heat_helpers_match_fast_formulas():
    """Check IsenRelPkg and SpecHeatPkg scalar helper formulas."""

    pt = pt_ps(101325, 0.5, 1.4)
    tt = tt_ts(288.15, 0.5, 1.4)

    assert_close(pt, 120192.99554984865)
    assert_close(ps_pt(pt, 0.5, 1.4), 101325)
    assert_close(tt, 302.5575)
    assert_close(ts_tt(tt, 0.5, 1.4), 288.15)
    assert_close(cp_air(300), 1007.1576666941079)
    assert_close(cp_air(300, 1200), 975796.019946149)
    assert_close(cp_jeta(300, 1200), 2721493.7956250105)


def test_remaining_engine_helper_formulas_match_fast_values():
    """Check remaining IsenRelPkg and SpecHeatPkg helper formulas."""

    astar = astar_a(2.0, 0.5, 1.4)
    gamma_result = new_gamma(1200, 0.7, 1.4)
    reynolds = local_reynolds(
        {
            "Ts": 300,
            "Ro": 0.5,
            "Ri": 0.2,
            "Ps": 101325,
            "Mach": 0.5,
            "Gam": 1.4,
        }
    )

    assert_close(astar, 1.4927113702623904)
    assert_close(a_astar(astar, 0.5, 1.4), 2.0)
    assert_close(mass_flow_parameter(0.5, 1.4), 0.0404260185570129)
    assert_close(rhos_rhot(1.5, 0.5, 1.4), 1.3277552012905212)
    assert_close(cv_air(300), 720.1576666941079)
    assert_close(gamma_result[0], 1111.382957596211)
    assert_close(gamma_result[1], 1168.9224813712344)
    assert_close(gamma_result[2], 881.9224813712344)
    assert_close(gamma_result[3], 1.325425426908004)
    assert_close(local_efficiency(1.0e6), 0.8443851671995364)
    assert_close(reynolds, 2740820.510093329)
    assert_close(newton_raphson_tt1(300, 100000), 382.6832178636457)
    assert_close(newton_raphson_tt3(1200, 100000), 1115.4905614685606)


def test_off_design_nozzle_matches_matlab_values():
    """Check ComponentOffPkg Nozzle against MATLAB oracle values."""

    assert_close(off_design_nozzle(1.5, 2.0, 0.3, 1.4), 0.219537929121)
    assert_close(off_design_nozzle(1.5, 0.8, 0.9, 1.4), 0.418404458924)
    assert_close(off_design_nozzle(1.0, 100.0, 0.1, 1.4), 1.0)


def test_diffuser_and_burner_match_representative_component_values():
    """Check ComponentOnPkg Diffuser and Burner representative outputs."""

    state = make_component_flow_state()
    eta = {
        "Diffusers": 0.99,
        "Combustor": 0.98,
    }
    diffused = diffuser(state, 0.3, "Outer", eta)
    burned, fuel_flow, fuel_air_ratio = burner(diffused, 1200, 43.17e6, eta)

    assert_close(diffused["Pt"], 118947.47625917323)
    assert_close(diffused["Ts"], 297.18394341984595)
    assert_close(diffused["Gam"], 1.3986219636254724)
    assert_close(diffused["Area"], 0.759466546754636)
    assert_close(diffused["Ri"], 0.8707779797067305)
    assert_close(fuel_flow, 0.24583591808788896)
    assert_close(fuel_air_ratio, 0.024583591808788895)
    assert_close(burned["MDot"], 10.24583591808789)
    assert_close(burned["Pt"], 99201.49373794645)
    assert_close(burned["Mach"], 0.11494447055893961)
    assert_close(burned["Ts"], 1197.470689872743)
    assert_close(burned["Gam"], 1.3196997982033598)
    assert_close(burned["Ps"], 98341.30928628524)
    assert_close(burned["Area"], 1.8938634030495547)
    assert_close(burned["Ri"], 0.6302099298865813)


def test_compressor_stage_and_wrapper_match_representative_values():
    """Check ComponentOnPkg CompStg and Compressor representative outputs."""

    eta = make_component_eta()
    state = make_component_flow_state()
    stage, work, tau = comp_stage(state, eta, 1.2, 12000)
    compressed, comp_object = compressor(state, 2.0, 12000, eta)

    assert_close(work, 180792.9454208882)
    assert_close(tau, 1.0533675294569007)
    assert_close(stage["Pt"], 144178.75910202815)
    assert_close(stage["Mach"], 0.48655521368317556)
    assert_close(stage["Area"], 0.4361099728492316)
    assert_close(stage["Eta"], 0.8551661261301973)
    assert_close(stage["Psi"], 0.011087935920011257)
    assert_close(stage["Phi"], 0.013478278621624784)
    assert_close(compressed["Pt"], 240297.9318367136)
    assert_close(compressed["Tt"], 368.56190489529325)
    assert_close(compressed["Mach"], 0.45098480537485214)
    assert_close(compressed["Area"], 0.29786479565867635)
    assert comp_object["NoStages"] == 2
    assert_close(comp_object["Pi"], 2.0)
    assert_close(comp_object["Tau"], 1.2183101975746407)
    assert_close(comp_object["ReqWork"], 740882.5348646322)


def test_turbine_stage_and_wrapper_match_representative_values():
    """Check ComponentOnPkg TurbStg and Turbine representative outputs."""

    eta = make_component_eta()
    hot_state = make_hot_turbine_state()
    stage, pi_stage, tau = turb_stage(hot_state, 1100, 12000, True, eta)
    expanded, turbine_object = turbine(
        hot_state,
        {
            "RPM": 12000,
            "ReqWork": 500000,
        },
        eta,
    )

    assert_close(pi_stage, 0.7247055612854424)
    assert_close(tau, 0.9166666666666666)
    assert_close(stage["Pt"], 217411.6683856327)
    assert_close(stage["Mach"], 0.45319319598010377)
    assert_close(stage["Area"], 0.05563524728061169)
    assert_close(stage["Ro"], 1.096841529247806)
    assert_close(expanded["Pt"], 258580.73335027043)
    assert_close(expanded["Tt"], 1152.7751801837906)
    assert_close(expanded["Mach"], 0.44107113277762755)
    assert_close(expanded["Area"], 0.048972764485143254)
    assert turbine_object["NoStages"] == 1
    assert_close(turbine_object["CPR"], 0.8619357778342348)
    assert_close(turbine_object["CTR"], 0.9606459834864921)
    assert_close(turbine_object["DelivWork"], 555555.5555555555)


def test_perf_ex_nozzle_matches_representative_values():
    """Check ComponentOnPkg PerfExNozzle representative outputs."""

    ambient = make_nozzle_ambient_state()
    hot = make_nozzle_hot_state(ambient)
    nozzle, thrust = perf_ex_nozzle(
        hot,
        ambient,
        {
            "Nozzles": 0.985,
        },
        "Prop",
    )

    assert_close(thrust, 8036.908647172055)
    assert_close(nozzle["Pt"], 242587.3562293636)
    assert_close(nozzle["Ps"], 136292.37593854472)
    assert_close(nozzle["Ts"], 864.4903760682575)
    assert_close(nozzle["Mach"], 1)
    assert_close(nozzle["Area"], 0.039363334927576876)
    assert_close(nozzle["Ro"], 0.11193631520025764)
    assert_close(nozzle["Ri"], 0)


def test_turboprop_cycle_and_nonlinear_sizing_match_representative_values():
    """Check TurbopropOnDesignCycle and nonlinear mass-flow iteration."""

    spec = make_turboprop_engine_spec()
    cycle = turboprop_on_design_cycle(spec, 17.46586075226333)
    nonlinear = turboprop_nonlinear_sizing(spec)
    row_spec = make_turboprop_engine_spec()
    row_spec["RPMs"] = MatlabRow(row_spec["RPMs"])
    row_nonlinear = turboprop_nonlinear_sizing(row_spec)

    assert_close(cycle["Power"], 4386284.506405142)
    assert_close(cycle["MDotAir"], 17.46586075226333)
    assert_close(cycle["BSFC"], 6.154583090847233e-08)
    assert_close(cycle["JetThrust"], 3332.6979886976487)
    assert_close(cycle["OPR"], 15.000000000000002)
    assert_close(cycle["Fuel"]["MDot"], 0.26995752454766286)
    assert_close(cycle["Fuel"]["FAR"], 0.015456296622122112)
    assert_close(nonlinear["Power"], 3000007.6329113096)
    assert_close(nonlinear["MDotAir"], 11.853887651947352)
    assert_close(nonlinear["BSFC"], 6.107224583825763e-08)
    assert_close(nonlinear["JetThrust"], 2287.2398904444913)
    assert_close(nonlinear["Fuel"]["MDot"], 0.18321720367380884)
    assert_close(row_nonlinear["Fuel"]["MDot"], nonlinear["Fuel"]["MDot"])


def test_turbofan_cycle_and_nonlinear_sizing_match_representative_values():
    """Check TurbofanOnDesignCycle and nonlinear mass-flow iteration."""

    spec = make_turbofan_engine_spec()
    cycle = turbofan_on_design_cycle(spec, 433.2063930096766)
    nonlinear = turbofan_nonlinear_sizing(spec)
    row_spec = make_turbofan_engine_spec()
    row_spec["RPMs"] = MatlabRow(row_spec["RPMs"])
    row_nonlinear = turbofan_nonlinear_sizing(row_spec)

    assert_close(cycle["Thrust"]["Net"], 137208.72538736585)
    assert_close(cycle["Fuel"]["MDot"], 1.748426802494049)
    assert_close(cycle["TSFC"], 1.2742825192478931e-05)
    assert_close(cycle["FanDiam"], 1.749384462306714)
    assert_close(cycle["OPRActual"], 30.0)
    assert_close(nonlinear["Thrust"]["Net"], 120085.6954884161)
    assert_close(nonlinear["MDotAir"], 378.6083189635311)
    assert_close(nonlinear["Fuel"]["MDot"], 1.5280682446167573)
    assert_close(nonlinear["TSFC"], 1.2724814878256339e-05)
    assert_close(nonlinear["FanDiam"], 1.6354335864082377)
    assert_close(row_nonlinear["Fuel"]["MDot"], nonlinear["Fuel"]["MDot"])


def test_turbofan_geared_fan_system_matches_representative_values():
    """Check geared fan/LPC branch in the native turbofan cycle."""

    result = turbofan_nonlinear_sizing(make_geared_turbofan_engine_spec())

    assert result["FanSysObject"]["Geared"] is True
    assert result["FanSysObject"]["Boosted"] is False
    assert result["FanSysObject"]["LPCObject"]["NoStages"] == 3
    assert_close(result["FanSysObject"]["FanObject"]["RPM"], 3461.2244897959185)
    assert_close(result["Thrust"]["Net"], 100309.98848788107)
    assert_close(result["MDotAir"], 371.85843204998577)
    assert_close(result["Fuel"]["MDot"], 1.0243561686733123)
    assert_close(result["TSFC"], 1.0211905953882845e-05)
    assert_close(result["FanDiam"], 1.6207896444407952)


def test_simple_off_design_matches_bada_polynomial():
    """Check SimpleOffDesign thrust adjustment and fuel-flow polynomial."""

    aircraft = make_simple_off_design_aircraft()
    off_params = {
        "FlightCon": {
            "Alt": 1000,
            "Mach": 0.2,
        },
        "Thrust": 10000,
    }
    result = simple_off_design(aircraft, off_params, 1000, 1, 0)
    temperature, _, _ = standard_atmosphere(1000)
    tas = 0.2 * np.sqrt(1.4 * 287 * temperature)
    thrust = 10000 - 1000 / tas
    thrust_frac = (thrust / 1000) / 20
    expected_fuel = 2 * thrust_frac

    assert abs(result["Fuel"] - expected_fuel) < 1.0e-12
    assert abs(result["Thrust"] - thrust) < 1.0e-12
    assert abs(result["TSFC"] - expected_fuel / thrust) < 1.0e-12


def test_turboprop_linear_sizing_matches_fast_example_values():
    """Check TurbopropLinearSizing representative outputs."""

    result = turboprop_linear_sizing(make_turboprop_engine_spec())

    assert_close(result["MDot0"], 17.46586075226333)
    assert_close(result["BSFC"], 7.824812225136761e-08)
    assert_close(result["BSFC_g_kW_hr"], 281.6932401049234)
    assert_close(result["m2"], 17.46586075226333)
    assert_close(result["mfuel"], 0.23474436675410282)
    assert_close(result["Tt7"], 617.358701252883)


def test_turbofan_linear_sizing_matches_representative_values():
    """Check TurbofanLinearSizing representative outputs."""

    result = turbofan_linear_sizing(make_turbofan_engine_spec())

    assert_close(result["TSFC"], 1.0864123130264473e-05)
    assert_close(result["MDot0"], 433.2063930096766)
    assert_close(result["MFuel"], 1.3036947756317367)
    assert_close(result["compwork"], 599105.6790273181)
    assert_close(result["Tt49"], 728.9837318295638)
    assert_close(result["DFan"], 1.915081230965144)
    assert_close(result["wdot"], 433.2063930096766)
    assert_close(result["u9"], 474.15139375271735)
    assert_close(result["f"], 0.014231230201119677)
    assert_close(result["TGT_Stagnation"], 728.9837318295638)


def make_simple_off_design_aircraft():
    """Return a compact aircraft structure for SimpleOffDesign."""

    return {
        "Specs": {
            "Propulsion": {
                "SLSThrust": [20000],
                "ThrustSupp": [0],
                "PropArch": {
                    "SrcType": [1],
                },
                "Engine": {
                    "Cff3": 0,
                    "Cff2": 0,
                    "Cff1": 2,
                    "Cffch": 0,
                    "HEcoeff": 1,
                },
            }
        },
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "Tav": [
                            [0, 20000],
                        ],
                    }
                }
            }
        },
    }


def make_component_flow_state():
    """Return a geometrically valid flow state for component tests."""

    temperature, pressure, _ = standard_atmosphere(0)
    cp = cp_air(temperature)
    cv = cv_air(temperature)
    gamma = cp / cv
    area = 0.5
    outer_radius = 1.0

    return {
        "MDot": 10.0,
        "Pt": pt_ps(pressure, 0.5, gamma),
        "Tt": tt_ts(temperature, 0.5, gamma),
        "Mach": 0.5,
        "Cp": cp,
        "Cv": cv,
        "Gam": gamma,
        "Area": area,
        "Ro": outer_radius,
        "Ri": np.sqrt(outer_radius ** 2 - area / np.pi),
        "Ps": pressure,
        "Ts": temperature,
    }


def make_hot_turbine_state():
    """Return a valid hot flow state for turbine component tests."""

    state = make_component_flow_state()
    state["Tt"] = 1200
    state["Mach"] = 0.4
    ts, cp, cv, gamma = new_gamma(state["Tt"], state["Mach"], state["Gam"])
    state["Ts"] = ts
    state["Cp"] = cp
    state["Cv"] = cv
    state["Gam"] = gamma
    state["Pt"] = 300000
    state["Ps"] = ps_pt(state["Pt"], state["Mach"], state["Gam"])
    state["Area"] = 0.8
    state["Ro"] = 1.2
    state["Ri"] = np.sqrt(state["Ro"] ** 2 - state["Area"] / np.pi)
    return state


def make_nozzle_ambient_state():
    """Return an ambient flow state for nozzle tests."""

    temperature, pressure, _ = standard_atmosphere(0)
    cp = cp_air(temperature)
    cv = cv_air(temperature)
    gamma = cp / cv

    return {
        "MDot": 10.0,
        "Pt": pt_ps(pressure, 0.05, gamma),
        "Tt": tt_ts(temperature, 0.05, gamma),
        "Mach": 0.05,
        "Cp": cp,
        "Cv": cv,
        "Gam": gamma,
        "Area": 1.0,
        "Ro": 1.0,
        "Ri": 0.0,
        "Ps": pressure,
        "Ts": temperature,
    }


def make_nozzle_hot_state(ambient):
    """Return a hot nozzle inlet state for nozzle tests."""

    hot = ambient.copy()
    hot["MDot"] = 12.0
    hot["Pt"] = 250000.0
    hot["Tt"] = 1000.0
    hot["Mach"] = 0.4
    ts, cp, cv, gamma = new_gamma(hot["Tt"], hot["Mach"], hot["Gam"])
    hot["Ts"] = ts
    hot["Cp"] = cp
    hot["Cv"] = cv
    hot["Gam"] = gamma
    hot["Ps"] = ps_pt(hot["Pt"], hot["Mach"], hot["Gam"])
    hot["Area"] = 0.7
    hot["Ro"] = 1.0
    hot["Ri"] = 0.0
    return hot


def make_component_eta():
    """Return representative component efficiencies."""

    return {
        "Compressors": 0.9,
        "Fan": 0.92,
        "Turbines": 0.9,
        "Diffusers": 0.99,
        "Combustor": 0.98,
    }


def make_turboprop_engine_spec():
    """Return FAST's compact ExampleTP-style engine spec."""

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


def make_turbofan_engine_spec():
    """Return a representative SLS turbofan engine spec."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 30,
        "BPR": 5,
        "FPR": 1.5,
        "Tt4Max": 1600,
        "DesignThrust": 120000,
        "NoSpools": 2,
        "RPMs": [7400, 17820],
        "FanGearRatio": np.nan,
        "FanBoosters": False,
        "MaxIter": 300,
        "CoreFlow": {
            "PaxBleed": 0.03,
            "Leakage": 0.01,
            "Cooling": 0.0,
        },
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Fan": 0.92,
            "Compressors": 0.9,
            "BypassNozzle": 0.98,
            "Combustor": 0.99,
            "Turbines": 0.9,
            "CoreNozzle": 0.98,
            "Nozzles": 0.99,
            "Mixing": 0.0,
        },
    }


def make_geared_turbofan_engine_spec():
    """Return a compact PW1919G-style geared turbofan spec."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 40,
        "BPR": 12,
        "FPR": 1.4,
        "Tt4Max": 2000,
        "DesignThrust": 100310,
        "NoSpools": 2,
        "RPMs": [10600, 24470],
        "FanGearRatio": 3.0625,
        "FanBoosters": False,
        "MaxIter": 300,
        "CoreFlow": {
            "PaxBleed": 0.03,
            "Leakage": 0.01,
            "Cooling": 0.0,
        },
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Fan": 0.99,
            "Compressors": 0.9,
            "BypassNozzle": 0.99,
            "Combustor": 0.995,
            "Turbines": 0.98,
            "CoreNozzle": 0.99,
            "Nozzles": 0.99,
            "Mixing": 0.95,
        },
    }
