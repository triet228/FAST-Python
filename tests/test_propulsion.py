# tests/test_propulsion.py

"""Tests for native PropulsionPkg ports."""

import numpy as np

from fast_python.engine import turboprop_nonlinear_sizing
from fast_python.markers import matlab_expr
from fast_python.propulsion import (
    create_prop_arch,
    eval_split,
    power_available,
    power_flow,
    prop_analysis,
    propulsion_sizing,
    recompute_splits,
    power_supplement_check,
    prop_arch_connections,
)


def assert_array_close(actual, expected, tolerance=1.0e-9):
    """Assert arrays match within tolerance."""

    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def test_eval_split_passes_declared_arguments():
    """Check split callables receive the correct number of values."""

    result = eval_split(
        lambda first, second: [[first, 1 - first], [second, 1 - second]],
        [0.25, 0.75, 0.5],
    )

    assert_array_close(result, [[0.25, 0.75], [0.75, 0.25]])


def test_eval_split_parses_constant_matlab_matrix_expression():
    """Check wrapper MATLAB matrix markers can drive split evaluation."""

    result = eval_split(matlab_expr("@()[0,1,0;0,0.5,0.5;0,0,0]"))

    assert_array_close(result, [[0, 1, 0], [0, 0.5, 0.5], [0, 0, 0]])


def test_power_flow_propagates_upstream_and_downstream():
    """Check PowerFlow matrix propagation in both directions."""

    arch = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    split_up = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    eta_up = [[1, 0.5, 1], [1, 1, 0.25], [1, 1, 1]]
    split_down = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    eta_down = [[1, 1, 1], [0.5, 1, 1], [1, 0.25, 1]]

    upstream = power_flow([100, 0, 0], arch, split_up, eta_up, 1)
    downstream = power_flow([0, 0, 100], np.asarray(arch).T, split_down, eta_down, -1)

    assert_array_close(upstream, [100, 50, 12.5])
    assert_array_close(downstream, [800, 400, 100])


def test_power_supplement_check_series_and_parallel_links():
    """Check gas turbine siphon and electric motor supplement behavior."""

    preq = [[100, 20, 80]]
    arch = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]
    split = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]
    eta = [
        [1, 0.5, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    trn_type = [1, 0, 2]

    result = power_supplement_check(preq, arch, split, eta, trn_type, 0.8)

    assert_array_close(result, [[16, 0, 0]])


def test_prop_arch_connections_finds_parallel_electric_motor():
    """Check parallel connection bookkeeping uses MATLAB transmitter offsets."""

    aircraft = {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                    ],
                    "SrcType": [1],
                    "TrnType": [1, 0, 2],
                }
            }
        }
    }

    result = prop_arch_connections(aircraft)

    assert result["Specs"]["Propulsion"]["PropArch"]["ParConns"] == [[2], [], []]


def test_create_prop_arch_conventional_two_engine_turboprop():
    """Check conventional architecture matrices against FAST test values."""

    aircraft = make_arch_aircraft("C", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]

    assert_array_close(
        prop_arch["Arch"],
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    assert_array_close(
        prop_arch["OperUps"](),
        [
            [0, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    assert_array_close(
        prop_arch["OperDwn"](),
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0],
        ],
    )
    assert prop_arch["SrcType"] == [1]
    assert prop_arch["TrnType"] == [1, 1, 2, 2]


def test_create_prop_arch_electric_two_engine_turboprop_efficiencies():
    """Check electric architecture source types and efficiencies."""

    aircraft = make_arch_aircraft("E", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]

    assert prop_arch["SrcType"] == [0]
    assert prop_arch["TrnType"] == [0, 0, 2, 2]
    assert_array_close(np.asarray(prop_arch["EtaUps"])[0, 1:3], [0.96, 0.96])
    assert_array_close(np.asarray(prop_arch["EtaUps"])[1, 3], 0.8)
    assert_array_close(np.asarray(prop_arch["EtaDwn"])[1:3, 0], [0.96, 0.96])
    assert_array_close(np.asarray(prop_arch["EtaDwn"])[3, 1], 0.8)


def test_create_prop_arch_parallel_hybrid_two_engine_turboprop():
    """Check PHE architecture matrices against FAST test values."""

    aircraft = make_arch_aircraft("PHE", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]
    lam = 0.07

    assert_array_close(
        prop_arch["Arch"],
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert_array_close(
        prop_arch["OperUps"](1),
        [
            [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert_array_close(
        prop_arch["OperDwn"](lam),
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1 - lam, 0, lam, 0, 0, 0, 0],
            [0, 0, 0, 1 - lam, 0, lam, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0],
        ],
    )
    assert prop_arch["SrcType"] == [1, 0]
    assert prop_arch["TrnType"] == [1, 1, 0, 0, 2, 2]
    assert result["Settings"]["nargOperUps"] == 1
    assert result["Settings"]["nargOperDwn"] == 1
    assert_array_close(np.asarray(prop_arch["EtaUps"])[1, 4:6], [0.96, 0.96])
    assert_array_close(np.asarray(prop_arch["EtaUps"])[2, 6], 0.8)
    assert_array_close(np.asarray(prop_arch["EtaUps"])[4, 6], 0.8)
    assert_array_close(np.asarray(prop_arch["EtaDwn"])[4:6, 1], [0.96, 0.96])
    assert_array_close(np.asarray(prop_arch["EtaDwn"])[6, [2, 4]], [0.8, 0.8])


def test_create_prop_arch_series_hybrid_two_engine_turboprop():
    """Check SHE architecture connections and split-dependent matrices."""

    aircraft = make_arch_aircraft("SHE", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]
    arch = np.asarray(prop_arch["Arch"])
    ups = np.asarray(prop_arch["OperUps"](0.25))
    dwn = np.asarray(prop_arch["OperDwn"](0.25))
    eta_ups = np.asarray(prop_arch["EtaUps"])
    eta_dwn = np.asarray(prop_arch["EtaDwn"])

    assert arch.shape == (13, 13)
    assert prop_arch["SrcType"] == [1, 0]
    assert prop_arch["TrnType"] == [1, 1, 3, 3, 4, 4, 0, 0, 2, 2]
    assert result["Settings"]["nargOperUps"] == 1
    assert result["Settings"]["nargOperDwn"] == 1
    assert_array_close(arch[0, 2:4], [1, 1])
    assert_array_close(arch[1, 6:8], [1, 1])
    assert_array_close(arch[2, 4], 1)
    assert_array_close(arch[4, 8], 1)
    assert_array_close(arch[6, 8], 1)
    assert_array_close(arch[8, 10], 1)
    assert_array_close(ups[0, 2:4], [0.5, 0.5])
    assert_array_close(ups[6, 8], 0.25)
    assert_array_close(dwn[8, [4, 6]], [0.75, 0.25])
    assert_array_close(dwn[12, 10:12], [0.5, 0.5])
    assert_array_close(eta_ups[2, 4], 0.96)
    assert_array_close(eta_ups[4, 8], 0.96)
    assert_array_close(eta_ups[6, 8], 0.96)
    assert_array_close(eta_ups[8, 10], 0.8)
    assert_array_close(eta_dwn[4, 2], 0.96)
    assert_array_close(eta_dwn[8, [4, 6]], [0.96, 0.96])
    assert_array_close(eta_dwn[10, 8], 0.8)


def test_create_prop_arch_turboelectric_two_engine_turboprop():
    """Check TE architecture connections and no-argument split matrices."""

    aircraft = make_arch_aircraft("TE", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]
    arch = np.asarray(prop_arch["Arch"])
    ups = np.asarray(prop_arch["OperUps"]())
    dwn = np.asarray(prop_arch["OperDwn"]())
    eta_ups = np.asarray(prop_arch["EtaUps"])
    eta_dwn = np.asarray(prop_arch["EtaDwn"])

    assert arch.shape == (10, 10)
    assert prop_arch["SrcType"] == [1]
    assert prop_arch["TrnType"] == [1, 1, 3, 3, 0, 0, 2, 2]
    assert result["Settings"]["nargOperUps"] == 0
    assert result["Settings"]["nargOperDwn"] == 0
    assert_array_close(arch[0, 1:3], [1, 1])
    assert_array_close(arch[1, 3], 1)
    assert_array_close(arch[3, 5], 1)
    assert_array_close(arch[5, 7], 1)
    assert_array_close(arch[7, 9], 1)
    assert_array_close(ups[0, 1:3], [0.5, 0.5])
    assert_array_close(dwn[9, 7:9], [0.5, 0.5])
    assert_array_close(eta_ups[1, 3], 0.96)
    assert_array_close(eta_ups[3, 5], 0.96)
    assert_array_close(eta_ups[5, 7], 0.8)
    assert_array_close(eta_dwn[3, 1], 0.96)
    assert_array_close(eta_dwn[5, 3], 0.96)
    assert_array_close(eta_dwn[7, 5], 0.8)


def test_create_prop_arch_partial_turboelectric_two_engine_turboprop():
    """Check PE architecture split-dependent inboard/outboard paths."""

    aircraft = make_arch_aircraft("PE", "Turboprop", 2)
    result = create_prop_arch(aircraft)
    prop_arch = result["Specs"]["Propulsion"]["PropArch"]
    arch = np.asarray(prop_arch["Arch"])
    ups = np.asarray(prop_arch["OperUps"](0.2))
    dwn = np.asarray(prop_arch["OperDwn"](0.2))
    eta_ups = np.asarray(prop_arch["EtaUps"])
    eta_dwn = np.asarray(prop_arch["EtaDwn"])

    assert arch.shape == (12, 12)
    assert prop_arch["SrcType"] == [1]
    assert prop_arch["TrnType"] == [1, 1, 3, 3, 0, 0, 2, 2, 2, 2]
    assert result["Settings"]["nargOperUps"] == 1
    assert result["Settings"]["nargOperDwn"] == 1
    assert_array_close(arch[0, 1:3], [1, 1])
    assert_array_close(arch[1, [3, 9]], [1, 1])
    assert_array_close(arch[3, 5], 1)
    assert_array_close(arch[5, 7], 1)
    assert_array_close(arch[7, 11], 1)
    assert_array_close(arch[9, 11], 1)
    assert_array_close(ups[1, [3, 9]], [0.8, 0.2])
    assert_array_close(dwn[11, 7:11], [0.4, 0.4, 0.1, 0.1])
    assert_array_close(eta_ups[1, 3], 0.96)
    assert_array_close(eta_ups[1, 9], 0.8)
    assert_array_close(eta_ups[3, 5], 0.96)
    assert_array_close(eta_ups[5, 7], 0.8)
    assert_array_close(eta_dwn[3, 1], 0.96)
    assert_array_close(eta_dwn[5, 3], 0.96)
    assert_array_close(eta_dwn[7, 5], 0.8)
    assert_array_close(eta_dwn[9, 1], 0.8)


def test_create_prop_arch_accepts_constant_matlab_expression_splits():
    """Check custom architectures accept wrapper expression matrices."""

    aircraft = {
        "Settings": {},
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Type": "O",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperUps": matlab_expr("@()[0,1,0;0,0,1;0,0,0]"),
                    "OperDwn": matlab_expr("@()[0,0,0;1,0,0;0,1,0]"),
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
                },
            }
        },
    }

    result = create_prop_arch(aircraft)

    assert result["Settings"]["nargOperUps"] == 0
    assert result["Settings"]["nargOperDwn"] == 0


def test_power_available_matches_fast_parallel_hybrid_cases():
    """Check two simple PowerAvailable cases from FAST TestPowerAvailable."""

    aircraft = make_power_available_aircraft()
    result_on = power_available(aircraft)
    tv_on = result_on["Mission"]["History"]["SI"]["Power"]["TV"]
    aircraft["Mission"]["History"]["SI"]["Power"]["LamUps"] = [[0], [0]]
    result_off = power_available(aircraft)
    tv_off = result_off["Mission"]["History"]["SI"]["Power"]["TV"]

    assert_array_close(tv_on, [100, 100])
    assert_array_close(tv_off, [82, 82])


def test_propulsion_sizing_electric_turboprop_weights():
    """Check electric PropulsionSizing power split and motor weight."""

    aircraft = make_propulsion_sizing_aircraft("E")
    result = propulsion_sizing(create_prop_arch(aircraft))
    specs = result["Specs"]

    assert_array_close(specs["Propulsion"]["SLSPower"], [625, 625, 500, 500])
    assert_array_close(specs["Propulsion"]["SLSThrust"], [6.25, 6.25, 5, 5])
    assert specs["Weight"]["Engines"] == 0
    assert specs["Weight"]["EM"] == 125
    assert specs["Weight"]["EG"] == 0
    assert specs["Weight"]["Cables"] == 0


def test_propulsion_sizing_conventional_turboprop_engine_weight():
    """Check conventional PropulsionSizing engine regression path."""

    aircraft = make_propulsion_sizing_aircraft("C")
    result = propulsion_sizing(create_prop_arch(aircraft))
    specs = result["Specs"]

    assert_array_close(specs["Propulsion"]["SLSPower"], [625, 625, 500, 500])
    assert abs(specs["Weight"]["Engines"] - 0.125) < 1.0e-9
    assert specs["Weight"]["EM"] == 0
    assert specs["Weight"]["EG"] == 0


def test_propulsion_sizing_parallel_hybrid_turboprop_power_split():
    """Check PropulsionSizing handles built-in PHE architecture splits."""

    aircraft = make_propulsion_sizing_aircraft("PHE")
    aircraft["Specs"]["Power"]["LamDwn"]["SLS"] = 0.3
    result = propulsion_sizing(create_prop_arch(aircraft))
    specs = result["Specs"]
    prop = specs["Propulsion"]

    assert_array_close(
        prop["SLSPower"],
        [437.5, 437.5, 187.5, 187.5, 500.0, 500.0],
    )
    assert_array_close(prop["PowerSupp"], [187.5, 187.5, 0, 0, 0, 0])
    assert_array_close(prop["SLSThrust"], [4.375, 4.375, 1.875, 1.875, 5.0, 5.0])
    assert_array_close(prop["ThrustSupp"], [1.875, 1.875, 0, 0, 0, 0])
    assert abs(specs["Weight"]["Engines"] - 0.08750000000002252) < 1.0e-9
    assert abs(specs["Weight"]["EM"] - 37.5) < 1.0e-9
    assert specs["Weight"]["EG"] == 0


def test_propulsion_sizing_conventional_turbofan_sizes_engine_cycle():
    """Check PropulsionSizing calls native turbofan nonlinear sizing."""

    aircraft = make_turbofan_propulsion_sizing_aircraft()
    result = propulsion_sizing(create_prop_arch(aircraft))
    specs = result["Specs"]
    prop = specs["Propulsion"]

    assert_array_close(prop["SLSPower"], [9913043.47826087, 9120000.0])
    assert_array_close(prop["SLSThrust"], [123913.04347826086, 114000.0])
    assert abs(specs["Weight"]["Engines"] - 1583.3278543150398) < 1.0e-9
    assert abs(prop["SizedEngine"]["Thrust"]["Net"] - 123968.40130659047) < 1.0e-9
    assert abs(prop["SizedEngine"]["MDotAir"] - 391.41111994860785) < 1.0e-9
    assert abs(prop["SizedEngine"]["Fuel"]["MDot"] - 1.5797405208123798) < 1.0e-9
    assert prop["Engine"]["Sizing"] == 0
    assert prop["SizedEngine"]["Specs"]["Sizing"] == 0


def test_recompute_splits_updates_parallel_downstream_split():
    """Check full-throttle split recomputation for a parallel motor."""

    aircraft = {
        "Settings": {
            "nargOperDwn": 1,
        },
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": [1],
                    "TrnType": [1, 0],
                    "ParConns": [[2], []],
                    "OperDwn": lambda lam: [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, lam, 0, 0],
                    ],
                }
            }
        },
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "Pav": [
                            [0, 80, 20, 100],
                            [0, 50, 50, 100],
                        ],
                        "LamUps": [
                            [1],
                            [1],
                        ],
                        "LamDwn": [
                            [0],
                            [0],
                        ],
                    }
                }
            }
        },
    }

    result = recompute_splits(aircraft, 1, 2)

    assert_array_close(result["Mission"]["History"]["SI"]["Power"]["LamDwn"], [[0.2], [0.5]])


def test_prop_analysis_tracks_electric_power_and_energy():
    """Check PropAnalysis core bookkeeping for an electric-only segment."""

    aircraft = {
        "Settings": {
            "nargOperDwn": 1,
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "E",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperDwn": [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                    "EtaDwn": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "SrcType": [0],
                    "TrnType": [0],
                }
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "MissID": 1,
                "SegBeg": [1],
                "SegEnd": [3],
            },
            "History": {
                "SI": {
                    "Performance": {
                        "Time": [0, 10, 20],
                        "TAS": [100, 100, 100],
                        "Mach": [0, 0, 0],
                        "Alt": [0, 0, 0],
                    },
                    "Weight": {
                        "CurWeight": [1000, 1000, 1000],
                        "Fburn": [0, 0, 0],
                    },
                    "Power": {
                        "Req": [100, 100, 100],
                        "Pav": [
                            [1000, 1000, 1000],
                            [1000, 1000, 1000],
                            [1000, 1000, 1000],
                        ],
                        "LamDwn": [[0], [0], [0]],
                    },
                    "Propulsion": {},
                    "Energy": {
                        "E_ES": [[0], [0], [0]],
                        "Eleft_ES": [[10000], [10000], [10000]],
                    },
                }
            },
        },
    }

    result = prop_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Power"]["Preq"], [[0, 100, 100]] * 3)
    assert_array_close(history["Power"]["Pout"], [[100, 100, 100]] * 3)
    assert_array_close(history["Power"]["Treq"], [[0, 1, 1]] * 3)
    assert_array_close(history["Energy"]["E_ES"], [[0], [1000], [2000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[10000], [9000], [8000]])
    assert_array_close(history["Weight"]["CurWeight"], [1000, 1000, 1000])


def test_prop_analysis_records_detailed_battery_histories():
    """Check detailed BatteryPkg.Discharging integration in PropAnalysis."""

    aircraft = make_detailed_battery_prop_analysis_aircraft()
    result = prop_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(
        history["Power"]["Voltage"],
        [[46.87763213296087], [46.87638872226717], [0.0]],
    )
    assert_array_close(
        history["Power"]["Current"],
        [[2.1332135487638553], [2.13327013291231], [0.0]],
    )
    assert_array_close(
        history["Power"]["Capacity"],
        [[0.0], [320.0], [319.994074406809]],
    )
    assert_array_close(
        history["Power"]["SOC"],
        [[100.0], [99.9981482521278], [99.99629645513743]],
    )
    assert_array_close(
        history["Power"]["C_rate"],
        [[0.006666292339887048], [0.0066664691653509685], [0.0]],
    )
    assert_array_close(history["Energy"]["E_ES"], [[0], [1000], [2000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[10000], [9000], [8000]])


def test_prop_analysis_uses_simple_turbofan_fuel_model():
    """Check PropAnalysis default turbofan fuel burn through SimpleOffDesign."""

    aircraft = make_turbofan_prop_analysis_aircraft()
    aircraft = power_available(aircraft)
    result = prop_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Propulsion"]["MDotFuel"], [[0.05, 0], [0, 0]])
    assert_array_close(history["Propulsion"]["TSFC"], [[0.00005, 0], [0, 0]])
    assert_array_close(history["Weight"]["Fburn"], [0, 0.5])
    assert_array_close(history["Weight"]["CurWeight"], [1000, 999.5])
    assert_array_close(history["Energy"]["E_ES"], [[0], [21600000]])
    assert_array_close(history["Energy"]["Eleft_ES"], [[432000000], [410400000]])


def test_prop_analysis_uses_segment_local_tav_cap_for_turbofans():
    """Check FAST's local SimpleOffDesign Tav index is preserved."""

    aircraft = make_turbofan_prop_analysis_aircraft()
    aircraft["Specs"]["Power"]["Battery"] = {
        "SerCells": np.nan,
        "ParCells": np.nan,
    }
    aircraft["Mission"]["Profile"] = {
        "SegsID": 2,
        "MissID": 1,
        "SegBeg": [1, 3],
        "SegEnd": [2, 4],
    }
    history = aircraft["Mission"]["History"]["SI"]
    history["Performance"]["Time"] = [0, 10, 20, 30]
    history["Performance"]["TAS"] = [100, 100, 100, 100]
    history["Performance"]["Mach"] = [0.3, 0.3, 0.3, 0.3]
    history["Performance"]["Alt"] = [0, 0, 0, 0]
    history["Performance"]["Rho"] = [1.225, 1.225, 1.225, 1.225]
    history["Weight"]["CurWeight"] = [1000, 1000, 1000, 1000]
    history["Weight"]["Fburn"] = [0, 0, 0, 0]
    history["Power"]["Req"] = [0, 0, 500000, 500000]
    history["Power"]["LamUps"] = [[0], [0], [0], [0]]
    history["Power"]["LamDwn"] = [[0], [0], [0], [0]]
    history["Power"]["Pav"] = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1000000, 1000000, 1000000],
        [0, 1000000, 1000000, 1000000],
    ]
    history["Power"]["Tav"] = [
        [0, 3000, 0, 0],
        [0, 3000, 0, 0],
        [0, 10000, 0, 0],
        [0, 10000, 0, 0],
    ]
    history["Energy"]["E_ES"] = [[0], [0], [0], [0]]
    history["Energy"]["Eleft_ES"] = [
        [432000000],
        [432000000],
        [432000000],
        [432000000],
    ]

    result = prop_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert_array_close(history["Propulsion"]["MDotFuel"], [[0, 0], [0, 0], [0.15, 0], [0, 0]])
    assert_array_close(history["Weight"]["Fburn"], [0, 0, 0, 1.5])


def test_prop_analysis_integrates_fuel_over_multiple_intervals():
    """Check fuel burn integration works for segments with more than two points."""

    aircraft = make_turbofan_prop_analysis_aircraft()
    aircraft["Mission"]["Profile"]["SegEnd"] = [4]
    history = aircraft["Mission"]["History"]["SI"]
    history["Performance"]["Time"] = [0, 10, 20, 30]
    history["Performance"]["TAS"] = [100, 100, 100, 100]
    history["Performance"]["Mach"] = [0.3, 0.3, 0.3, 0.3]
    history["Performance"]["Alt"] = [0, 0, 0, 0]
    history["Performance"]["Rho"] = [1.225, 1.225, 1.225, 1.225]
    history["Weight"]["CurWeight"] = [1000, 1000, 1000, 1000]
    history["Weight"]["Fburn"] = [0, 0, 0, 0]
    history["Power"]["Req"] = [10000, 10000, 10000, 10000]
    history["Power"]["LamUps"] = [[0], [0], [0], [0]]
    history["Power"]["LamDwn"] = [[0], [0], [0], [0]]
    history["Energy"]["E_ES"] = [[0], [0], [0], [0]]
    history["Energy"]["Eleft_ES"] = [
        [432000000],
        [432000000],
        [432000000],
        [432000000],
    ]
    aircraft = power_available(aircraft)
    result = prop_analysis(aircraft)

    assert_array_close(result["Mission"]["History"]["SI"]["Weight"]["Fburn"], [0, 0.5, 1.0, 1.5])


def test_prop_analysis_uses_turboprop_nonlinear_fuel_model():
    """Check PropAnalysis estimates fuel for conventional turboprops."""

    aircraft = make_turboprop_prop_analysis_aircraft()
    aircraft = power_available(aircraft)
    result = prop_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]
    engine_spec = make_turboprop_engine_spec()
    engine_spec["ReqPower"] = 10000
    expected_flow = turboprop_nonlinear_sizing(engine_spec)["Fuel"]["MDot"]

    assert_array_close(history["Propulsion"]["MDotFuel"], [[expected_flow, 0], [0, 0]])
    assert_array_close(history["Weight"]["Fburn"], [0, expected_flow * 10])


def make_power_available_aircraft():
    """Return the parallel hybrid test aircraft from FAST TestPowerAvailable."""

    return {
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "SegBeg": [1],
                "SegEnd": [2],
            },
            "History": {
                "SI": {
                    "Performance": {
                        "TAS": [50, 100],
                        "Rho": [1.225, 1.225],
                    },
                    "Power": {
                        "TV": [0, 0],
                        "LamUps": [[1], [1]],
                    },
                }
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Propulsion": {
                "SLSPower": [82, 18, 100],
                "SLSThrust": [0, 0, 0],
                "PropArch": {
                    "Arch": [
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    "OperUps": lambda lam: [
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, lam, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    "EtaUps": np.ones((6, 6)).tolist(),
                    "SrcType": [1, 0],
                    "TrnType": [1, 0, 2],
                },
            },
        },
    }


def make_propulsion_sizing_aircraft(arch_type):
    """Return a small aircraft for PropulsionSizing tests."""

    aircraft = make_arch_aircraft(arch_type, "Turboprop", 2)
    aircraft["Specs"]["Performance"] = {
        "Vels": {
            "Tko": 100,
        }
    }
    aircraft["Specs"]["Power"]["SLS"] = 1000
    aircraft["Specs"]["Power"]["P_W"] = {
        "SLS": 1,
        "EM": 10,
        "EG": 20,
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "SLS": 0,
    }
    aircraft["Specs"]["Weight"] = {}
    aircraft["HistData"] = {
        "Eng": {
            "E1": {
                "Power_SLS": 100,
                "DryWeight": 10,
            },
            "E2": {
                "Power_SLS": 200,
                "DryWeight": 20,
            },
            "E3": {
                "Power_SLS": 300,
                "DryWeight": 30,
            },
        }
    }
    return aircraft


def make_turbofan_propulsion_sizing_aircraft():
    """Return a small turbofan aircraft for PropulsionSizing tests."""

    aircraft = make_arch_aircraft("C", "Turbofan", 1)
    aircraft["Specs"]["Performance"] = {
        "Vels": {
            "Tko": 80,
        }
    }
    aircraft["Specs"]["Power"]["P_W"] = {
        "EM": 10,
        "EG": 20,
        "Cables": 1,
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "SLS": 0,
    }
    aircraft["Specs"]["Weight"] = {}
    aircraft["Specs"]["Propulsion"]["Thrust"] = {
        "SLS": 120000,
    }
    aircraft["Specs"]["Propulsion"]["Engine"] = {
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
    aircraft["HistData"] = {
        "Eng": {
            "E1": {
                "Thrust_Max": 80000,
                "DryWeight": 1000,
            },
            "E2": {
                "Thrust_Max": 120000,
                "DryWeight": 1500,
            },
            "E3": {
                "Thrust_Max": 160000,
                "DryWeight": 2200,
            },
        }
    }
    return aircraft


def make_detailed_battery_prop_analysis_aircraft():
    """Return a compact electric aircraft with detailed battery cells."""

    aircraft = {
        "Settings": {
            "nargOperDwn": 1,
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Battery": {
                "MaxExtVolCell": 4.088,
                "IntResist": 0.01,
                "ExpVol": 0.6,
                "ExpCap": 3.0,
                "CapCell": 3.2,
                "Degradation": 0,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000,
                },
                "Battery": {
                    "SerCells": 10,
                    "ParCells": 100,
                },
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "E",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperDwn": [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                    "EtaDwn": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "SrcType": [0],
                    "TrnType": [0],
                }
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "MissID": 1,
                "SegBeg": [1],
                "SegEnd": [3],
            },
            "History": {
                "SI": {
                    "Performance": {
                        "Time": [0, 10, 20],
                        "TAS": [100, 100, 100],
                        "Mach": [0, 0, 0],
                        "Alt": [0, 0, 0],
                    },
                    "Weight": {
                        "CurWeight": [1000, 1000, 1000],
                        "Fburn": [0, 0, 0],
                    },
                    "Power": {
                        "Req": [100, 100, 100],
                        "Pav": [
                            [1000, 1000, 1000],
                            [1000, 1000, 1000],
                            [1000, 1000, 1000],
                        ],
                        "LamDwn": [[0], [0], [0]],
                    },
                    "Propulsion": {},
                    "Energy": {
                        "E_ES": [[0], [0], [0]],
                        "Eleft_ES": [[10000], [10000], [10000]],
                    },
                }
            },
        },
    }
    return aircraft


def make_turbofan_prop_analysis_aircraft():
    """Return a compact conventional turbofan for PropAnalysis fuel tests."""

    aircraft = create_prop_arch(
        {
            "Settings": {
                "Analysis": {
                    "Type": 1,
                },
            },
            "Specs": {
                "TLAR": {
                    "Class": "Turbofan",
                },
                "Power": {
                    "SpecEnergy": {
                        "Fuel": 43200000,
                        "Batt": 1000,
                    },
                    "Eta": {
                        "EM": 1,
                    },
                },
                "Propulsion": {
                    "NumEngines": 1,
                    "SLSPower": [2000000, 0],
                    "SLSThrust": [20000, 0],
                    "ThrustSupp": [0, 0],
                    "Thrust": {
                        "SLS": 20000,
                    },
                    "MDotCF": 1,
                    "Engine": {
                        "EtaPoly": {
                            "Fan": 1,
                        },
                        "Cff3": 0,
                        "Cff2": 0,
                        "Cff1": 1,
                        "Cffch": 0,
                        "HEcoeff": 1,
                    },
                    "PropArch": {
                        "Type": "C",
                    },
                },
            },
        }
    )
    aircraft["Mission"] = {
        "Profile": {
            "SegsID": 1,
            "MissID": 1,
            "SegBeg": [1],
            "SegEnd": [2],
        },
        "History": {
            "SI": {
                "Performance": {
                    "Time": [0, 10],
                    "TAS": [100, 100],
                    "Mach": [0.3, 0.3],
                    "Alt": [0, 0],
                    "Rho": [1.225, 1.225],
                },
                "Weight": {
                    "CurWeight": [1000, 1000],
                    "Fburn": [0, 0],
                },
                "Power": {
                    "Req": [10000, 10000],
                    "LamUps": [[0], [0]],
                    "LamDwn": [[0], [0]],
                },
                "Propulsion": {},
                "Energy": {
                    "E_ES": [[0], [0]],
                    "Eleft_ES": [[432000000], [432000000]],
                },
            }
        },
    }
    return aircraft


def make_turboprop_prop_analysis_aircraft():
    """Return a compact conventional turboprop for PropAnalysis fuel tests."""

    aircraft = create_prop_arch(
        {
            "Settings": {
                "Analysis": {
                    "Type": 1,
                },
            },
            "Specs": {
                "TLAR": {
                    "Class": "Turboprop",
                },
                "Power": {
                    "SLS": 200000,
                    "SpecEnergy": {
                        "Fuel": 43200000,
                        "Batt": 1000,
                    },
                    "Eta": {
                        "EM": 1,
                        "Propeller": 1,
                    },
                },
                "Propulsion": {
                    "NumEngines": 1,
                    "SLSPower": [200000, 0],
                    "SLSThrust": [0, 0],
                    "MDotCF": 1,
                    "Engine": make_turboprop_engine_spec(),
                    "PropArch": {
                        "Type": "C",
                    },
                },
            },
        }
    )
    aircraft["Mission"] = {
        "Profile": {
            "SegsID": 1,
            "MissID": 1,
            "SegBeg": [1],
            "SegEnd": [2],
        },
        "History": {
            "SI": {
                "Performance": {
                    "Time": [0, 10],
                    "TAS": [100, 100],
                    "Mach": [0.3, 0.3],
                    "Alt": [0, 0],
                    "Rho": [1.225, 1.225],
                },
                "Weight": {
                    "CurWeight": [1000, 1000],
                    "Fburn": [0, 0],
                },
                "Power": {
                    "Req": [10000, 10000],
                    "LamUps": [[0], [0]],
                    "LamDwn": [[0], [0]],
                },
                "Propulsion": {},
                "Energy": {
                    "E_ES": [[0], [0]],
                    "Eleft_ES": [[432000000], [432000000]],
                },
            }
        },
    }
    return aircraft


def make_turboprop_engine_spec():
    """Return a compact turboprop engine spec."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 15,
        "Tt4Max": 1200,
        "ReqPower": 10000,
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


def make_arch_aircraft(arch_type, aircraft_class, num_engines):
    """Return a small aircraft dictionary for CreatePropArch tests."""

    return {
        "Settings": {},
        "Specs": {
            "TLAR": {
                "Class": aircraft_class,
            },
            "Power": {
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                    "Propeller": 0.8,
                }
            },
            "Propulsion": {
                "NumEngines": num_engines,
                "Engine": {
                    "EtaPoly": {
                        "Fan": 0.99,
                    }
                },
                "PropArch": {
                    "Type": arch_type,
                },
            },
        },
    }
