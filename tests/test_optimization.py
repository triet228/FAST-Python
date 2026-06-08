# tests/test_optimization.py

"""Tests for native OptimizationPkg helper ports."""

import numpy as np

from fast_python.optimization import (
    ConSizeOpt,
    DesOptimize,
    ObjPowerManagement,
    OpsOptimize,
    OptimizationError,
    check_flag,
    con_size_opt,
    des_optimize,
    feas_step,
    gauss_elim,
    get_splits,
    golden_section,
    hess_upd,
    interior_point,
    merit_function,
    obj_power_management,
    ops_optimize,
    simplex_post,
    simplex_setup,
    simplex_solve,
)


def test_check_flag_matches_fast_behavior():
    """Check CheckFlag returns one only for existing enabled flags."""

    flags = {"A": 1, "B": 0}

    assert check_flag(flags, "A") == 1
    assert check_flag(flags, "B") == 0
    assert check_flag(flags, "C") == 0


def test_feas_step_matches_matlab_value():
    """Check FeasStep against a MATLAB oracle value."""

    assert abs(feas_step(3, [1, 2, 3], [-1, 4, -2]) - 0.995) < 1.0e-12


def test_gauss_elim_performs_documented_pivot_step():
    """Check GaussElim's documented elimination behavior."""

    result = gauss_elim(
        [
            [2, 1],
            [4, 5],
        ],
        1,
        1,
    )

    np.testing.assert_allclose(
        result,
        [
            [1, 0.5],
            [0, 3],
        ],
        atol=1.0e-12,
        rtol=0,
    )


def test_hess_upd_matches_matlab_value():
    """Check HessUpd against a MATLAB oracle value."""

    result = hess_upd(
        np.eye(2),
        [1, 2],
        [3, 4],
    )

    np.testing.assert_allclose(
        result,
        [
            [1.618181818182, 0.690909090909],
            [0.690909090909, 1.654545454545],
        ],
        atol=1.0e-12,
        rtol=0,
    )


def test_golden_section_matches_fast_phase_one_search():
    """Check GoldenSection returns FAST's phase-I bounded midpoint."""

    def objective(x):
        return (x[0] - 2) ** 2

    amin, fval = golden_section(objective, [0], [1], 10)

    assert abs(amin - 1.844299065734) < 1.0e-12
    assert fval is None


def test_merit_function_unconstrained_matches_objective():
    """Check MeritFunction without constraints returns the objective."""

    def objective(x, _):
        return x[0] ** 2, None, None

    assert merit_function(objective, [3]) == 9


def test_simplex_solve_finds_bounded_linear_program_solution():
    """Check SimplexSolve using the documented simplex algorithm."""

    tableau = [
        [1, 1, 1, 0, 0, 4],
        [1, 0, 0, 1, 0, 2],
        [0, 1, 0, 0, 1, 3],
        [-3, -2, 0, 0, 0, 0],
    ]
    xopt, iterations = simplex_solve(tableau)

    np.testing.assert_allclose(
        xopt,
        [2, 2, 0, 0, 1],
        atol=1.0e-12,
        rtol=0,
    )
    assert iterations == 2


def test_simplex_setup_phe_fuel_burn_matches_matlab_tableau():
    """Check SimplexSetup PHE/FuelBurn against a MATLAB oracle tableau."""

    result = simplex_setup(make_simplex_aircraft(), [1, 2, 3])

    expected = [
        [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.45],
        [-1388.888888888889, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1388.888888888889, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 20000],
        [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.45],
        [0, -1666.666666666667, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1666.666666666667, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 20000],
        [-13888.888888888889, -25000, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [13888.888888888889, 25000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 24000000],
        [-6000, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    np.testing.assert_allclose(result, expected, atol=1.0e-9, rtol=0)


def test_simplex_setup_she_energy_matches_matlab_objective_row():
    """Check SimplexSetup SHE/Energy objective coefficients."""

    aircraft = make_simplex_aircraft()
    aircraft["Specs"]["Propulsion"]["Arch"]["Type"] = "SHE"
    aircraft["PowerOpt"]["ObjFun"] = "Energy"
    result = simplex_setup(aircraft, [1, 2, 3])

    np.testing.assert_allclose(
        result[-1],
        [
            -257999986111.111114501953,
            -3869975000,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        atol=1.0e-3,
        rtol=0,
    )


def test_simplex_solve_rejects_unbounded_tableau():
    """Check SimplexSolve detects an unbounded pivot column."""

    with np.testing.assert_raises(OptimizationError):
        simplex_solve(
            [
                [-1, 1, -1],
                [-1, 0, 0],
            ]
        )


def test_simplex_post_writes_one_based_power_split_indices():
    """Check SimplexPost writes optimized splits to mission history."""

    aircraft = {
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "Phi": [0, 0, 0, 0],
                    },
                },
            },
        },
    }
    result = simplex_post(aircraft, [2, 3, 4], [0.2, 0.4, 99])

    assert result["Mission"]["History"]["SI"]["Power"]["Phi"] == [
        0,
        0.2,
        0.4,
        0,
    ]


def test_get_splits_matches_matlab_values():
    """Check GetSplits against a MATLAB oracle fixture."""

    aircraft = {
        "PowerOpt": {
            "SegIndex": [1, 2, 3],
            "LamIndex": [1, 2, 3],
            "npoint": 3,
            "Splits": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "Settings": {
                "OperTS": 1,
                "OperTSPS": 1,
                "OperPSPS": 0,
                "OperPSES": 1,
            },
        },
        "Settings": {
            "nargTS": 1,
            "nargTSPS": 1,
            "nargPSPS": 1,
            "nargPSES": 1,
        },
    }
    lam_ts, lam_tsps, lam_psps, lam_pses = get_splits(
        aircraft,
        1,
        4,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
    )

    np.testing.assert_allclose(lam_ts.reshape(-1), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(lam_tsps.reshape(-1), [0.4, 0.5, 0.6])
    np.testing.assert_allclose(lam_psps.reshape(-1), [0, 0, 0])
    np.testing.assert_allclose(lam_pses.reshape(-1), [0.7, 0.8, 0.9])

    lam_ts, _, _, _ = get_splits(
        aircraft,
        4,
        6,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
    )

    np.testing.assert_allclose(lam_ts.reshape(-1), [0, 0, 0])


def test_con_size_opt_assembles_constraint_vector():
    """Check ConSizeOpt's power, design split, and operation split constraints."""

    g, h, dgdx, dhdx = con_size_opt(
        [0.2, 0.3, 0.4, 0.5, 0.6],
        0,
        make_con_size_aircraft(),
    )

    expected = [
        -0.1,
        -0.2,
        -0.9,
        -0.8,
        -0.3,
        -0.25,
        -0.7,
        -0.75,
        -0.25,
        -0.75,
        -0.6,
        -0.4,
        -0.2,
        -0.3,
        -0.4,
        -0.5,
        -2 / 3,
        -0.5,
        -0.5,
        -0.375,
        -0.1,
    ]

    np.testing.assert_allclose(g, expected, atol=1.0e-12, rtol=0)
    assert len(h) == 0
    assert len(dgdx) == 0
    assert len(dhdx) == 0


def test_con_size_opt_returns_split_and_sensitivity_gradients():
    """Check ConSizeOpt gradient rows for finite differences and split bounds."""

    aircraft = make_con_size_aircraft()
    g, h, dgdx, dhdx = ConSizeOpt([0.2, 0.3, 0.4, 0.5, 0.6], 1, aircraft)

    assert len(g) == 21
    assert len(h) == 0
    assert dhdx.shape == (0, 5)
    assert dgdx.shape == (21, 5)
    np.testing.assert_allclose(dgdx[0:2, 0], [-0.05, -0.1], atol=1.0e-10, rtol=0)
    np.testing.assert_allclose(dgdx[2:4, 0], [0.05, 0.1], atol=1.0e-10, rtol=0)
    np.testing.assert_allclose(dgdx[10, :], [0, 0, 0, 0, -1], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[11, :], [0, 0, 0, 0, 1], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[12:16, :4], -np.eye(4), atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[16, [0, 4]], [1 / 0.6, -0.2 / 0.6 ** 2], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[17, [1, 4]], [1 / 0.6, -0.3 / 0.6 ** 2], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[18, 2], 1 / 0.8, atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[19, 3], 1 / 0.8, atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(dgdx[20, :], [0, 0, 0, 0, 0], atol=1.0e-12, rtol=0)


def test_obj_power_management_updates_constraints_without_gradient():
    """Check ObjPowerManagement objective scaling and constraint bookkeeping."""

    aircraft = make_power_management_aircraft()
    fvalue, grad, result = obj_power_management(
        [0.2, 0.3],
        0,
        aircraft,
        fake_power_management_runner,
    )
    constraints = result["PowerOpt"]["Constraints"]

    assert fvalue == 1.5
    assert len(grad) == 0
    np.testing.assert_allclose(result["PowerOpt"]["Splits"], [0.2, 0.3], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(constraints["DesPem"], [1.5, 3.5], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(constraints["DesPgtAv"], [10.5, 30.5], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(constraints["DesPgt"], [5.5, 7.5], atol=1.0e-12, rtol=0)
    assert constraints["DesPemAv"] == 20
    assert constraints["DesEbattAv"] == 500
    assert constraints["DesEbatt"] == 1050
    assert constraints["DesPavGT"] == 20.5
    assert constraints["DesCrsPow"] == 6.5
    assert result["PowerOpt"]["Results"]["FlownAC"]["PowerOpt"]["SegIndex"] == [1, 3]


def test_obj_power_management_returns_gradient_and_sensitivities():
    """Check ObjPowerManagement finite-difference objective and sensitivities."""

    aircraft = make_power_management_aircraft()
    fvalue, grad, result = ObjPowerManagement(
        [0.2, 0.3],
        1,
        aircraft,
        fake_power_management_runner,
    )
    constraints = result["PowerOpt"]["Constraints"]

    assert fvalue == 1.5
    np.testing.assert_allclose(grad, [1, 1], atol=1.0e-9, rtol=0)
    assert constraints["EPS"] == 1.0e-06
    np.testing.assert_allclose(
        constraints["SenPem"],
        [
            [1.500001, 1.500001],
            [3.500001, 3.500001],
        ],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(constraints["SenPavGT"], [20.500001, 20.500001], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(constraints["SenCrsPow"], [6.500001, 6.500001], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(constraints["SenEbatt"], [1050.0001, 1050.0001], atol=1.0e-12, rtol=0)
    np.testing.assert_allclose(result["PowerOpt"]["Splits"], [0.2, 0.3], atol=1.0e-12, rtol=0)


def test_des_optimize_prepares_indices_and_stores_results():
    """Check DesOptimize setup and result bookkeeping with a fake optimizer."""

    aircraft = make_design_optimization_aircraft()
    result = des_optimize(
        aircraft,
        optimizer=lambda obj_fun, x0, con_fun: fake_design_optimizer(aircraft, obj_fun, x0, con_fun),
    )
    power_opt = result["PowerOpt"]

    assert power_opt["npoint"] == 2
    assert power_opt["nopers"] == 4
    assert power_opt["ndesns"] == 1
    assert power_opt["ndvars"] == 5
    assert power_opt["SegIndex"] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert power_opt["LamIndex"] == [1, 1, 2, 2, 1, 1, 2, 2]
    assert power_opt["Results"]["ObjFunVal"] == 17207 * 1.5
    np.testing.assert_allclose(power_opt["Results"]["OptParams"], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1.0e-12, rtol=0)
    assert power_opt["Results"]["Optimality"] == 0.01
    assert power_opt["Results"]["Feasiblity"] == 0.02
    assert power_opt["Results"]["ParamHist"] == [[0, 1], [2, 3]]
    assert power_opt["Results"]["ObjFnHist"] == [2, 1.5]


def test_des_optimize_alias_and_validation_error():
    """Check DesOptimize alias and obvious invalid split requests."""

    aircraft = make_design_optimization_aircraft()
    aircraft["Settings"]["nargPSES"] = 0

    with np.testing.assert_raises(OptimizationError):
        DesOptimize(
            aircraft,
            optimizer=lambda obj_fun, x0, con_fun: fake_design_optimizer(aircraft, obj_fun, x0, con_fun),
        )


def test_ops_optimize_iterates_simplex_until_power_split_convergence():
    """Check OpsOptimize loop, one-based indices, and convergence cleanup."""

    aircraft = make_ops_optimization_aircraft()
    solver = SequentialPhiSolver([[0.5, 0.5, 99], [0.55, 0.55, 99]])
    result = ops_optimize(
        aircraft,
        mission_runner=fake_ops_mission_runner,
        tableau_builder=fake_ops_tableau_builder,
        simplex_solver=solver,
    )

    assert result["Specs"]["Power"]["Phi"] == {
        "Tko": 0,
        "Clb": 0,
        "Crs": 0,
        "Des": 0,
        "Lnd": 0,
    }
    assert solver.calls == 2
    assert result["PowerOpt"]["PhiCount"] == 1
    assert result["PowerOpt"]["Iter"] == 1
    assert "PhiHist" not in result["PowerOpt"]
    assert result["PowerOpt"]["LastObjFunVal"] == 42
    assert result["Mission"]["History"]["SI"]["Power"]["Phi"] == [0.55, 0.55, 0]


def test_ops_optimize_alias_rejects_missing_segments():
    """Check OpsOptimize reports missing target segments clearly."""

    aircraft = make_ops_optimization_aircraft()
    aircraft["PowerOpt"]["Segments"] = ["Descent"]

    with np.testing.assert_raises(OptimizationError):
        OpsOptimize(
            aircraft,
            mission_runner=fake_ops_mission_runner,
            tableau_builder=fake_ops_tableau_builder,
            simplex_solver=SequentialPhiSolver([[0.5, 0.5, 99]]),
        )


def test_interior_point_matches_matlab_test_problem():
    """Check InteriorPoint against FAST's original test problem."""

    xopt, fopt, xhist, _, optim, feas, _ = interior_point(
        sample_objective,
        [-15, 17],
        sample_constraints,
    )

    np.testing.assert_allclose(
        xopt,
        [0.009959395356, 0.005022809433],
        atol=1.0e-9,
        rtol=0,
    )
    assert abs(fopt - 0.020005014222) < 1.0e-9
    assert abs(optim - 0.000913326512) < 1.0e-9
    assert feas == 0
    assert xhist.shape[1] == 12


def test_interior_point_matches_matlab_new_test_problem():
    """Check InteriorPoint against FAST's newer test problem."""

    xopt, fopt, xhist, _, optim, feas, _ = interior_point(
        sample_new_objective,
        [-10, 15],
        sample_new_constraints,
    )

    np.testing.assert_allclose(
        xopt,
        [0.250418110707, 3.331348279243],
        atol=1.0e-9,
        rtol=0,
    )
    assert abs(fopt + 22.130386327889) < 1.0e-9
    assert abs(optim - 0.000659739344) < 1.0e-9
    assert feas == 0
    assert xhist.shape[1] == 14


def sample_objective(x, need_grad=1):
    """Return OptimizationPkg.TestObj values."""

    fvalue = x[0] + 2 * x[1]

    if need_grad == 1:
        return fvalue, np.asarray([1, 2]), None

    return fvalue, np.asarray([]), None


def sample_constraints(x, need_grad=1):
    """Return OptimizationPkg.TestCon values."""

    g = np.asarray([
        0.25 * x[0] ** 2 + x[1] ** 2 - 1,
        -x[1],
        -x[0],
    ])
    h = np.asarray([])

    if need_grad == 1:
        dgdx = np.asarray(
            [
                [0.5 * x[0], 2 * x[1]],
                [0, -1],
                [-1, 0],
            ]
        )
        return g, h, dgdx, np.zeros((0, 2))

    return g, h, np.asarray([]), np.asarray([])


def sample_new_objective(x, need_grad=1):
    """Return OptimizationPkg.NewTestObj values."""

    l1 = 12
    l2 = 8
    k1 = 1
    k2 = 10
    mg = 7
    left = np.sqrt((l1 + x[0]) ** 2 + x[1] ** 2)
    right = np.sqrt((l2 - x[0]) ** 2 + x[1] ** 2)
    fvalue = 0.5 * k1 * (left - l1) ** 2
    fvalue += 0.5 * k2 * (right - l2) ** 2
    fvalue -= mg * x[1]

    if need_grad == 1:
        dfdx1 = k2 * (l2 - right) * (l2 - x[0]) / right
        dfdx1 -= k1 * (l1 - left) * (l1 + x[0]) / left
        dfdx2 = -k2 * (l2 - right) * x[1] / right
        dfdx2 -= k1 * (l1 - left) * x[1] / left
        dfdx2 -= mg
        return fvalue, np.asarray([dfdx1, dfdx2]), None

    return fvalue, np.asarray([]), None


def sample_new_constraints(x, need_grad=1):
    """Return OptimizationPkg.NewTestCon values."""

    lc1 = 9
    lc2 = 6
    yc = 2
    xc1 = 7
    xc2 = 3
    g1_root = np.sqrt((x[0] + xc1) ** 2 + (x[1] + yc) ** 2)
    g2_root = np.sqrt((x[0] - xc2) ** 2 + (x[1] + yc) ** 2)
    g = np.asarray([g1_root - lc1, g2_root - lc2])
    h = np.asarray([])

    if need_grad == 1:
        dgdx = np.asarray(
            [
                [(x[0] + xc1) / g1_root, (x[1] + yc) / g1_root],
                [(x[0] - xc2) / g2_root, (x[1] + yc) / g2_root],
            ]
        )
        return g, h, dgdx, np.zeros((0, 2))

    return g, h, np.asarray([]), np.asarray([])


def make_simplex_aircraft():
    """Return a small aircraft fixture for SimplexSetup tests."""

    return {
        "Specs": {
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43e6,
                    "Batt": 1.2e6,
                },
                "Eta": {
                    "EM": 0.9,
                },
                "Phi": {
                    "SLS": 0.5,
                },
                "P_W": {
                    "EM": 2000,
                },
            },
            "Propulsion": {
                "Eta": {
                    "Prop": 0.8,
                },
                "Arch": {
                    "Type": "PHE",
                },
            },
            "Weight": {
                "EM": 10,
                "Batt": 20,
            },
            "TLAR": {
                "Class": "Turbofan",
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "Out": [1000, 1200, 900],
                        "Av": [2000, 2200, 2100],
                    },
                    "Propulsion": {
                        "TSFC": [0.6, 0.5, 0.4],
                    },
                    "Performance": {
                        "Alt": [0, 1000, 2000],
                        "Time": [0, 10, 25],
                        "TAS": [0, 100, 120],
                    },
                },
            },
        },
        "PowerOpt": {
            "ObjFun": "FuelBurn",
            "Segments": ["Takeoff", "Climb"],
        },
    }


def make_con_size_aircraft():
    """Return a compact PowerOpt fixture for ConSizeOpt tests."""

    eps = 1.0e-06
    sen_pem = np.tile(np.asarray([[10.0], [20.0]]), (1, 5))
    sen_pem[:, 0] += eps * np.asarray([5.0, 10.0])

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "PSType": [0, 1],
                    "ESType": [0],
                },
            },
            "Power": {
                "LamPSES": {
                    "SLS": 0.8,
                },
            },
        },
        "Settings": {
            "Analysis": {
                "Type": 1,
            },
            "nargTS": 1,
            "nargTSPS": 0,
            "nargPSPS": 0,
            "nargPSES": 1,
        },
        "PowerOpt": {
            "Settings": {
                "DesnTS": 1,
                "OperTS": 1,
                "DesnTSPS": 0,
                "OperTSPS": 0,
                "DesnPSPS": 0,
                "OperPSPS": 0,
                "DesnPSES": 0,
                "OperPSES": 1,
            },
            "nopers": 4,
            "ndesns": 1,
            "ndvars": 5,
            "npoint": 2,
            "Constraints": {
                "DesPem": [10, 20],
                "DesPemAv": 100,
                "DesPgt": [15, 25],
                "DesPgtAv": [50, 100],
                "DesEbatt": 30,
                "DesEbattAv": 120,
                "DesPavGT": 50,
                "DesCrsPow": 45,
                "SenPem": sen_pem,
                "SenPemAv": [100, 100, 100, 100, 100],
                "SenPgt": np.tile(np.asarray([[15.0], [25.0]]), (1, 5)),
                "SenPgtAv": np.tile(np.asarray([[50.0], [100.0]]), (1, 5)),
                "SenEbatt": [30, 30, 30, 30, 30],
                "SenEbattAv": [120, 120, 120, 120, 120],
                "SenPavGT": [50, 50, 50, 50, 50],
                "SenCrsPow": [45, 45, 45, 45, 45],
            },
        },
    }


def make_power_management_aircraft():
    """Return a compact fixture for ObjPowerManagement tests."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 1,
                "MaxIter": 3,
            },
        },
        "PowerOpt": {
            "ObjFun": "FuelBurn",
            "ndvars": 2,
            "Constraints": {},
        },
    }


def fake_power_management_runner(aircraft, analysis_type, max_iter):
    """Return a flown aircraft whose histories depend on the split sum."""

    assert analysis_type == 1
    assert max_iter == 3
    split_sum = float(np.sum(aircraft["PowerOpt"]["Splits"]))
    return {
        "Specs": {
            "Power": {
                "P_W": {
                    "EM": 10,
                },
                "SpecEnergy": {
                    "Batt": 5,
                },
            },
            "Weight": {
                "EM": 2,
                "Batt": 100,
            },
            "Propulsion": {
                "PropArch": {
                    "PSType": [0, 1],
                },
            },
        },
        "PowerOpt": {
            "SegIndex": [1, 3],
        },
        "Mission": {
            "History": {
                "Segment": ["Climb", "Cruise", "Cruise"],
                "SI": {
                    "Weight": {
                        "Fburn": [0, 100, 17207 * (1 + split_sum)],
                    },
                    "Energy": {
                        "Fuel": [0, 10, 1000 + 10 * split_sum],
                        "Batt": [0, 20, 1000 + 100 * split_sum],
                    },
                    "Power": {
                        "EM": [1 + split_sum, 2 + split_sum, 3 + split_sum],
                        "AvGT": [10 + split_sum, 20 + split_sum, 30 + split_sum],
                        "GT": [5 + split_sum, 6 + split_sum, 7 + split_sum],
                    },
                },
            },
        },
    }


def make_design_optimization_aircraft():
    """Return a compact fixture for DesOptimize setup tests."""

    return {
        "Settings": {
            "nargTS": 1,
            "nargTSPS": 0,
            "nargPSPS": 0,
            "nargPSES": 1,
        },
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "TSPS": [
                        [1, 0],
                        [0, 1],
                    ],
                    "PSPS": [
                        [1],
                        [1],
                    ],
                    "PSES": [
                        [1, 1],
                    ],
                },
            },
        },
        "Mission": {
            "Profile": {
                "PowerOpt": [2],
                "SegBeg": [1],
                "SegEnd": [5],
                "SegPts": [5],
            },
        },
        "PowerOpt": {
            "ObjFun": "FuelBurn",
            "Settings": {
                "DesnTS": 1,
                "OperTS": 1,
                "DesnTSPS": 0,
                "OperTSPS": 0,
                "DesnPSPS": 0,
                "OperPSPS": 0,
                "DesnPSES": 0,
                "OperPSES": 1,
            },
        },
    }


def fake_design_optimizer(aircraft, obj_fun, x0, con_fun):
    """Return a deterministic DesOptimize result after checking setup."""

    assert callable(obj_fun)
    assert callable(con_fun)
    assert aircraft["PowerOpt"]["SegIndex"] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert aircraft["PowerOpt"]["LamIndex"] == [1, 1, 2, 2, 1, 1, 2, 2]
    np.testing.assert_allclose(x0, [0, 0, 0, 0, 0], atol=1.0e-12, rtol=0)
    return (
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        1.5,
        [[0, 1], [2, 3]],
        [2, 1.5],
        0.01,
        0.02,
        aircraft,
    )


def make_ops_optimization_aircraft():
    """Return a compact fixture for OpsOptimize loop tests."""

    return {
        "Specs": {
            "Power": {
                "Phi": {
                    "Tko": 7,
                    "Clb": 7,
                    "Crs": 7,
                    "Des": 7,
                    "Lnd": 7,
                },
            },
        },
        "PowerOpt": {
            "MaxIter": 4,
            "Tol": 0.2,
            "ObjFun": "FuelBurn",
            "Segments": ["Climb"],
        },
    }


def fake_ops_mission_runner(aircraft):
    """Populate a small mission history for OpsOptimize tests."""

    aircraft["Mission"] = {
        "History": {
            "Segment": ["Climb", "Climb", "Cruise"],
            "SI": {
                "Performance": {
                    "Alt": [0, 1000, 2000],
                },
                "Power": {
                    "Phi": [0, 0, 0],
                },
                "Weight": {
                    "Fburn": [0, 10, 42],
                },
                "Energy": {
                    "Fuel": [0, 1, 2],
                    "Batt": [0, 3, 4],
                },
            },
        },
    }
    return aircraft


def fake_ops_tableau_builder(aircraft, ielem):
    """Return a small marker tableau after checking OpsOptimize indices."""

    assert ielem == [1, 2, 3]
    return [["tableau"]]


class SequentialPhiSolver:
    """Return a sequence of simplex results for OpsOptimize tests."""

    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __call__(self, tableau):
        assert tableau == [["tableau"]]
        value = self.values[self.calls]
        self.calls += 1
        return value
