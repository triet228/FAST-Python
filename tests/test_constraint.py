# tests/test_constraint.py

"""Tests for native ConstraintDiagramPkg helper ports."""

import numpy as np
import pytest

from fast_python.constraint import (
    ConstraintError,
    constraint_diagram,
    jet25_111,
    jet25_119,
    jet25_121a,
    jet25_121b,
    jet25_121c,
    jet25_121d,
    jet_app,
    jet_aeo_climb,
    jet_ceil,
    jet_crs,
    jet_div,
    jet_lfl,
    jet_tofl,
    oei_multiplier,
    sigmoid,
)


def test_sigmoid_and_oei_multiplier_match_matlab_values():
    """Check scalar ConstraintDiagramPkg helpers."""

    aircraft = make_constraint_aircraft()

    assert oei_multiplier(aircraft) == 2

    aircraft["Settings"]["ConstraintType"] = 1

    assert abs(oei_multiplier(aircraft) - 1.09405) < 1.0e-12
    assert abs(sigmoid(aircraft, 10, 2, 0.2, 1) - 0.064983399731) < 1.0e-12

    aircraft["Settings"]["ConstraintType"] = 3

    with pytest.raises(ConstraintError, match="Type"):
        oei_multiplier(aircraft)


def test_jet_app_matches_matlab_values():
    """Check JetApp against MATLAB oracle values."""

    result = jet_app([400, 500], [0.3, 0.4], make_constraint_aircraft())

    np.testing.assert_allclose(
        result,
        [-22.803506042708, -2.321891680456],
        atol=1.0e-12,
        rtol=0,
    )


def test_jet_tofl_matches_matlab_values():
    """Check JetTOFL against MATLAB oracle values."""

    result = jet_tofl([400, 500], [0.3, 0.4], make_constraint_aircraft())

    np.testing.assert_allclose(
        result,
        [-0.039794640519, -0.074743300649],
        atol=1.0e-12,
        rtol=0,
    )


def test_jet_cruise_diversion_and_ceiling_match_matlab_values():
    """Check cruise-style constraints against MATLAB oracle values."""

    aircraft = make_constraint_aircraft()

    np.testing.assert_allclose(
        jet_crs([400, 500], [0.3, 0.4], aircraft),
        [-0.160177693635, -0.274807516069],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet_div([400, 500], [0.3, 0.4], aircraft),
        [-0.183407062464, -0.294389109285],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet_ceil([400, 500], [0.3, 0.4], aircraft),
        [0.055780976647, -0.109972997754],
        atol=1.0e-12,
        rtol=0,
    )

    aircraft["Specs"]["TLAR"]["ReqType"] = 0

    np.testing.assert_allclose(
        jet_ceil([400, 500], [0.3, 0.4], aircraft),
        [-0.174678195815, -0.274678195815],
        atol=1.0e-12,
        rtol=0,
    )


def test_jet_landing_field_length_matches_matlab_values():
    """Check JetLFL against MATLAB oracle values."""

    aircraft = make_constraint_aircraft()

    np.testing.assert_allclose(
        jet_lfl([400, 500], [0.3, 0.4], aircraft),
        [-10.836774394655, 9.651836588065],
        atol=1.0e-12,
        rtol=0,
    )

    aircraft["Specs"]["TLAR"]["ReqType"] = 0

    np.testing.assert_allclose(
        jet_lfl([400, 500], [0.3, 0.4], aircraft),
        [-15.399655189081, 5.088955793639],
        atol=1.0e-12,
        rtol=0,
    )


def test_far25_climb_constraints_match_matlab_values():
    """Check FAR 25 climb constraints against MATLAB oracle values."""

    aircraft = make_constraint_aircraft()

    np.testing.assert_allclose(
        jet25_111([400, 500], [0.3, 0.4], aircraft),
        [-0.100878686629, -0.177724169905],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet25_119([400, 500], [0.3, 0.4], aircraft),
        [-0.149301440563, -0.251070221137],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet25_121a([400, 500], [0.3, 0.4], aircraft),
        [-0.078647426633, -0.160092805326],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet25_121b([400, 500], [0.3, 0.4], aircraft),
        [-0.074478686629, -0.151324169905],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet25_121c([400, 500], [0.3, 0.4], aircraft),
        [-0.138253241254, -0.219881387235],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        jet25_121d([400, 500], [0.3, 0.4], aircraft),
        [-0.032326052330, -0.144376696157],
        atol=1.0e-12,
        rtol=0,
    )


def test_jet_all_engines_climb_matches_matlab_values():
    """Check JetAEOClimb against MATLAB oracle values."""

    aircraft = make_constraint_aircraft()

    np.testing.assert_allclose(
        jet_aeo_climb([400, 500], [0.3, 0.4], aircraft),
        [-0.115071861226, -0.192404053164],
        atol=1.0e-12,
        rtol=0,
    )

    aircraft["Specs"]["TLAR"]["ReqType"] = 0

    np.testing.assert_allclose(
        jet_aeo_climb([400, 500], [0.3, 0.4], aircraft),
        [-0.127356207968, -0.227356207968],
        atol=1.0e-12,
        rtol=0,
    )

    aircraft["Specs"]["TLAR"]["ReqType"] = 2

    np.testing.assert_allclose(
        jet_aeo_climb([400, 500], [0.3, 0.4], aircraft),
        [-0.213678103984, -0.313678103984],
        atol=1.0e-12,
        rtol=0,
    )

    aircraft["Specs"]["TLAR"]["Class"] = "Turboprop"
    aircraft["Specs"]["TLAR"]["ReqType"] = 1

    np.testing.assert_allclose(
        jet_aeo_climb([400, 500], [0.3, 0.4], aircraft),
        [0.138631842478, 0.172873724613],
        atol=1.0e-12,
        rtol=0,
    )


def test_constraint_diagram_evaluates_named_constraints():
    """Check the constraint-diagram driver grid and residual assembly."""

    aircraft = make_constraint_aircraft()
    result = constraint_diagram(aircraft, npoints=3)

    np.testing.assert_allclose(
        result["horizontal_range"],
        [500, 1500, 2500],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        result["vertical_range"],
        [0.15, 0.35, 0.55],
        atol=1.0e-12,
        rtol=0,
    )
    assert result["horizontal_label"] == "Wing Loading (kg/m^2)"
    assert result["vertical_label"] == "Thrust-Weight Ratio (N/N)"
    assert result["constraint_names"] == ["JetTOFL", "JetCrs"]
    assert result["constraint_labels"] == ["TOFL", "Cruise"]
    assert result["constraints"].shape == (3, 3, 2)
    assert result["feasible"].shape == (3, 3)

    np.testing.assert_allclose(
        result["constraints"][:, :, 0],
        jet_tofl(
            result["horizontal_grid"],
            result["vertical_grid"],
            aircraft,
        ),
        atol=1.0e-12,
        rtol=0,
    )


def test_constraint_diagram_converts_part25_power_aircraft_axis():
    """Check the Part 25 turboprop axis conversion after evaluation."""

    aircraft = make_constraint_aircraft()
    aircraft["Specs"]["TLAR"]["Class"] = "Turboprop"
    aircraft["Specs"]["TLAR"]["CFRPart"] = 25
    aircraft["Specs"]["Power"]["P_W"]["SLS"] = 0.03
    result = constraint_diagram(aircraft, npoints=3)

    np.testing.assert_allclose(
        result["horizontal_range"],
        np.array([500, 1500, 2500]) * 9.81 / 1000,
        atol=1.0e-12,
        rtol=0,
    )
    assert result["horizontal_label"] == "Wing Loading (kN/m^2)"
    assert result["vertical_label"] == "Power Loading (N/W)"


def make_constraint_aircraft():
    """Return a minimal aircraft for constraint helper tests."""

    return {
        "Settings": {
            "ConstraintType": 0,
        },
        "Specs": {
            "TLAR": {
                "Class": "Turbofan",
                "CFRPart": 25,
                "ReqType": 1,
            },
            "Propulsion": {
                "NumEngines": 2,
                "T_W": {
                    "SLS": 0.35,
                },
            },
            "Power": {
                "P_W": {
                    "SLS": 0.3,
                },
            },
            "Performance": {
                "PsLoss": 0.3,
                "TempInc": 1.1,
                "MaxCont": 0.92,
                "ExtraGrad": 0.018,
                "Wland_MTOW": 0.85,
                "Vels": {
                    "App": 70,
                    "Crs": 0.78,
                    "Div": 0.65,
                    "Stl": 60,
                },
                "Alts": {
                    "Crs": 10000,
                    "Div": 8000,
                    "Srv": 11000,
                },
                "TOFL": 1800,
                "LFL": 1500,
                "ObstLen": 15,
                "ConstraintFuns": ["JetTOFL", "JetCrs"],
                "ConstraintLabs": ["TOFL", "Cruise"],
            },
            "Aero": {
                "W_S": {
                    "SLS": 1500,
                },
                "AR": 9,
                "CD0": {
                    "Tko": 0.05,
                    "Crs": 0.02,
                    "Lnd": 0.08,
                },
                "CL": {
                    "Crs": 1.5,
                    "Lnd": 2.4,
                    "Tko": 2.0,
                },
                "e": {
                    "Tko": 0.75,
                    "Crs": 0.8,
                    "Lnd": 0.7,
                },
            },
        },
    }
