# tests/test_database.py

"""Tests for FAST historical database loading."""

import numpy as np

from fast_python.database import (
    bundled_database_path,
    calc_fan_vals,
    calc_prop_vals,
    load_ideas_database,
    randomize_db,
    resolve_database_path,
    struct_tree_search,
    tree_branch,
)
from fast_python.regression import search_db


def test_resolve_database_path_finds_bundled_database():
    """Check the package-local FAST database can be resolved."""

    path = resolve_database_path()

    assert path == bundled_database_path()
    assert path.exists()


def test_load_ideas_database_supports_regression_searches():
    """Check database dictionaries are compatible with search_db()."""

    database = load_ideas_database()
    turbofan_ac = database["TurbofanAC"]
    turboprop_engines = database["TurbopropEngines"]
    _, max_pax_rows = search_db(turbofan_ac, ["Specs", "TLAR", "MaxPax"])
    _, power_rows = search_db(turboprop_engines, ["Power_SLS"])

    assert len(turbofan_ac) > 300
    assert len(database["TurbopropAC"]) > 50
    assert len(max_pax_rows) == len(turbofan_ac)
    assert isinstance(max_pax_rows[0][0], str)
    assert isinstance(max_pax_rows[0][1], int) or isinstance(max_pax_rows[0][1], float)
    assert len(power_rows) == len(turboprop_engines)


def test_tree_branch_splits_nested_branches_and_leaves():
    """Check TreeBranch separates sub-structures from scalar leaves."""

    branches, leaves = tree_branch(
        {
            "Specs": {
                "Weight": {
                    "MTOW": 100,
                },
            },
            "Name": "Example",
        },
        "Plane",
    )

    assert branches == [("Specs", {"Weight": {"MTOW": 100}})]
    assert leaves == [("Plane_Name", "Example")]


def test_struct_tree_search_returns_transposed_leaf_cells():
    """Check StructTreeSearch returns MATLAB-style name/value rows."""

    leaves = struct_tree_search(
        {
            "Name": "Example",
            "Specs": {
                "Weight": {
                    "MTOW": 100,
                    "Fuel": 10,
                },
                "Class": "Turbofan",
            },
        }
    )

    assert leaves == [
        ["Plane_Name", "Specs_Class", "Weight_MTOW", "Weight_Fuel"],
        ["Example", "Turbofan", 100, 10],
    ]


def test_randomize_db_uses_fast_threshold_rule_without_mutating_input():
    """Check RandomizeDB assigns Training/Validation from random draws."""

    database = {
        "A": {
            "Settings": {},
        },
        "B": {
            "Settings": {},
        },
    }
    randomized = randomize_db(database, 50, FakeRng([20, 80]))

    assert randomized["A"]["Settings"]["DataTypeValidation"] == "Training"
    assert randomized["B"]["Settings"]["DataTypeValidation"] == "Validation"
    assert "DataTypeValidation" not in database["A"]["Settings"]


def test_calc_fan_vals_units_assigns_metadata_without_mutating_input():
    """Check CalcFanVals Units mode assigns turbofan metadata."""

    plane = {
        "Overview": {
            "KeyWords": "baseline single",
        },
        "Specs": {
            "Weight": {},
        },
    }
    result = calc_fan_vals(plane, "Units")

    assert result["Specs"]["Performance"]["Range"] == "m"
    assert result["Specs"]["Aero"]["L_D"]["CrsBRE"] == "ratio"
    assert result["Specs"]["Propulsion"]["Thrust"]["Crs"] == "N"
    assert result["Specs"]["Power"]["P_W"]["AC"] == "W/kg"
    assert result["Settings"]["DataTypeValidation"] == "data type"
    assert "KeyWords" not in result["Overview"]
    assert "KeyWords" in plane["Overview"]


def test_calc_fan_vals_values_match_matlab_oracle():
    """Check CalcFanVals Vals mode against a MATLAB oracle fixture."""

    result = calc_fan_vals(make_fan_plane(), "Vals")

    assert abs(result["Specs"]["Aero"]["L_D"]["CrsBRE"] - 8.879962788885) < 1.0e-12
    assert abs(result["Specs"]["Aero"]["L_D"]["Crs"] - 5.6084791875) < 1.0e-12
    assert abs(result["Specs"]["Aero"]["L_D"]["CrsMAC"] - 18.764320102772) < 1.0e-12
    assert abs(result["Specs"]["Aero"]["L_D"]["CrsMAC2"] - 20.156524304003) < 1.0e-12
    assert result["Specs"]["Weight"]["Airframe"] == 38000
    assert result["Specs"]["Weight"]["OEW_MTOW"] == 0.6
    assert abs(result["Specs"]["Weight"]["MZFW_MTOW"] - 0.785714285714) < 1.0e-12
    assert abs(result["Specs"]["Propulsion"]["T_W"]["SLS"] - 0.349497597204) < 1.0e-12
    assert result["Specs"]["Propulsion"]["Thrust"]["SLS"] == 200000
    assert result["Specs"]["Propulsion"]["Thrust"]["Max"] == 240000
    assert result["Specs"]["Propulsion"]["Thrust"]["Crs"] == 120000
    assert abs(result["Specs"]["Aero"]["W_S"]["SLS"] - 583.333333333333) < 1.0e-12
    assert abs(result["Specs"]["Weight"]["EngineFrac"] - 0.057142857143) < 1.0e-12
    assert abs(result["Specs"]["Weight"]["FuelFrac"] - 0.214285714286) < 1.0e-12
    assert result["Overview"]["Aisle"] == "Single"
    assert result["Overview"]["RangeCapability"] == "Long Range"
    assert result["Overview"]["ModelType"] == "Baseline"
    assert result["Overview"]["PayloadType"] == "Passenger"
    assert result["Overview"]["PassengerType"] == "Business"
    assert result["Overview"]["Monikers"] == "N/A"
    assert result["Overview"]["AlternateDesignation"] == "N/A"
    assert result["Specs"]["Aero"]["TaperRatio"] == 0.4
    assert abs(result["Specs"]["Aero"]["AR"] - 10.208333333333) < 1.0e-12
    assert abs(result["Specs"]["Performance"]["Vels"]["Tko"] - 72.022222222222) < 1.0e-12
    assert result["Specs"]["Weight"]["Cargo"] == 0
    assert result["Specs"]["Weight"]["Payload"] == 14250
    assert result["Specs"]["Performance"]["Range"] == 3000000
    assert result["Specs"]["TLAR"]["Class"] == "Turbofan"
    assert result["Specs"]["Propulsion"]["Arch"]["Type"] == "C"
    assert result["Specs"]["Power"]["SpecEnergy"]["Fuel"] == 12
    assert abs(result["Specs"]["Power"]["P_W"]["SLS"] - 25.171593611961) < 1.0e-12
    assert result["Specs"]["TLAR"]["ADG"] == "III"
    assert abs(result["Specs"]["TLAR"]["ADGPercent"] - 0.973130477334) < 1.0e-12
    assert result["Specs"]["TLAR"]["ADGLimitor"] == "Wingspan"
    assert result["Specs"]["Weight"]["Burden"] == 33250
    assert abs(result["Specs"]["Weight"]["Structure_Burden"] - 1.105263157895) < 1.0e-12
    assert result["Settings"]["DataTypeValidation"] in ("Training", "Validation")
    assert "KeyWords" not in result["Overview"]


def test_calc_prop_vals_units_assigns_metadata_and_removes_max_payload():
    """Check CalcPropVals Units mode assigns turboprop metadata."""

    plane = {
        "Overview": {
            "KeyWords": "baseline",
        },
        "Specs": {
            "Weight": {
                "MaxPayload": 1234,
            },
        },
    }
    result = calc_prop_vals(plane, "Units")

    assert result["Specs"]["Performance"]["Vels"]["Crs"] == "Mach"
    assert result["Specs"]["Power"]["SLS"] == "W"
    assert result["Specs"]["Power"]["P_W"]["SLS"] == "kW/kg"
    assert result["Specs"]["Propulsion"]["T_W"]["SLS"] == "ratio"
    assert result["Settings"]["Analysis"]["Type"] == "flag"
    assert "KeyWords" not in result["Overview"]
    assert "MaxPayload" not in result["Specs"]["Weight"]
    assert "MaxPayload" in plane["Specs"]["Weight"]


def test_calc_prop_vals_values_match_matlab_oracle():
    """Check CalcPropVals Vals mode against a MATLAB oracle fixture."""

    result = calc_prop_vals(make_prop_plane(), "Vals")

    assert result["Specs"]["Weight"]["Airframe"] == 9000
    assert abs(result["Specs"]["Weight"]["OEW_MTOW"] - 0.555555555556) < 1.0e-12
    assert result["Specs"]["Power"]["SLS"] == 2400000
    assert result["Specs"]["Power"]["Cont"] == 2000000
    assert result["Specs"]["Power"]["Clb"] == 1800000
    assert result["Specs"]["Power"]["Crs"] == 1400000
    assert abs(result["Specs"]["Power"]["P_W"]["SLS"] - 0.133333333333) < 1.0e-12
    assert result["Specs"]["Aero"]["W_S"]["SLS"] == 450
    assert abs(result["Specs"]["Weight"]["EngineFrac"] - 0.055555555556) < 1.0e-12
    assert abs(result["Specs"]["Weight"]["FuelFrac"] - 0.138888888889) < 1.0e-12
    assert result["Overview"]["ModelType"] == "Baseline"
    assert result["Overview"]["PayloadType"] == "Passenger"
    assert result["Overview"]["PassengerType"] == "Business"
    assert result["Overview"]["Monikers"] == "N/A"
    assert result["Overview"]["AlternateDesignation"] == "N/A"
    assert result["Specs"]["Aero"]["TaperRatio"] == 0.5
    assert result["Specs"]["Aero"]["AR"] == 10
    assert abs(result["Specs"]["Aero"]["L_D"]["Crs"] - 15.335566325083) < 1.0e-12
    assert abs(result["Specs"]["Performance"]["Vels"]["Tko"] - 51.444444444444) < 1.0e-12
    assert result["Specs"]["Weight"]["Cargo"] == 0
    assert result["Specs"]["Weight"]["Payload"] == 3800
    assert result["Specs"]["TLAR"]["ADG"] == "II"
    assert abs(result["Specs"]["TLAR"]["ADGPercent"] - 0.830592378484) < 1.0e-12
    assert result["Specs"]["TLAR"]["ADGLimitor"] == "Wingspan"
    assert result["Specs"]["TLAR"]["Class"] == "Turboprop"
    assert result["Specs"]["Propulsion"]["Arch"]["Type"] == "C"
    assert result["Specs"]["Power"]["SpecEnergy"]["Fuel"] == 12
    assert abs(result["Specs"]["Propulsion"]["T_W"]["SLS"] - 0.000264199048) < 1.0e-12
    assert "KeyWords" not in result["Overview"]
    assert "MaxPayload" not in result["Specs"]["Weight"]
    assert np.isnan(result["Settings"]["OEW"]["Tol"])


def make_prop_plane():
    """Return the MATLAB oracle fixture for CalcPropVals Vals mode."""

    return {
        "Overview": {
            "KeyWords": "baseline business",
            "PayloadType": "P",
            "Monikers": np.nan,
            "AlternateDesignation": np.nan,
        },
        "Specs": {
            "Weight": {
                "OEW": 10000,
                "MTOW": 18000,
                "Fuel": 2500,
                "Cargo": np.nan,
                "MaxPayload": np.nan,
            },
            "TLAR": {
                "MaxPax": 40,
            },
            "Propulsion": {
                "Engine": {
                    "DryWeight": 500,
                    "Power_SLS": 1200,
                    "Power_SLS_Eq": np.nan,
                    "Power_Cont_Eq": 1000,
                },
                "NumEngines": 2,
            },
            "Power": {
                "SLS": np.nan,
                "Cont": np.nan,
                "Clb": 900,
                "Crs": 700,
            },
            "Aero": {
                "S": 40,
                "TipChord": 2,
                "RootChord": 4,
                "Span": 20,
                "Height": 6,
            },
            "Performance": {
                "Vels": {
                    "Crs": 0.4,
                    "Tko": 100,
                },
            },
        },
    }


def make_fan_plane():
    """Return the MATLAB oracle fixture for CalcFanVals Vals mode."""

    return {
        "Overview": {
            "KeyWords": "baseline single long business",
            "PayloadType": "P",
            "Monikers": np.nan,
            "AlternateDesignation": np.nan,
        },
        "Specs": {
            "Weight": {
                "MTOW": 70000,
                "Fuel": 15000,
                "OEW": 42000,
                "MZFW": 55000,
                "Cargo": np.nan,
            },
            "TLAR": {
                "MaxPax": 150,
            },
            "Propulsion": {
                "Engine": {
                    "TSFC_Crs": 0.6,
                    "Thrust_Crs": 60000,
                    "Thrust_Max": 120000,
                    "Thrust_SLS": 100000,
                    "DryWeight": 2000,
                },
                "NumEngines": 2,
                "Thrust": {
                    "SLS": np.nan,
                    "Max": np.nan,
                },
            },
            "Performance": {
                "Range": 3000,
                "Alts": {
                    "Crs": 10000,
                },
                "Vels": {
                    "Crs": 0.78,
                    "Tko": 140,
                },
            },
            "Aero": {
                "Span": 35,
                "S": 120,
                "MAC": 4,
                "WingtipDevice": "Winglets",
                "TipChord": 2,
                "RootChord": 5,
                "Height": 12,
            },
        },
    }


class FakeRng:
    """Small deterministic RNG for RandomizeDB tests."""

    def __init__(self, values):
        self.values = list(values)

    def integers(self, _low, _high):
        return self.values.pop(0)
