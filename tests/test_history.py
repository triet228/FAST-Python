# tests/test_history.py

"""Tests for MissionHistTable port."""

from fast_python.history import MISSION_HISTORY_COLUMNS, MissionHistTable, mission_history_table


def test_mission_history_table_scales_and_names_columns():
    """Check MissionHistTable column names and unit conversions."""

    result = mission_history_table(make_history_aircraft())
    rows = result["rows"]

    assert result["columns"] == MISSION_HISTORY_COLUMNS
    assert len(rows) == 2
    assert rows[0]["Time (min)"] == 1
    assert rows[1]["Segment"] == "Cruise"
    assert rows[1]["Distance (km)"] == 2
    assert rows[0]["Thrust Available (kN)"] == 3
    assert rows[1]["Power Available (MW)"] == [2, 4]
    assert rows[1]["Energy Remaining (MJ)"] == 5
    assert MissionHistTable(make_history_aircraft())["rows"][0]["Mach"] == 0.2


def make_history_aircraft():
    """Return a compact aircraft with mission history arrays."""

    return {
        "Mission": {
            "History": {
                "Segment": ["Climb", "Cruise"],
                "SI": {
                    "Performance": {
                        "Time": [60, 120],
                        "Dist": [1000, 2000],
                        "TAS": [100, 110],
                        "EAS": [90, 95],
                        "RC": [5, 0],
                        "Alt": [1000, 2000],
                        "Acc": [0.1, 0.0],
                        "FPA": [2, 0],
                        "Mach": [0.2, 0.3],
                        "Rho": [1.1, 1.0],
                        "Ps": [10, 0],
                    },
                    "Propulsion": {
                        "TSFC": [0.5, 0.4],
                        "MDotFuel": [0.2, 0.1],
                    },
                    "Weight": {
                        "CurWeight": [1000, 990],
                        "Fburn": [0, 10],
                    },
                    "Power": {
                        "Tav": [3000, 4000],
                        "Treq": [2500, 3500],
                        "TV": [1.0e6, 2.0e6],
                        "Req": [1.5e6, 2.5e6],
                        "Pav": [[1.0e6, 3.0e6], [2.0e6, 4.0e6]],
                        "Preq": [[0.5e6, 1.5e6], [1.0e6, 2.0e6]],
                        "SOC": [100, 95],
                    },
                    "Energy": {
                        "KE": [1.0e6, 2.0e6],
                        "PE": [3.0e6, 4.0e6],
                        "E_ES": [0.5e6, 1.5e6],
                        "Eleft_ES": [6.0e6, 5.0e6],
                    },
                },
            },
        },
    }
