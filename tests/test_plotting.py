# tests/test_plotting.py

"""Tests for PlotPkg data-formatting helper ports."""

from fast_python.plotting import PlotMission, PlotPerfParam, plot_mission_data, plot_perf_param_data


def test_plot_perf_param_data_preserves_state_series():
    """Check PlotPerfParam data wrapper for non-instantaneous values."""

    result = plot_perf_param_data([1, 2, 3], [10, 20, 30], 0, "x", "y", "Demo")

    assert result == {
        "x": [1, 2, 3],
        "y": [10, 20, 30],
        "xlabel": "x",
        "ylabel": "y",
        "title": "Demo Profile",
    }


def test_plot_perf_param_data_steps_instantaneous_values():
    """Check PlotPerfParam instantaneous step-like reshaping."""

    result = PlotPerfParam([1, 2, 3], [10, 20, 30], 1, "time", "power", "Power")

    assert result["x"] == [1, 2, 2, 3, 3]
    assert result["y"] == [10, 10, 20, 20, 30]
    assert result["title"] == "Power Profile"


def test_plot_perf_param_data_steps_matrix_columns():
    """Check column-wise step formatting for matrix inputs."""

    result = plot_perf_param_data(
        [[1, 10], [2, 20], [3, 30]],
        [[4, 40], [5, 50], [6, 60]],
        1,
        "x",
        "y",
        "Matrix",
    )

    assert result["x"] == [
        [1, 10],
        [2, 20],
        [2, 20],
        [3, 30],
        [3, 30],
    ]
    assert result["y"] == [
        [4, 40],
        [4, 40],
        [5, 50],
        [5, 50],
        [6, 60],
    ]


def test_plot_perf_param_data_broadcasts_time_for_matrix_series():
    """Check MATLAB-style plotting of one x vector against matrix y columns."""

    result = plot_perf_param_data(
        [1, 2, 3],
        [[4, 40], [5, 50], [6, 60]],
        1,
        "time",
        "power",
        "Power",
    )

    assert result["x"] == [
        [1, 1],
        [2, 2],
        [2, 2],
        [3, 3],
        [3, 3],
    ]
    assert result["y"] == [
        [4, 40],
        [4, 40],
        [5, 50],
        [5, 50],
        [6, 60],
    ]


def test_plot_mission_data_converts_history_and_filters_propellers():
    """Check PlotMission's unit conversions and transmitter mask."""

    result = plot_mission_data(make_plot_aircraft())
    values = result["values"]

    assert values["time_min"] == [1, 2, 3]
    assert abs(values["altitude_ft"][1] - 3280.83989501312) < 1.0e-12
    assert abs(values["distance_nmi"][2] - 5.39956803455724) < 1.0e-12
    assert values["power_output_mw"] == [0.2, 0.3, 0.4]
    assert values["thrust_available_kn"] == [1, 2, 3]
    assert values["transmitter_mask"] == [False, True, False, False]
    assert result["figures"][0]["name"] == "flight_performance"
    assert result["figures"][2]["plots"][2]["y"] == [0.2, 0.2, 0.3, 0.3, 0.4]
    assert abs(PlotMission(make_plot_aircraft())["values"]["fuel_burn_lbm"][2] - 22.0462262184878) < 1.0e-12


def make_plot_aircraft():
    """Return a compact aircraft with the fields consumed by PlotMission."""

    return {
        "Specs": {
            "TLAR": {
                "Class": "Turbofan",
            },
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0, 1, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    "SrcType": [0],
                    "TrnType": [0, 2],
                },
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Performance": {
                        "Time": [60, 120, 180],
                        "Alt": [0, 1000, 2000],
                        "Dist": [0, 5000, 10000],
                        "TAS": [100, 110, 120],
                        "RC": [5, 0, -3],
                        "Ps": [2, 1, 0],
                    },
                    "Weight": {
                        "CurWeight": [1000, 995, 990],
                        "Fburn": [0, 5, 10],
                    },
                    "Power": {
                        "Req": [1.0e6, 2.0e6, 3.0e6],
                        "TV": [1.5e6, 2.5e6, 3.5e6],
                        "Preq": [
                            [0, 1.0e5, 9.0e5, 0],
                            [0, 2.0e5, 8.0e5, 0],
                            [0, 3.0e5, 7.0e5, 0],
                        ],
                        "Pav": [
                            [0, 1.5e5, 1.0e6, 0],
                            [0, 2.5e5, 1.0e6, 0],
                            [0, 3.5e5, 1.0e6, 0],
                        ],
                        "Pout": [
                            [0, 2.0e5, 1.0e6, 0],
                            [0, 3.0e5, 1.0e6, 0],
                            [0, 4.0e5, 1.0e6, 0],
                        ],
                        "Treq": [
                            [0, 500, 900, 0],
                            [0, 600, 900, 0],
                            [0, 700, 900, 0],
                        ],
                        "Tav": [
                            [0, 1000, 900, 0],
                            [0, 2000, 900, 0],
                            [0, 3000, 900, 0],
                        ],
                        "Tout": [
                            [0, 1500, 900, 0],
                            [0, 2500, 900, 0],
                            [0, 3500, 900, 0],
                        ],
                    },
                    "Propulsion": {
                        "TSFC": [0.5, 0.4, 0.3],
                        "MDotFuel": [0.2, 0.15, 0.1],
                    },
                },
            },
        },
    }
