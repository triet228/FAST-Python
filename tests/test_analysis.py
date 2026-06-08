# tests/test_analysis.py

"""Tests for native EAPAnalysis workflow."""

from fast_python.analysis import eap_analysis
from fast_python.io import build_json_data, write_json_file
from fast_python.main import main
from fast_python.mission import process_profile
from fast_python.native import run_native
from fast_python.propulsion import create_prop_arch, prop_arch_connections


def test_eap_analysis_flies_fixed_battery_off_design_mission():
    """Check EAPAnalysis runs a fixed-battery all-electric mission."""

    aircraft = make_analysis_aircraft()
    aircraft = create_prop_arch(aircraft)
    aircraft = prop_arch_connections(aircraft)
    aircraft = process_profile(aircraft)
    result = eap_analysis(aircraft)
    history = result["Mission"]["History"]["SI"]

    assert abs(history["Performance"]["Dist"][-1] - 20000) < 1.0e-5
    assert result["Specs"]["Weight"]["MTOW"] == 1000
    assert result["Specs"]["Weight"]["Batt"] == 200
    assert result["Settings"]["DetailedBatt"] == 0
    assert result["Settings"]["Converged"] == 1
    assert result["Mission"]["History"]["Flags"]["SOCOff"] == [0]


def test_run_native_executes_ported_workflow():
    """Check the native top-level runner on a covered workflow."""

    result = run_native(make_analysis_aircraft())
    history = result["aircraft"]["Mission"]["History"]["SI"]

    assert result["status"] == "success"
    assert result["backend"] == "native"
    assert abs(result["mtow"] - 1000) < 1.0e-9
    assert abs(history["Performance"]["Dist"][-1] - 20000) < 1.0e-5


def test_run_native_accepts_separate_scalar_target_mission():
    """Check native runner accepts Mission.json scalar target fields."""

    aircraft = make_analysis_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    mission["Target"] = {
        "Valu": 20000,
        "Type": "Dist",
    }
    result = run_native(aircraft, mission)
    history = result["aircraft"]["Mission"]["History"]["SI"]

    assert result["backend"] == "native"
    assert abs(history["Performance"]["Dist"][-1] - 20000) < 1.0e-5


def test_cli_main_runs_native_json_inputs(tmp_path):
    """Check CLI main can run the native backend from JSON input files."""

    aircraft = make_analysis_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    mission["Target"] = {
        "Valu": 20000,
        "Type": "Dist",
    }
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    write_json_file(input_dir / "InputAircraft.json", build_json_data(aircraft))
    write_json_file(input_dir / "Mission.json", build_json_data(mission))

    result = main(INPUT_DIR=input_dir, OUTPUT_DIR=output_dir, backend="native")

    assert result["backend"] == "native"
    assert (output_dir / "OutputAircraft.json").exists()
    assert (output_dir / "OutputAircraftStructure.json").exists()


def test_cli_main_runs_native_case_factory(tmp_path, monkeypatch):
    """Check CLI main can source aircraft and mission from native factories."""

    captured = {}

    def fake_run_native(aircraft, mission):
        captured["aircraft"] = aircraft
        captured["mission"] = mission
        return {
            "status": "success",
            "backend": "native",
            "mtow": 123,
            "aircraft": {
                "Specs": {
                    "TLAR": {
                        "Class": "Turboprop",
                    },
                    "Weight": {
                        "MTOW": 123,
                        "Fuel": 4,
                    },
                    "Aero": {
                        "S": 5,
                    },
                },
                "Mission": {
                    "Profile": captured["mission"],
                },
            },
        }

    monkeypatch.setattr("fast_python.main.run_native", fake_run_native)
    result = main(OUTPUT_DIR=tmp_path, native_case_name="ATR42")

    assert result["backend"] == "native"
    assert captured["aircraft"]["Specs"]["TLAR"]["Class"] == "Turboprop"
    assert captured["mission"]["Segs"][0] == "Takeoff"
    assert (tmp_path / "OutputAircraft.json").exists()


def make_analysis_aircraft():
    """Return a compact all-electric aircraft for EAPAnalysis."""

    return {
        "Settings": {
            "ClbPoints": 3,
            "CrsPoints": 3,
            "DesPoints": 3,
            "Analysis": {
                "Type": -2,
                "MaxIter": 3,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
                "MaxPax": 0,
            },
            "Performance": {
                "Vels": {
                    "Tko": 100,
                },
                "RCMax": 1000,
                "Range": 20000,
            },
            "Aero": {
                "L_D": {
                    "Clb": 10,
                    "Crs": 10,
                    "Des": 10,
                },
                "W_S": {
                    "SLS": 10,
                },
            },
            "Weight": {
                "MTOW": 1000,
                "OEW": 800,
                "Crew": 0,
                "Payload": 0,
                "Fuel": 0,
                "Batt": 200,
            },
            "Power": {
                "SLS": 1000000,
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000000,
                },
                "Eta": {
                    "EM": 1,
                    "EG": 1,
                    "Propeller": 1,
                },
                "P_W": {
                    "SLS": 1,
                    "EM": 1000000,
                    "EG": 1000000,
                },
                "LamDwn": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "LamUps": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "NumEngines": 1,
                "SLSPower": [1000000, 1000000],
                "SLSThrust": [0, 0],
                "PropArch": {
                    "Type": "E",
                },
            },
        },
        "Mission": {
            "Profile": {
                "Segs": ["Climb", "Cruise", "Descent"],
                "ID": [1, 1, 1],
                "Target": {
                    "Valu": [20000],
                    "Type": ["Dist"],
                },
                "AltBeg": [0, 1000, 1000],
                "AltEnd": [1000, 1000, 0],
                "VelBeg": [100, 100, 100],
                "VelEnd": [100, 100, 100],
                "TypeBeg": ["TAS", "TAS", "TAS"],
                "TypeEnd": ["TAS", "TAS", "TAS"],
                "ClbRate": [10, float("nan"), -10],
            }
        },
    }
