# tests/test_oew.py

"""Tests for native OEWPkg ports."""

from fast_python.oew import oew_iteration
from fast_python.propulsion import create_prop_arch


def test_oew_iteration_updates_turboprop_weights_and_wing_area():
    """Check turboprop OEW iteration updates coupled weight fields."""

    aircraft = make_turboprop_oew_aircraft()
    result = oew_iteration(create_prop_arch(aircraft))
    weight = result["Specs"]["Weight"]

    assert weight["Airframe"] > 0
    assert weight["OEW"] == weight["Airframe"] + weight["Engines"]
    assert weight["MTOW"] > 0
    assert abs(result["Specs"]["Aero"]["S"] - weight["MTOW"] / 2) < 1.0e-6
    assert result["Specs"]["Propulsion"]["SLSPower"]
    assert result["Settings"]["OEWIterations"] <= 20


def make_turboprop_oew_aircraft():
    """Return a compact turboprop aircraft with synthetic historical data."""

    return {
        "Settings": {
            "OEW": {
                "Tol": 1.0e-9,
                "MaxIter": 20,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
                "EIS": 2021,
            },
            "Aero": {
                "W_S": {
                    "SLS": 2,
                },
            },
            "Performance": {
                "Vels": {
                    "Tko": 100,
                }
            },
            "Weight": {
                "OEW": 400,
                "MTOW": 600,
                "Fuel": 100,
                "Batt": 0,
                "Payload": 50,
                "Crew": 10,
                "Engines": 0,
                "EM": 0,
                "EG": 0,
                "EAP": 0,
                "Cables": 0,
                "WairfCF": 1,
            },
            "Power": {
                "SLS": 600,
                "P_W": {
                    "SLS": 1,
                    "EM": 10,
                    "EG": 20,
                },
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                    "Propeller": 0.8,
                },
                "LamDwn": {
                    "SLS": 0,
                },
            },
            "Propulsion": {
                "NumEngines": 2,
                "Engine": {
                    "EtaPoly": {
                        "Fan": 0.99,
                    }
                },
                "PropArch": {
                    "Type": "C",
                },
            },
        },
        "HistData": {
            "AC": {
                "A": {
                    "Specs": {
                        "Weight": {
                            "MTOW": 100,
                            "Airframe": 50,
                        }
                    }
                },
                "B": {
                    "Specs": {
                        "Weight": {
                            "MTOW": 200,
                            "Airframe": 100,
                        }
                    }
                },
                "C": {
                    "Specs": {
                        "Weight": {
                            "MTOW": 300,
                            "Airframe": 150,
                        }
                    }
                },
            },
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
            },
        },
    }
