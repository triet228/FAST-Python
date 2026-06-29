# tests/test_matlab_mission_segs_parity.py

"""Optional MATLAB parity tests for MissionSegsPkg helpers and evaluators."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fast_python.atmosphere import gravity, standard_atmosphere
from fast_python.compare import compare_json_value
from fast_python.data_struct import init_mission_history
from fast_python.io import build_json_data
from fast_python import mission


SEGMENT_FIXTURE_PATH = Path(__file__).resolve().parent / "test_mission_segments.py"
CRUISE_BRE_POST = """
matlab_aircraft.Specs.Power.Battery.SerCells = NaN;
matlab_aircraft.Specs.Power.Battery.ParCells = NaN;
"""
CRUISE_BRE_PREP = """
matlab_aircraft.Specs.Power.Battery.SerCells = 1;
matlab_aircraft.Specs.Power.Battery.ParCells = 1;
matlab_aircraft.Mission.History.SI.Performance.Time = [];
matlab_aircraft.Mission.History.SI.Performance.Dist = [];
matlab_aircraft.Mission.History.SI.Performance.TAS = [];
matlab_aircraft.Mission.History.SI.Performance.EAS = [];
matlab_aircraft.Mission.History.SI.Performance.RC = [];
matlab_aircraft.Mission.History.SI.Performance.Alt = [];
matlab_aircraft.Mission.History.SI.Performance.Acc = [];
matlab_aircraft.Mission.History.SI.Performance.FPA = [];
matlab_aircraft.Mission.History.SI.Performance.Mach = [];
matlab_aircraft.Mission.History.SI.Performance.Rho = [];
matlab_aircraft.Mission.History.SI.Propulsion.Treq = [];
matlab_aircraft.Mission.History.SI.Propulsion.Eta = [];
matlab_aircraft.Mission.History.SI.Propulsion.TSFC = [];
matlab_aircraft.Mission.History.SI.Propulsion.MDotFuel = [];
matlab_aircraft.Mission.History.SI.Weight.CurWeight = [];
matlab_aircraft.Mission.History.SI.Weight.Fburn = [];
matlab_aircraft.Mission.History.SI.Power.Av = [];
matlab_aircraft.Mission.History.SI.Power.Req = [];
matlab_aircraft.Mission.History.SI.Power.Out = [];
matlab_aircraft.Mission.History.SI.Power.Fuel = [];
matlab_aircraft.Mission.History.SI.Power.Batt = [];
matlab_aircraft.Mission.History.SI.Power.EM = [];
matlab_aircraft.Mission.History.SI.Power.EG = [];
matlab_aircraft.Mission.History.SI.Power.Prop = [];
matlab_aircraft.Mission.History.SI.Power.Phi = [];
matlab_aircraft.Mission.History.SI.Power.SOC = [];
matlab_aircraft.Mission.History.SI.Energy.KE = [];
matlab_aircraft.Mission.History.SI.Energy.PE = [];
matlab_aircraft.Mission.History.SI.Energy.Fuel = [];
matlab_aircraft.Mission.History.SI.Energy.Batt = [];
matlab_aircraft.Mission.History.Segment = strings(0, 1);
"""
MATLAB_SPLIT_PREP = """
if isfield(matlab_aircraft, "Specs") && ...
   isfield(matlab_aircraft.Specs, "Propulsion") && ...
   isfield(matlab_aircraft.Specs.Propulsion, "PropArch")

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperUps") && ...
       ~isa(matlab_aircraft.Specs.Propulsion.PropArch.OperUps, "function_handle")

        oper_ups_value = matlab_aircraft.Specs.Propulsion.PropArch.OperUps;
        matlab_aircraft.Specs.Propulsion.PropArch.OperUps = @(varargin) oper_ups_value;
    end

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperDwn") && ...
       ~isa(matlab_aircraft.Specs.Propulsion.PropArch.OperDwn, "function_handle")

        oper_dwn_value = matlab_aircraft.Specs.Propulsion.PropArch.OperDwn;
        matlab_aircraft.Specs.Propulsion.PropArch.OperDwn = @(varargin) oper_dwn_value;
    end

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "ParConns") && ...
       ~iscell(matlab_aircraft.Specs.Propulsion.PropArch.ParConns)

        ntrn = numel(matlab_aircraft.Specs.Propulsion.PropArch.TrnType);
        matlab_aircraft.Specs.Propulsion.PropArch.ParConns = cell(1, ntrn);
    end

    if ~isfield(matlab_aircraft.Specs.Propulsion.PropArch, "WhichProp")
        ntrn = numel(matlab_aircraft.Specs.Propulsion.PropArch.TrnType);
        matlab_aircraft.Specs.Propulsion.PropArch.WhichProp = zeros(1, ntrn);
    end
end

if isfield(matlab_aircraft, "Specs") && ...
   isfield(matlab_aircraft.Specs, "Aero") && ...
   isfield(matlab_aircraft.Specs.Aero, "L_D")

    if ~isfield(matlab_aircraft.Specs.Aero.L_D, "Method") || ...
       ~isa(matlab_aircraft.Specs.Aero.L_D.Method, "function_handle")

        matlab_aircraft.Specs.Aero.L_D.Method = ...
            @(Aircraft) AerodynamicsPkg.ConstantLD(Aircraft);
    end
end
"""
MATLAB_SPLIT_POST = """
if isfield(matlab_aircraft, "Specs") && ...
   isfield(matlab_aircraft.Specs, "Propulsion") && ...
   isfield(matlab_aircraft.Specs.Propulsion, "PropArch")

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperUps") && ...
       isa(matlab_aircraft.Specs.Propulsion.PropArch.OperUps, "function_handle")

        matlab_aircraft.Specs.Propulsion.PropArch.OperUps = ...
            matlab_aircraft.Specs.Propulsion.PropArch.OperUps();
    end

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperDwn") && ...
       isa(matlab_aircraft.Specs.Propulsion.PropArch.OperDwn, "function_handle")

        matlab_aircraft.Specs.Propulsion.PropArch.OperDwn = ...
            matlab_aircraft.Specs.Propulsion.PropArch.OperDwn();
    end

    if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "ParConns") && ...
       iscell(matlab_aircraft.Specs.Propulsion.PropArch.ParConns)

        matlab_aircraft.Specs.Propulsion.PropArch.ParConns = ...
            cellfun(@(x) x, matlab_aircraft.Specs.Propulsion.PropArch.ParConns, ...
            "UniformOutput", false);
    end
end
"""
MISSION_SEGMENT_CASES = [
    (
        "EvalTakeoff",
        mission.eval_takeoff,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_takeoff_aircraft())
        ),
        "",
        "",
    ),
    (
        "EvalDetailedTakeoff",
        mission.eval_detailed_takeoff,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_detailed_takeoff_aircraft())
        ),
        "",
        "",
    ),
    (
        "EvalLanding",
        mission.eval_landing,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_landing_aircraft())
        ),
        "",
        "",
    ),
    (
        "EvalCruise",
        mission.eval_cruise,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_cruise_aircraft())
        ),
        "",
        "",
    ),
    (
        "EvalCruiseBRE",
        mission.eval_cruise_breguet,
        lambda fixtures: init_mission_history(
            cruise_bre_aircraft(fixtures.make_cruise_breguet_aircraft())
        ),
        CRUISE_BRE_PREP,
        CRUISE_BRE_POST,
    ),
    (
        "EvalClimb",
        mission.eval_climb,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_climb_aircraft())
        ),
        "",
        "",
    ),
    (
        "EvalDescent",
        mission.eval_descent,
        lambda fixtures: init_mission_history(
            complete_segment_profile(fixtures.make_descent_aircraft())
        ),
        "",
        "",
    ),
]

@pytest.fixture(scope="module")
def mission_fixtures():
    """Load the ordinary MissionSegs fixture builders without package-name clashes."""

    spec = importlib.util.spec_from_file_location(
        "fast_python_mission_segment_fixtures",
        SEGMENT_FIXTURE_PATH,
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return fixtures


def test_gravity_and_standard_atmosphere_match_matlab(matlab_wrapper):
    """Compare Gravity and StdAtm against MATLAB at representative altitudes."""

    wrapper, _ = matlab_wrapper

    for altitude in (0, 1000, 10000, 50000):
        wrapper.engine.evalc(
            f"""
            matlab_gravity = MissionSegsPkg.Gravity({altitude});
            [matlab_temp, matlab_press, matlab_rho] = MissionSegsPkg.StdAtm({altitude});
            """,
            nargout=1,
        )
        expected = (
            wrapper.engine.workspace["matlab_temp"],
            wrapper.engine.workspace["matlab_press"],
            wrapper.engine.workspace["matlab_rho"],
        )

        assert np.isclose(gravity(altitude), wrapper.engine.workspace["matlab_gravity"])
        np.testing.assert_allclose(standard_atmosphere(altitude), expected)


def test_standard_atmosphere_array_shape_matches_matlab(matlab_wrapper):
    """Compare StdAtm vector input shape against MATLAB."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    altitudes = np.asarray([[0, 11000, 20000]])
    actual = build_json_data(standard_atmosphere(altitudes))
    wrapper.engine.evalc(
        """
        [matlab_temp, matlab_press, matlab_rho] = ...
            MissionSegsPkg.StdAtm([0, 11000, 20000]);
        matlab_atmosphere = {matlab_temp, matlab_press, matlab_rho};
        """,
        nargout=1,
    )
    expected = wrapper_build_json_data(
        wrapper._to_python_data(wrapper.engine.workspace["matlab_atmosphere"])
    )
    assert_parity(actual, expected, "Atmosphere")


@pytest.mark.parametrize(
    "velocity_type, velocity",
    [
        ("TAS", 120),
        ("EAS", 120),
        ("Mach", 0.4),
    ],
)
def test_compute_flight_conditions_matches_matlab(matlab_wrapper, velocity_type, velocity):
    """Compare ComputeFltCon for TAS, EAS, and Mach input modes."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    actual = build_json_data(
        mission.compute_flight_conditions(1000, 0, velocity_type, velocity)
    )
    wrapper.engine.evalc(
        f"""
        [eas, tas, mach, temp, press, rho, visc] = ...
            MissionSegsPkg.ComputeFltCon(1000, 0, "{velocity_type}", {velocity});
        matlab_flight_conditions = [eas, tas, mach, temp, press, rho, visc];
        """,
        nargout=1,
    )
    expected = wrapper_build_json_data(
        wrapper._to_python_data(wrapper.engine.workspace["matlab_flight_conditions"])
    )
    assert_parity(actual, expected, "FlightConditions")


def test_process_profile_matches_matlab(matlab_wrapper, mission_fixtures):
    """Compare ProcessProfile profile indexing and target annotations."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    aircraft = mission_fixtures.make_full_mission_aircraft()
    actual = build_json_data(mission.process_profile(deepcopy(aircraft))["Mission"]["Profile"])
    wrapper.engine.evalc(
        f"""
        matlab_aircraft = {wrapper._to_matlab_literal(aircraft)};
        matlab_aircraft = MissionSegsPkg.ProcessProfile(matlab_aircraft);
        """,
        nargout=1,
    )
    expected = wrapper_build_json_data(
        wrapper._to_python_data(wrapper.engine.workspace["matlab_aircraft"])[
            "Mission"
        ]["Profile"]
    )
    assert_parity(actual, expected, "Profile")


@pytest.mark.parametrize(
    "matlab_name, python_evaluator, aircraft_factory, extra_prep, extra_post",
    MISSION_SEGMENT_CASES,
)
def test_mission_segment_evaluator_matches_matlab(
    matlab_wrapper,
    mission_fixtures,
    matlab_name,
    python_evaluator,
    aircraft_factory,
    extra_prep,
    extra_post,
):
    """Compare one MissionSegsPkg evaluator against MATLAB FAST output."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    aircraft = aircraft_factory(mission_fixtures)
    actual = build_json_data(python_evaluator(aircraft))
    expected = matlab_eval_aircraft(
        wrapper,
        wrapper_build_json_data,
        aircraft,
        matlab_name,
        extra_prep,
        extra_post,
    )
    assert_parity(actual, expected, "Aircraft")


def test_fly_mission_matches_matlab(matlab_wrapper, mission_fixtures):
    """Compare FlyMission dispatch and cruise-distance iteration against MATLAB."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    aircraft = mission.process_profile(mission_fixtures.make_full_mission_aircraft())
    aircraft = init_mission_history(aircraft)
    actual = build_json_data(mission.fly_mission(aircraft))
    expected = matlab_eval_aircraft(
        wrapper,
        wrapper_build_json_data,
        aircraft,
        "FlyMission",
        "",
        "",
    )
    assert_parity(actual, expected, "Aircraft")


def matlab_eval_aircraft(
    wrapper,
    wrapper_build_json_data,
    aircraft,
    matlab_name,
    extra_prep,
    extra_post,
):
    """Evaluate one MATLAB MissionSegsPkg function and return comparable JSON."""

    wrapper.engine.evalc(
        f"""
        matlab_aircraft = {wrapper._to_matlab_literal(aircraft)};
        {MATLAB_SPLIT_PREP}
        {extra_prep}
        matlab_aircraft = MissionSegsPkg.{matlab_name}(matlab_aircraft);
        {extra_post}
        {MATLAB_SPLIT_POST}
        """,
        nargout=1,
    )
    matlab_aircraft = wrapper._to_python_data(wrapper.engine.workspace["matlab_aircraft"])
    return wrapper_build_json_data(matlab_aircraft)


def complete_segment_profile(aircraft):
    """Fill no-op profile fields that MATLAB reads even when unused."""

    aircraft = deepcopy(aircraft)
    profile = aircraft["Mission"]["Profile"]
    specs = aircraft["Specs"]
    power = specs.setdefault("Power", {})
    windmill = power.setdefault("Windmill", {})
    nseg = len(profile.get("SegPts", [1]))
    defaults = {
        "AltBeg": 0,
        "AltEnd": 0,
        "VelBeg": 0,
        "VelEnd": 0,
        "TypeBeg": "TAS",
        "TypeEnd": "TAS",
        "ClbRate": float("nan"),
    }

    for key, value in defaults.items():
        if key not in profile:
            profile[key] = [value for _ in range(nseg)]

    for key in ("Tko", "Clb", "Crs", "Des", "Lnd"):
        windmill.setdefault(key, 0)

    aero = specs.get("Aero", {})

    if isinstance(aero, dict) and "L_D" in aero and "S" not in aero and "W_S" in aero:
        aero["S"] = specs["Weight"]["MTOW"] / aero["W_S"]["SLS"]

    return aircraft


def cruise_bre_aircraft(aircraft):
    """Add the deprecated CruiseBRE architecture field MATLAB still expects."""

    aircraft = complete_segment_profile(aircraft)
    aircraft["Specs"]["Propulsion"]["Arch"] = "AC"
    return aircraft


def assert_parity(actual, expected, path):
    """Fail with a compact recursive comparison report."""

    failures, compared = compare_json_value(actual, expected, path)
    assert compared > 0

    if failures:
        preview = "\n".join(failures[:25])
        pytest.fail(f"{path} parity failures:\n{preview}")
