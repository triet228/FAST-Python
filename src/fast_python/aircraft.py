# src/fast_python/aircraft.py

"""Aircraft input normalization helpers.

The wrapper accepts compact JSON fields that MATLAB expands before running
FAST. This module performs the same expansion for Python callers so downstream
modules can work with one PropArch dictionary shape.
"""

from copy import deepcopy

from fast_python.markers import MatlabRow


def prepare_aircraft(aircraft):
    """Normalize aircraft input the same way the wrapper does before FAST.

    Inputs:
        aircraft: FAST aircraft dictionary, or any other value passed through
            by callers that are still validating their input type.

    Outputs:
        A deep-copied aircraft dictionary when normalization is possible.
        Non-dictionaries and aircraft without Specs.Propulsion are returned
        unchanged to preserve the wrapper's forgiving pre-validation behavior.

    Side effects:
        None on the caller's object. For custom "O" architectures, the JSON
        PropArchGraph is promoted into Specs.Propulsion.PropArch and MATLAB row
        markers are restored for SrcType and TrnType so architecture matrices
        keep their intended row-vector meaning.
    """

    if not isinstance(aircraft, dict):
        return aircraft

    aircraft = deepcopy(aircraft)

    try:
        propulsion = aircraft["Specs"]["Propulsion"]
    except KeyError:
        return aircraft

    prop_arch = propulsion.get("PropArch")

    if not isinstance(prop_arch, str):
        return aircraft

    arch_type = prop_arch.upper()

    if arch_type == "O":
        graph = propulsion.get("PropArchGraph")

        if graph is None:
            raise ValueError('PropArchGraph is required when "PropArch" is "O".')

        prop_arch = deepcopy(graph)
        prop_arch["Type"] = "O"

        for field_name in ("SrcType", "TrnType"):
            if field_name in prop_arch:
                value = prop_arch[field_name]

                if (
                    not isinstance(value, MatlabRow)
                    and (isinstance(value, list) or isinstance(value, tuple))
                ):
                    prop_arch[field_name] = MatlabRow(prop_arch[field_name])

        propulsion["PropArch"] = prop_arch
        del propulsion["PropArchGraph"]
        return aircraft

    propulsion["PropArch"] = {"Type": arch_type}
    propulsion.pop("PropArchGraph", None)
    return aircraft
