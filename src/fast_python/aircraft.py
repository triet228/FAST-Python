# src/fast_python/aircraft.py

"""Aircraft input normalization helpers."""

from copy import deepcopy

from fast_python.markers import MatlabRow


def prepare_aircraft(aircraft):
    """Normalize aircraft input the same way the wrapper does before FAST."""

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
