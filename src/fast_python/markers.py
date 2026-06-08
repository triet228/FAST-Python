# src/fast_python/markers.py

"""Marker objects used to preserve MATLAB-oriented FAST JSON inputs.

The wrapper needs to distinguish ordinary JSON arrays from MATLAB row vectors
and trusted MATLAB expression strings. Marker classes keep that information
available to Python code without evaluating arbitrary MATLAB syntax.
"""


class MatlabExpression:
    """Store trusted MATLAB expression text from wrapper-compatible JSON.

    Inputs:
        value: Expression text such as an engine specification package name.

    Outputs:
        Marker object that round-trips through build_json_data().

    Assumptions:
        The current Python backend does not evaluate MATLAB expressions; it
        preserves them so inputs remain byte-compatible with wrapper fixtures
        and future algorithm ports can resolve them explicitly.
    """

    def __init__(self, value):
        self.value = value


class MatlabExpressionError(ValueError):
    """Report unsupported MATLAB expression markers."""


class MatlabRow:
    """Mark a one-dimensional sequence that was intended as a MATLAB row.

    Inputs:
        value: Sequence loaded from a _matlab_row JSON marker.

    Outputs:
        Marker object that round-trips through build_json_data().

    Assumptions:
        Row orientation matters for some propulsion graph metadata. Preserving
        this marker keeps Python inputs compatible with the wrapper contract.
    """

    def __init__(self, value):
        self.value = value


def matlab_expr(value):
    """Return a MATLAB expression marker for compatibility with wrapper specs."""

    return MatlabExpression(value)


def parse_constant_matrix_expression(value):
    """Parse trusted constant MATLAB matrix function handles.

    Inputs:
        value: MatlabExpression or expression text such as
            ``@()[1,0;0,1]``.

    Outputs:
        Nested Python list containing numeric matrix rows.

    Assumptions:
        This intentionally supports only the constant matrix form used in
        wrapper propulsion graph fixtures. Expressions with variables,
        function calls, or arithmetic remain unsupported.
    """

    if isinstance(value, MatlabExpression):
        value = value.value

    if not isinstance(value, str):
        raise MatlabExpressionError("MATLAB expression must be text.")

    text = value.strip()

    if text.startswith("@()"):
        text = text[3:].strip()

    if not text.startswith("[") or not text.endswith("]"):
        raise MatlabExpressionError(
            f"Unsupported MATLAB expression marker: {value!r}."
        )

    body = text[1:-1].strip()

    if not body:
        return []

    rows = []

    for row_text in body.split(";"):
        row = []

        for item in row_text.replace(",", " ").split():
            try:
                row.append(float(item))
            except ValueError as error:
                raise MatlabExpressionError(
                    f"Unsupported MATLAB matrix token: {item!r}."
                ) from error

        rows.append(row)

    width = len(rows[0]) if rows else 0

    if any(len(row) != width for row in rows):
        raise MatlabExpressionError("MATLAB matrix rows must have equal length.")

    return rows
