# src/fast_python/regression.py

"""Regression utilities ported from FAST RegressionPkg."""

from copy import deepcopy

import numpy as np


MISSING = object()


class RegressionError(ValueError):
    """Report invalid regression database inputs."""


def search_db(main_struct, substruct_list, desired_value=MISSING):
    """Search a nested database dictionary for a parameter path.

    Inputs:
        main_struct: Dictionary whose top-level keys are aircraft or engine
            names and whose values are nested dictionaries.
        substruct_list: String or sequence of path fields to read.
        desired_value: Optional value used to filter returned entries.

    Outputs:
        A tuple of filtered dictionary and output rows. Each row contains the
        entry name and the requested value.

    Assumptions:
        This mirrors RegressionPkg.SearchDB for one-to-four-level paths, but it
        accepts any path length because Python dictionary walking is cheap.
    """

    path = normalize_path(substruct_list)
    output_rows = []

    try:
        for name, entry in main_struct.items():
            value = get_path(entry, path)

            if desired_value is MISSING or values_equal(value, desired_value):
                output_rows.append([name, value])
    except KeyError as error:
        joined = ", ".join(path)
        raise RegressionError(
            f"Invalid parameter search path. <{joined}> is not found."
        ) from error

    new_struct = {
        name: deepcopy(main_struct[name])
        for name, _ in output_rows
    }
    return new_struct, output_rows


def square_exp_kernel(x_value, y_value, hyperparams):
    """Evaluate FAST's squared exponential covariance kernel."""

    x_value = as_2d(x_value)
    y_value = as_2d(y_value)
    hyperparams = as_2d(hyperparams)

    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = -0.3 * np.sum(
            (x_value - y_value) ** 2 / hyperparams[:, :-1],
            axis=1,
        )

    return hyperparams[:, -1] * np.exp(exponent)


def prior_calculation(data_struct, io_space):
    """Return the mean of non-NaN output data for a regression path."""

    _, prior_data = search_db(data_struct, io_space[-1])
    values = numeric_column([row[1] for row in prior_data])
    values = values[~np.isnan(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def build_data(data_struct, io_space, weights):
    """Return regression data matrix and weighted hyperparameters."""

    columns = []

    for path in io_space:
        _, rows = search_db(data_struct, path)
        columns.append(numeric_column([row[1] for row in rows]))

    data_matrix = np.column_stack(columns)
    data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]
    hyperparams = np.zeros(len(io_space))

    for index, path in enumerate(io_space):
        _, rows = search_db(data_struct, path)
        values = numeric_column([row[1] for row in rows])
        values = values[~np.isnan(values)]
        hyperparams[index] = sample_variance(values)

    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights) * len(weights)
    hyperparams[:-1] = hyperparams[:-1] / weights
    return data_matrix, hyperparams


def reg_processing(data_struct, io_space, prior, weights):
    """Build regression matrices and inverse covariance term."""

    data_matrix, hyperparams = build_data(data_struct, io_space, weights)
    prior = as_vector(prior)
    noise_variance = (np.mean(prior) * 5.0e-2) ** 2
    size = data_matrix.shape[0]
    kbarbar = np.zeros((size, size))

    for irow in range(size):
        for jrow in range(size):
            kbarbar[irow, jrow] = square_exp_kernel(
                data_matrix[irow, :-1],
                data_matrix[jrow, :-1],
                hyperparams,
            )[0]

    inverse_term = np.linalg.inv(kbarbar + noise_variance * np.eye(size))
    return data_matrix, hyperparams, inverse_term


def nlgpr(data_struct, io_space, target, weights=None, prior=None, preprocessing=None):
    """Run FAST's non-linear Gaussian process regression.

    Inputs:
        data_struct: Historical aircraft or engine database dictionary.
        io_space: Sequence of input paths followed by one output path.
        target: One or more rows of known input values.
        weights: Optional relevance weights for the input dimensions.
        prior: Optional prior mean values for each target row.
        preprocessing: Optional dictionary containing DataMatrix, HyperParams,
            and InverseTerm from reg_processing().

    Outputs:
        A tuple of posterior mean and posterior variance arrays.

    Assumptions:
        This follows RegressionPkg.NLGPR, including the same covariance kernel
        and noise-augmented inverse term.
    """

    target = target_matrix(target)
    ninputs = len(io_space) - 1

    if weights is None:
        weights = np.ones(ninputs)

    if prior is None:
        prior = np.ones(target.shape[0]) * prior_calculation(data_struct, io_space)
    else:
        prior = as_vector(prior)

    if preprocessing is None:
        data_matrix, hyperparams, inverse_term = reg_processing(
            data_struct,
            io_space,
            prior,
            weights,
        )
    else:
        data_matrix = preprocessing["DataMatrix"]
        hyperparams = preprocessing["HyperParams"]
        inverse_term = preprocessing["InverseTerm"]

    post_mu = np.zeros(target.shape[0])
    post_var = np.zeros(target.shape[0])

    for index in range(len(post_mu)):
        repeated_target = np.tile(target[index, :], (data_matrix.shape[0], 1))
        repeated_hyper = np.tile(hyperparams, (data_matrix.shape[0], 1))
        kbarstar = square_exp_kernel(
            data_matrix[:, :-1],
            repeated_target,
            repeated_hyper,
        ).reshape(1, -1)
        centered = data_matrix[:, -1] - prior[index]
        post_mu[index] = prior[index] + (kbarstar @ inverse_term @ centered)[0]
        post_var[index] = (
            square_exp_kernel(target[index, :], target[index, :], hyperparams)[0]
            - (kbarstar @ inverse_term @ kbarstar.T)[0, 0]
        )

    return post_mu, post_var


def vary_user_inputs(aircraft, aircraft_class):
    """Classify regression inputs as known or unknown from an aircraft dict."""

    values = {
        "vto": get_path(aircraft, ["Specs", "Performance", "Vels", "Tko"]),
        "vcr": get_path(aircraft, ["Specs", "Performance", "Vels", "Crs"]),
        "hcr": get_path(aircraft, ["Specs", "Performance", "Alts", "Crs"]),
        "mtow": get_path(aircraft, ["Specs", "Weight", "MTOW"]),
        "ws": get_path(aircraft, ["Specs", "Aero", "W_S", "SLS"]),
        "ldcr": get_path(aircraft, ["Specs", "Aero", "L_D", "Crs"]),
    }
    names = {
        "vto": ["Specs", "Performance", "Vels", "Tko"],
        "vcr": ["Specs", "Performance", "Vels", "Crs"],
        "hcr": ["Specs", "Performance", "Alts", "Crs"],
        "mtow": ["Specs", "Weight", "MTOW"],
        "ws": ["Specs", "Aero", "W_S", "SLS"],
        "ldcr": ["Specs", "Aero", "L_D", "Crs"],
    }

    if aircraft_class == "Turbofan":
        values["tw"] = get_path(aircraft, ["Specs", "Propulsion", "T_W", "SLS"])
        values["tsls"] = get_path(aircraft, ["Specs", "Propulsion", "Thrust", "SLS"])
        names["tw"] = ["Specs", "Propulsion", "T_W", "SLS"]
        names["tsls"] = ["Specs", "Propulsion", "Thrust", "Crs"]
    elif aircraft_class == "Turboprop":
        values["pw"] = get_path(aircraft, ["Specs", "Power", "P_W", "SLS"])
        values["psls"] = get_path(aircraft, ["Specs", "Power", "SLS"])
        names["pw"] = ["Specs", "Power", "P_W", "SLS"]
        names["psls"] = ["Specs", "Power", "Crs"]

    known_names = []
    known_values = []
    unknown = []

    for key, value in values.items():
        if is_nan(value):
            unknown.append(names[key])
        else:
            known_names.append(names[key])
            known_values.append(value)

    unknown.append(["Specs", "Weight", "Fuel"])
    return {
        "names": known_names,
        "values": known_values,
    }, unknown


def get_path(value, path):
    """Return a nested dictionary value for path."""

    current = value

    for key in normalize_path(path):
        current = current[key]

    return current


def normalize_path(path):
    """Return a search path as a list of strings."""

    if isinstance(path, str):
        return [path]

    return [str(item) for item in path]


def values_equal(actual, expected):
    """Return True when two database values match MATLAB isequal semantics."""

    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected

    return actual == expected


def numeric_column(values):
    """Return values as a numeric column vector."""

    return np.asarray([numeric_scalar(value) for value in values], dtype=float)


def numeric_scalar(value):
    """Return the first numeric scalar represented by value."""

    array = np.asarray(value)

    if array.size == 0:
        return np.nan

    return float(array.reshape(-1)[0])


def sample_variance(values):
    """Return MATLAB-style sample variance for one numeric vector."""

    if len(values) < 2:
        return 0.0

    return float(np.var(values, ddof=1))


def target_matrix(target):
    """Return target as a two-dimensional numeric matrix."""

    array = np.asarray(target, dtype=float)

    if array.ndim == 0:
        return array.reshape(1, 1)

    if array.ndim == 1:
        return array.reshape(1, -1)

    return array


def as_vector(value):
    """Return value as a one-dimensional numeric array."""

    return np.asarray(value, dtype=float).reshape(-1)


def as_2d(value):
    """Return value as a two-dimensional numeric array."""

    array = np.asarray(value, dtype=float)

    if array.ndim == 1:
        return array.reshape(1, -1)

    return array


def is_nan(value):
    """Return True when value is a NaN scalar."""

    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


SearchDB = search_db
SquareExKernel = square_exp_kernel
PriorCalculation = prior_calculation
BuildData = build_data
RegProcessing = reg_processing
NLGPR = nlgpr
VaryUserInputs = vary_user_inputs
