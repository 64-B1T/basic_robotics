"""
Conversion helpers that let filters operate generically over basic_robotics tm
objects, numpy arrays, and raw floats/doubles.

Every filter in this package tracks state internally as a flat numpy vector, since
that is what the underlying linear algebra needs, and converts back to whichever
representation (tm, ndarray, or scalar) it was originally given whenever the state
is read back out. These functions are the boundary that conversion happens at.
"""
import numpy as np

from ..general import tm


def to_vector(state) -> np.ndarray:
    """
    Flatten a state into a 1D float vector.

    Args:
        state: tm, np.ndarray, list/tuple, or scalar (int/float) to flatten.

    Returns:
        np.ndarray: 1D float vector representation of state.
    """
    if isinstance(state, tm):
        return state.gTAA().flatten().astype(float)
    if isinstance(state, np.ndarray):
        return state.flatten().astype(float)
    if isinstance(state, (list, tuple)):
        return np.array(state, dtype=float).flatten()
    if np.isscalar(state):
        return np.array([float(state)])
    raise TypeError(f'Unsupported state type for filtering: {type(state)}')


def from_vector(vector, reference):
    """
    Reconstruct a state in the same representation as `reference` from a flat vector.

    Args:
        vector: 1D (or column) numpy vector to convert back.
        reference: tm, np.ndarray, list/tuple, or scalar whose type/shape is matched.

    Returns:
        A tm, np.ndarray, or float matching the type of `reference`.
    """
    vector = np.asarray(vector, dtype=float).flatten()
    if isinstance(reference, tm):
        return tm(vector.reshape((6, 1)).copy())
    if isinstance(reference, np.ndarray):
        return vector.reshape(reference.shape).copy()
    if isinstance(reference, (list, tuple)):
        return vector.reshape(np.array(reference).shape).copy()
    if np.isscalar(reference):
        return float(vector[0])
    raise TypeError(f'Unsupported reference type for filtering: {type(reference)}')


def as_matrix(value, rows, cols=None) -> np.ndarray:
    """
    Coerce a scalar or array-like into a `rows` x `cols` numpy matrix.

    A scalar is expanded to `value * identity(rows)`, the common shorthand for an
    isotropic noise or gain matrix (requires rows == cols). Anything else is passed
    through `np.atleast_2d` unchanged, so an explicit (possibly non-square) matrix
    can always be supplied instead.

    Args:
        value: scalar, or array-like matrix. May be None, in which case None passes
            through (used by filters to detect "no override supplied").
        rows: number of rows the resulting matrix should have.
        cols: number of columns. Defaults to `rows`, i.e. a square matrix.

    Returns:
        np.ndarray: `rows` x `cols` matrix, or None if `value` is None.
    """
    if value is None:
        return None
    if cols is None:
        cols = rows
    if np.isscalar(value):
        if rows != cols:
            raise ValueError(
                    'A scalar shorthand can only be used for a square matrix '
                    f'(requested {rows}x{cols}); supply an explicit matrix instead.')
        return np.eye(rows) * float(value)
    return np.atleast_2d(np.asarray(value, dtype=float))


def numerical_jacobian(vector_function, x0, delta=1e-6) -> np.ndarray:
    """
    Estimate the Jacobian of `vector_function` at `x0` via central differences.

    Args:
        vector_function: function mapping a 1D vector to a 1D vector.
        x0: 1D vector at which to linearize.
        delta (float): perturbation size used for the central difference.

    Returns:
        np.ndarray: (m, n) Jacobian, where m is the output dimension of
        `vector_function` and n = len(x0).
    """
    x0 = np.asarray(x0, dtype=float).flatten()
    f0 = np.asarray(vector_function(x0), dtype=float).flatten()
    jacobian = np.zeros((len(f0), len(x0)))
    for i in range(len(x0)):
        perturbation = np.zeros(len(x0))
        perturbation[i] = delta
        f_plus = np.asarray(vector_function(x0 + perturbation), dtype=float).flatten()
        f_minus = np.asarray(vector_function(x0 - perturbation), dtype=float).flatten()
        jacobian[:, i] = (f_plus - f_minus) / (2 * delta)
    return jacobian
