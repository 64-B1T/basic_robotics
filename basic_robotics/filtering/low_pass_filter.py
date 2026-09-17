"""Exponential (first-order) low pass filter."""
import numpy as np


class LowPassFilter:
    """
    A simple exponential low pass filter.

    Operates generically on any type supporting scalar multiplication and addition,
    which includes basic_robotics tm objects, numpy arrays, and raw floats/doubles -
    no vector conversion is needed since tm and ndarray both already overload `*`
    and `+` appropriately.
    """

    def __init__(self, alpha: float, initial_state=None):
        """
        Create a new Low Pass Filter.

        Args:
            alpha (float): smoothing factor in [0, 1]. Higher values track new
                measurements more closely; lower values smooth more aggressively.
            initial_state: optional tm, np.ndarray, list/tuple, or float to seed the
                filter with. If omitted, the filter initializes itself from the first
                measurement passed to `update`.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha
        self.state = self._coerce(initial_state) if initial_state is not None else None

    def _coerce(self, value):
        """Convert plain lists/tuples to numpy arrays; leave tm/ndarray/scalars as-is."""
        if isinstance(value, (list, tuple)):
            return np.array(value, dtype=float)
        return value

    def update(self, measurement):
        """
        Fold a new measurement into the filter and return the updated state.

        Args:
            measurement: tm, np.ndarray, list/tuple, or float measurement.

        Returns:
            The filtered state, in the same representation as `measurement`.
        """
        measurement = self._coerce(measurement)
        if self.state is None:
            self.state = measurement
        else:
            self.state = self.state * (1 - self.alpha) + measurement * self.alpha
        return self.state

    def reset(self, initial_state=None):
        """
        Clear the filter state.

        Args:
            initial_state: optional new seed state. If omitted, the filter goes back
                to initializing itself from the next measurement it receives.
        """
        self.state = self._coerce(initial_state) if initial_state is not None else None
