"""Generic particle filter (Sequential Importance Resampling)."""
import numpy as np

from .state_utils import as_matrix, from_vector, to_vector


class ParticleFilter:
    """
    A basic Sequential Importance Resampling (SIR) particle filter.

    Like the other filters in this module, particles are tracked internally as flat
    vectors and the aggregate state estimate is exposed in whichever representation
    (tm, numpy array, or float) the filter was initialized with. Process and
    measurement models operate on individual particles in that same native
    representation, so a tm-based motion or observation model can freely use tm's
    overloaded operators.

    Note that the state estimate is a weighted arithmetic mean of the particles'
    vector representations; for tm particles that spread across a wide range of
    orientations this is only an approximation (not a proper rotational average), but
    is fine for tracking that stays reasonably localized.
    """

    def __init__(self, initial_state, initial_covariance=1.0, num_particles=200, seed=None):
        """
        Create a new Particle Filter, sampling particles from a Gaussian around
        `initial_state`.

        Args:
            initial_state: tm, np.ndarray, list/tuple, or float initial state estimate.
            initial_covariance: covariance (n, n), or scalar shorthand, used to spread
                the initial particles around `initial_state`.
            num_particles (int): number of particles to track.
            seed: optional seed (int, or an `np.random.Generator`) for reproducibility.
        """
        self._template = initial_state
        self.n = len(to_vector(initial_state))
        self.num_particles = num_particles
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        mean = to_vector(initial_state)
        covariance = as_matrix(initial_covariance, self.n)
        self.particles = self.rng.multivariate_normal(mean, covariance, size=num_particles)
        self.weights = np.full(num_particles, 1.0 / num_particles)

    @property
    def state(self):
        """Return the weighted-mean state estimate, in the filter's native representation."""
        mean = np.average(self.particles, axis=0, weights=self.weights)
        return from_vector(mean, self._template)

    def predict(self, f, u=None, process_noise=1e-4):
        """
        Propagate every particle forward through a process model, then add process noise.

        Args:
            f: process model. Called as `f(state)`, or `f(state, u)` if `u` is given,
                where `state` is in the filter's native representation; must return a
                new state in that same representation. May be nonlinear and/or itself
                stochastic.
            u: optional control input passed through to `f`.
            process_noise: covariance (n, n), or scalar shorthand, of the Gaussian
                noise added to each particle after propagation.

        Returns:
            The (pre-update) weighted-mean state estimate, in the filter's native representation.
        """
        Q = as_matrix(process_noise, self.n)
        noise = self.rng.multivariate_normal(np.zeros(self.n), Q, size=self.num_particles)
        for i in range(self.num_particles):
            particle_state = from_vector(self.particles[i], self._template)
            propagated = f(particle_state, u) if u is not None else f(particle_state)
            self.particles[i] = to_vector(propagated)
        self.particles = self.particles + noise
        return self.state

    def update(self, measurement, h, measurement_noise=1.0):
        """
        Reweight particles by how well they explain a new measurement, then resample.

        Args:
            measurement: tm, np.ndarray, list/tuple, or float measurement.
            h: measurement model. Called as `h(state)`, where `state` is in the
                filter's native representation; must return a predicted measurement in
                the same representation as `measurement`.
            measurement_noise: covariance (m, m), or scalar shorthand, describing
                measurement noise; used to weight particles by Gaussian likelihood.

        Returns:
            The updated (post-resampling) state estimate, in the filter's native representation.
        """
        z = to_vector(measurement)
        m = len(z)
        R = as_matrix(measurement_noise, m)
        R_inv = np.linalg.inv(R)
        normalizer = 1.0 / np.sqrt((2 * np.pi) ** m * np.linalg.det(R))

        likelihoods = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            particle_state = from_vector(self.particles[i], self._template)
            predicted_z = to_vector(h(particle_state))
            residual = z - predicted_z
            likelihoods[i] = normalizer * np.exp(-0.5 * residual @ R_inv @ residual)

        self.weights = self.weights * likelihoods
        weight_sum = np.sum(self.weights)
        if weight_sum <= 0:
            # Degenerate case: every particle had ~zero likelihood under this
            # measurement. Reset to uniform rather than dividing by zero, so
            # filtering can continue (if noisily) instead of raising.
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        else:
            self.weights = self.weights / weight_sum

        self._resample_if_needed()
        return self.state

    def _resample_if_needed(self, threshold=0.5):
        """
        Resample particles (systematic resampling) if the effective sample size drops
        below `threshold * num_particles`, to avoid weight degeneracy.

        Args:
            threshold (float): fraction of `num_particles` below which resampling triggers.
        """
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size >= threshold * self.num_particles:
            return

        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles
        cumulative_weights = np.cumsum(self.weights)
        cumulative_weights[-1] = 1.0  # Guard against floating point drift
        indexes = np.searchsorted(cumulative_weights, positions)

        self.particles = self.particles[indexes]
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
