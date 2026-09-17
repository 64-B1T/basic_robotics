"""
Time-parametrization of geometric paths.

RRTStar and IKPath produce purely geometric paths: a sequence of waypoints with
no notion of *when* each one is reached. This module turns such a path into a
time-stamped trajectory that respects per-axis velocity and acceleration limits
(a trapezoidal profile), or additionally jerk limits (a jerk-limited S-curve
profile), so it can be handed directly to a real or simulated controller.

Two path shapes are supported:
    - `JointTrajectory` for waypoints in R^n (e.g. joint angles), retiming each
      axis independently and synchronizing segment durations across axes.
    - `CartesianTrajectory` for `tm` waypoints, retiming along the screw motion
      between consecutive poses (the same construction `mr.ScrewTrajectory`
      uses, but timed by velocity/acceleration/jerk limits rather than a fixed
      duration).
"""
import numpy as np

from ..general import tm
from ..modern_robotics_numba import modern_high_performance as mr


def _check_limits(v_max, a_max):
    if v_max is None or a_max is None or v_max <= 0 or a_max <= 0 or \
            not np.isfinite(v_max) or not np.isfinite(a_max):
        raise ValueError(
                'Time-parametrization requires finite, positive velocity and '
                'acceleration limits, got v_max=%r, a_max=%r. Set them explicitly '
                '(e.g. Arm.setJointProperties(max_vels=..., max_accels=...)).'
                % (v_max, a_max))


class TrapezoidalProfile:
    """
    Rest-to-rest trapezoidal (velocity-limited) motion profile covering `distance`
    starting and ending at zero velocity, subject to a peak velocity `v_max` and a
    constant acceleration magnitude `a_max`. Falls back to a triangular profile
    (never reaching `v_max`) when `distance` is too short.

    Args:
        distance (float): signed distance to cover
        v_max (float): maximum speed, must be finite and positive
        a_max (float): maximum acceleration magnitude, must be finite and positive
    """

    def __init__(self, distance, v_max, a_max):
        _check_limits(v_max, a_max)
        self.sign = 1.0 if distance >= 0 else -1.0
        self.distance = abs(float(distance))
        self.v_max = float(v_max)
        self.a_max = float(a_max)

        if self.distance == 0.0:
            self.t_accel = self.t_flat = self.v_peak = self.duration = 0.0
            return

        t_accel = self.v_max / self.a_max
        d_accel = 0.5 * self.a_max * t_accel ** 2

        if 2 * d_accel >= self.distance:
            # Triangular profile: v_max is never reached.
            t_accel = np.sqrt(self.distance / self.a_max)
            v_peak = self.a_max * t_accel
            t_flat = 0.0
        else:
            v_peak = self.v_max
            t_flat = (self.distance - 2 * d_accel) / self.v_max

        self.t_accel = t_accel
        self.t_flat = t_flat
        self.v_peak = v_peak
        self.duration = 2 * t_accel + t_flat

    def position(self, t):
        if self.duration == 0.0:
            return 0.0
        t = min(max(t, 0.0), self.duration)
        t_dec_start = self.t_accel + self.t_flat
        if t <= self.t_accel:
            s = 0.5 * self.a_max * t ** 2
        elif t <= t_dec_start:
            s = 0.5 * self.a_max * self.t_accel ** 2 + self.v_peak * (t - self.t_accel)
        else:
            td = t - t_dec_start
            s_dec_start = 0.5 * self.a_max * self.t_accel ** 2 + self.v_peak * self.t_flat
            s = s_dec_start + self.v_peak * td - 0.5 * self.a_max * td ** 2
        return self.sign * s

    def velocity(self, t):
        if self.duration == 0.0:
            return 0.0
        t = min(max(t, 0.0), self.duration)
        t_dec_start = self.t_accel + self.t_flat
        if t <= self.t_accel:
            v = self.a_max * t
        elif t <= t_dec_start:
            v = self.v_peak
        else:
            v = self.v_peak - self.a_max * (t - t_dec_start)
        return self.sign * v

    def acceleration(self, t):
        if self.duration == 0.0:
            return 0.0
        t = min(max(t, 0.0), self.duration)
        t_dec_start = self.t_accel + self.t_flat
        if t < self.t_accel:
            a = self.a_max
        elif t < t_dec_start:
            a = 0.0
        else:
            a = -self.a_max
        return self.sign * a


def _accel_phase(v_peak, a_max, j_max):
    """
    Duration/shape of a rest-to-`v_peak` jerk-limited acceleration ramp.

    Returns (Tj, Ta, a_reached): Tj is the duration of each of the two jerk
    segments, Ta is the total ramp duration, and a_reached is the peak
    acceleration actually attained (== a_max unless v_peak is too small).
    """
    if a_max ** 2 / j_max <= v_peak:
        Tj = a_max / j_max
        Ta = Tj + v_peak / a_max
        a_reached = a_max
    else:
        Tj = np.sqrt(v_peak / j_max)
        Ta = 2 * Tj
        a_reached = j_max * Tj
    return Tj, Ta, a_reached


class SCurveProfile:
    """
    Rest-to-rest, jerk-limited ("S-curve") motion profile covering `distance`,
    subject to peak velocity `v_max`, peak acceleration magnitude `a_max`, and
    peak jerk magnitude `j_max`. Reduces to a triangular acceleration ramp when
    `a_max` can't be reached, and to a peak-velocity ramp with no cruise phase
    when `v_max` can't be reached (found by bisection, since the reduced peak
    velocity has no simple closed form once the accel-ramp case also changes).

    Args:
        distance (float): signed distance to cover
        v_max (float): maximum speed, must be finite and positive
        a_max (float): maximum acceleration magnitude, must be finite and positive
        j_max (float): maximum jerk magnitude, must be finite and positive
    """

    def __init__(self, distance, v_max, a_max, j_max):
        _check_limits(v_max, a_max)
        if j_max is None or j_max <= 0 or not np.isfinite(j_max):
            raise ValueError('SCurveProfile requires a finite, positive j_max, got %r' % (j_max,))

        self.sign = 1.0 if distance >= 0 else -1.0
        self.distance = abs(float(distance))
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.j_max = float(j_max)

        if self.distance == 0.0:
            self.duration = 0.0
            self._phases = []
            return

        _, Ta_full, _ = _accel_phase(self.v_max, self.a_max, self.j_max)
        if self.v_max * Ta_full <= self.distance:
            v_peak = self.v_max
            Ta = Ta_full
            Tv = (self.distance - self.v_max * Ta) / self.v_max
        else:
            # v_max is never reached: solve for the peak velocity that makes
            # the (cruise-free) accel+decel distance exactly match `distance`.
            lo, hi = 0.0, self.v_max
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                _, Ta_mid, _ = _accel_phase(mid, self.a_max, self.j_max)
                if mid * Ta_mid < self.distance:
                    lo = mid
                else:
                    hi = mid
            v_peak = 0.5 * (lo + hi)
            _, Ta, _ = _accel_phase(v_peak, self.a_max, self.j_max)
            Tv = 0.0

        Tj, _, a_reached = _accel_phase(v_peak, self.a_max, self.j_max)
        self.v_peak = v_peak
        self.duration = 2 * Ta + Tv

        j = self.j_max
        # Each phase: (duration, jerk, (s0, v0, a0) state at its start).
        self._phases = []
        s0, v0, a0 = 0.0, 0.0, 0.0
        for dur, jerk in (
                (Tj, j), (Ta - 2 * Tj, 0.0), (Tj, -j),
                (Tv, 0.0),
                (Tj, -j), (Ta - 2 * Tj, 0.0), (Tj, j)):
            dur = max(dur, 0.0)
            self._phases.append((dur, jerk, s0, v0, a0))
            s0, v0, a0 = self._eval(dur, jerk, s0, v0, a0)

    @staticmethod
    def _eval(dt, j, s0, v0, a0):
        s = s0 + v0 * dt + 0.5 * a0 * dt ** 2 + (j * dt ** 3) / 6.0
        v = v0 + a0 * dt + 0.5 * j * dt ** 2
        a = a0 + j * dt
        return s, v, a

    def _locate(self, t):
        t = min(max(t, 0.0), self.duration)
        t_cursor = 0.0
        for dur, jerk, s0, v0, a0 in self._phases:
            if t <= t_cursor + dur:
                return t - t_cursor, jerk, s0, v0, a0
            t_cursor += dur
        dur, jerk, s0, v0, a0 = self._phases[-1]
        return dur, jerk, s0, v0, a0

    def position(self, t):
        if self.duration == 0.0:
            return 0.0
        dt, jerk, s0, v0, a0 = self._locate(t)
        return self.sign * self._eval(dt, jerk, s0, v0, a0)[0]

    def velocity(self, t):
        if self.duration == 0.0:
            return 0.0
        dt, jerk, s0, v0, a0 = self._locate(t)
        return self.sign * self._eval(dt, jerk, s0, v0, a0)[1]

    def acceleration(self, t):
        if self.duration == 0.0:
            return 0.0
        dt, jerk, s0, v0, a0 = self._locate(t)
        return self.sign * self._eval(dt, jerk, s0, v0, a0)[2]


def timeScaleProfile(distance, v_max, a_max, j_max=None):
    """
    Build a `TrapezoidalProfile`, or a `SCurveProfile` if `j_max` is given.

    Args:
        distance (float): signed distance to cover
        v_max (float): maximum speed
        a_max (float): maximum acceleration magnitude
        j_max (float, optional): maximum jerk magnitude. Defaults to None (no jerk limit).

    Returns:
        TrapezoidalProfile | SCurveProfile
    """
    if j_max is None or not np.isfinite(j_max) or j_max <= 0:
        return TrapezoidalProfile(distance, v_max, a_max)
    return SCurveProfile(distance, v_max, a_max, j_max)


class _RetimedProfile:
    """Wraps a profile so it spans exactly `duration`, by scaling time and derivatives."""

    def __init__(self, base_profile, duration):
        self.base = base_profile
        self.duration = duration
        if base_profile.duration <= 0.0 or duration <= 0.0:
            self._scale = 0.0
        else:
            self._scale = base_profile.duration / duration

    def position(self, t):
        if self._scale == 0.0:
            return self.base.position(0.0)
        t = min(max(t, 0.0), self.duration)
        return self.base.position(t * self._scale)

    def velocity(self, t):
        if self._scale == 0.0:
            return 0.0
        t = min(max(t, 0.0), self.duration)
        return self.base.velocity(t * self._scale) * self._scale

    def acceleration(self, t):
        if self._scale == 0.0:
            return 0.0
        t = min(max(t, 0.0), self.duration)
        return self.base.acceleration(t * self._scale) * (self._scale ** 2)


def _broadcast_limits(value, dof, name):
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(dof, arr[0])
    if arr.size != dof:
        raise ValueError('%s must be a scalar or length-%d array, got shape %s'
                % (name, dof, arr.shape))
    return arr


class JointTrajectory:
    """
    Time-parametrizes a sequence of R^n waypoints (e.g. joint angles) into a
    trajectory that respects per-axis velocity/acceleration (and optionally
    jerk) limits.

    Each segment (the straight line between two consecutive waypoints) is
    timed independently per axis, then every axis in that segment is retimed
    to match the slowest axis, so all axes start and stop moving together at
    each waypoint. Waypoints are therefore visited at rest (stop-and-go); this
    does not blend corners into a single smooth pass.

    Args:
        waypoints (list[np.ndarray]): at least two points in R^n
        max_vel (float | np.ndarray): per-axis peak velocity (scalar broadcasts)
        max_accel (float | np.ndarray): per-axis peak acceleration magnitude
        max_jerk (float | np.ndarray, optional): per-axis peak jerk magnitude.
            Defaults to None (trapezoidal, not jerk-limited).

    Attributes:
        duration (float): total trajectory duration, in seconds
    """

    def __init__(self, waypoints, max_vel, max_accel, max_jerk=None):
        wps = [np.asarray(w, dtype=float).flatten() for w in waypoints]
        if len(wps) < 2:
            raise ValueError('JointTrajectory needs at least two waypoints')
        dof = wps[0].shape[0]
        for w in wps:
            if w.shape[0] != dof:
                raise ValueError('All waypoints must have the same dimension')

        max_vel = _broadcast_limits(max_vel, dof, 'max_vel')
        max_accel = _broadcast_limits(max_accel, dof, 'max_accel')
        max_jerk = None if max_jerk is None else _broadcast_limits(max_jerk, dof, 'max_jerk')

        self.waypoints = wps
        self.dof = dof
        self._segments = []
        t_cursor = 0.0
        for w0, w1 in zip(wps[:-1], wps[1:]):
            delta = w1 - w0
            base_profiles = []
            seg_duration = 0.0
            for d in range(dof):
                jmax = None if max_jerk is None else max_jerk[d]
                prof = timeScaleProfile(delta[d], max_vel[d], max_accel[d], jmax)
                base_profiles.append(prof)
                seg_duration = max(seg_duration, prof.duration)
            retimed = [_RetimedProfile(p, seg_duration) for p in base_profiles]
            self._segments.append((t_cursor, seg_duration, w0, retimed))
            t_cursor += seg_duration
        self._seg_ends = np.array([t_start + dur for t_start, dur, _, _ in self._segments])
        self.duration = t_cursor

    def _find_segment(self, t):
        t = min(max(t, 0.0), self.duration)
        idx = int(np.searchsorted(self._seg_ends, t, side='left'))
        idx = min(idx, len(self._segments) - 1)
        t_start, dur, w0, profs = self._segments[idx]
        return w0, profs, t - t_start

    def position(self, t):
        """Joint-space position (np.ndarray, shape (dof,)) at time `t`."""
        w0, profs, local_t = self._find_segment(t)
        return w0 + np.array([p.position(local_t) for p in profs])

    def velocity(self, t):
        """Joint-space velocity (np.ndarray, shape (dof,)) at time `t`."""
        _, profs, local_t = self._find_segment(t)
        return np.array([p.velocity(local_t) for p in profs])

    def acceleration(self, t):
        """Joint-space acceleration (np.ndarray, shape (dof,)) at time `t`."""
        _, profs, local_t = self._find_segment(t)
        return np.array([p.acceleration(local_t) for p in profs])

    def sample(self, dt):
        """
        Uniformly sample the trajectory.

        Args:
            dt (float): sample spacing, in seconds

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): times (N,),
            positions (N, dof), velocities (N, dof), accelerations (N, dof)
        """
        n = max(2, int(np.ceil(self.duration / dt)) + 1)
        times = np.linspace(0.0, self.duration, n)
        positions = np.array([self.position(t) for t in times])
        velocities = np.array([self.velocity(t) for t in times])
        accelerations = np.array([self.acceleration(t) for t in times])
        return times, positions, velocities, accelerations


class CartesianTrajectory:
    """
    Time-parametrizes a sequence of `tm` waypoints into a trajectory that
    respects a velocity/acceleration (and optionally jerk) limit on the
    combined screw-motion rate between consecutive poses - the same
    construction `mr.ScrewTrajectory` uses to interpolate a single segment,
    but timed by motion limits instead of a fixed duration.

    As with `JointTrajectory`, waypoints are visited at rest; corners are not
    blended.

    Args:
        waypoints (list[tm]): at least two poses
        v_max (float): maximum screw-motion rate
        a_max (float): maximum screw-motion acceleration magnitude
        j_max (float, optional): maximum screw-motion jerk magnitude.
            Defaults to None (trapezoidal, not jerk-limited).

    Attributes:
        duration (float): total trajectory duration, in seconds
    """

    def __init__(self, waypoints, v_max, a_max, j_max=None):
        if len(waypoints) < 2:
            raise ValueError('CartesianTrajectory needs at least two waypoints')

        self.waypoints = list(waypoints)
        self._segments = []
        t_cursor = 0.0
        for X0, X1 in zip(self.waypoints[:-1], self.waypoints[1:]):
            screw_vec = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X0.gTM()) @ X1.gTM()))
            theta = float(np.linalg.norm(screw_vec))
            profile = timeScaleProfile(theta, v_max, a_max, j_max)
            self._segments.append((t_cursor, profile.duration, X0, screw_vec, theta, profile))
            t_cursor += profile.duration
        self._seg_ends = np.array([t_start + dur for t_start, dur, *_ in self._segments])
        self.duration = t_cursor

    def _find_segment(self, t):
        t = min(max(t, 0.0), self.duration)
        idx = int(np.searchsorted(self._seg_ends, t, side='left'))
        idx = min(idx, len(self._segments) - 1)
        t_start, dur, X0, screw_vec, theta, profile = self._segments[idx]
        return X0, screw_vec, theta, profile, t - t_start

    def position(self, t):
        """Pose (`tm`) at time `t`."""
        X0, screw_vec, theta, profile, local_t = self._find_segment(t)
        if theta == 0.0:
            return X0
        s = profile.position(local_t) / theta
        return tm(X0.gTM() @ mr.MatrixExp6(mr.VecTose3(screw_vec * s)))

    def twist(self, t):
        """Body-frame spatial velocity (np.ndarray, shape (6,)) at time `t`."""
        X0, screw_vec, theta, profile, local_t = self._find_segment(t)
        if theta == 0.0:
            return np.zeros(6)
        sdot = profile.velocity(local_t) / theta
        return screw_vec * sdot

    def sample(self, dt):
        """
        Uniformly sample the trajectory.

        Args:
            dt (float): sample spacing, in seconds

        Returns:
            (np.ndarray, list[tm], np.ndarray): times (N,), poses (length N),
            body-frame twists (N, 6)
        """
        n = max(2, int(np.ceil(self.duration / dt)) + 1)
        times = np.linspace(0.0, self.duration, n)
        poses = [self.position(t) for t in times]
        twists = np.array([self.twist(t) for t in times])
        return times, poses, twists
