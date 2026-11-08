"""
PID controller implementation for Underfloor Heating Controller.

This module provides a pure Python PID controller with anti-windup
for temperature regulation in heating zones.
"""

from dataclasses import dataclass, field

DUTY_MIN = 0.0
DUTY_MAX = 100.0


@dataclass(frozen=True)
class PIDState:
    """
    Complete state of the PID controller.

    This frozen dataclass contains both the output values from the last
    update and the internal accumulator state (integral serves as the
    integral accumulator, error serves as last_error for derivative).
    """

    error: float
    proportional: float
    integral: float
    derivative: float
    duty_cycle: float


@dataclass
class PIDController:
    """
    PID controller with anti-windup for temperature control.

    The controller calculates a duty cycle (0-100%) based on the
    temperature error (setpoint - current temperature).

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        integral_min: Minimum integral term contribution in % (anti-windup).
        integral_max: Maximum integral term contribution in % (anti-windup).

    """

    kp: float = 50.0
    ki: float = 0.001
    kd: float = 0.0
    integral_min: float = 0.0
    integral_max: float = 100.0

    _state: PIDState | None = field(default=None, init=False, repr=False)

    @property
    def state(self) -> PIDState | None:
        """Return the current PID state."""
        return self._state

    def update(self, setpoint: float, current: float, dt: float) -> PIDState | None:
        """
        Calculate duty cycle from temperature error.

        Args:
            setpoint: Target temperature.
            current: Current temperature.
            dt: Time delta in seconds since last update.

        Returns:
            PIDState with all terms and clamped output (0.0 to 100.0),
            or None if dt <= 0 and no prior state exists.

        """
        if dt <= 0:
            return self._state

        error = setpoint - current

        # Proportional term
        proportional = self.kp * error

        # Derivative term - use previous error from state
        last_error = self._state.error if self._state else 0.0
        derivative = self.kd * (error - last_error) / dt

        # Integral term with anti-windup
        # Use previous integral as the integral accumulator
        prev_integral = self._state.integral if self._state else 0.0
        # Conditional integration: while the output is already saturated, do
        # not integrate further in the direction of saturation. Otherwise a
        # long rise (cold start, recovery from a setback) keeps charging the
        # integral although the valve cannot open any wider, and the excess
        # shows up as overshoot after the setpoint is reached.
        unclamped = proportional + prev_integral + derivative
        saturated_high = unclamped >= DUTY_MAX and error > 0
        saturated_low = unclamped <= DUTY_MIN and error < 0
        if saturated_high or saturated_low:
            integral = prev_integral
        else:
            integral = prev_integral + self.ki * error * dt
        integral = max(self.integral_min, min(self.integral_max, integral))

        # Output clamped to 0-100%
        duty_cycle = max(DUTY_MIN, min(DUTY_MAX, proportional + integral + derivative))

        self._state = PIDState(
            error=error,
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            duty_cycle=duty_cycle,
        )
        return self._state

    def set_state(self, state: PIDState) -> None:
        """
        Set the PID state directly (for state restoration).

        Args:
            state: The PIDState to restore.

        """
        self._state = state
