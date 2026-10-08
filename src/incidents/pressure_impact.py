from dataclasses import dataclass


@dataclass(frozen=True)
class IncidentPressureAdjustment:
    """
    Adjusts traffic pressure without modifying the
    underlying observed traffic measurements.
    """

    capacity_reduction: float = 0.0
    spillback_risk: float = 0.0
    severity: float = 0.0
    blocked: bool = False

    def adjusted_pressure(
        self,
        base_pressure: float,
    ) -> float:

        pressure = max(
            0.0,
            float(base_pressure),
        )

        # Physical blockage has the strongest effect.
        if self.blocked:
            return pressure * 2.0 + 100.0

        # Capacity loss increases effective pressure.
        pressure *= (
            1.0 + self.capacity_reduction
        )

        # Spillback risk represents downstream
        # congestion propagation.
        pressure *= (
            1.0 + 0.5 * self.spillback_risk
        )

        # Severity provides a small additional
        # urgency component.
        pressure += (
            20.0 * self.severity
        )

        return pressure