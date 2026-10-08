from dataclasses import dataclass, field
from typing import List


@dataclass
class Junction:
    """
    Represents one traffic junction.

    This model does not assume:
        - 4 approaches
        - fixed directions
        - fixed lanes
        - fixed movements

    The junction structure comes from configuration/topology.
    """

    junction_id: str
    name: str
    approaches: List[str] = field(default_factory=list)

    def get_approaches(self) -> List[str]:
        """Return approaches belonging to this junction."""
        return list(self.approaches)

    def has_approach(self, approach: str) -> bool:
        """Check whether this junction has the given approach."""
        return approach in self.approaches

    def approach_count(self) -> int:
        """Return the number of approaches."""
        return len(self.approaches)

    def get_metadata(self) -> dict:
        """Return junction metadata."""
        return {
            "junction_id": self.junction_id,
            "name": self.name,
            "approaches": self.get_approaches(),
            "approach_count": self.approach_count(),
        }