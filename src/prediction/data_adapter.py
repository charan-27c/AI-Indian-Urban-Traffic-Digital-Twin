from typing import Iterable, List

import numpy as np

from .models import TrafficHistoryRecord


class TrafficDataAdapter:
    """
    Converts tabular traffic measurements into
    TrafficHistoryRecord objects.

    The adapter is intentionally independent of
    junction/lane/movement IDs.
    """

    REQUIRED_COLUMNS = (
        "timestamp",
        "vehicle_count",
        "queue_length",
        "average_speed",
        "density",
        "occupancy",
        "waiting_time",
        "delay",
        "flow_rate",
        "pressure",
    )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[dict],
    ) -> List[TrafficHistoryRecord]:

        records = []

        for row in rows:

            missing = [
                column
                for column in cls.REQUIRED_COLUMNS
                if column not in row
            ]

            if missing:
                raise ValueError(
                    "Missing traffic columns: "
                    + ", ".join(missing)
                )

            records.append(
                TrafficHistoryRecord(
                    timestamp=float(
                        row["timestamp"]
                    ),
                    vehicle_count=float(
                        row["vehicle_count"]
                    ),
                    queue_length=float(
                        row["queue_length"]
                    ),
                    average_speed=float(
                        row["average_speed"]
                    ),
                    density=float(
                        row["density"]
                    ),
                    occupancy=float(
                        row["occupancy"]
                    ),
                    waiting_time=float(
                        row["waiting_time"]
                    ),
                    delay=float(
                        row["delay"]
                    ),
                    flow_rate=float(
                        row["flow_rate"]
                    ),
                    pressure=float(
                        row["pressure"]
                    ),
                )
            )

        if not records:
            raise ValueError(
                "At least one traffic row is required."
            )

        return records

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
    ) -> List[TrafficHistoryRecord]:

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        if data.ndim != 2:
            raise ValueError(
                "Traffic data must be a 2D array."
            )

        if data.shape[1] != 10:
            raise ValueError(
                "Traffic data must contain "
                "10 columns."
            )

        rows = []

        for row in data:

            rows.append(
                {
                    "timestamp": row[0],
                    "vehicle_count": row[1],
                    "queue_length": row[2],
                    "average_speed": row[3],
                    "density": row[4],
                    "occupancy": row[5],
                    "waiting_time": row[6],
                    "delay": row[7],
                    "flow_rate": row[8],
                    "pressure": row[9],
                }
            )

        return cls.from_rows(rows)