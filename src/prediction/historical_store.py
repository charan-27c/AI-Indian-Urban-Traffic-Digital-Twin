from typing import List, Optional

from .models import TrafficHistoryRecord


class HistoricalTrafficStore:
    """
    Runtime time-series storage for traffic states.

    The store keeps historical NetworkTrafficState values
    that will later be converted into LSTM sequences.

    No fixed junctions or lanes are assumed.
    """

    def __init__(
        self,
        max_records: Optional[int] = None,
    ) -> None:

        if max_records is not None and max_records <= 0:
            raise ValueError(
                "max_records must be greater than zero."
            )

        self.max_records = max_records

        self._records: List[
            TrafficHistoryRecord
        ] = []

    # ========================================================
    # ADD RECORD
    # ========================================================

    def add(
        self,
        record: TrafficHistoryRecord,
    ) -> None:

        self._records.append(record)

        if (
            self.max_records is not None
            and len(self._records) > self.max_records
        ):
            excess = (
                len(self._records)
                - self.max_records
            )

            del self._records[:excess]

    # ========================================================
    # ADD FROM TRAFFIC STATE
    # ========================================================

    def add_state(
        self,
        state,
    ) -> TrafficHistoryRecord:
        """
        Convert a NetworkTrafficState into a historical record.
        """

        record = TrafficHistoryRecord(
            timestamp=state.timestamp,

            vehicle_count=state.vehicle_count,
            queue_length=state.queue_length,

            average_speed=state.average_speed,

            density=state.density,
            occupancy=state.occupancy,

            waiting_time=state.waiting_time,
            delay=state.delay,

            flow_rate=state.flow_rate,
            pressure=state.pressure,
        )

        self.add(record)

        return record

    # ========================================================
    # GET ALL
    # ========================================================

    def get_all(
        self,
    ) -> List[TrafficHistoryRecord]:

        return list(self._records)

    # ========================================================
    # LAST N
    # ========================================================

    def get_last(
        self,
        count: int,
    ) -> List[TrafficHistoryRecord]:

        if count <= 0:
            return []

        return list(
            self._records[-count:]
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:

        return len(self._records)

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:

        self._records.clear()

    # ========================================================
    # EMPTY
    # ========================================================

    def is_empty(self) -> bool:

        return not self._records