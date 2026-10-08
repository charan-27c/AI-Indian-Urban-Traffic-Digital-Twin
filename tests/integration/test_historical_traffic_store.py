from src.prediction.models import (
    TrafficHistoryRecord,
)

from src.prediction.historical_store import (
    HistoricalTrafficStore,
)


def test_historical_store_adds_records():

    store = HistoricalTrafficStore()

    record = TrafficHistoryRecord(
        timestamp=1.0,

        vehicle_count=10,
        queue_length=3,

        average_speed=8.0,

        density=10.0,
        occupancy=0.3,

        waiting_time=20.0,
        delay=15.0,

        flow_rate=10.0,
        pressure=13.0,
    )

    store.add(record)

    assert store.count() == 1

    assert (
        store.get_all()[0]
        == record
    )


def test_historical_store_keeps_latest_records():

    store = HistoricalTrafficStore(
        max_records=3
    )

    for timestamp in range(5):

        store.add(
            TrafficHistoryRecord(
                timestamp=float(timestamp),

                vehicle_count=timestamp,

                queue_length=0,

                average_speed=10.0,

                density=0.0,
                occupancy=0.0,

                waiting_time=0.0,
                delay=0.0,

                flow_rate=0.0,
                pressure=0.0,
            )
        )

    assert store.count() == 3

    records = store.get_all()

    assert records[0].timestamp == 2.0
    assert records[-1].timestamp == 4.0


def test_add_state():

    store = HistoricalTrafficStore()

    class State:

        timestamp = 10.0

        vehicle_count = 20
        queue_length = 5

        average_speed = 7.5

        density = 20.0
        occupancy = 0.25

        waiting_time = 30.0
        delay = 25.0

        flow_rate = 20.0
        pressure = 35.0

    record = store.add_state(State())

    assert store.count() == 1

    assert record.timestamp == 10.0
    assert record.vehicle_count == 20
    assert record.queue_length == 5
    assert record.pressure == 35.0