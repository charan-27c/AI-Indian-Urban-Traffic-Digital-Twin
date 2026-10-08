class IncidentManager:

    def __init__(self):

        self.active_incidents = {}

    def update(self, events):

        for event in events:

            key = (
                event.anomaly_type.value,
                event.junction_id,
                event.movement_id,
                event.lane_id,
            )

            self.active_incidents[key] = event

        return list(
            self.active_incidents.values()
        )

    def clear(self, key):

        self.active_incidents.pop(
            key,
            None,
        )

    def reset(self):

        self.active_incidents.clear()