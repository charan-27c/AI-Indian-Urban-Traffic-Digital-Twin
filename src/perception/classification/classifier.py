from .traffic_classes import TRAFFIC_CLASSES


class IndianTrafficClassifier:

    def classify(self, detection):

        detected_class = detection["class"]

        information = TRAFFIC_CLASSES.get(
            detected_class
        )

        if information is None:

            return {
                **detection,
                "traffic_class": "unknown",
                "is_emergency": False,
            }

        result = {
            **detection,
            "traffic_class": information["category"],
            "is_emergency": information["is_emergency"],
        }

        if "emergency_type" in information:

            result["emergency_type"] = (
                information["emergency_type"]
            )

        return result