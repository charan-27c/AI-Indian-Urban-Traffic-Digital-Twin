from src.perception.classification.classifier import (
    IndianTrafficClassifier,
)


def test_car_classification():

    classifier = IndianTrafficClassifier()

    detection = {
        "vehicle_id": 1,
        "class": "car",
        "confidence": 0.93,
        "bbox": [100, 200, 300, 400],
    }

    result = classifier.classify(detection)

    assert result["class"] == "car"
    assert result["traffic_class"] == "four_wheeler"
    assert result["is_emergency"] is False


def test_motorcycle_classification():

    classifier = IndianTrafficClassifier()

    detection = {
        "vehicle_id": 2,
        "class": "motorcycle",
        "confidence": 0.88,
        "bbox": [400, 200, 500, 350],
    }

    result = classifier.classify(detection)

    assert result["class"] == "motorcycle"
    assert result["traffic_class"] == "two_wheeler"
    assert result["is_emergency"] is False


def test_ambulance_classification():

    classifier = IndianTrafficClassifier()

    detection = {
        "vehicle_id": 3,
        "class": "ambulance",
        "confidence": 0.95,
        "bbox": [600, 200, 800, 400],
    }

    result = classifier.classify(detection)

    assert result["class"] == "ambulance"
    assert result["traffic_class"] == "emergency_vehicle"
    assert result["is_emergency"] is True
    assert result["emergency_type"] == "ambulance"