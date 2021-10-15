import mediapipe

from classifiers import gestures, Gesture


class Classifier:
    def __init__(self, image_width: int, image_height: int, pinch_proximity_radius: int):
        self.image_width = image_width
        self.image_height = image_height
        self.pinch_proximity_radius = pinch_proximity_radius

    def classify(self, landmarks) -> str:
        # Pinch detection
        if gestures.pinch(
                landmarks,
                mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mediapipe.solutions.hands.HandLandmark.THUMB_TIP,
                self.pinch_proximity_radius,
                self.image_width,
                self.image_height
        ):
            return Gesture.PINCH
        elif gestures.middle_finger(landmarks):
            return Gesture.MIDDLE_FINGER
        elif gestures.index_finger(landmarks):
            return Gesture.INDEX_FINGER
        return ""
