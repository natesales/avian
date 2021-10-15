import mediapipe


class Finger:
    INDEX = [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_MCP
    ]

    MIDDLE = [
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP
    ]

    RING = [
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_MCP
    ]

    PINKY = [
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_DIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_PIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_MCP
    ]

    THUMB = [
        mediapipe.solutions.hands.HandLandmark.THUMB_TIP,
        mediapipe.solutions.hands.HandLandmark.THUMB_IP,
        mediapipe.solutions.hands.HandLandmark.THUMB_MCP,
        mediapipe.solutions.hands.HandLandmark.THUMB_CMC
    ]
