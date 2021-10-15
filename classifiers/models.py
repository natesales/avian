from typing import List

import mediapipe


class Finger:
    segments: List[mediapipe.solutions.hands.HandLandmark] = []


class Index(Finger):
    segments = [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_MCP
    ]


class Middle(Finger):
    segments = [
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP
    ]


class Ring(Finger):
    segments = [
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_MCP
    ]


class Pinky(Finger):
    segments = [
        mediapipe.solutions.hands.HandLandmark.PINKY_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_FINGER_DIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_FINGER_PIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_FINGER_MCP
    ]


class Thumb(Finger):
    segments = [
        mediapipe.solutions.hands.HandLandmark.THUMB_TIP,
        mediapipe.solutions.hands.HandLandmark.THUMB_IP,
        mediapipe.solutions.hands.HandLandmark.THUMB_MCP,
        mediapipe.solutions.hands.HandLandmark.THUMB_CMC
    ]
