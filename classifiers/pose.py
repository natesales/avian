import collections
from typing import List

import mediapipe


def hand_vector(points: collections.deque):
    pass # TODO


def lm_avg(
        landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        targets: List[mediapipe.solutions.hands.HandLandmark],
        height: int,
        width: int) -> (int, int):
    """Find average pose of multiple hand landmarks

    :param landmarks: List of landmarks in hand
    :param targets: List of hand landmarks to target
    :param width: Image width
    :param height: Image height

    :returns: Coordinate pair of average
    """
    x, y = 0, 0
    for target_lm in targets:
        x += int(width * landmarks.landmark[target_lm].x)
        y += int(height * landmarks.landmark[target_lm].y)
    return int(x / len(targets)), int(y / len(targets))


def hand_pose(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, width: int, height: int):
    """Get the overall average hand position as a single point"""

    return lm_avg(landmarks, [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ], width, height)
