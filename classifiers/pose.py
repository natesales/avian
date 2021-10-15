import collections
from typing import List

import mediapipe

from . import Direction


def hand_vector(points: collections.deque):
    pass  # TODO


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
    """
    Get the overall average hand position as a single point
    :param landmarks:
    :param width:
    :param height:
    :return:
    """

    return lm_avg(landmarks, [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ], width, height)


def finger_extended(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
                    finger: List[mediapipe.solutions.hands.HandLandmark]) -> bool:
    """Determine if a finger is extended vertically
    :param landmarks: List of landmarks in hand
    :param finger: Finger model
    :returns: True if the finger is extended
    """

    segments = [landmarks.landmark[segment].y for segment in finger]
    return segments == sorted(segments)


def lm_threshold(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
                 lm: mediapipe.solutions.hands.HandLandmark,
                 lms: List[mediapipe.solutions.hands.HandLandmark],
                 direction: str) -> bool:
    """
    Check if a landmark is above or below a list of landmarks
    :param landmarks:
    :param lm: Landmark to evaluate
    :param lms: List of landmarks to evaluate
    :param direction: Above or Below
    :return:
    """

    # Validate that all other fingers are below the middle finger
    lm_y = landmarks.landmark[lm].y
    for lm_case in lms:
        finger_lm_y = landmarks.landmark[lm_case].y
        if (direction == Direction.BELOW and finger_lm_y < lm_y) or (direction == Direction.ABOVE and finger_lm_y > lm_y):
            return False
    return True
