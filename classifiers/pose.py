import collections
from typing import List, Tuple

import mediapipe

from . import Direction, SwipeDirection


def hand_vector(points: collections.deque):
    pass  # TODO


def lm_avg(
        landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        targets: List[mediapipe.solutions.hands.HandLandmark],
        width: int,
        height: int) -> Tuple[int, int]:
    """Find average pose of multiple hand landmarks
    :param landmarks: List of landmarks in hand
    :param targets: List of hand landmarks to target
    :param width: Image width
    :param height: Image height
    :returns: Coordinate pair of average
    """
    x, y = 0, 0
    for target_lm in targets:
        # landmarks are between 0 and 1
        x += landmarks.landmark[target_lm].x
        y += landmarks.landmark[target_lm].y

    return int(x / len(targets) * width), int(y / len(targets) * height)


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
    :param landmarks: List of landmarks in hand
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


def lm_slope(
        landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        lm1: mediapipe.solutions.hands.HandLandmark,
        lm2: mediapipe.solutions.hands.HandLandmark
) -> int:
    """
    Calculate the slope of the line between two landmarks
    :param landmarks: List of landmarks in hand
    :param lm1: First landmark
    :param lm2: Second landmark
    :return:
    """

    lm1_pose = landmarks.landmark[lm1]
    lm2_pose = landmarks.landmark[lm2]
    return (lm2_pose.y - lm1_pose.y) / (lm2_pose.x - lm1_pose.x)


def hand_swipe(history: collections.deque, distance_threshold: int, direction: SwipeDirection) -> bool:
    """
    Calculate whether a hand has moved a certain distance within the length of the cache
    :param history: Historical cache of hand positions
    :param distance_threshold: Swipe distance threshold
    :param direction: Direction of swipe (left or right)
    :return:
    """

    startX = history[len(history) - 1][0]
    endX = history[0][0]
    distance_traveled = endX - startX
    if direction == SwipeDirection.LEFT:
        return distance_traveled > distance_threshold
    return distance_traveled < -distance_threshold
