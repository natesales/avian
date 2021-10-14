import math

import mediapipe


class Hand:
    LEFT = "Left"
    RIGHT = "Right"


class Axis:
    X = "x"
    Y = "y"


def circle_intersection(x0, y0, x1, y1, radius) -> bool:
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < (radius * 2)


def average_finger(
        landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        fingers,
        scalar: int,
        axis: str):
    average = 0
    for fingertip in fingers:
        if axis == Axis.X:
            average += int(scalar * landmarks.landmark[fingertip].x)
        elif axis == Axis.Y:
            average += int(scalar * landmarks.landmark[fingertip].y)
        else:
            raise AttributeError("Unknown axis, must be X or Y")
    return int(average / len(fingers))


def hand_pose(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, width: int, height: int):
    """Get the overall average hand position as a single point"""

    all_fingers = [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ]

    avg_x, avg_y = 0, 0
    for finger in all_fingers:
        avg_x += int(width * landmarks.landmark[finger].x)
        avg_y += int(height * landmarks.landmark[finger].y)

    return int(avg_x / len(all_fingers)), int(avg_y / len(all_fingers))


def pinch(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
          finger1: mediapipe.solutions.hands.HandLandmark,
          finger2: mediapipe.solutions.hands.HandLandmark,
          radius: int,
          w: int,
          h: int,
          ) -> bool:
    """Detect a pinch gesture between two fingers

    :param landmarks: List of landmarks in hand
    :param finger1: HandLandmark of one finger
    :param finger2: HandLandmark of other finger
    :param radius: Pinch detection radius
    :param w: Image width
    :param h: Image height

    :returns: True if the fingers are touching
    """

    finger1_lm = landmarks.landmark[finger1]
    finger2_lm = landmarks.landmark[finger2]

    return circle_intersection(
        int(finger1_lm.x * w), int(finger1_lm.y * h),
        int(finger2_lm.x * w), int(finger2_lm.y * h),
        radius
    )


def fist(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, height: int) -> bool:
    """Detect fist gesture

    :param landmarks: List of landmarks in hand
    :param height: Image height

    :returns: True if the hand is making a fist
    """

    # Average of all fingers
    fingertip_average_y = average_finger(landmarks, [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ], height, Axis.Y)
    middle_finger_base_y = int(height * landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y)
    wrist_y = int(height * landmarks.landmark[mediapipe.solutions.hands.HandLandmark.WRIST].y)

    # If the full hand isn't visible, we can't detect a fist
    if middle_finger_base_y == 0 or wrist_y == 0 or fingertip_average_y == 0:
        return False

    if middle_finger_base_y < wrist_y:  # Hand facing up
        return fingertip_average_y > middle_finger_base_y
    else:  # Hand facing down
        return fingertip_average_y < middle_finger_base_y


def middle_finger(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList) -> bool:
    """Detect middle finger

    :param landmarks: List of landmarks in hand

    :returns: True if the hand is making a middle finger gesture
    """

    # Validate that all other fingers are below the middle finger
    middle_finger_segment_y = landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y
    for finger in [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ]:
        finger_lm_y = landmarks.landmark[finger].y
        if finger_lm_y < middle_finger_segment_y:
            return False

    # Validate the middle finger is extended
    middle_finger_segments = [
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y,
    ]
    # If the list isn't sorted in ascending order, then the finger isn't extended vertically
    return middle_finger_segments == sorted(middle_finger_segments)


def index_finger(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList) -> bool:
    """Detect index finger

    :param landmarks: List of landmarks in hand

    :returns: True if the hand is making a index finger raised gesture
    """

    # Validate that all other fingers are below the middle finger
    index_finger_segment_y = landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y
    for finger in [
        mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ]:
        finger_lm_y = landmarks.landmark[finger].y
        if finger_lm_y < index_finger_segment_y:
            return False

    # Validate the index finger is extended
    index_finger_segments = [
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y,
    ]
    # If the list isn't sorted in ascending order, then the finger isn't extended vertically
    return index_finger_segments == sorted(index_finger_segments)
