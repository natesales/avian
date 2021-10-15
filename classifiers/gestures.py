import mediapipe

from . import pose, models, Direction, circle_intersection


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


def middle_finger(landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList) -> bool:
    """Detect middle finger

    :param landmarks: List of landmarks in hand

    :returns: True if the hand is making a middle finger gesture
    """

    # Validate that all other fingers are below the middle finger
    if not pose.lm_threshold(landmarks, mediapipe.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, [
        mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mediapipe.solutions.hands.HandLandmark.PINKY_TIP
    ], Direction.BELOW):
        return False

    # Validate the middle finger is extended
    return pose.finger_extended(landmarks, models.Finger.MIDDLE)


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
    return pose.finger_extended(landmarks, models.Finger.INDEX)
