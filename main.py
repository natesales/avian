import collections
import logging
import sys
import time

import cv2
import mediapipe
import numpy

from classifiers import Hand
from classifiers import gestures
from classifiers import pose

from smartdashboard import SmartDashboard

NETWORKTABLES_SERVER = sys.argv[1]
logging.info(f"Using NetworkTables server {NETWORKTABLES_SERVER}")

detections_available = ["pinch", "fist", "middle_finger", "index_finger"]

DETECTIONS_ENABLED = detections_available  # Detections can be disabled for performance improvements
DETECT_HAND_POSE = True  # Detect historical hand pose
POSE_CACHE_SIZE = 20  # Cache last 10 hand positions
DRAW = True  # Display graphics on screen

SD_DEFAULTS = {
    "avian/max_num_hands": 2,
    "avian/pinch_proximity_radius": 7,
    "avian/invert_horizontal": False,
    "avian/invert_vertical": False,
    "avian/black_background": False,

    # Immutable
    "avian/fps": 0,
    "avian/detected_hands": 0,
    "avian/calibrated": False,
}

for detection in detections_available:
    SD_DEFAULTS[f"avian/left_{detection}"] = False
    SD_DEFAULTS[f"avian/right_{detection}"] = False

cap = cv2.VideoCapture(0)


def sd_change_listener(table, key, value, isNew):
    # Create new hand processor if max_num_hands changes
    if key == "avian/max_num_hands":
        global hands
        hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("avian/max_num_hands")))
        logging.info("Updated hand detection target")


sd = SmartDashboard(NETWORKTABLES_SERVER, SD_DEFAULTS, sd_change_listener, "avian/max_num_hands")
sd.set("avian/detections", ",".join(detections_available))

hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("avian/max_num_hands")))

results = None
p_time = 0

left_hand_poses = collections.deque(maxlen=POSE_CACHE_SIZE)
right_hand_poses = collections.deque(maxlen=POSE_CACHE_SIZE)

while True:
    success, img = cap.read()
    if not success:
        logging.fatal("Unable to read from video capture device")
    if sd.get("avian/invert_vertical"):
        img = cv2.flip(img, 0)
    if sd.get("avian/invert_horizontal"):
        img = cv2.flip(img, 1)
    h, w, c = img.shape

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if sd.get("avian/black_background"):
        h, w, _ = img.shape
        img = numpy.zeros((h, w, 3), numpy.uint8)

    if results.multi_hand_landmarks:
        sd.set("avian/detected_hands", len(results.multi_hand_landmarks))
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand matrix
            if DRAW:
                mediapipe.solutions.drawing_utils.draw_landmarks(img, hand_landmarks,
                                                                 mediapipe.solutions.hands.HAND_CONNECTIONS)

            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            # Calibration
            if not sd.get("avian/calibrated"):
                if handedness == Hand.LEFT:
                    sd.set("avian/invert_horizontal", True)
            sd.set("avian/calibrated", True)

            # Add to pose cache
            if DETECT_HAND_POSE:
                hand_pose = pose.hand_pose(hand_landmarks, w, h)
                if handedness == Hand.LEFT:
                    left_hand_poses.append(hand_pose)
                elif handedness == Hand.RIGHT:
                    right_hand_poses.append(hand_pose)

            # Pinch detection
            if "pinch" in DETECTIONS_ENABLED:
                index_thumb_pinch = gestures.pinch(
                    hand_landmarks,
                    mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                    mediapipe.solutions.hands.HandLandmark.THUMB_TIP,
                    int(sd.get("avian/pinch_proximity_radius")),
                    w,
                    h
                )
                if handedness == Hand.LEFT:
                    sd.set("avian/left_pinch", index_thumb_pinch)
                elif handedness == Hand.RIGHT:
                    sd.set("avian/right_pinch", index_thumb_pinch)

                # Draw pinch radii
                if DRAW:
                    for digit in [mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                                  mediapipe.solutions.hands.HandLandmark.THUMB_TIP]:
                        if handedness == Hand.LEFT:
                            color = (255, 0, 0)
                        elif handedness == Hand.RIGHT:
                            color = (0, 0, 255)
                        else:
                            color = (255, 255, 255)

                        lm = hand_landmarks.landmark[digit]
                        cv2.circle(img, (int(lm.x * w), int(lm.y * h)),
                                   int(sd.get("avian/pinch_proximity_radius")),
                                   color, thickness=2)

            # Detect middle finger
            if "middle_finger" in DETECTIONS_ENABLED:
                middle_finger = gestures.middle_finger(hand_landmarks)
                if handedness == Hand.LEFT:
                    sd.set("avian/left_middle_finger", middle_finger)
                elif handedness == Hand.RIGHT:
                    sd.set("avian/right_middle_finger", middle_finger)

            # Detect index finger
            if "index_finger" in DETECTIONS_ENABLED:
                index_finger = gestures.index_finger(hand_landmarks)
                if handedness == Hand.LEFT:
                    sd.set("avian/left_index_finger", index_finger)
                elif handedness == Hand.RIGHT:
                    sd.set("avian/right_index_finger", index_finger)

            # Fist detection
            if "fist" in DETECTIONS_ENABLED:
                fist = gestures.fist(hand_landmarks, h)
                if handedness == Hand.LEFT:
                    sd.set("avian/left_fist", fist)
                elif handedness == Hand.RIGHT:
                    sd.set("avian/right_fist", fist)

    else:  # No hands detected
        sd.set("avian/detected_hands", 0)
        for detection in detections_available:
            sd.set(f"avian/left_{detection}", False)
            sd.set(f"avian/right_{detection}", False)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    sd.set("avian/fps", fps)

    # print(left_hand_poses, "  ", right_hand_poses)

    if DRAW:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
