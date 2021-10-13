import logging
import time

import cv2
import mediapipe
import numpy

from smartdashboard import SmartDashboard
from classifiers import gestures

SD_DEFAULTS = {
    "vision/max_num_hands": 2,
    "vision/pinch_proximity_radius": 7,
    "vision/invert_horizontal": False,
    "vision/invert_vertical": False,
    "vision/black_background": False,

    # Immutable
    "vision/fps": 0,
    "vision/detected_hands": 0,
    "vision/calibrated": False,
}

SD_DETECTIONS = ["pinch", "fist", "middle_finger", "index_finger"]

for detection in SD_DETECTIONS:
    SD_DEFAULTS[f"vision/left_{detection}"] = False
    SD_DEFAULTS[f"vision/right_{detection}"] = False

cap = cv2.VideoCapture(0)


def sd_change_listener(table, key, value, isNew):
    # Create new hand processor if max_num_hands changes
    if key == "vision/max_num_hands":
        global hands
        hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("vision/max_num_hands")))
        logging.info("Updated hand detection target")


sd = SmartDashboard("localhost", SD_DEFAULTS, sd_change_listener, "vision/max_num_hands")

hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("vision/max_num_hands")))

results = None
p_time = 0

while True:
    success, img = cap.read()
    if not success:
        logging.fatal("Unable to read from video capture device")
    if sd.get("vision/invert_vertical"):
        img = cv2.flip(img, 0)
    if sd.get("vision/invert_horizontal"):
        img = cv2.flip(img, 1)
    h, w, c = img.shape

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if sd.get("vision/black_background"):
        h, w, _ = img.shape
        img = numpy.zeros((h, w, 3), numpy.uint8)

    if results.multi_hand_landmarks:
        sd.set("vision/detected_hands", len(results.multi_hand_landmarks))
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # Draw hand matrix
            mediapipe.solutions.drawing_utils.draw_landmarks(img, handLms, mediapipe.solutions.hands.HAND_CONNECTIONS)

            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            # Calibration
            if not sd.get("vision/calibrated"):
                if handedness == gestures.Hand.LEFT:
                    sd.set("vision/invert_horizontal", True)
            sd.set("vision/calibrated", True)

            # Pinch detection
            index_thumb_pinch = gestures.pinch(
                handLms,
                mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mediapipe.solutions.hands.HandLandmark.THUMB_TIP,
                int(sd.get("vision/pinch_proximity_radius")),
                w,
                h
            )
            if handedness == gestures.Hand.LEFT:
                sd.set("vision/left_pinch", index_thumb_pinch)
            elif handedness == gestures.Hand.RIGHT:
                sd.set("vision/right_pinch", index_thumb_pinch)

            # Draw pinch radii
            for digit in [mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                          mediapipe.solutions.hands.HandLandmark.THUMB_TIP]:
                if handedness == gestures.Hand.LEFT:
                    color = (255, 0, 0)
                elif handedness == gestures.Hand.RIGHT:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)

                lm = handLms.landmark[digit]
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)),
                           int(sd.get("vision/pinch_proximity_radius")),
                           color, thickness=2)

            # Detect middle finger
            middle_finger = gestures.middle_finger(handLms)
            if handedness == gestures.Hand.LEFT:
                sd.set("vision/left_middle_finger", middle_finger)
            elif handedness == gestures.Hand.RIGHT:
                sd.set("vision/right_middle_finger", middle_finger)

            # Detect index finger
            index_finger = gestures.index_finger(handLms)
            if handedness == gestures.Hand.LEFT:
                sd.set("vision/left_index_finger", index_finger)
            elif handedness == gestures.Hand.RIGHT:
                sd.set("vision/right_index_finger", index_finger)

            # Fist detection
            fist = gestures.fist(handLms, h)
            if handedness == gestures.Hand.LEFT:
                sd.set("vision/left_fist", fist)
            elif handedness == gestures.Hand.RIGHT:
                sd.set("vision/right_fist", fist)

    else:  # No hands detected
        sd.set("vision/detected_hands", 0)
        for detection in SD_DETECTIONS:
            sd.set(f"vision/left_{detection}", False)
            sd.set(f"vision/right_{detection}", False)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    sd.set("vision/fps", fps)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
