import logging
import time

import cv2
import mediapipe
import numpy
from networktables import NetworkTables

from classifiers import gestures

cap = cv2.VideoCapture(0)

NetworkTables.initialize(server="localhost")
sd = NetworkTables.getTable("SmartDashboard")
NetworkTables.addConnectionListener(
    lambda connected, info: print(info, "; Connected=%s" % connected),
    immediateNotify=True
)


def sd_change_listener(table, key, value, isNew):
    # Create new hand processor if max_num_hands changes
    if key == "vision/max_num_hands":
        global hands
        hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd_get_value("vision/max_num_hands")))
        logging.info("Updated hand detection target")


sd.addEntryListener(sd_change_listener, key="vision/max_num_hands", immediateNotify=True)

SD_DEFAULTS = {
    "vision/max_num_hands": 1,
    "vision/pinch_proximity_radius": 7,
    "vision/invert_horizontal": False,
    "vision/invert_vertical": False,
    "vision/black_background": False,

    # Immutable
    "vision/fps": 0,
    "vision/frame_processing_time": 0,
    "vision/detected_hands": 0,
    "vision/left_pinch": False,
    "vision/right_pinch": False,
    "vision/left_fist": False,
    "vision/right_fist": False
}

for key in SD_DEFAULTS:
    sd.putValue(key, SD_DEFAULTS[key])


def sd_get_value(key: str):
    return sd.getValue(key, SD_DEFAULTS[key])


hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd_get_value("vision/max_num_hands")))
henum = mediapipe.solutions.hands.HandLandmark

results = None
p_time = 0

while True:
    success, img = cap.read()
    if not success:
        logging.fatal("Unable to read from video capture device")
    if sd_get_value("vision/invert_vertical"):
        img = cv2.flip(img, 0)
    if sd_get_value("vision/invert_horizontal"):
        img = cv2.flip(img, 1)
    h, w, c = img.shape

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if sd_get_value("vision/black_background"):
        h, w, _ = img.shape
        img = numpy.zeros((h, w, 3), numpy.uint8)

    if results.multi_hand_landmarks:
        sd.putNumber("vision/detected_hands", len(results.multi_hand_landmarks))
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # Draw hand matrix
            mediapipe.solutions.drawing_utils.draw_landmarks(img, handLms, mediapipe.solutions.hands.HAND_CONNECTIONS)

            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            # Pinch detection
            index_thumb_pinch = gestures.pinch(
                handLms,
                henum.INDEX_FINGER_TIP,
                henum.THUMB_TIP,
                int(sd_get_value("vision/pinch_proximity_radius")),
                w,
                h
            )
            if handedness == gestures.Hand.LEFT:
                sd.putBoolean("vision/left_pinch", index_thumb_pinch)
            elif handedness == gestures.Hand.RIGHT:
                sd.putBoolean("vision/right_pinch", index_thumb_pinch)

            # Draw pinch radii
            for digit in [henum.INDEX_FINGER_TIP, henum.THUMB_TIP]:
                if handedness == gestures.Hand.LEFT:
                    color = (0, 0, 255)
                elif handedness == gestures.Hand.RIGHT:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)

                lm = handLms.landmark[digit]
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), int(sd_get_value("vision/pinch_proximity_radius")),
                           color, thickness=2)

            # Detect middle finger
            middle_finger = gestures.middle_finger(handLms)
            print(middle_finger)

            # Fist detection
            fist = gestures.fist(handLms, h)
            if handedness == gestures.Hand.LEFT:
                sd.putBoolean("vision/left_fist", fist)
            elif handedness == gestures.Hand.RIGHT:
                sd.putBoolean("vision/right_fist", fist)

            # # Draw fist detection line
            # cv2.line(
            #     img,
            #     (fingertip_average_x - 10, fingertip_average_y),
            #     (fingertip_average_x + 10, fingertip_average_y),
            #     (0, 255, 0),
            #     thickness=3
            # )

    else:  # No hands detected
        sd.putNumber("vision/detected_hands", 0)

    c_time = time.time()
    frame_processing_time = c_time - p_time
    fps = 1 / frame_processing_time
    p_time = c_time
    sd.putNumber("vision/fps", fps)
    sd.putNumber("vision/frame_processing_time", frame_processing_time * 1000)  # convert to milliseconds

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
