import logging
import math
import time

import cv2
import mediapipe
import numpy

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (255, 255, 255)
FONT_SCALE = 2
FONT_THICKNESS = 2
INVERT_VERTICAL = False
INVERT_HORIZONTAL = True
MAX_NUM_HANDS = 1

PINCH_PROXIMITY_RADIUS = 10

cap = cv2.VideoCapture(0)

hands = mediapipe.solutions.hands.Hands(max_num_hands=MAX_NUM_HANDS)

results = None
p_time = 0


def put_text(img, text, pos):
    cv2.putText(img, text, pos, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)


def circle_intersection(x0, y0, x1, y1, radius) -> bool:
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < (radius * 2)


def pinch_detect(lms, finger1: mediapipe.solutions.hands.HandLandmark, finger2: mediapipe.solutions.hands.HandLandmark) -> bool:
    finger1_lm = lms.landmark[finger1]
    finger2_lm = lms.landmark[finger2]
    h, w, _ = img.shape

    return circle_intersection(
        int(finger1_lm.x * w), int(finger1_lm.y * h),
        int(finger2_lm.x * w), int(finger2_lm.y * h),
        PINCH_PROXIMITY_RADIUS
    )


while True:
    success, img = cap.read()
    if not success:
        logging.fatal("Unable to read from video capture device")
    if INVERT_VERTICAL:
        img = cv2.flip(img, 0)
    if INVERT_HORIZONTAL:
        img = cv2.flip(img, 1)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        h, w, _ = img.shape
        img = numpy.zeros((h, w, 3), numpy.uint8)

        put_text(img, "Hands: " + str(len(results.multi_hand_landmarks)), (5, 120))
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # Draw hand matrix
            mediapipe.solutions.drawing_utils.draw_landmarks(img, handLms, mediapipe.solutions.hands.HAND_CONNECTIONS)

            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            # Pinch detection
            index_thumb_pinch = pinch_detect(handLms, mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mediapipe.solutions.hands.HandLandmark.THUMB_TIP)

            if handedness == "Left":
                put_text(img, "Left pinch: " + str(index_thumb_pinch), (5, 60))
            elif handedness == "Right":
                put_text(img, "Right pinch: " + str(index_thumb_pinch), (5, 90))

            # Draw white dot
            for digit in [mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mediapipe.solutions.hands.HandLandmark.THUMB_TIP]:
                if handedness == "Left":
                    color = (0, 0, 255)
                elif handedness == "Right":
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)

                lm = handLms.landmark[digit]
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), PINCH_PROXIMITY_RADIUS, color, thickness=2)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # FPS indicator
    h, w, c = img.shape
    put_text(img, str(int(fps)), (w - 50, 30))

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
