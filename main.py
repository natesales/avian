import logging
import math
import time

import cv2
import mediapipe

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_COLOR = (255, 255, 255)
FONT_SCALE = 2
FONT_THICKNESS = 2
INVERT_VERTICAL = False
INVERT_HORIZONTAL = True

PINCH_PROXIMITY_RADIUS = 10

cap = cv2.VideoCapture(0)

hands_sl = mediapipe.solutions.hands
hands = hands_sl.Hands()

results = None
p_time = 0


def circle_intersection(x0, y0, x1, y1, radius) -> bool:
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < (radius * 2)


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
        cv2.putText(img, "Hands: " + str(len(results.multi_hand_landmarks)), (5, 120), FONT, FONT_SCALE, FONT_COLOR,
                    FONT_THICKNESS)
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # Draw hand matrix
            mediapipe.solutions.drawing_utils.draw_landmarks(img, handLms, hands_sl.HAND_CONNECTIONS)

            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            # Pinch detection
            h, w, c = img.shape
            index_finger = handLms.landmark[hands_sl.HandLandmark.INDEX_FINGER_TIP]
            thumb = handLms.landmark[hands_sl.HandLandmark.THUMB_TIP]
            intersecting = circle_intersection(
                int(index_finger.x * w), int(index_finger.y * h),
                int(thumb.x * w), int(thumb.y * h),
                PINCH_PROXIMITY_RADIUS
            )
            if intersecting:
                if handedness == "Left":
                    cv2.putText(img, "Left pinch", (5, 60), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                elif handedness == "Right":
                    cv2.putText(img, "Right pinch", (5, 90), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

            # Draw white dot
            for digit in [hands_sl.HandLandmark.INDEX_FINGER_TIP, hands_sl.HandLandmark.THUMB_TIP]:
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
    cv2.putText(img, str(int(fps)), (w-50, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
