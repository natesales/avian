import collections
import sys
import time

import cv2
import mediapipe
import numpy

from classifiers import Hand, GESTURES, Gesture
from classifiers import pose
from classifiers.classify import Classifier
from smartdashboard import SmartDashboard

NETWORKTABLES_SERVER = sys.argv[1]
print(f"Using NetworkTables server {NETWORKTABLES_SERVER}")

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
    "avian/angle": 0,
}

for detection in GESTURES:
    SD_DEFAULTS[f"avian/left_{detection}"] = False
    SD_DEFAULTS[f"avian/right_{detection}"] = False

cap = cv2.VideoCapture(0)
fps_deque = collections.deque(maxlen=10)


def sd_change_listener(table, key, value, isNew):
    # Create new hand processor if max_num_hands changes
    if key == "avian/max_num_hands":
        global hands
        hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("avian/max_num_hands")))


sd = SmartDashboard(NETWORKTABLES_SERVER, SD_DEFAULTS, sd_change_listener, "avian/max_num_hands")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cam_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Opened {width}x{height}@{cam_fps} camera")

classifier = Classifier(width, height, int(sd.get("avian/pinch_proximity_radius")))
hands = mediapipe.solutions.hands.Hands(max_num_hands=int(sd.get("avian/max_num_hands")))

results = None
p_time = 0

left_hand_poses = collections.deque(maxlen=POSE_CACHE_SIZE)
right_hand_poses = collections.deque(maxlen=POSE_CACHE_SIZE)

while True:
    success, img = cap.read()
    if not success:
        print("Unable to read from video capture device")
        exit(1)
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
            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label.lower()

            # Draw hand matrix
            if DRAW:
                mediapipe.solutions.drawing_utils.draw_landmarks(img, hand_landmarks,
                                                                 mediapipe.solutions.hands.HAND_CONNECTIONS)

                # Draw pinch radii
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

            # Calibration
            if not sd.get("avian/calibrated"):
                # If the first hand seen is the left (actually the right), flip the image
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

            # Gesture classifier
            gesture = classifier.classify(hand_landmarks)
            if gesture != "":
                sd.set(f"avian/{handedness}_{gesture}", True)

            if len(right_hand_poses) > 0:
                # print(len(left_hand_poses))
                handX, handY = right_hand_poses[0]
                angle = numpy.arctan((handY*2-1) / (handX*2-1)) * (180/numpy.pi)
                sd.set("avian/angle", int(angle))

            # Set all other gestures to false
            for g in GESTURES:
                if g != gesture:
                    sd.set(f"avian/{handedness}_{g}", False)

    else:  # No hands detected
        sd.set("avian/detected_hands", 0)
        for detection in GESTURES:
            sd.set(f"avian/left_{detection}", False)
            sd.set(f"avian/right_{detection}", False)

    # Calculate FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    fps_deque.append(fps)
    sd.set("avian/fps", int(sum(fps_deque) / len(fps_deque)))

    if DRAW:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            exit(0)
