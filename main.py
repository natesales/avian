import collections
import sys
import time

import cv2
import mediapipe
import numpy

from classifiers import SWIPE_DIRECTIONS, Hand, GESTURES, SwipeDirection
from classifiers import pose
from classifiers.classify import Classifier
from smartdashboard import SmartDashboard

NETWORKTABLES_SERVER = sys.argv[1]
print(f"Using NetworkTables server {NETWORKTABLES_SERVER}")

# TODO: move to smartdashboard
DRAW_TO_ORIGIN = True

SD_DEFAULTS = {
    "avian/max_num_hands": 2,
    "avian/pinch_proximity_radius": 7,
    "avian/invert_horizontal": False,
    "avian/invert_vertical": False,
    "avian/black_background": False,
    "avian/pose_cache_size": 5,
    "avian/draw": True,
    "avian/detect_hand_pose": True,

    # Deadzones are in pixels above and below.
    # Not cumulative; the space between {forward,backward}_thresh is 2*deadzone)
    "avian/tank_middle_deadzone": 30,
    "avian/tank_top_bottom_deadzone": 50,

    # Immutable
    "avian/fps": 0,
    "avian/detected_hands": 0,
    "avian/calibrated": False,
}

for detection in GESTURES:
    SD_DEFAULTS[f"avian/left_{detection}"] = False
    SD_DEFAULTS[f"avian/right_{detection}"] = False

for direction in SWIPE_DIRECTIONS:
    SD_DEFAULTS[f"avian/{direction}_swipe_detected"] = False

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

pose_cache_size = int(sd.get("avian/pose_cache_size"))
left_hand_poses = collections.deque(maxlen=pose_cache_size)
right_hand_poses = collections.deque(maxlen=pose_cache_size)

# testing
swiped_distance = 0

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

    draw = sd.get("avian/draw")
    detect_hand_pose = sd.get("avian/detect_hand_pose")

    # Resize pose caches if needed
    if detect_hand_pose:
        pose_cache_size = int(sd.get("avian/pose_cache_size"))
        if pose_cache_size != left_hand_poses.maxlen:
            left_hand_poses = collections.deque(left_hand_poses, maxlen=pose_cache_size)
        if pose_cache_size != right_hand_poses.maxlen:
            right_hand_poses = collections.deque(right_hand_poses, maxlen=pose_cache_size)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if sd.get("avian/black_background"):
        h, w, _ = img.shape
        img = numpy.zeros((h, w, 3), numpy.uint8)

    # Draw segment lines
    color = (255, 255, 255)
    size = 3
    middle_deadzone = int(sd.get("avian/tank_middle_deadzone"))
    top_bottom_deadzone = int(sd.get("avian/tank_top_bottom_deadzone"))
    forward_thresh = int(h / 2) - middle_deadzone
    backward_thresh = int(h / 2) + middle_deadzone
    forward_max = top_bottom_deadzone
    backward_max = h - top_bottom_deadzone
    # cv2.line(img, (int(w / 2), 0), (int(w / 2), h), color, size)  # Vertical line
    cv2.line(img, (0, forward_thresh), (w, forward_thresh), color, size)  # Forward area
    cv2.line(img, (0, backward_thresh), (w, backward_thresh), color, size)  # Backward area
    cv2.line(img, (0, forward_max), (w, forward_max), color, size)  # Forward max
    cv2.line(img, (0, backward_max), (w, backward_max), color, size)  # Backward max

    if results.multi_hand_landmarks:
        sd.set("avian/detected_hands", len(results.multi_hand_landmarks))
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Detect left/right hand
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label.lower()

            if draw:
                # Draw hand matrix
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
            if detect_hand_pose:
                hand_pose = pose.hand_pose(hand_landmarks, w, h)
                if handedness == Hand.LEFT:
                    left_hand_poses.append(hand_pose)
                elif handedness == Hand.RIGHT:
                    right_hand_poses.append(hand_pose)

            # Gesture classifier
            gesture = classifier.classify(hand_landmarks)
            if gesture != "":
                sd.set(f"avian/{handedness}_{gesture}", True)

            # Set all other gestures to false
            for g in GESTURES:
                if g != gesture:
                    sd.set(f"avian/{handedness}_{g}", False)

            # # draw line to origin
            # if DRAW_TO_ORIGIN and len(right_hand_poses) > 0:
            #     # print(right_hand_poses[0])
            #     lm = hand_landmarks.landmark[mediapipe.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            #     angle = numpy.arctan((-lm.y + 0.5) / (lm.x - 0.5))
            #     if lm.x < 0.5:
            #         angle += numpy.pi
            #     if angle < 0:
            #         angle = 2 * numpy.pi + angle
            #     angle *= (180 / numpy.pi)
            #     cv2.line(img, (int(lm.x * w), int(lm.y * h)), (int(w / 2), int(h / 2)), (255, 255, 255), 10)
            #     cv2.putText(img, str(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #
            if len(right_hand_poses) > 0:
                sd.set("avian/right_swipe_detected", pose.hand_swipe(right_hand_poses, 100, SwipeDirection.LEFT))
            if len(left_hand_poses) > 0:
                sd.set("avian/left_swipe_detected", pose.hand_swipe(left_hand_poses, 100, SwipeDirection.RIGHT))

            if detect_hand_pose:
                # Tank drive
                if handedness == Hand.LEFT:
                    poses = left_hand_poses
                elif handedness == Hand.RIGHT:
                    poses = right_hand_poses
                else:
                    raise RuntimeError(f"Invalid handedness value: unexpected \"{handedness}\"")
                avg_y = int(sum([p[1] for p in poses]) / len(poses))  # p[1] for y coordinate in (x, y) tuple
                if draw:
                    x = int(w / 2)
                    cv2.drawMarker(img, ((x - 15 if handedness == Hand.LEFT else x + 15), avg_y), color, thickness=size)
                if avg_y < forward_thresh:
                    sd.set(f"avian/{handedness}_tank", (avg_y - forward_thresh) / (forward_max - forward_thresh))
                elif avg_y > backward_thresh:
                    sd.set(f"avian/{handedness}_tank",
                           -1 * ((avg_y - backward_thresh) / (backward_max - backward_thresh)))
                else:  # Deadzone
                    sd.set(f"avian/{handedness}_tank", 0)

    else:  # No hands detected
        sd.set("avian/detected_hands", 0)
        sd.set(f"avian/left_tank", 0)
        sd.set(f"avian/right_tank", 0)
        for detection in GESTURES:
            sd.set(f"avian/left_{detection}", False)
            sd.set(f"avian/right_{detection}", False)

    # Calculate FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    fps_deque.append(fps)
    sd.set("avian/fps", int(sum(fps_deque) / len(fps_deque)))
    cv2.putText(img, str(swiped_distance), (w-100, h-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if draw:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27:  # escape for exit
            cv2.destroyAllWindows()
            exit(0)
        elif key == 105:  # i for invert
            sd.set("avian/invert_horizontal", not sd.get("avian/invert_horizontal"))
    else:
        cv2.destroyAllWindows()
