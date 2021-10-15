import math


class Hand:
    LEFT = "left"
    RIGHT = "right"


class Axis:
    X = "x"
    Y = "y"


class Direction:
    ABOVE = "above"
    BELOW = "below"


class Gesture:
    PINCH = "pinch"
    FIST = "fist"
    MIDDLE_FINGER = "middle_finger"
    INDEX_FINGER = "index_finger"


def circle_intersection(x0, y0, x1, y1, radius) -> bool:
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < (radius * 2)
