import math


class Hand:
    LEFT = "Left"
    RIGHT = "Right"


class Axis:
    X = "x"
    Y = "y"


def circle_intersection(x0, y0, x1, y1, radius) -> bool:
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < (radius * 2)
