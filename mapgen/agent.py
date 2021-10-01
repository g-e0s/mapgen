from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import List, Tuple

import numpy as np


@dataclass
class Position:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))


class Orientation(IntEnum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3


class Move(IntEnum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = -1



class Agent:
    def __init__(self,
        position: Position,
        orientation: Orientation,
        vision_radius: float = 5.0,
        vision_angle: float = math.pi / 2
    ):
        self.position = position
        self.orientation = orientation
        self._vision_radius = vision_radius
        self._vision_angle = vision_angle

        self._visible_area_canonical = self._construct_canonical_visible_area()

    @property
    def visible_area(self) -> np.ndarray:
        return np.rot90(self._visible_area_canonical, k=int(self.orientation))


    def _construct_canonical_visible_area(self) -> np.ndarray:
        # set area
        area_size = 2 * self._vision_radius + 1
        x = np.zeros((area_size, area_size), dtype=bool)

        # limit by vision radius

        xs = ys = np.arange(area_size)
        x0 = y0 = self._vision_radius
        inside_area = (xs[None, :] - x0) ** 2 + (ys[:, None] - y0) ** 2 <= (self._vision_radius ** 2)

        # limit by vision angle
        if self._vision_angle <= math.pi:
            fov_area_pos = ((xs[None, :] - x0) * math.tan(self._vision_angle / 2) >= ys[:, None] - y0)  & (ys[:, None] - y0 >= 0)
            fov_area_neg = ((xs[None, :] - x0) * math.tan(-self._vision_angle / 2) <= ys[:, None] - y0)  & (ys[:, None] - y0 < 0)

        else:

            fov_area_pos = ((xs[None, :] - x0) * math.tan(self._vision_angle / 2) <= ys[:, None] - y0)  & (ys[:, None] - y0 >= 0)
            fov_area_neg = ((xs[None, :] - x0) * math.tan(-self._vision_angle / 2) >= ys[:, None] - y0)  & (ys[:, None] - y0 < 0)


        return x | (inside_area & (fov_area_pos | fov_area_neg))
