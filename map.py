from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import List, Tuple

import numpy as np

from agent import Agent, Move, Orientation, Position
from dungeon import TileKind


TILES = {
    TileKind.OCCUPIED: "█",
    TileKind.FREE: ".",
    TileKind.UNKNOWN: "×",
    TileKind.AGENT: "O",
    TileKind.EXPLORED: "_"
}

@dataclass
class Slice:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def __iter__(self):
        return iter((self.x_min, self.x_max, self.y_min, self.y_max))

class Map:
    def __init__(
        self,
        tiles: List[List[TileKind]],
        explored_area: List[List[bool]] = None
    ):
        self.tiles = np.array(tiles).astype(int)
        self._explored_area = np.array(explored_area) if explored_area is not None else np.zeros_like(self.tiles, dtype=bool)

        self._render = '\n'.join([''.join([TILES[TileKind(tile)] for tile in row]) for row in tiles])

        self._visible_cells = np.sum(np.where(self.tiles != TileKind.UNKNOWN, True, False))
        self._total_explored = 0

    @property
    def size(self):
        return self.tiles.shape

    def __str__(self):
        return self._render

    def show(self, agent: Agent):
        tiles = self.tiles.copy()
        tiles = np.where(self._explored_area, tiles, TileKind.UNKNOWN)
        tiles[agent.position.y, agent.position.x] = TileKind.AGENT
        render = '\n'.join([''.join([TILES[TileKind(tile)] for tile in row]) for row in tiles])
        return render


    @property
    def explored_area(self):
        return '\n'.join([''.join(row) for row in np.where(self._explored_area, ".", "×")])


    def step(self, agent: Agent, move: Move, observation_size: int):
        position = deepcopy(agent.position)
        moved = False
        if move == Move.FORWARD:
            if agent.orientation == Orientation.EAST:
                position.x += 1
            elif agent.orientation == Orientation.NORTH:
                position.y += 1
            elif agent.orientation == Orientation.WEST:
                position.x -= 1
            elif agent.orientation == Orientation.SOUTH:
                position.y -= 1
            
            # check validity
            try:
                if self.tiles[position.y, position.x] == TileKind.FREE:
                    agent.position = position
                    moved = True
                else:
                    moved = False
            except IndexError as err:
                # print(self.show(agent))
                # print(agent.position)
                # raise err
                moved = False

        else:
            orientation = agent.orientation + move
            if orientation > Orientation.SOUTH:
                orientation = Orientation.EAST
            elif orientation < Orientation.EAST:
                orientation = Orientation.SOUTH
            agent.orientation = orientation
            moved = True

        explored = self.update_explored_area(agent, align_with_map=True)
        success = self._total_explored == self._visible_cells
        obs = self.get_observation(agent, observation_size)
        return obs, explored, success


    def get_random_free_position(self) -> Tuple[int, int]:

        ys, xs = np.nonzero(np.where(self.tiles == TileKind.FREE, True, False))
        idx = np.random.randint(len(xs))
        return xs[idx], ys[idx]


    def get_map_slice_coords(self, agent: Agent, slice_size: int) -> Tuple[Slice, Slice]:
        x0, y0 = agent.position

        w = h = slice_size
        x = x0 - (slice_size - 1) // 2
        y = y0 - (slice_size - 1) // 2

        x_max, y_max = self.size


        x_min, x_max = np.clip([x, x+h], a_min=0, a_max=self.size[1])
        y_min, y_max = np.clip([y, y+w], a_min=0, a_max=self.size[0])

        map_slice = Slice(x_min, x_max, y_min, y_max)
        observation_slice = Slice(max(-x, 0), max(-x, 0) + x_max - x_min, max(-y, 0), max(-y, 0) + y_max - y_min)

        return map_slice, observation_slice


    def update_explored_area(self, agent: Agent, align_with_map: bool = False) -> int:

        map_slice, obs_slice = self.get_map_slice_coords(agent, agent.visible_area.shape[0])

        # get observation_slice
        x_min, x_max, y_min, y_max = obs_slice
        vis_area_slice = agent.visible_area[y_min : y_max, x_min : x_max]

        # add new observation
        x_min, x_max, y_min, y_max = map_slice

        old_explored = np.sum(self._explored_area)
        self._explored_area[y_min : y_max, x_min : x_max] += vis_area_slice.astype(bool)


        # align with map
        if align_with_map:
            self._explored_area *= np.where(self.tiles != TileKind.UNKNOWN, True, False)

        new_explored = np.sum(self._explored_area)

        self._total_explored = new_explored
        return new_explored - old_explored



    def get_observation(self, agent, observation_size: int):
        observation = np.zeros((3, observation_size, observation_size), dtype=int)

        map_slice, obs_slice = self.get_map_slice_coords(agent, observation_size)

        # get map slices
        x_min, x_max, y_min, y_max = map_slice
        map_slice = self.tiles[y_min : y_max, x_min : x_max]
        explored_slice = self._explored_area[y_min : y_max, x_min : x_max]

        # set by observation slice
        x_min, x_max, y_min, y_max = obs_slice

        observation[0, y_min : y_max, x_min : x_max] = map_slice
        observation[1, y_min : y_max, x_min : x_max] = explored_slice


        # rotate
        observation[0] = np.rot90(observation[0], k=-agent.orientation + 1)
        observation[1] = np.rot90(observation[1], k=-agent.orientation + 1)

        return Map(tiles=observation[0], explored_area=observation[1])
