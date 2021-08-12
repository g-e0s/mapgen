from dataclasses import dataclass
import math
from enum import Enum
from typing import List
import gym
import numpy as np
from dungeon import DungeonGenerator, TileKind, Map


@dataclass
class Position:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))


class Orientation(Enum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3


@dataclass
class Agent:
    position: Position
    orientation: Orientation







class Dungeon(gym.Env):
    def __init__(self,
        width=64,
        height=64,
        max_rooms=25,
        min_room_xy=10,
        max_room_xy=25,
        observation_size: int = 11,
        view_radius: float = 5.0
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_rooms = max_rooms
        self.min_room_xy = min_room_xy
        self.max_room_xy = max_room_xy

        # observation parameters
        self.observation_size = observation_size
        self.view_radius = view_radius

        # map generator
        self._gen = DungeonGenerator(width, height, max_rooms, min_room_xy, max_room_xy)
        self._map = None
        self._explored_area = None
        self._agent_pos = None


    
        # The main API methods that users of this class need to know are:
        #     step
        #     reset
        #     render
        #     close
        #     seed

        # And set the following attributes:
        #     action_space: The Space object corresponding to valid actions
        #     observation_space: The Space object corresponding to valid observations
        #     reward_range: A tuple corresponding to the min and max possible rewards

    def reset(self):
        self._map = np.zeros((self._gen.height, self._gen.width))
        self._explored_area = np.zeros_like(self._map)

        self._gen.gen_level()
        # self._gen.gen_tiles_level()
        #print(self._gen.tiles["floor"])
        for i, row in enumerate(self._gen.level):
            #print((np.array(list(row)) == self._gen.tiles["floor"]).sum())
            self._map[i] = (np.array(list(row)) == TileKind.FREE)

        #self._map = self._gen.gen_level()

        # place agent
        xs, ys = np.nonzero(self._map)
        idx = np.random.randint(len(xs))
        agent_pos = Position(xs[idx], ys[idx])
        agent_orientation = Orientation(np.random.randint(4))
        self._agent_pos = Agent(agent_pos, agent_orientation)

        # df: place agent
        # row = list(self._gen.tiles_level[self._agent_pos.position.x])
        # row[self._agent_pos.position.y] = 'O'
        # self._gen.tiles_level[self._agent_pos.position.x] = ''.join(row)

        self._gen.level[self._agent_pos.position.x][self._agent_pos.position.y] = TileKind.AGENT

        obs = self.get_observation()
        return obs


    def get_observation(self):
        observation = np.zeros((3, self.observation_size, self.observation_size), dtype=int)
        x0, y0 = self._agent_pos.position
        x = x0 - (self.observation_size - 1) // 2
        y = y0 - (self.observation_size - 1) // 2
        w = h = self.observation_size

        x_max, y_max = self._map.shape

        # = max(-x, 0), max(-y, 0)

        xs_slice = np.clip([x, x+h], a_min=0, a_max=x_max)
        ys_slice = np.clip([y, y+w], a_min=0, a_max=y_max)

        map_slice = self._map[xs_slice[0] : xs_slice[1], ys_slice[0] : ys_slice[1]]
        explored_slice = self._explored_area[xs_slice[0] : xs_slice[1], ys_slice[0] : ys_slice[1]]

        observation[0, max(-x, 0) : max(-x, 0)+map_slice.shape[0], max(-y, 0) : max(-y, 0)+map_slice.shape[1]] = map_slice
        observation[1, max(-x, 0) : max(-x, 0)+explored_slice.shape[0], max(-y, 0) : max(-y, 0)+explored_slice.shape[1]] = explored_slice

        #observation = map_slice

        return Map(observation[0])









    def show_map(self):
        self._gen.show()
        print(self._map)
        print(f"Free area: {self._map.sum() / self._map.size * 100 :.2f}%")


def ascii_arr(arr):
    for row in arr:
        print(''.join(np.where(row, ".", "#")))




def visible_area(r, phi, size):
    x = np.zeros((size+1, size+1), dtype=bool)
    xs = ys = np.arange(size+1)
    x0 = y0 = size // 2
    inside_area = (xs[None, :] - x0) ** 2 + (ys[:, None] - y0) ** 2 <= (r ** 2)

    fov_area_pos = x

    if phi <= math.pi:
        fov_area_pos = ((xs[None, :] - x0) * math.tan(phi/2) >= ys[:, None] - y0)  & (ys[:, None] - y0 >= 0)
        fov_area_neg = ((xs[None, :] - x0) * math.tan(-phi/2) <= ys[:, None] - y0)  & (ys[:, None] - y0 < 0)

    else:

        fov_area_pos = ((xs[None, :] - x0) * math.tan(phi/2) <= ys[:, None] - y0)  & (ys[:, None] - y0 >= 0)
        fov_area_neg = ((xs[None, :] - x0) * math.tan(-phi/2) >= ys[:, None] - y0)  & (ys[:, None] - y0 < 0)


    return x | (inside_area & (fov_area_pos | fov_area_neg))



if __name__ == "__main__":
    env = Dungeon(32, 32)
    obs = env.reset()
    env.show_map()

    a = np.eye(5)
    ascii_arr(visible_area(5, math.pi / 2, 12))
    ascii_arr(visible_area(5, 3 * math.pi / 2, 12))

    # obs = env.reset()

    #ascii_arr(obs)
    print(obs)
    print(env._agent_pos)



