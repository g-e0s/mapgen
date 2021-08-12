from dataclasses import dataclass
import math
from enum import Enum
from typing import List
import gym
import numpy as np
from dungeon import DungeonGenerator, TileKind
from map import Map, Move
from agent import Agent, Position, Orientation







class Dungeon(gym.Env):
    def __init__(self,
        width=64,
        height=64,
        max_rooms=25,
        min_room_xy=10,
        max_room_xy=25,
        observation_size: int = 11,
        vision_radius: int = 5
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_rooms = max_rooms
        self.min_room_xy = min_room_xy
        self.max_room_xy = max_room_xy

        # observation parameters
        self.observation_size = observation_size
        self.vision_radius = vision_radius

        # map generator
        self._gen = DungeonGenerator(width, height, max_rooms, min_room_xy, max_room_xy)
        self._map: Map = None
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
        # generate level

        self._gen.gen_level()
        self._map = Map(self._gen.level)

        # place agent
        x, y = self._map.get_random_free_position()
        self._agent = Agent(Position(x, y), Orientation(np.random.randint(4)), vision_radius=self.vision_radius, vision_angle=math.pi / 2)
        _ = self._map.update_explored_area(self._agent, align_with_map=True)

        obs = self._map.get_observation(self._agent, self.observation_size)
        return obs

    def step(self, action: Move):
        obs, explored, success = self._map.step(self._agent, action, self.observation_size)
        return obs, explored, success


if __name__ == "__main__":
    env = Dungeon(32, 32)
    obs = env.reset()
    for _ in range(100):
        action = np.random.choice(3)
        obs, explored, success = env.step(action)
        if success:
            break

        print(env._map.explored_area)
