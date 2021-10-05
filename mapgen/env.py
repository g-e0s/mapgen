import math
import gym
from gym import spaces
import numpy as np
from mapgen.dungeon import DungeonGenerator
from mapgen.map import Map, Move
from mapgen.agent import Agent, Position, Orientation


class Dungeon(gym.Env):
    def __init__(self,
        width=64,
        height=64,
        max_rooms=25,
        min_room_xy=10,
        max_room_xy=25,
        observation_size: int = 11,
        vision_radius: int = 5,
        max_steps: int = 2000
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
        self._step = 0
        self._max_steps = max_steps

        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 4])
        self.action_space = spaces.Discrete(3)

    def reset(self):

        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        # generate level

        self._gen.gen_level()
        self._map = Map(self._gen.level)

        # place agent
        x, y = self._map.get_random_free_position()
        self._agent = Agent(Position(x, y), Orientation(np.random.randint(4)), vision_radius=self.vision_radius, vision_angle=math.pi / 2)
        _ = self._map.update_explored_area(self._agent, align_with_map=True)
        self._step = 1

        observation = self._map.get_observation(self._agent, self.observation_size)
        return observation

    def step(self, action: int):
        
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
              - step: current step number
              - total_cells: total number of visible cells for current map
              - total_explored: total number of explored cells (map is solved when total_explored == total_cells)
              - new_explored: number of explored cells during this step
              - moved: whether an agent made a move (didn't collide with an obstacle)
        """
        action = Move(action-1)
        observation, explored, done, moved = self._map.step(self._agent, action, self.observation_size)

        # set reward as a fraction of new explored cells (so total reward is 1.0)
        reward = explored /  self._map._visible_cells

        info = {
            "step": self._step,
            "total_cells": self._map._visible_cells,
            "total_explored": self._map._total_explored,
            "new_explored": explored,
            "avg_explored_per_step": self._map._total_explored / self._step,
            "moved": moved
        }
        self._step += 1

        return observation, reward , done or self._step == self._max_steps, info


    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == "rgb_array":
            raise NotImplementedError
        elif mode == "human":
            print(f"Step {self._step}")
            print(self._map.show(self._agent))
            print(f"Explored: {self._map._total_explored}/{self._map._visible_cells} cells ({self._map._total_explored / self._map._visible_cells * 100:.2f}%)")
