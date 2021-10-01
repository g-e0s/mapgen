import pytest
import numpy as np
from mapgen.agent import Agent, Move, Position, Orientation
from mapgen.map import Map, TileKind, TILES, Slice


sample_map = \
    """×××××××××××███××
       ×××××××××××█.█××
       ××××××××████.██×
       █████████.....█×
       █.......█.....█×
       █.............█×
       █.......█.....█×
       █.......███████×
       █.......█×××××××
       █████████×××××××"""

tile_dict = {v: k for k, v in TILES.items()}

sample_map_array = [[tile_dict[tile] for tile in row.strip()] for row in sample_map.split("\n")]



@pytest.mark.parametrize("observation_size, result",[
    (5, (Slice(10, 15, 0, 5), Slice(0, 5, 0, 5))),
    (7, (Slice(9, 16, 0, 6), Slice(0, 7, 1, 7))),
    (9, (Slice(8, 16, 0, 7), Slice(0, 8, 2, 9))),
])
def test_slice_coords(observation_size, result):
    map = Map(sample_map_array)
    agent = Agent(Position(x=12, y=2), Orientation(0), vision_radius=3)
    slice_coords = map.get_map_slice_coords(agent, observation_size)
    assert slice_coords == result


@pytest.mark.parametrize("orientation", [0, 1, 2, 3])
def test_get_observation(orientation):
    map = Map(sample_map_array)
    agent = Agent(Position(x=12, y=2), Orientation(orientation), vision_radius=3)


    obs = map.get_observation(agent, observation_size=7)

    expected_obs = \
    """×××××××
       ××███××
       ××█.█××
       ███.██×
       .....█×
       .....█×
       .....█×""".replace(' ', '')

    # rotate expected observation
    expected_obs_array = np.array([[x for x in row] for row in expected_obs.split("\n")])
    expected_obs_array = np.rot90(expected_obs_array, k=-orientation+1)
    expected_obs = "\n".join([''.join(row) for row in expected_obs_array])


    assert str(obs) == expected_obs


@pytest.mark.parametrize("orientation", [0, 1, 2, 3])
def test_visible_area(orientation):
    agent = Agent(Position(x=12, y=2), Orientation(orientation), vision_radius=3)

    expected_obs = \
        """×××.×××
           ××...××
           ×××.×××
           ×××.×××
           ×××××××
           ×××××××
           ×××××××""".replace(' ', '')

    # rotate expected observation
    expected_obs_array = np.array([[x for x in row] for row in expected_obs.split("\n")])
    expected_obs_array = np.rot90(expected_obs_array, k=orientation-1)
    expected_obs = "\n".join([''.join(row) for row in expected_obs_array])

    obs = '\n'.join([''.join(row) for row in np.where(agent.visible_area, ".", "×")])

    assert obs == expected_obs



@pytest.mark.parametrize("orientation", [0, 2, 3])
def test_get_explored_area(orientation):
    map = Map(sample_map_array)
    agent = Agent(Position(x=12, y=2), Orientation(orientation), vision_radius=3)

    map.update_explored_area(agent)


    obs = map.get_observation(agent, observation_size=7)

    expected_obs = \
    """×××.×××
       ××...××
       ×××.×××
       ×××.×××
       ×××××××
       ×××××××
       ×××××××""".replace(' ', '')

    print(map.explored_area)
    print(obs.explored_area)
    print(obs)


    assert obs.explored_area == expected_obs


def test_rotation():
    map = Map(sample_map_array)
    agent = Agent(Position(x=12, y=2), Orientation(0), vision_radius=3, vision_angle=np.deg2rad(100.0))

    map.update_explored_area(agent)

    total_explored = 0

    for _ in range(3):
        obs, explored, _ = map.step(agent, Move.TURN_LEFT, observation_size=7)
        total_explored += explored
        print(agent.orientation)

    expected_obs = \
    """×××.×××
       ×.....×
       ×.....×
       ×......
       ××...××
       ××...××
       ×××××××""".replace(' ', '')

    print(map.explored_area)
    print(obs.explored_area)
    print(obs)
    print(total_explored)


    assert obs.explored_area == expected_obs
