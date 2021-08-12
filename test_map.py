from map import Map, TileKind


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

tile_dict = {
    "█": TileKind.OCCUPIED,
    ".": TileKind.FREE,
    "×": TileKind.UNKNOWN
}

sample_map_array = [[tile_dict[tile] for tile in row] for row in sample_map.split("\n")]



def test_slice_coords():
    map = Map(sample_map_array)
    assert True
