import numpy as np


def ascii_render_tile_data_map(
        tile_data_map: TileDataMap,
        visited_map: Optional[np.ndarray] = None,
        tile_ascii=None,
        enable_row_col_delimiters=False,
) -> str:
    str_arr = np.empty(tile_data_map.tiles_map.shape, dtype=np.object)
    grid_x, grid_y = tile_data_map.tiles_map.shape
    for x in range(grid_x):
        for y in range(grid_y):
            tile: Tile = tile_data_map.tiles_map[x, y]
            if tile_ascii is None:
                if tile.kind is TileKind.Empty:
                    str_arr[x, y] = '×'
                elif tile.kind is TileKind.Unobstructed:
                    if visited_map is not None and visited_map[x, y]:
                        str_arr[x, y] = '_'
                    else:
                        str_arr[x, y] = '▒'
                elif tile.kind is TileKind.ThickWall:
                    str_arr[x, y] = '█'
                elif tile.kind is TileKind.Table:
                    str_arr[x, y] = '█'
                elif tile.kind is TileKind.Chair:
                    str_arr[x, y] = '█'
            else:
                str_arr[x, y] = tile_ascii(tile)
    for agent in tile_data_map.agents:
        if agent.direction is Direction.East:
            str_arr[agent.position.x, agent.position.y] = '►' if agent.kind is AgentKind.Robot else '→'
        elif agent.direction is Direction.North:
            str_arr[agent.position.x, agent.position.y] = '▲' if agent.kind is AgentKind.Robot else '↑'
        elif agent.direction is Direction.West:
            str_arr[agent.position.x, agent.position.y] = '◄' if agent.kind is AgentKind.Robot else '←'
        elif agent.direction is Direction.South:
            str_arr[agent.position.x, agent.position.y] = '▼' if agent.kind is AgentKind.Robot else '↓'
    if enable_row_col_delimiters:
        rows = list(np.apply_along_axis(lambda row: '|'.join([''] + list(row) + ['']), 1, str_arr))
        row_delim = '—'.join([''] + ['—'] * grid_y + ['']) + "\n"
        return row_delim.join([''] + [row + '\n' for row in rows] + [''])
    else:
        return '\n'.join(
            ['▓' * (2 * len(str_arr[0]) + 5)] +
            ['|'.join(['▓▓'] + row_arr + ['▓▓']) for row_arr in typing.cast(list, str_arr.tolist())] +
            ['▓' * (2 * len(str_arr[0]) + 5)]
        )