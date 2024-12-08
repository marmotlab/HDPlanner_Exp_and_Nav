import numpy as np
import imageio
import os
from skimage.morphology import label

from parameter import *


def get_cell_position_from_coords(coords, map_info):
    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)
    if cell_position.shape[0] == 1:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == 255)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = (map_info.map == 255).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_local_node_coords(location, local_map_info):
    x_min = (local_map_info.map_origin_x // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    y_min = (local_map_info.map_origin_y // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    x_max = ((local_map_info.map_origin_x + local_map_info.map.shape[1] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
    y_max = ((local_map_info.map_origin_y + local_map_info.map.shape[0] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = get_free_and_connected_map(location, local_map_info)

    indices = []
    nodes_cells = get_cell_position_from_coords(nodes, local_map_info)
    for i, cell in enumerate(nodes_cells):
        if cell[1] >= free_connected_map.shape[0] or cell[0] >= free_connected_map.shape[1]:
            continue
        # if free_connected_map[cell[1], cell[0]] == 1:
        if local_map_info.map[cell[1], cell[0]] == 255:
            indices.append(i)
    indices = np.array(indices)
    nodes = nodes[indices]
        
    return nodes, free_connected_map

def get_all_node_coords(ground_truth_info):
    x_min = (ground_truth_info.map_origin_x // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    y_min = (ground_truth_info.map_origin_y // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    x_max = ((ground_truth_info.map_origin_x + ground_truth_info.map.shape[1] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
    y_max = ((ground_truth_info.map_origin_y + ground_truth_info.map.shape[0] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    indices = []
    nodes_cells = get_cell_position_from_coords(nodes, ground_truth_info)
    for i, cell in enumerate(nodes_cells):
        if ground_truth_info.map[cell[1], cell[0]] == 255:
            indices.append(i)
    indices = np.array(indices)
    nodes = nodes[indices]
        
    return nodes


def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == 127) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == 255)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info)
    return frontier_coords


def get_partial_map_from_center(original_map_info, center_coords, partial_map_size):
    partial_map_origin_x = (center_coords[
                              0] - partial_map_size / 2) // NODE_RESOLUTION * NODE_RESOLUTION
    partial_map_origin_y = (center_coords[
                              1] - partial_map_size / 2) // NODE_RESOLUTION * NODE_RESOLUTION
    partial_map_top_x = partial_map_origin_x + partial_map_size
    partial_map_top_y = partial_map_origin_y + partial_map_size

    min_x = original_map_info.map_origin_x
    min_y = original_map_info.map_origin_y
    max_x = original_map_info.map_origin_x + original_map_info.cell_size * original_map_info.map.shape[1]
    max_y = original_map_info.map_origin_y + original_map_info.cell_size * original_map_info.map.shape[0]

    if partial_map_origin_x < min_x:
        partial_map_origin_x = min_x
    if partial_map_origin_y < min_y:
        partial_map_origin_y = min_y
    if partial_map_top_x > max_x:
        partial_map_top_x = max_x
    if partial_map_top_y > max_y:
        partial_map_top_y = max_y

    partial_map_origin_x = np.around(partial_map_origin_x, 1)
    partial_map_origin_y = np.around(partial_map_origin_y, 1)
    partial_map_top_x = np.around(partial_map_top_x, 1)
    partial_map_top_y = np.around(partial_map_top_y, 1)

    partial_map_origin = np.array([partial_map_origin_x, partial_map_origin_y])
    partial_map_origin_in_global_map = get_cell_position_from_coords(partial_map_origin, original_map_info)

    partial_map_top = np.array([partial_map_top_x, partial_map_top_y])
    partial_map_top_in_global_map = get_cell_position_from_coords(partial_map_top, original_map_info)

    partial_map = original_map_info.map[
                partial_map_origin_in_global_map[1]:partial_map_top_in_global_map[1],
                partial_map_origin_in_global_map[0]:partial_map_top_in_global_map[0]]

    partial_map_info = Map_info(partial_map, partial_map_origin_x, partial_map_origin_y, original_map_info.cell_size)

    return partial_map_info


def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == 1:
            collision = True
            break
        if k == 127:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class Map_info:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

