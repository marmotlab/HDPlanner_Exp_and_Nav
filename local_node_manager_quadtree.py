import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils import *
from parameter import *
import quads


class Local_node_manager:
    def __init__(self, plot=False):
        self.local_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.all_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.init_target_frontiers = True
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.local_nodes_dict.find(key)
        return exist

    def check_node_exist_in_all_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.all_nodes_dict.find(key)
        return exist
    
    def add_node_to_dict(self, coords, local_frontiers, extended_local_map_info):
        key = (coords[0], coords[1])
        node = Local_node(coords, local_frontiers, extended_local_map_info)
        self.local_nodes_dict.insert(point=key, data=node)

    def add_node_to_all_dict(self, coords):
        key = (coords[0], coords[1])
        node = All_node(coords)
        self.all_nodes_dict.insert(point=key, data=node)

    def update_local_graph(self, robot_location, local_frontiers, local_map_info, extended_local_map_info, global_map_info, target_location):
        local_node_coords, _ = get_local_node_coords(robot_location, local_map_info)
        for coords in local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                self.add_node_to_dict(coords, local_frontiers, extended_local_map_info)
            else:
                node = node.data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(local_frontiers, extended_local_map_info)
        x_min = (extended_local_map_info.map_origin_x // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        y_min = (extended_local_map_info.map_origin_y // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        x_max = ((extended_local_map_info.map_origin_x + extended_local_map_info.map.shape[1] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
        y_max = ((extended_local_map_info.map_origin_y + extended_local_map_info.map.shape[0] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
        
        if x_min <= target_location[0] <= x_max and y_min <= target_location[1] <= y_max:
            target_cell = get_cell_position_from_coords(target_location, global_map_info)
            if global_map_info.map[target_cell[1], target_cell[0]] == 255:
                # print("update target frontier")
                node = self.check_node_exist_in_dict(target_location)
                node = node.data
                if self.init_target_frontiers:
                    node.initialize_observable_frontiers(local_frontiers, extended_local_map_info)
                    self.init_target_frontiers = False
                elif node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(local_frontiers, extended_local_map_info)
            
        for coords in local_node_coords:
            node = self.local_nodes_dict.find((coords[0], coords[1])).data

            plot_x = self.x if self.plot else None
            plot_y = self.y if self.plot else None
            node.update_neighbor_nodes(extended_local_map_info, global_map_info, self.local_nodes_dict, target_location, plot_x, plot_y)

    def update_all_graph(self, ground_truth_info, target_location):
        self.add_node_to_all_dict(target_location)
        all_node_coords = get_all_node_coords(ground_truth_info)
        new_all_node_coords = []
        for coords in all_node_coords:
            node = self.check_node_exist_in_all_dict(coords)
            if node is None:
                self.add_node_to_all_dict(coords)
        for node in self.all_nodes_dict.__iter__():
            node = node.data
            node.update_neighbor_nodes(ground_truth_info, self.all_nodes_dict, target_location)

    def get_all_node_graph(self, robot_location, target_location, global_map_info):
        all_node_coords = []
        for node in self.local_nodes_dict.__iter__():
            coords = node.data.coords
            if coords[0] == target_location[0] and coords[1] == target_location[1]:
                all_node_coords.append(coords)
                continue
            cell = get_cell_position_from_coords(coords, global_map_info)
            if cell[1] < global_map_info.map.shape[0] and cell[0] < global_map_info.map.shape[1]:
                if global_map_info.map[cell[1], cell[0]] == 255:
                    all_node_coords.append(coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)
        # sorted_centers = self.find_sorted_centers(utility, all_node_coords, target_location, global_map_info)
        sorted_centers = self.find_sorted_centers_from_all_nodes_dict(utility, all_node_coords, target_location, global_map_info)
        # optimal_center = self.find_optimal_center(sorted_centers, target_location, global_map_info)
        optimal_center, optimal_center_index_in_center_lst = self.find_optimal_center_from_all_nodes_dict(sorted_centers, target_location, global_map_info)
        adjacent_matrix = self.find_centers_of_target(all_node_coords, sorted_centers, target_location, adjacent_matrix)
        center_beacon = np.zeros((n_nodes, 1))
        for node in sorted_centers:
            index = np.argwhere(local_node_coords_to_check == node[0] + node[1]*1j)
            if index or index == [[0]]:
                index = index[0][0]
            center_beacon[index] = 1
        current_index = np.argwhere(local_node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices, sorted_centers, center_beacon, optimal_center, optimal_center_index_in_center_lst

    def find_sorted_centers(self, utility, all_node_coords, target_location, global_map_info):
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        center_indices = []
        non_zero_utility_node_indices = np.argwhere(utility > 0)[:, 0].tolist()
        non_zero_utility_node_coords = all_node_coords[non_zero_utility_node_indices]
        centers = non_zero_utility_node_coords
        if centers.shape[0] >= MIN_CENTERS_BEFORE_SPARSIFY:
            knn = NearestNeighbors(radius=SPARSIFICATION_CENTERS_KNN_RAD)
            knn.fit(centers)
            key_center_indices = []
            coverd_center_indices = []
            for i, center in enumerate(centers):
                if i in coverd_center_indices:
                    pass
                else:
                    _, indices = knn.radius_neighbors(center.reshape(1,2))
                    key_center_indices.append(i)
                    for index in indices[0]:
                        node = centers[index]
                        if not check_collision(center, node, global_map_info):
                            coverd_center_indices.append(index)
                    #coverd_center_indices += indices[0].tolist()
            for i in key_center_indices:
                tmp = centers[i]
                center_indices.append(np.argwhere(local_node_coords_to_check == tmp[0] + tmp[1] * 1j)[0][0])
        else:
            for center in centers:
                center_indices.append(np.argwhere(local_node_coords_to_check == center[0] + center[1] * 1j)[0][0])
        target_cell = get_cell_position_from_coords(target_location, global_map_info)
        if global_map_info.map[target_cell[1], target_cell[0]] == 255:
            # print("add target as target")
            center_indices.append(np.argwhere(local_node_coords_to_check == target_location[0] + target_location[1] * 1j)[0][0])
        center_indices = list(set(center_indices))
        centers = all_node_coords[center_indices]
        sorted_centers = np.array(sorted(centers, key=lambda center: np.linalg.norm(center - target_location, axis=0)))
        return sorted_centers

    def find_sorted_centers_from_all_nodes_dict(self, utility, all_node_coords, target_location, global_map_info):
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        center_indices = []
        non_zero_utility_node_indices = np.argwhere(utility > 0)[:, 0].tolist()
        non_zero_utility_node_coords = all_node_coords[non_zero_utility_node_indices]
        centers = non_zero_utility_node_coords
        if centers.shape[0] >= MIN_CENTERS_BEFORE_SPARSIFY:
            knn = NearestNeighbors(radius=SPARSIFICATION_CENTERS_KNN_RAD)
            knn.fit(centers)
            key_center_indices = []
            coverd_center_indices = []
            for i, center in enumerate(centers):
                if i in coverd_center_indices:
                    pass
                else:
                    _, indices = knn.radius_neighbors(center.reshape(1,2))
                    key_center_indices.append(i)
                    for index in indices[0]:
                        node = centers[index]
                        if not check_collision(center, node, global_map_info):
                            coverd_center_indices.append(index)
                    #coverd_center_indices += indices[0].tolist()
            for i in key_center_indices:
                tmp = centers[i]
                center_indices.append(np.argwhere(local_node_coords_to_check == tmp[0] + tmp[1] * 1j)[0][0])
        else:
            for center in centers:
                center_indices.append(np.argwhere(local_node_coords_to_check == center[0] + center[1] * 1j)[0][0])
        target_cell = get_cell_position_from_coords(target_location, global_map_info)
        if global_map_info.map[target_cell[1], target_cell[0]] == 255:
            # print("add target as target")
            center_indices.append(np.argwhere(local_node_coords_to_check == target_location[0] + target_location[1] * 1j)[0][0])
        center_indices = list(set(center_indices))
        # tmp bug fix
        if len(center_indices) == 0:
            print("should add target to centers")
            center_indices.append(np.argwhere(local_node_coords_to_check == target_location[0] + target_location[1] * 1j)[0][0])
        centers = all_node_coords[center_indices]
        sorted_centers = np.array(sorted(centers, key=lambda center: self.a_star_for_all_nodes_dict(center, target_location)[1]))
        return sorted_centers

    def find_optimal_center_from_all_nodes_dict(self, centers, target_location, global_map_info):
        assert len(centers) > 0, "should add more centers"
        target_cell = get_cell_position_from_coords(target_location, global_map_info)
        if global_map_info.map[target_cell[1], target_cell[0]] == 255:
            centers_to_check = centers[:, 0] + centers[:, 1] * 1j
            optimal_center_index_in_center_lst = np.argwhere(centers_to_check == target_location[0] + target_location[1] * 1j)[0][0]
            return target_location, optimal_center_index_in_center_lst
        optimal_center = min(centers, key=lambda center: self.a_star_for_all_nodes_dict(center, target_location)[1])
        centers_to_check = centers[:, 0] + centers[:, 1] * 1j
        optimal_center_index_in_center_lst = np.argwhere(centers_to_check == optimal_center[0] + optimal_center[1] * 1j)[0][0]
        return optimal_center, optimal_center_index_in_center_lst

    def find_optimal_center(self, centers, target_location, global_map_info):
        target_cell = get_cell_position_from_coords(target_location, global_map_info)
        if global_map_info.map[target_cell[1], target_cell[0]] == 255:
            # print("optimal center is target")
            return target_location
        dist_list = np.linalg.norm((target_location - centers), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        while k < sorted_index.shape[0]:
            optimal_center_index = sorted_index[k]
            optimal_center = centers[optimal_center_index]
            if optimal_center[0] != target_location[0] or optimal_center[1] != target_location[1]:
                return optimal_center
            k += 1
        print("cannot find the optimal center")
        return None   
    
    def find_centers_of_target(self, all_node_coords, centers, target_location, adjacent_matrix):
        self.x_center, self.y_center = [], []
        dist_list = np.linalg.norm((target_location-centers), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        a = np.argwhere(local_node_coords_to_check == target_location[0] + target_location[1]*1j)
        if a or a == [[0]]:
            a = a[0][0]
        while k < CENTER_SIZE and k< centers.shape[0]:
            neighbor_index = sorted_index[k]
            dist = dist_list[k]
            center = centers[neighbor_index]
            b = np.argwhere(local_node_coords_to_check == center[0] + center[1]*1j)
            if b or b == [[0]]:
                b = b[0][0]
            adjacent_matrix[a, b] = 0
            adjacent_matrix[b, a] = 0
            k += 1
            self.x_center.append([center[0], target_location[0]])
            self.y_center.append([center[1], target_location[1]])
        
        return adjacent_matrix
    
    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        h = ((coords_1[0]-coords_2[0])**2 + (coords_1[1] - coords_2[1])**2)**(1/2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=1e8):
        if not self.check_node_exist_in_dict(start):
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [start, destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.local_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.local_nodes_dict.find(n).data
                    n_coords = node.coords

            # if g[n] > max_dist:
            #     return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.append(start)
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0]-n_coords[0])**2 + (neighbor_node_coords[1] - n_coords[1])**2)**(1/2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')

        return [], 1e8

    def a_star_for_all_nodes_dict(self, start, destination, max_dist=1e8):
        if not self.check_node_exist_in_all_dict(start):
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_all_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [start, destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.all_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.all_nodes_dict.find(n).data
                    n_coords = node.coords

            # if g[n] > max_dist:
            #     return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.append(start)
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0]-n_coords[0])**2 + (neighbor_node_coords[1] - n_coords[1])**2)**(1/2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')
        dist = round(((start[0]-destination[0])**2 + (start[1] - destination[1])**2)**(1/2), 1)
        # return [], 1e8
        return [], dist
    
class Local_node:
    def __init__(self, coords, local_frontiers, extended_local_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_frontiers = self.initialize_observable_frontiers(local_frontiers, extended_local_map_info)
        self.utility = 1 if self.observable_frontiers.shape[0] > MIN_UTILITY else 0
        self.utility_share = [self.utility]
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, local_frontiers, extended_local_map_info):
        if local_frontiers == []:
            self.utility = 0
            return []
        else:
            observable_frontiers = []
            dist_list = np.linalg.norm(local_frontiers - self.coords, axis=-1)
            frontiers_in_range = local_frontiers[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    observable_frontiers.append(point)
            observable_frontiers = np.array(observable_frontiers)
            return observable_frontiers

    def update_neighbor_nodes(self, extended_local_map_info, global_map_info, nodes_dict, target_location, plot_x=None, plot_y=None):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        cell = get_cell_position_from_coords(neighbor_coords, extended_local_map_info)
                        if cell[0] < extended_local_map_info.map.shape[1] and cell[1] < extended_local_map_info.map.shape[0]:
                            if extended_local_map_info.map[cell[1], cell[0]] == 1:
                                self.neighbor_matrix[i, j] = 1
                            continue
                    else:
                        # if neighbor_coords[0] == target_location[0] and neighbor_coords[1] == target_location[1]:
                        #     print("find target node as neighbor in uniform points")
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, global_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

                            if plot_x is not None and plot_y is not None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])
                                
        if not check_collision(self.coords, target_location, global_map_info):
            self.neighbor_list.append(target_location)
            target_node = nodes_dict.find((target_location[0], target_location[1]))
            target_node = target_node.data
            target_node.neighbor_list.append(self.coords)
            if plot_x is not None and plot_y is not None:
                plot_x.append([self.coords[0], target_location[0]])
                plot_y.append([self.coords[1], target_location[1]])

    def update_node_observable_frontiers(self, local_frontiers, extended_local_map_info):
        
        # remove observed frontiers in the observable frontiers
        if local_frontiers.shape[0] == 0:
            self.utility = 0
            self.utility_share[0] = self.utility
            self.observable_frontiers = []
            return

        local_frontiers = local_frontiers.reshape(-1, 2)
        old_frontier_to_check = self.observable_frontiers[:, 0] + self.observable_frontiers[:, 1] * 1j
        local_frontiers_to_check = local_frontiers[:, 0] + local_frontiers[:, 1] * 1j
        to_observe_index = np.where(
            np.isin(old_frontier_to_check, local_frontiers_to_check, assume_unique=True) == True)
        new_frontier_index = np.where(
            np.isin(local_frontiers_to_check, old_frontier_to_check, assume_unique=True) == False)
        self.observable_frontiers = self.observable_frontiers[to_observe_index]
        new_frontiers = local_frontiers[new_frontier_index]

        # add new frontiers in the observable frontiers
        if new_frontiers != []:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    self.observable_frontiers = np.concatenate((self.observable_frontiers, point.reshape(1, 2)), axis=0)
        self.utility = self.observable_frontiers.shape[0]
        if self.utility > MIN_UTILITY:
            self.utility = 1
        else:
            self.utility = 0
        self.utility_share[0] = self.utility

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = []
        self.utility = 0
        self.utility_share[0] = self.utility


class All_node:
    def __init__(self, coords):
        self.coords = coords
        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True

    def update_neighbor_nodes(self, ground_truth_info, nodes_dict, target_location):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        cell = get_cell_position_from_coords(neighbor_coords, ground_truth_info)
                        if cell[0] < ground_truth_info.map.shape[1] and cell[1] < ground_truth_info.map.shape[0]:
                            if ground_truth_info.map[cell[1], cell[0]] == 1:
                                self.neighbor_matrix[i, j] = 1
                            continue
                    else:
                        # if neighbor_coords[0] == target_location[0] and neighbor_coords[1] == target_location[1]:
                        #     print("find target node as neighbor in uniform points")
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)
                                
        if not check_collision(self.coords, target_location, ground_truth_info):
            self.neighbor_list.append(target_location)
            target_node = nodes_dict.find((target_location[0], target_location[1]))
            target_node = target_node.data
            target_node.neighbor_list.append(self.coords)