import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import sensor_work
from parameter import *
from utils import *


class Env:
    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        # self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth, self.robot_cell, self.target_cell = self.import_ground_truth_pp_640_480(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter
        
        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0

        self.done = False
        
        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)
        self.ground_truth_info = Map_info(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.belief_info = Map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.global_frontiers = get_frontier_in_map(self.belief_info)
        self.target_location = get_coords_from_cell_position(self.target_cell, self.belief_info)
        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        map_dir = f'maps'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def import_ground_truth_pp_640_480(self, episode_index):
        map_dir = f'train_640_480'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)
        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 209)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])  
        
        target_cell = np.nonzero(ground_truth == 68)
        target_cell = np.array([np.array(target_cell)[1, 10], np.array(target_cell)[0, 10]])
        
        ground_truth = (ground_truth > 150)|((ground_truth<=80)&(ground_truth>=60))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell, target_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self, astar_dist_cur, astar_dist_next):
        reward = 0
        global_frontiers = get_frontier_in_map(self.belief_info)
        if global_frontiers.shape[0] == 0:
            delta_num = self.global_frontiers.shape[0]
        else:
            global_frontiers = global_frontiers.reshape(-1, 2)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        reward -= 1.0
        reward += (astar_dist_cur - astar_dist_next) / 32
        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, astar_dist_cur, asta_dist_next):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        dist_to_target = np.linalg.norm(next_waypoint - self.target_location)
        self.travel_dist += dist
        self.evaluate_exploration_rate()
        self.robot_location = next_waypoint

        self.update_robot_location(next_waypoint)
        self.update_robot_belief()
        reward = self.calculate_reward(astar_dist_cur, asta_dist_next)

        return reward, dist_to_target

    def plot_env(self, step, optimal_center, centers):

        plt.subplot(1, 2, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'ro', markersize=10, zorder=5)
        plt.plot((self.target_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.target_location[1] - self.belief_origin_y) / self.cell_size, 'rs', markersize=10, zorder=5)
        for center in centers:
            plt.plot((center[0] - self.belief_origin_x) / self.cell_size,
                    (center[1] - self.belief_origin_y) / self.cell_size, 'b^', markersize=6, zorder=6)
        # plt.plot((optimal_center[0] - self.belief_origin_x) / self.cell_size,
        #          (optimal_center[1] - self.belief_origin_y) / self.cell_size, 'r^', markersize=6, zorder=7)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        self.frame_files.append(frame)

