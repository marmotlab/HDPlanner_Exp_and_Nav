from time import time

import numpy as np
import torch

from env import Env
from agent import Agent
from parameter import *
from utils import *
from model import PolicyNet

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)
        # self.robot.update_ground_truth_map(self.env.ground_truth_info)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(21):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        local_observation = self.robot.get_local_observation()
        for i in range(MAX_EPISODE_STEP):
            self.save_observation(local_observation)
            self.save_optimal_center(self.robot.optimal_center)
            next_location, action_index = self.robot.select_next_waypoint(local_observation, i)
            self.save_action(action_index)

            if self.save_image:
                self.robot.plot_local_env()
                self.env.plot_env(i, self.robot.optimal_center, self.robot.centers)

            node = self.robot.local_node_manager.local_nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(node.data.neighbor_list)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_list)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            _, astar_dist_cur = self.robot.local_node_manager.a_star(self.robot.location, self.robot.optimal_center)
            _, astar_dist_next = self.robot.local_node_manager.a_star(next_location, self.robot.optimal_center)
            
            # add momentum reward
            direction = next_location - self.robot.location
            dist = np.linalg.norm(direction)
            # if dist == 0:
            #     momentum = np.zeros(2)
            # else:
            #     momentum = direction / dist
            # momentum_reward = np.dot(momentum, self.robot.momentum) * 0.1
            # self.robot.momentum = momentum
            reward = self.env.step(next_location, astar_dist_cur, astar_dist_next)
            # reward += momentum_reward
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            # code here! ! !
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            local_observation = self.robot.get_local_observation()
            self.save_next_observations(local_observation)
            self.save_next_optimal_center(self.robot.optimal_center)

            if done:
                if self.save_image:
                    self.robot.plot_local_env()
                    self.env.plot_env(i, self.robot.optimal_center, self.robot.centers)
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, local_observation):
        local_node_inputs, current_local_edge, current_local_index, all_center_index, local_node_padding_mask, local_edge_padding_mask, local_edge_mask, center_padding_mask = local_observation
        self.episode_buffer[0] += local_node_inputs
        self.episode_buffer[1] += local_node_padding_mask.bool()
        self.episode_buffer[2] += local_edge_mask.bool()
        self.episode_buffer[3] += current_local_index
        self.episode_buffer[4] += current_local_edge
        self.episode_buffer[5] += local_edge_padding_mask.bool()
        self.episode_buffer[6] += all_center_index
        self.episode_buffer[7] += center_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[8] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[9] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[10] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, local_observation):
        local_node_inputs, current_local_edge, current_local_index, all_center_index, local_node_padding_mask, local_edge_padding_mask, local_edge_mask, center_padding_mask = local_observation
        self.episode_buffer[11] += local_node_inputs
        self.episode_buffer[12] += local_node_padding_mask.bool()
        self.episode_buffer[13] += local_edge_mask.bool()
        self.episode_buffer[14] += current_local_index
        self.episode_buffer[15] += current_local_edge
        self.episode_buffer[16] += local_edge_padding_mask.bool()
        self.episode_buffer[17] += all_center_index
        self.episode_buffer[18] += center_padding_mask.bool()

    def save_optimal_center(self, optimal_center):
        local_node_coords_to_check = self.robot.local_node_coords[:, 0] + self.robot.local_node_coords[:, 1] * 1j
        optimal_center_index = np.argwhere(local_node_coords_to_check == optimal_center[0] + optimal_center[1] * 1j)
        if optimal_center_index or optimal_center_index == [[0]]:
            optimal_center_index = optimal_center_index[0][0]
        optimal_center_index = torch.tensor([optimal_center_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1) 
        # optimal_center_index_in_center_list = self.env.find_index_from_center_list(self.env.centers, optimal_center_node) 
        # optimal_center_index_in_center_list = torch.tensor([optimal_center_index_in_center_list]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)   
        self.episode_buffer[19] += optimal_center_index
        # self.episode_buffer[23] += copy.deepcopy(optimal_center_index_in_center_list)

    def save_next_optimal_center(self, optimal_center):
        local_node_coords_to_check = self.robot.local_node_coords[:, 0] + self.robot.local_node_coords[:, 1] * 1j
        optimal_center_index = np.argwhere(local_node_coords_to_check == optimal_center[0] + optimal_center[1] * 1j)
        if optimal_center_index or optimal_center_index == [[0]]:
            optimal_center_index = optimal_center_index[0][0]
        optimal_center_index = torch.tensor([optimal_center_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1) 
        # optimal_center_index_in_center_list = self.env.find_index_from_center_list(self.env.centers, optimal_center_node) 
        # optimal_center_index_in_center_list = torch.tensor([optimal_center_index_in_center_list]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)   
        self.episode_buffer[20] += optimal_center_index
        # self.episode_buffer[23] += copy.deepcopy(optimal_center_index_in_center_list)

if __name__ == "__main__":
    model = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 0, save_image=True)
    worker.run_episode()
