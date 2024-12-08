import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to HDPlanner-Nav!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha1 = torch.FloatTensor([-2]).to(device)  # not trainable when loaded from checkpoint, manually tune it for now
    log_alpha1.requires_grad = True
    log_alpha2 = torch.FloatTensor([-2]).to(device)  # not trainable when loaded from checkpoint, manually tune it for now
    log_alpha2.requires_grad = True

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer1 = optim.Adam([log_alpha1], lr=1e-4)
    log_alpha_optimizer2 = optim.Adam([log_alpha2], lr=1e-4)

    # target entropy for SAC, manually tune it for now
    entropy_target1 = 0.02 * (-np.log(1 / LOCAL_K_SIZE))
    entropy_target2 = 0.02 * (-np.log(1 / LOCAL_K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha1 = checkpoint['log_alpha1']  # not trainable when loaded from checkpoint, manually tune it for now
        log_alpha_optimizer1 = optim.Adam([log_alpha1], lr=1e-4)
        log_alpha2 = checkpoint['log_alpha2']  # not trainable when loaded from checkpoint, manually tune it for now
        log_alpha_optimizer2 = optim.Adam([log_alpha2], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer1.load_state_dict(checkpoint['log_alpha_optimizer1'])
        log_alpha_optimizer2.load_state_dict(checkpoint['log_alpha_optimizer2'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha1, log_alpha1.requires_grad)
        print(log_alpha2, log_alpha2.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    il_data = []
    contrastive_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(24):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs_batch = torch.stack(rollouts[0]).to(device)
                    node_padding_mask_batch = torch.stack(rollouts[1]).to(device)
                    edge_mask_batch = torch.stack(rollouts[2]).to(device)
                    current_index_batch = torch.stack(rollouts[3]).to(device)
                    edge_inputs_batch = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask_batch = torch.stack(rollouts[5]).to(device)
                    target_index_batch = torch.stack(rollouts[6]).to(device)
                    center_index_batch = torch.stack(rollouts[7]).to(device)
                    center_padding_mask_batch = torch.stack(rollouts[8]).to(device)
                    action_batch = torch.stack(rollouts[9]).to(device)
                    reward_batch = torch.stack(rollouts[10]).to(device)
                    done_batch = torch.stack(rollouts[11]).to(device)
                    next_node_inputs_batch = torch.stack(rollouts[12]).to(device)
                    next_node_padding_mask_batch = torch.stack(rollouts[13]).to(device)
                    next_edge_mask_batch = torch.stack(rollouts[14]).to(device)
                    next_current_index_batch = torch.stack(rollouts[15]).to(device)
                    next_edge_inputs_batch = torch.stack(rollouts[16]).to(device)
                    next_edge_padding_mask_batch = torch.stack(rollouts[17]).to(device)
                    next_target_index_batch = torch.stack(rollouts[18]).to(device)
                    next_center_index_batch = torch.stack(rollouts[19]).to(device)
                    next_center_padding_mask_batch = torch.stack(rollouts[20]).to(device)
                    optimal_center_index_batch = torch.stack(rollouts[21]).to(device)
                    next_optimal_center_index_batch = torch.stack(rollouts[22]).to(device)
                  
                    observation = [node_inputs_batch, edge_inputs_batch, current_index_batch, target_index_batch, center_index_batch, node_padding_mask_batch, edge_padding_mask_batch,
                                   edge_mask_batch, center_padding_mask_batch]
                    next_observation = [next_node_inputs_batch, next_edge_inputs_batch, next_current_index_batch, next_target_index_batch, next_center_index_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch,
                                   next_edge_mask_batch, next_center_padding_mask_batch]
                    q_observation = [node_inputs_batch, edge_inputs_batch, current_index_batch, optimal_center_index_batch, center_index_batch, target_index_batch, node_padding_mask_batch, edge_padding_mask_batch,
                                   edge_mask_batch]
                    q_next_observation = [next_node_inputs_batch, next_edge_inputs_batch, next_current_index_batch, next_optimal_center_index_batch, next_center_index_batch, next_target_index_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch,
                                   next_edge_mask_batch]
                    
                    with torch.no_grad():
                        center_q_values1, _, action_q_values1, _ = dp_q_net1(*q_observation)
                        center_q_values2, _, action_q_values2, _ = dp_q_net2(*q_observation)
                        center_q_values = torch.min(center_q_values1, center_q_values2)
                        action_q_values = torch.min(action_q_values1, action_q_values2)

                    center_logp, action_logp, _, _, _, _, _, _ = dp_policy(*observation)
                    policy_center_loss = torch.sum(
                        (center_logp.exp().unsqueeze(2) * (log_alpha1.exp().detach() * center_logp.unsqueeze(2) - center_q_values.detach())),
                        dim=1).mean()
                    policy_action_loss = torch.sum(
                        (action_logp.exp().unsqueeze(2) * (log_alpha2.exp().detach() * action_logp.unsqueeze(2) - action_q_values.detach())),
                        dim=1).mean()
                    policy_loss = policy_center_loss + policy_action_loss
                    
                    with torch.no_grad():
                        next_center_logp, next_action_logp, _, _, _, _, _, _ = dp_policy(*next_observation)
                        next_center_q_values1, _, next_action_q_values1, _ = dp_target_q_net1(*q_next_observation)
                        next_center_q_values2, _, next_action_q_values2, _ = dp_target_q_net2(*q_next_observation)
                        next_center_q_values = torch.min(next_center_q_values1, next_center_q_values2)
                        next_action_q_values = torch.min(next_action_q_values1, next_action_q_values2)
                        center_value_prime_batch = torch.sum(next_center_logp.unsqueeze(2).exp() * (next_center_q_values - log_alpha1.exp() * next_center_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_center_q_batch = reward_batch + GAMMA * (1 - done_batch) * center_value_prime_batch
                        action_value_prime_batch = torch.sum(next_action_logp.unsqueeze(2).exp() * (next_action_q_values - log_alpha2.exp() * next_action_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_action_q_batch = reward_batch + GAMMA * (1 - done_batch) * action_value_prime_batch
                        
                    center_q_values1, _, action_q_values1, _ = dp_q_net1(*q_observation)
                    center_q_values2, _, action_q_values2, _ = dp_q_net2(*q_observation)
                    center_q1 = torch.gather(center_q_values1, 1, action_batch)
                    center_q2 = torch.gather(center_q_values2, 1, action_batch)
                    mse_loss = nn.MSELoss()
                    center_q1_loss = mse_loss(center_q1, target_center_q_batch.detach()).mean()
                    center_q2_loss = mse_loss(center_q2, target_center_q_batch.detach()).mean()
                    action_q1 = torch.gather(action_q_values1, 1, action_batch)
                    action_q2 = torch.gather(action_q_values2, 1, action_batch)
                    mse_loss = nn.MSELoss()
                    action_q1_loss = mse_loss(action_q1, target_action_q_batch.detach()).mean()
                    action_q2_loss = mse_loss(action_q2, target_action_q_batch.detach()).mean()
                    
                    q1_loss = center_q1_loss + action_q1_loss
                    q2_loss = center_q2_loss + action_q2_loss

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    center_entropy = (center_logp * center_logp.exp()).sum(dim=-1)
                    center_alpha_loss = -(log_alpha1 * (center_entropy.detach() + entropy_target1)).mean()
                    
                    action_entropy = (action_logp * action_logp.exp()).sum(dim=-1)
                    action_alpha_loss = -(log_alpha2 * (action_entropy.detach() + entropy_target2)).mean()

                    log_alpha_optimizer1.zero_grad()
                    center_alpha_loss.backward()
                    log_alpha_optimizer1.step()

                    log_alpha_optimizer2.zero_grad()
                    action_alpha_loss.backward()
                    log_alpha_optimizer2.step()
                    
                    target_q_update_counter += 1

                    # contrastive learning for center and action
                    center_logp, action_logp, \
                        selected_center_index, selected_action_index, \
                            center_node_features, neighboring_features, selected_center_feature, selected_action_feature = global_policy_net(*observation)
                    epsilon = 0.5
                    if torch.rand(1) < epsilon:
                        cl_center_q_values, _, cl_action_q_values, _ = dp_q_net1(*q_observation)
                    else:
                        cl_center_q_values, _, cl_action_q_values, _ = dp_target_q_net1(*q_observation)
                    center_index = torch.argmax(cl_center_q_values, dim=1)
                    action_index = torch.argmax(cl_action_q_values, dim=1)

                    center_positive_node_features, center_negative_node_features = get_contrastive_pairs(center_logp, center_node_features, selected_center_index, center_index)
                    action_positive_node_features, action_negative_node_features = get_contrastive_pairs(action_logp, neighboring_features, selected_action_index, action_index)
                    triplet_loss_center = get_triplet_loss(selected_center_feature, center_positive_node_features, center_negative_node_features)
                    triplet_loss_action = get_triplet_loss(selected_action_feature, action_positive_node_features, action_negative_node_features)
                    triplet_loss = triplet_loss_center + triplet_loss_action
                    
                    global_policy_optimizer.zero_grad()
                    triplet_loss.backward()
                    policy_contrastive_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), 
                                                                                  max_norm=10000, norm_type=2)
                    global_policy_optimizer.step()

                # data record to be written in tensorboard
                data = [triplet_loss_center.item(), triplet_loss_action.item(), triplet_loss.item(), policy_contrastive_grad_norm.item()]
                contrastive_data.append(data)
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), center_value_prime_batch.mean().item(), action_value_prime_batch.mean().item(), policy_loss.item(), q1_loss.item(),
                        center_entropy.mean().item(), action_entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha1.item(), log_alpha2.item(), 
                        center_alpha_loss.item(), action_alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            if len(contrastive_data) >= SUMMARY_WINDOW:
                write_contrastive_to_tensor_board(writer, contrastive_data, curr_episode)
                contrastive_data = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 1024:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 50 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha1": log_alpha1,
                              "log_alpha2": log_alpha2,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer1": log_alpha_optimizer1.state_dict(),
                              "log_alpha_optimizer2": log_alpha_optimizer2.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
            
def get_contrastive_pairs(action_logp, center_node_features, center_index, index):
    valid_indices = torch.nonzero(action_logp > -1e7, as_tuple=False)
    has_valid_values = valid_indices[:, 0].unique()
    selected_indices = []
    for row_index in has_valid_values:
        row_valid_indices = valid_indices[valid_indices[:, 0] == row_index, 1]
        selected_index = torch.randint(0, row_valid_indices.size(0), (1,))
        selected_indices.append(row_valid_indices[selected_index].unsqueeze(0))
    center_index = torch.cat(selected_indices, dim=0)
    # print("Center Index:", center_index.size()) # [batch_size, 1]
    negative_node_features = center_node_features[torch.arange(center_node_features.size(0)), center_index.squeeze(1), :].unsqueeze(1)
    negative_node_features = negative_node_features.view(negative_node_features.size(0), -1)
    # print("Negative Node Features:", negative_node_features.size()) # [batch_size, 128]
    positive_node_features = center_node_features[torch.arange(center_node_features.size(0)), index.squeeze(1), :].unsqueeze(1)
    positive_node_features = positive_node_features.view(positive_node_features.size(0), -1)
    # print("Positive Node Features:", positive_node_features.size()) # [batch_size, 128]
    return positive_node_features, negative_node_features

def get_triplet_loss(anchor, positive, negative, margin=0.5):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = F.relu(distance_positive - distance_negative + margin)
    loss = loss.mean()
    # print("Triplet Loss:", loss.item())
    return loss

def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, center_value, action_value, policy_loss, q_value_loss, center_entropy, action_entropy, policy_grad_norm, q_value_grad_norm, \
        log_alpha1, log_alpha2, center_alpha_loss, action_alpha_loss, travel_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Center Value', scalar_value=center_value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Action Value', scalar_value=action_value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Center Alpha Loss', scalar_value=center_alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Action Alpha Loss', scalar_value=action_alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Center Entropy', scalar_value=center_entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Action Entropy', scalar_value=action_entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Center Log Alpha', scalar_value=log_alpha1, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Action Log Alpha', scalar_value=log_alpha2, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)

def write_contrastive_to_tensor_board(writer, contrastive_data,  curr_episode):
    contrastive_data = np.array(contrastive_data)
    contrastive_data = list(np.nanmean(contrastive_data, axis=0))
    policy_contrastive_center_loss, policy_contrastive_action_loss, policy_contrastive_loss, policy_contrastive_norm  = contrastive_data
    writer.add_scalar(tag='Contrastive Losses/Policy Contrastive Loss', scalar_value=policy_contrastive_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Contrastive Losses/Policy Contrastive Center Loss', scalar_value=policy_contrastive_center_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Contrastive Losses/Policy Contrastive Action Loss', scalar_value=policy_contrastive_action_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Contrastive Losses/Policy Contrastive Grad Norm', scalar_value=policy_contrastive_norm,
                      global_step=curr_episode)
    
def write_imitation_to_tensor_board(writer, imitation_data,  curr_episode):
    imitation_data = np.array(imitation_data)
    imitation_data = list(np.nanmean(imitation_data, axis=0))
    policy_imitation_center_loss, policy_imitation_action_loss, policy_imitation_loss, policy_imitation_norm  = imitation_data
    writer.add_scalar(tag='Imitation Losses/Policy Imitation Loss', scalar_value=policy_imitation_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Imitation Losses/Policy Imitation Center Loss', scalar_value=policy_imitation_center_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Imitation Losses/Policy Imitation Action Loss', scalar_value=policy_imitation_action_loss, 
                      global_step=curr_episode)
    writer.add_scalar(tag='Imitation Losses/Policy Imitation Grad Norm', scalar_value=policy_imitation_norm,
                      global_step=curr_episode)

if __name__ == "__main__":
    main()
