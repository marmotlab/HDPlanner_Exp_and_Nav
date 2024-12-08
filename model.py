import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from parameter import *


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PolicyNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.target_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.current_embedding2 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.target_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_node_decoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.pointer1 = SingleHeadAttention(embedding_dim)
        self.pointer2 = SingleHeadAttention(embedding_dim)

    def graph_encoder_and_center_decoder(self, node_inputs, node_padding_mask, edge_mask, center_mask, center_index, target_index, current_index, edge_inputs):
        # encoder
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)
        # decoder1 - select center
        center_index = center_index.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]
        target_node_feature = torch.gather(enhanced_node_feature, 1, target_index.repeat(1, 1, embedding_dim))
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        center_node_features = torch.gather(enhanced_node_feature, 1, center_index.repeat(1, 1, embedding_dim))

        enhanced_target_node_feature, _ = self.target_decoder(target_node_feature, enhanced_node_feature, node_padding_mask)
        embedding_target_node_feature = self.target_embedding(torch.cat((enhanced_target_node_feature, target_node_feature), dim=-1))
        center_logp = self.pointer1(embedding_target_node_feature, center_node_features, center_mask)
        center_logp = center_logp.squeeze(1) # batch_size*k_size
        center_logp_index = torch.argmax(center_logp, dim=1).long()
        selected_center_index = center_index[torch.arange(center_index.size(0)), center_logp_index, :]
        selected_center_node_feature = center_node_features[torch.arange(center_node_features.size(0)), center_logp_index, :]
        selected_center_node_feature = selected_center_node_feature.unsqueeze(1)
        return enhanced_node_feature, current_node_feature, selected_center_index, selected_center_node_feature, center_logp, center_node_features

    def output_policy(self, enhanced_node_feature, current_node_feature, edge_inputs, edge_padding_mask, selected_center_feature, node_padding_mask):
        # decoder2 - select next move based on the selected center
        current_edge = edge_inputs
        embedding_dim = enhanced_node_feature.size()[2]
        if edge_padding_mask is not None:
            current_mask = edge_padding_mask
            # print(current_mask)
        else:
            current_mask = None
        current_mask[:,:,0] = 1 # don't stay at current position
        neighboring_feature = torch.gather(enhanced_node_feature, 1,  current_edge.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.current_node_decoder2(current_node_feature, enhanced_node_feature, node_padding_mask)
        # selected_center_feature = selected_center_feature.unsqueeze(1)
        embedding_current_node_feature = self.current_embedding2(torch.cat((enhanced_current_node_feature, current_node_feature, selected_center_feature), dim=-1))
        action_logp = self.pointer2(embedding_current_node_feature, neighboring_feature, current_mask)
        action_logp= action_logp.squeeze(1) # batch_size*k_size
        action_logp_index = torch.argmax(action_logp, dim=1).long()
        selected_action_index = current_edge[torch.arange(current_edge.size(0)), action_logp_index, :]
        selected_action_feature = neighboring_feature[torch.arange(neighboring_feature.size(0)), action_logp_index, :]
        selected_action_feature = selected_action_feature.unsqueeze(1)       
        return action_logp, neighboring_feature, selected_action_index, selected_action_feature

    def forward(self, node_inputs, edge_inputs, current_index, target_index, all_center_index, node_padding_mask=None, edge_padding_mask=None, edge_mask=None, center_mask=None):
        enhanced_node_feature, enhanced_current_node_feature, selected_center_index, selected_center_feature, center_logp, center_node_features = self.graph_encoder_and_center_decoder(\
            node_inputs, node_padding_mask, edge_mask, center_mask, all_center_index, target_index, current_index, edge_inputs)
        action_logp, neighboring_features, selected_action_index, selected_action_feature = self.output_policy(enhanced_node_feature, enhanced_current_node_feature, edge_inputs, edge_padding_mask, selected_center_feature, node_padding_mask)
        return center_logp, action_logp, \
            selected_center_index, selected_action_index, \
            center_node_features, neighboring_features, selected_center_feature, selected_action_feature

class QNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(QNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.current_embedding1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.current_embedding2 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.target_node_decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_node_decoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.pointer1 = SingleHeadAttention(embedding_dim)
        self.pointer2 = SingleHeadAttention(embedding_dim)
        self.neighbor_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.q_values_layer1 = nn.Linear(embedding_dim * 2, 1)
        self.q_values_layer2 = nn.Linear(embedding_dim * 2, 1)

    def graph_encoder_and_center_decoder(self, node_inputs, node_padding_mask, edge_mask, optimal_center_index, center_index, target_index, current_index, edge_inputs):
        # q encoder
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)
        # decoder1 - select center
        center_index = center_index.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]
        target_node_feature = torch.gather(enhanced_node_feature, 1, target_index.repeat(1, 1, embedding_dim))
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        center_node_features = torch.gather(enhanced_node_feature, 1, center_index.repeat(1, 1, embedding_dim))
        enhanced_target_node_feature, attention_weights = self.target_node_decoder(target_node_feature, enhanced_node_feature, node_padding_mask)
        embedding_target_node_feature = self.current_embedding1(torch.cat((enhanced_target_node_feature, target_node_feature), dim=-1))
        center_feature = torch.cat((embedding_target_node_feature.repeat(1, LOCAL_K_SIZE, 1), center_node_features), dim=-1) # batch_size*k_size*embedding_dim
        q_values = self.q_values_layer1(center_feature)
        selected_center_index = torch.argmax(q_values, dim=1).long()
        selected_center_node_feature = torch.gather(center_node_features, 1, selected_center_index.unsqueeze(1).repeat(1, 1, embedding_dim))
        # print("selected_center_node_feature", selected_center_node_feature.size()) # batch_size*1*embedding_dim
        
        return q_values, attention_weights, selected_center_index, selected_center_node_feature, enhanced_node_feature, current_node_feature
    
    def output_q_values(self, enhanced_node_feature, current_node_feature, edge_inputs, edge_padding_mask, selected_center_feature, node_padding_mask):
        # q decoder2 - select next move based on the selected center
        k_size = edge_inputs.size()[2]
        current_edge = edge_inputs
        # current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]
        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, attention_weights = self.current_node_decoder2(current_node_feature, enhanced_node_feature, node_padding_mask)
        embedding_current_node_feature = self.current_embedding2(torch.cat((enhanced_current_node_feature, current_node_feature, selected_center_feature), dim=-1))
        action_features = torch.cat((embedding_current_node_feature.repeat(1, LOCAL_K_SIZE, 1), neigboring_feature), dim=-1)
        q_values = self.q_values_layer2(action_features)

        # if edge_padding_mask is not None:
        #     current_mask = edge_padding_mask
        # else:
        #     current_mask = None
        # current_mask[:, :, 0] = 1  # don't stay at current position
        # #assert 0 in current_mask
        # current_mask = current_mask.permute(0, 2, 1)
        # zero = torch.zeros_like(q_values).to(q_values.device)
        # q_values = torch.where(current_mask == 1, zero, q_values)

        return q_values, attention_weights

    def forward(self, node_inputs, edge_inputs, current_index, optimal_center_index, center_index, target_index, node_padding_mask=None, edge_padding_mask=None, edge_mask=None, center_mask=None):
        centers_q_values, attention_weights1, selected_center_index, selected_center_feature, enhanced_node_feature, current_node_feature  = self.graph_encoder_and_center_decoder(\
            node_inputs, node_padding_mask, edge_mask, optimal_center_index, center_index, target_index, current_index, edge_inputs)
        action_q_values, attention_weights2 = self.output_q_values(enhanced_node_feature, current_node_feature, edge_inputs, edge_padding_mask, selected_center_feature, node_padding_mask)
        return centers_q_values, attention_weights1, \
            action_q_values, attention_weights2