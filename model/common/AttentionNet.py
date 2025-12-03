import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.amp.autocast_mode import autocast


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

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            # U[mask.bool()] = -1e8
            U[mask.bool()] = -1e4
        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


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

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            # attnc[mask.bool()] = 0
            attnc[mask] = 0
            attention = attnc
        # print(attention)

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


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

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
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

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class NodeEncoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(NodeEncoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class AgentEncoder(nn.Module):
    def __init__(self, embedding_dim, n_head=8, n_layer=3):
        super(AgentEncoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class NodeAgentEncoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(NodeAgentEncoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.node_layers = nn.ModuleList([NodeEncoder(embedding_dim, n_head) for i in range(n_layer)])
        self.agent_layers = nn.ModuleList([AgentEncoder(embedding_dim, n_head) for i in range(n_layer)])
        self.node_agent_layers = nn.ModuleList([NodeAgentEncoder(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, node_embedding, mask=None):
        for layer in self.node_layers:
            node_embedding = layer(node_embedding, mask)
        aggregated_embedding = torch.mean(node_embedding, dim=1, keepdim=True)

        return aggregated_embedding, node_embedding


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, current_node_feature, connected_node_agent_embedding, mask):
        final_candidate_embedding = 0

        for layer in self.layers:
            final_candidate_embedding = layer(current_node_feature, connected_node_agent_embedding, mask=mask)
        return final_candidate_embedding


class AttentionNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(AttentionNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim)  # layer for non-end position
        self.end_embedding = nn.Linear(input_dim, embedding_dim)  # embedding layer for end position
        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim)
        self.value_output = nn.Linear(embedding_dim, 1)
        self.pos_embedding = nn.Linear(32, embedding_dim)

        # self.nodes_update_layers = nn.ModuleList([DecoderLayer(embedding_dim, 8) for i in range(3)])
        self.current_embedding = nn.Linear(embedding_dim, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

        # agent embedding
        self.agent_embedding = nn.Linear(2, embedding_dim)

    def graph_embedding(self, node_inputs, edge_inputs, pos_encoding, mask=None):
        # current_position (batch, 1, 2)
        # end_position (batch, 1,2)
        # node_inputs (batch, sample_size+2, 2) end position and start position are the first two in the inputs
        # edge_inputs (batch, sample_size+2, k_size)
        # mask (batch, sample_size+2, k_size)
        # agent embedding
        # end_position = node_inputs[:, 0, :].unsqueeze(1)
        # embedding_feature = torch.cat(
        #     (self.end_embedding(end_position), self.initial_embedding(node_inputs[:, 1:, :])), dim=1)
        embedding_feature = self.initial_embedding(node_inputs)
        pos_encoding = self.pos_embedding(pos_encoding)
        embedding_feature = embedding_feature + pos_encoding

        aggregated_embedding, node_embedding = self.encoder(embedding_feature)

        return aggregated_embedding, node_embedding

    def select_next_node(self, aggregated_embedding, node_embedding, edge_inputs, budget_inputs,
                         current_index, LSTM_h, LSTM_c, mask):
        """
        change from single ipp:
        embedding feature -> node_agent_embedding
        current_node_feature -> aggregated_embedding
        connected_node_feature comes from node_agent_embedding rather than embedding feature
        """

        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        batch_size = edge_inputs.size()[0]
        sample_size = edge_inputs.size()[1]
        k_size = edge_inputs.size()[2]
        current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, k_size))
        # print(current_edge)
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = node_embedding.size()[2]

        th = torch.FloatTensor([0.4]).unsqueeze(0).unsqueeze(0).repeat(batch_size, sample_size, 1).to(
            node_embedding.device)

        # change from embedding feather in single ipp to nodes_agent_embedding
        node_embedding = self.budget_embedding(torch.cat((node_embedding, budget_inputs, th), dim=-1))
        # connected node feature from node_agent_embedding
        connected_nodes_agent_embedding = torch.gather(node_embedding, 1,
                                                       current_edge.repeat(1, 1, embedding_dim))

        connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)

        # print(embedding_feature)
        # print(connected_nodes_feature)

        current_node_feature, (LSTM_h, LSTM_c) = self.LSTM(aggregated_embedding, (LSTM_h, LSTM_c))

        current_node_feature = self.current_embedding(current_node_feature)
        # print(current_node_feature)
        if mask is not None:
            current_mask = torch.gather(mask, 1, current_index.repeat(1, 1, k_size)).to(node_embedding.device)
            # print(current_mask)
        else:
            current_mask = torch.zeros((batch_size, 1, k_size), dtype=torch.int64).to(node_embedding.device)
        one = torch.ones_like(current_mask, dtype=torch.int64).to(node_embedding.device)
        current_mask = torch.where(connected_nodes_budget.permute(0, 2, 1) > 0, current_mask, one)
        # print(current_mask)
        current_mask[:, :, 0] = 1  # don't stay at current position
        assert 0 in current_mask

        # connected_nodes_feature = self.encoder(connected_nodes_feature, current_mask)
        current_feature_prime = self.decoder(current_node_feature, connected_nodes_agent_embedding,
                                             current_mask)
        logp_list = self.pointer(current_feature_prime, connected_nodes_agent_embedding, current_mask)
        logp_list = logp_list.squeeze(1)
        value = self.value_output(current_feature_prime)

        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        return logp_list, value, LSTM_h, LSTM_c

    def forward(self, node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c,
                pos_encoding, mask=None):
        # with autocast():
        aggregated_embedding, node_embedding = self.graph_embedding(node_inputs, edge_inputs, pos_encoding,
                                                                    mask=None)
        logp_list, value, LSTM_h, LSTM_c = self.select_next_node(aggregated_embedding, node_embedding, edge_inputs,
                                                                    budget_inputs,
                                                                    current_index, LSTM_h, LSTM_c, mask)
        return logp_list, value, LSTM_h, LSTM_c


class GraphAttentionEncoder(nn.Module):
    def __init__(self, node_dim, pos_dim, agent_dim, embedding_dim):
        super(GraphAttentionEncoder, self).__init__()
        # For node inputs
        self.node_embedder = nn.Linear(node_dim, embedding_dim)  # 5-dim node inputs
        self.pos_embedder = nn.Linear(pos_dim, embedding_dim)  # 32-dim positional encoding
        self.graph_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        # For agent inputs
        self.agent_embedder = nn.Linear(agent_dim, embedding_dim) # 4-dim agent inputs

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

    def forward(self, node_inputs, pos_embedding, agent_inputs, mask=None):
        '''
        node_inputs: (batch_size, num_nodes, num_features) x, y, info, std, intent
        pos_encoding: (batch_size, num_nodes, pos_encoding_dim) 32-dim positional encoding
        agent_inputs: (batch_size, 1, agent_dim) x, y, budget, threshold (TBD prev intent mean and std)
        '''
        # Encode node inputs
        node_embedding = self.node_embedder(node_inputs) # project node inputs to embedding space
        pos_embedding = self.pos_embedder(pos_embedding) # project pos encoding to embedding space
        node_embedding = node_embedding + pos_embedding
        aggregated_embedding, enhanced_node_embedding = self.graph_encoder(node_embedding, mask=mask)

        # Encode agent inputs
        agent_embedding = self.agent_embedder(agent_inputs) # project agent features to embedding space

        # Decode
        final_features = self.decoder(agent_embedding, enhanced_node_embedding, mask=mask)
        return final_features
    
class GraphAttentionEncoder2(nn.Module):
    def __init__(self, node_dim, pos_dim, agent_dim, embedding_dim):
        """
        use aggregate as current node feature
        add agent embedding to each node embedding before decoding
        """
        super(GraphAttentionEncoder2, self).__init__()
        # For node inputs
        self.node_embedder = nn.Linear(node_dim, embedding_dim) # 5-dim node inputs
        self.pos_embedder = nn.Linear(pos_dim, embedding_dim) # 32-dim positional encoding
        self.graph_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # For agent inputs
        self.agent_embedder = nn.Linear(agent_dim, embedding_dim) # 4-dim agent inputs
        
        # Fuse node and agent embeddings
        self.node_and_agent_encoder = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

    def forward(self, node_inputs, pos_embedding, agent_inputs, mask=None):
        '''
        node_inputs: (batch_size, num_nodes, node_dim) x, y, info, std, intent
        pos_encoding: (batch_size, num_nodes, pos_dim) 32-dim positional encoding
        agent_inputs: (batch_size, 1, agent_dim) x, y, budget, threshold (TBD prev intent mean and std)
        '''
        # Encode node inputs
        node_embedding = self.node_embedder(node_inputs)
        pos_embedding = self.pos_embedder(pos_embedding)
        node_embedding = node_embedding + pos_embedding
        aggregated_embedding, enhanced_node_embedding = self.graph_encoder(node_embedding)
        aggregated_embedding = self.aggregate_encoder(aggregated_embedding)

        # Encode agent inputs
        agent_embedding = self.agent_embedder(agent_inputs)

        # Fuse node and agent embeddings
        agent_embedding = agent_embedding.repeat(1, node_inputs.size()[1], 1)
        node_and_agent_embedding = torch.cat((enhanced_node_embedding, agent_embedding), dim=-1)
        node_and_agent_embedding = self.node_and_agent_encoder(node_and_agent_embedding)

        # Decode
        final_features = self.decoder(aggregated_embedding, node_and_agent_embedding, mask=mask)

        return final_features
    
class GraphAttentionEncoder3(nn.Module):
    def __init__(self, node_dim, pos_dim, agent_dim, embedding_dim):
        """
        Fuse aggregate and agent embeddings as current knowledge
        """
        super(GraphAttentionEncoder3, self).__init__()
        # For node inputs
        self.node_embedder = nn.Linear(node_dim, embedding_dim)  # 5-dim node inputs
        self.pos_embedder = nn.Linear(pos_dim, embedding_dim)  # 32-dim positional encoding
        self.graph_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        # For agent inputs
        self.agent_embedder = nn.Linear(agent_dim, embedding_dim) # 4-dim agent inputs

        # Fuse aggregate and agent embeddings
        self.fuse_layer = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

    def forward(self, node_inputs, pos_embedding, agent_inputs, mask=None):
        '''
        node_inputs: (batch_size, num_nodes, num_features) x, y, info, std, intent
        pos_encoding: (batch_size, num_nodes, pos_encoding_dim) 32-dim positional encoding
        agent_inputs: (batch_size, 1, agent_dim) x, y, budget, threshold (TBD prev intent mean and std)
        '''
        # Encode node inputs
        node_embedding = self.node_embedder(node_inputs) # project node inputs to embedding space
        pos_embedding = self.pos_embedder(pos_embedding) # project pos encoding to embedding space
        node_embedding = node_embedding + pos_embedding
        aggregated_embedding, enhanced_node_embedding = self.graph_encoder(node_embedding, mask=mask)

        # Encode agent inputs
        agent_embedding = self.agent_embedder(agent_inputs) # project agent features to embedding space

        current_knowledge = torch.cat((aggregated_embedding, agent_embedding), dim=-1)
        current_knowledge = self.fuse_layer(current_knowledge)

        # Decode
        final_features = self.decoder(current_knowledge, enhanced_node_embedding, mask=mask)
        return final_features
    
class GraphAttentionEncoder4(nn.Module):
    def __init__(self, node_dim, pos_dim, agent_dim, embedding_dim):
        """
        With budget inputs and threshold to enhance node embeddings
        """
        super(GraphAttentionEncoder4, self).__init__()
        # For node inputs
        self.node_embedder = nn.Linear(node_dim, embedding_dim)  # 5-dim node inputs
        self.pos_embedder = nn.Linear(pos_dim, embedding_dim)  # 32-dim positional encoding
        self.graph_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        # For agent inputs
        self.agent_embedder = nn.Linear(agent_dim, embedding_dim) # 4-dim agent inputs

        # To fuse enhanced node embeddings and budget info and threshold
        self.budget_embedder = nn.Linear(embedding_dim + 2, embedding_dim)

        # Fuse aggregate and agent embeddings
        self.fuse_layer = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

    def forward(self, node_inputs, pos_embedding, agent_inputs, mask=None):
        '''
        node_inputs: (batch_size, num_nodes, num_features) x, y, info, std, intent
        pos_encoding: (batch_size, num_nodes, pos_encoding_dim) 32-dim positional encoding
        agent_inputs: (batch_size, 1, agent_dim) x, y, budget, threshold (TBD prev intent mean and std)
        '''
        batch_size = node_inputs.size()[0]
        sample_size = node_inputs.size()[1]

        # Encode node inputs
        node_embedding = self.node_embedder(node_inputs) # project node inputs to embedding space
        pos_embedding = self.pos_embedder(pos_embedding) # project pos encoding to embedding space
        node_embedding = node_embedding + pos_embedding
        aggregated_embedding, enhanced_node_embedding = self.graph_encoder(node_embedding, mask=mask)

        # Compute budget info and thres to add to all node embeddings
        distance_inputs = node_inputs[:, :, 0:2]  - agent_inputs[:, :, 0:2]  # (batch_size, sample_size, 2)
        distance_inputs = torch.norm(distance_inputs, dim=-1, keepdim=True)  # (batch_size, sample_size, 1)
        budget_inputs = agent_inputs[:, :, 2].unsqueeze(-1).repeat(1, sample_size, 1) - distance_inputs  # (batch_size, sample_size, 1)
        th = agent_inputs[:, :, 3].unsqueeze(-1).repeat(1, sample_size, 1)  # (batch_size, sample_size, 1)

        enhanced_node_embedding = self.budget_embedder(torch.cat((enhanced_node_embedding, budget_inputs, th), dim=-1))

        # Encode agent inputs
        agent_embedding = self.agent_embedder(agent_inputs) # project agent features to embedding space

        current_knowledge = torch.cat((aggregated_embedding, agent_embedding), dim=-1)
        current_knowledge = self.fuse_layer(current_knowledge)

        # Decode
        final_features = self.decoder(current_knowledge, enhanced_node_embedding, mask=mask)
        return final_features
    
class GraphAttentionEncoder5(nn.Module):
    def __init__(self, node_dim, pos_dim, agent_dim, embedding_dim):
        """
        Simplifing GE4
        """
        super(GraphAttentionEncoder5, self).__init__()
        # For node inputs
        self.node_embedder = nn.Linear(node_dim, embedding_dim)  # 5-dim node inputs
        self.pos_embedder = nn.Linear(pos_dim, embedding_dim)  # 32-dim positional encoding
        self.graph_encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        # For agent inputs
        self.agent_embedder = nn.Linear(agent_dim, embedding_dim) # 4-dim agent inputs

        # To fuse enhanced node embeddings and budget info and threshold
        self.budget_embedder = nn.Linear(embedding_dim + 2, embedding_dim)

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

    def forward(self, node_inputs, pos_embedding, agent_inputs, mask=None):
        '''
        node_inputs: (batch_size, num_nodes, num_features) x, y, info, std, intent
        pos_encoding: (batch_size, num_nodes, pos_encoding_dim) 32-dim positional encoding
        agent_inputs: (batch_size, 1, agent_dim) x, y, budget, threshold (TBD prev intent mean and std)
        '''
        batch_size = node_inputs.size()[0]
        sample_size = node_inputs.size()[1]

        # Encode node inputs
        node_embedding = self.node_embedder(node_inputs) # project node inputs to embedding space
        pos_embedding = self.pos_embedder(pos_embedding) # project pos encoding to embedding space
        node_embedding = node_embedding + pos_embedding
        aggregated_embedding, enhanced_node_embedding = self.graph_encoder(node_embedding, mask=mask)

        # Compute budget info and thres to add to all node embeddings
        with torch.no_grad(): # these computations are to prepare the budget inputs and thres so no gradient needed
            distance_inputs = node_inputs[:, :, 0:2]  - agent_inputs[:, :, 0:2]  # (batch_size, sample_size, 2)
            distance_inputs = torch.norm(distance_inputs, dim=-1, keepdim=True)  # (batch_size, sample_size, 1)
            budget_inputs = agent_inputs[:, :, 2].unsqueeze(-1).repeat(1, sample_size, 1) - distance_inputs  # (batch_size, sample_size, 1)
            th = agent_inputs[:, :, 3].unsqueeze(-1).repeat(1, sample_size, 1)  # (batch_size, sample_size, 1)

        enhanced_node_embedding = self.budget_embedder(torch.cat((enhanced_node_embedding, budget_inputs, th), dim=-1))

        # Encode agent inputs
        agent_embedding = self.agent_embedder(agent_inputs) # project agent features to embedding space

        # Decode
        final_features = self.decoder(agent_embedding, enhanced_node_embedding, mask=mask)
        return final_features

def padding_inputs(inputs):
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    seq = seq.permute(2, 1, 0)
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    mask = torch.where(seq != 1, mask, ones)
    # print(mask)
    # print(seq.size())
    return seq, mask


if __name__ == '__main__':
    model = AttentionNet(2, 8, greedy=True)
    node_inputs = torch.torch.rand((128, 10, 2))
    # print(node_inputs)
    edge_inputs = torch.randint(0, 10, (128, 10, 5))
    edge_inputs_list = []
    # for i in range(edge_inputs.size()[1]):
    #     edge_inputs_list.append(edge_inputs[:,i].permute(1,0))
    # edge_inputs_list.append(torch.randint(0, 10, (8, 1)))
    # edge_inputs, mask = padding_inputs(edge_inputs_list)
    current_index = torch.ones(size=(128, 1, 1), dtype=torch.int64)
    next_node, logp_list, value = model(node_inputs, edge_inputs, current_index)
    print(next_node.size())
    print(logp_list.size())
    print(value.size())
