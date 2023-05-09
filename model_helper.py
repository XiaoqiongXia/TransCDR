import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
np.random.seed(1)


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size) #每个词和位置先随机初始化
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, inputs):
        seq_length = inputs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs)
        words_embeddings = self.word_embeddings(inputs)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value
        v = score @ v
        return v,score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out,score = self.attention(q, k, v, mask=mask)
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        # 5. visualize attention map
        # TODO : we should implement visualization
        return out,score
    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        return tensor
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x,_ = self.attention(q=x, k=x, v=x, mask=s_mask)        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)       
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    def forward(self, dec, enc, t_mask, s_mask):
        # 1. compute self attention
        _x = dec
        x,_ = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x,_ = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)           
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x) 
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, x, s_mask):
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output
    
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         in_ch = [1] + [16,64,64]
#         layer_size = len(in_ch)-1
#         kernels = [3,5,7]
#         self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
#                                         out_channels = in_ch[i+1], 
#                                         kernel_size = kernels[i],padding='same') for i in range(layer_size)])
#         self.fc = nn.Linear(in_ch[-1], 64)

#     def forward(self, v):
#         for l in self.conv:
#             v = F.relu(l(v))
#         v = v.transpose(2, 1)
#         v = self.fc(v)
#         return v

class CNN(nn.Sequential):
    def __init__(self):
        super(CNN, self).__init__()
        in_ch = [63] + [32,64,96]
        kernels = [4,6,8]
        layer_size = 3
        self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                out_channels = in_ch[i+1], 
                                                kernel_size = kernels[i]) for i in range(layer_size)])
        self.conv = self.conv.double()
        n_size_d = self._get_conv_output((63, 100))
        self.fc1 = nn.Linear(n_size_d, 256)
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x
    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class RNN(nn.Sequential):
    def __init__(self):
        super(RNN, self).__init__()
        in_ch = [63] + [32,64,96]
        self.in_ch = in_ch[-1]
        kernels = [4,6,8]
        layer_size = 3
        self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                out_channels = in_ch[i+1], 
                                                kernel_size = kernels[i]) for i in range(layer_size)])
        self.conv = self.conv.double()
        n_size_d = self._get_conv_output((63, 100)) # auto get the seq_len of CNN output

        # if config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
        self.rnn = nn.LSTM(input_size = in_ch[-1], 
                        hidden_size = 64,
                        num_layers = 2,
                        batch_first = True,
                        bidirectional = True)
        
        # elif config['rnn_Use_GRU_LSTM_drug'] == 'GRU':
        #     self.rnn = nn.GRU(input_size = in_ch[-1], 
        #                     hidden_size = config['rnn_drug_hid_dim'],
        #                     num_layers = config['rnn_drug_n_layers'],
        #                     batch_first = True,
        #                     bidirectional = config['rnn_drug_bidirectional'])
        # else:
        #     raise AttributeError('Please use LSTM or GRU.')
        # direction = 2 if config['rnn_drug_bidirectional'] else 1
        direction = 2
        self.rnn = self.rnn.double()
        self.fc1 = nn.Linear(64 * direction * n_size_d,256)
        # self.config = config

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, self.in_ch, -1).size(2)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        return x

    def forward(self, v):
        for l in self.conv:
            v = F.relu(l(v.double()))
        batch_size = v.size(0)
        v = v.view(v.size(0), v.size(2), -1)
        # if self.config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
        # direction = 2 if self.config['rnn_drug_bidirectional'] else 1
        direction = 2
        h0 = torch.randn(2 * direction, batch_size, 64).to(device)
        c0 = torch.randn(2 * direction, batch_size, 64).to(device)
        v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
        # else:
        #     # GRU
        #     direction = 2 if self.config['rnn_drug_bidirectional'] else 1
        #     h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size, self.config['rnn_drug_hid_dim']).to(device)
        #     v, hn = self.rnn(v.double(), h0.double())
        v = torch.flatten(v, 1)
        v = self.fc1(v.float())
        return v


class Transformer(nn.Sequential):
    def __init__(self):
        super(Transformer, self).__init__()
        self.emb = Embeddings(2586, 256, 50, 0.1)
        self.encoder = Encoder(256,256,8,8,0.1,device)
    def forward(self, v_d, mask):
        v_d = self.emb(v_d)
        # mask = mask.unsqueeze(1).unsqueeze(2)
        # mask = (1.0 - mask) * -10000.0
        encoded_layers = self.encoder(v_d.float(), mask.float())
        return encoded_layers[:,0]


class GCN_encoder(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/gcn_predictor.py#L16
    def __init__(self, in_feats=74, hidden_feats=[64]*3, activation=[F.relu]*3, predictor_dim=256):
        super(GCN_encoder, self).__init__()
        from dgllife.model.gnn.gcn import GCN
        from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
        self.gnn = GCN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        activation=activation
                        )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, predictor_dim)

    def forward(self, bg):
        bg = bg.to(device)
        feats = bg.ndata.pop('h') 
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.transform(graph_feats)


class NeuralFP(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/gat_predictor.py
    def __init__(self, in_feats=74, hidden_feats=[63]*3, max_degree = 10, activation=[F.relu]*3, predictor_hidden_size = 128, predictor_activation = torch.tanh, predictor_dim=256):
        super(NeuralFP, self).__init__()
        from dgllife.model.gnn.nf import NFGNN
        from dgllife.model.readout.sum_and_max import SumAndMax

        self.gnn = NFGNN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        max_degree=max_degree,
                        activation=activation
                        )
        gnn_out_feats = self.gnn.gnn_layers[-1].out_feats
        self.node_to_graph = nn.Linear(gnn_out_feats, predictor_hidden_size)
        self.predictor_activation = predictor_activation

        self.readout = SumAndMax()
        self.transform = nn.Linear(predictor_hidden_size * 2, predictor_dim)

    def forward(self, bg):
        bg = bg.to(device)
        feats = bg.ndata.pop('h') 
        node_feats = self.gnn(bg, feats)
        node_feats = self.node_to_graph(node_feats)
        graph_feats = self.readout(bg, node_feats)
        graph_feats = self.predictor_activation(graph_feats)
        return self.transform(graph_feats)


class AttentiveFP(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/python/dgllife/model/model_zoo/attentivefp_predictor.py#L17
    def __init__(self, node_feat_size=39, edge_feat_size=11, num_layers=3, num_timesteps = 2, graph_feat_size = 64, predictor_dim=256):
        super(AttentiveFP, self).__init__()
        from dgllife.model.gnn import AttentiveFPGNN
        from dgllife.model.readout import AttentiveFPReadout

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)

        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)

        self.transform = nn.Linear(graph_feat_size, predictor_dim)

    def forward(self, bg):
        bg = bg.to(device)                
        node_feats = bg.ndata.pop('h')
        edge_feats = bg.edata.pop('e')

        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats, False)
        return self.transform(graph_feats)
