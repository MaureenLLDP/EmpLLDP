import math
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if (attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size

        self.embed = nn.Embedding(vocab_size, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        self.wordattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)
        emb_out = emb_out.permute(1, 0, 2)
        out_state, hid_state = self.wordRNN(emb_out, hid_state)
        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)
        sent = attention_mul(out_state, attn)
        return sent, hid_state


class FusedBilstm(nn.Module):
    def __init__(self, embedsize, hid_size, num_layers, device, manual_feature_dim=32, use_fusion=False):
        super(FusedBilstm, self).__init__()
        self.embedsize = embedsize
        self.hidsize = hid_size
        self.num_layers = num_layers
        self.device = device
        self.manual_feature_dim = manual_feature_dim
        self.use_fusion = use_fusion
        
        if self.use_fusion:
            lstm_input_size = self.embedsize + self.manual_feature_dim
        else:
            lstm_input_size = self.embedsize
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=self.hidsize, 
            num_layers=self.num_layers,
            dropout=0.5,
            bidirectional=True, 
            batch_first=True
        )
        
        classifier_input_dim = self.hidsize * 2
        self.classifier = nn.Linear(classifier_input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, ast_input):
        batch_size = ast_input.size(0)
        
        h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)

        lstm_output, (h_0, c_0) = self.lstm(ast_input, (h_0, c_0))
        
        query = self.dropout(lstm_output)
        deep_features, alpha_n = self.attention_net(lstm_output, query) 
        
        logit = self.classifier(deep_features)
        
        logit = self.sigmoid(logit)
        
        return logit
    
    def extract_deep_features(self, ast_input):
        batch_size = ast_input.size(0)
        
        h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)

        lstm_output, (h_0, c_0) = self.lstm(ast_input, (h_0, c_0))
        
        query = self.dropout(lstm_output)
        deep_features, alpha_n = self.attention_net(lstm_output, query)
        
        return deep_features


class Bilstm(nn.Module):
    def __init__(self, embedsize, hid_size, num_layars, device):
        super(Bilstm, self).__init__()
        self.embedsize = embedsize
        self.hidsize = hid_size
        self.num_layers = num_layars
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.embedsize, 
            hidden_size=self.hidsize, 
            num_layers=self.num_layers,
            dropout=0.5,
            bidirectional=True, 
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidsize * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, input):
        embeded_input = input
        batch_size = len(embeded_input)
        
        h_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidsize)).to(self.device)

        lstm_output, (h_0, c_0) = self.lstm(embeded_input, (h_0, c_0))

        query = self.dropout(lstm_output)
        attention_output, alpha_n = self.attention_net(lstm_output, query)

        logit = self.fc(attention_output)
        logit = self.sigmoid(logit) 

        return logit