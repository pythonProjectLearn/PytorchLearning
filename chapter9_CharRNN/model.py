#coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim,num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input,hidden=None):
        seq_len,batch_size = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0,c_0 = Variable(h_0),Variable(c_0)
        else:
            h_0,c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0,c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len*batch_size, -1))
        return output,hidden
    
"""
创建RNN
"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size+hidden_size, out_features=hidden_size)
        self.i2o = nn.Linear(in_features=input_size+hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1) # 输出1维

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined) # 循环时，新产生hidden重新与input进行combined
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
"""
LSTM
"""
class LSTM_rnn(nn.Module):
    def __init__(self, n_vocab, hidden_size, n_cat, bs=1, n1=2):
        super(LSTM_rnn, self).__init__()
        self.hidden_size = hidden_size
        self.bs=bs
        self.n1=n1
        self.e = nn.Embedding(num_embeddings=n_vocab,embedding_dim=hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n1)
        self.fc2 = nn.Linear(hidden_size, n_cat)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        h0 = c0= Variable(e_out.data.new(*(self.n1, self.bs, self.hidden_size)).zero_())
        rnn_o,_ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.8)
        return self.softmax(fc)

"""
CNN
"""
class CnnText(nn.Module):
    def __init__(self, n_vocab, hidden_size, n_cat, bs=1, kernel_size=3, max_len=200):
        super(CnnText, self).__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.e = nn.Embedding(num_embeddings=n_vocab, embedding_dim=hidden_size)
        self.cnn = nn.Conv1d(in_channels=max_len, out_channels=hidden_size, kernel_size=kernel_size)
        self.avg = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(in_features=1000, out_features=n_cat)
        self.sofmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp):
        bs = inp.size()[0]
        if bs != self.bs:
            self.bs = bs

        e_out = self.e(inp)
        cnn_o = self.cnn(e_out)
        cnn_avg = self.cnn(cnn_o)
        cnn_avg = cnn_avg.view(self.bs, -1)
        fc = F.dropout(self.fc(cnn_avg), p=0.5)
        return self.softmax(fc)
