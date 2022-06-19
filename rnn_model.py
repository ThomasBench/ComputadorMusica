import torch.nn as nn 
import torch 



### ENCODER & DECODER MODULES
# ENCODER ENCODES THE LYRICS     DECODER OUTPUTS THE CHORDS
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.inLinear = nn.Linear(input_size,hidden_size).to(device)
        self.out = nn.Linear(hidden_size,hidden_size).to(device)
        self.gru = nn.GRU(hidden_size, hidden_size).to(device)

    def forward(self, input, hidden):
        embedded = self.inLinear(input.view(1, 1, -1))
        output = nn.functional.relu(embedded)
        output, hidden = self.gru(output, hidden)
        hidden = self.out(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.gru = nn.GRU(hidden_size, hidden_size).to(device)
        # self.out = nn.Linear(hidden_size, output_size)
        self.out2 = nn.Linear(hidden_size,output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=2).to(device)

    def forward(self, input, hidden):
        output = self.embedding(input.to(self.device)).view(1, 1, -1)
        output = nn.functional.relu(output).to(self.device)
        output, hidden = self.gru(output, hidden)
        # output = self.out(output)
        output = self.out2(nn.functional.relu(output).to(self.device))
        output = self.softmax(output)
        # print("Input of decoder after softmax  + sum",output.shape, output.sum())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)



