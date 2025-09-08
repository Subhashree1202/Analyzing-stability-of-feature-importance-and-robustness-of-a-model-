import torch.nn as nn

class SingleLayerSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerSoftmax, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, output_dim,bias=False)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the last dimension (dim=1)
        
    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        out = self.fc1(out)
        out = self.softmax(out)
        return out