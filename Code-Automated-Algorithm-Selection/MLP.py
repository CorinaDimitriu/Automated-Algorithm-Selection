import torch
from torch import Tensor


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = [torch.nn.Linear(16, 144),
                       torch.nn.ReLU(),
                       torch.nn.Linear(36, 72),
                       torch.nn.ReLU(),
                       torch.nn.Linear(72, 144),
                       torch.nn.ReLU(),
                       torch.nn.Linear(144, 10),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 144),
                       torch.nn.ReLU(),
                       torch.nn.Linear(144, 6),
                       torch.nn.Softmax(dim=1)]
        torch.nn.init.xavier_normal_(self.layers[0].weight)
        torch.nn.init.xavier_normal_(self.layers[2].weight)
        torch.nn.init.xavier_normal_(self.layers[4].weight)
        torch.nn.init.xavier_normal_(self.layers[6].weight)
        torch.nn.init.xavier_normal_(self.layers[8].weight)
        torch.nn.init.xavier_normal_(self.layers[10].weight)
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, features: Tensor):
        features = self.layers(features)
        return features
