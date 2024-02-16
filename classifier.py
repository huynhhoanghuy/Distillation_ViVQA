"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):
        super(SimpleClassifier, self).__init__()
        activation_dict = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(), 'tanh': nn.Tanh()}
        try:
            activation_func = activation_dict[args.activation]
        except:
            raise AssertionError(args.activation + " is not supported yet!")
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            activation_func,
            nn.Dropout(args.dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



class StudentSimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):
        super(StudentSimpleClassifier, self).__init__()
        activation_dict = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(), 'tanh':nn.Tanh()}
        try:
            activation_func = activation_dict[args.activation]
        except:
            raise AssertionError(args.activation + " is not supported yet!")
        layers = [
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(hid_dim, hid_dim//2),
            nn.Tanh(),
            nn.Linear(hid_dim//2, out_dim),
        ]
        
        self.main = nn.Sequential(*layers)
        self.main.apply(weights_init)
        # nn.init.xavier_uniform_(self.main) 
        # self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

    def forward(self, x, y, c):
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        x = x + self.bn2(self.relu(self.lin_c(c)))
        x = self.lin2(self.drop(self.bn(x)))
        return x