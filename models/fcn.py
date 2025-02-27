import torch
import torch.nn as nn
# bulid a fully connected network for regression

class FCN(nn.Module):
    def __init__(self, in_ch, out_ch,
                 hidden_ch, num_layers, bn=False, sigmoid=False):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential()
        self.fcn.add_module('fc0', nn.Linear(in_ch, hidden_ch))
        self.fcn.add_module('relu0', nn.ReLU())
        if bn:
            self.fcn.add_module('bn0', nn.BatchNorm1d(hidden_ch, affine=True))
        for i in range(num_layers - 1):
            self.fcn.add_module('fc{}'.format(i+1),
                                nn.Linear(hidden_ch, hidden_ch))
            self.fcn.add_module('relu{}'.format(i+1), nn.ReLU())
            if bn:
                self.fcn.add_module('bn{}'.format(i+1),
                                    nn.BatchNorm1d(hidden_ch, affine=True))
            
        self.fcn.add_module('fc{}'.format(num_layers),
                            nn.Linear(hidden_ch, out_ch))
        
        if sigmoid:
            self.fcn.add_module('sigmoid', nn.Sigmoid())
        
        
    def forward(self, x):
        return self.fcn(x)