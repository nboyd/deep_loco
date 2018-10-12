import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepLoco(nn.Module):
    def __init__(self, net_filters = 16, conv_block_depth = 2, initial_filter_size = 5, filter_size = 3, resnet_out_dim = 1024, resnet_depth = 2, max_points = 64, min_coords = [0,0,-750], max_coords = [6400,6400,750] ):
        super(DeepLoco, self).__init__()
        print("Creating deep_loco, resnet_out_dim={}, max_points={}".format(resnet_out_dim, max_points))
        feature_net = nn.Sequential(DeepConvNet(1,net_filters, conv_block_depth, initial_filter_size),
                                    nn.Conv2d(net_filters,net_filters*4, 2,2),
                                    DeepConvNet(net_filters*4,net_filters*4,conv_block_depth, filter_size),
                                    nn.Conv2d(net_filters*4,net_filters*4*4, 2,2),
                                    DeepConvNet(net_filters*4*4,net_filters*4*4,conv_block_depth, filter_size),
                                    nn.Conv2d(net_filters*4*4,net_filters*4*4, 4,4))
        self.net = nn.Sequential(feature_net,ResNet(net_filters*4*4*4*4, resnet_out_dim, resnet_depth),
                            WeightsAndLocations(resnet_out_dim, max_points,len(min_coords), min_coords,max_coords ))
    def forward(self, input):
        return self.net(input.unsqueeze(1))

class DeepConvNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, depth, kernel_size =5):
        assert(depth >= 0)
        super(DeepConvNet, self).__init__()
        self.blocks = nn.Sequential(*([nn.Conv2d(input_channels, hidden_channels, kernel_size = kernel_size, padding=(kernel_size-1)//2, bias=True), nn.ReLU()] + [nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size = kernel_size, padding=(kernel_size-1)//2, bias=True), nn.ReLU()) for i in range(depth-1)]))

    def forward(self, input):
        return self.blocks(input)

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        assert(depth >= 0)
        super(ResNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for i in range(depth)])
        self.output_dim = hidden_dim

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.linear(input)
        for res_block in self.res_blocks:
            x = res_block(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias = False)
        self.l1.bias.data.uniform_(-1E-7,1E-7)
        self.l1.weight.data.uniform_(-1E-7,1E-7) #?!
        self.l2.weight.data.uniform_(-1E-7,1E-7)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))

class WeightsAndLocations(nn.Module):
    def __init__(self, feature_dim, MAX_SOURCES, source_dim, min_coord, max_coord):
        super(WeightsAndLocations, self).__init__()
        self.locations = nn.Linear(feature_dim, source_dim*MAX_SOURCES)
        self.weights = nn.Linear(feature_dim, MAX_SOURCES)
        self.source_dim = source_dim
        self.min_coord = torch.Tensor(min_coord).view(1,1,-1)
        self.max_coord = torch.Tensor(max_coord).view(1,1,-1)
    def forward(self, input):
        input = input.view(input.size(0), -1)
        thetas = F.sigmoid(self.locations(input).view(input.size(0), -1,self.source_dim))
        if thetas.is_cuda:
            thetas = thetas*Variable(self.max_coord-self.min_coord).cuda()
            thetas = thetas + Variable(self.min_coord).cuda()
        else:
            thetas = thetas*Variable(self.max_coord-self.min_coord)
            thetas = thetas + Variable(self.min_coord)
        return thetas, F.relu(self.weights(input))
