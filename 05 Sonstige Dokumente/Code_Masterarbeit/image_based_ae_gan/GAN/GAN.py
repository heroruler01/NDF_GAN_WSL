import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_scatter

class NDF(nn.Module):


    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()

        # feature_size = (1 + 16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        feature_size = 131
        # self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)  # should i change the input size = size of the latent vector?
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)


    def generator(self, p, latent_vector):

        p_features = p.transpose(1, -1)

        features = torch.cat((latent_vector, p_features), dim=1)  # (B, featue_size, samples_num) # [4,3482,50000]

        # net = self.z_lin1(self.latent_channels).unsqueeze(1) + p # p should be of dimension [batch size, num points, 3]

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return out

    def forward(self, p, latent_vector):
        out_fake = self.generator(p, latent_vector)
        return out_fake


# Input to D is (4,50 000) as Distances and corresponding points (x, 3, 50000) because we have the Distance values for each point, and the points are the same.
class Discriminator(nn.Module):

    def __init__(self, out_channels=1):
        super(Discriminator, self).__init__()

        self.nn1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels), # out_channels = 1, fake - real as D_loss: implementation of Wassersein Loss
        )

    # give the Discriminator the position and the distance from the training file
    def forward(self, p, dist, batch=None):
        dist = dist.unsqueeze(-1) if dist.size(-1) != 1 else dist

        x = torch.cat([p, dist], dim=-1)

        x = self.nn1(x)

        # why the max? implemented as in the paper
        x = x.max(dim=-2)[0] # maxpool over number of points

        '''
        if batch is None:
            x = x.max(dim=-2)[0]
        else:
            x = scatter_max(x, batch, dim=-2)[0]
        '''

        x = self.nn2(x)

        return x




