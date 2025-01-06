import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn.functional as F
from torch.distributions.normal import Normal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EMWeights(nn.Module):
    def __init__(self, n_components=32):
        super(EMWeights, self).__init__()
        self.n_components = n_components
        temp = np.random.rand(n_components)
        self.weights = temp / temp.sum()  # Initialize weights not uniformly
        #self.weights = np.ones(n_components) / n_components  # Initialize weights uniformly

        self.fc1 = nn.Linear(n_components, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, n_components)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the features axis

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.softmax(x)  # Ensure outputs sum to 1
        return x

    def _gaussian_pdf(self, x, mean, variance):

        std = torch.sqrt(variance)

        # avoid zero std
        if std == 0:
            std = torch.tensor(1e-6, device=std.device)
                
        distribution = Normal(mean, std)
        return torch.exp(distribution.log_prob(x))


    def compute_weights(self, data, weights, means, variances):

        # calculate responsibilities for eack k
        responsibilities = torch.zeros((data.shape[0], self.n_components), device=data.device)
        for k in range(self.n_components):
            responsibilities[:, k] = weights[k] * self._gaussian_pdf(data, means[k], variances[k])
        responsibilities_sum = responsibilities.sum(dim=1, keepdim=True)

        # avoid divide by 0
        if torch.any(responsibilities_sum == 0):
            responsibilities_sum = torch.where(responsibilities_sum == 0, torch.tensor(1e-6, device=responsibilities_sum.device), responsibilities_sum)

        # Normalize responsibilities
        responsibilities /= responsibilities_sum
        weights = responsibilities.mean(dim=0)
        return weights.to(dtype=torch.float)
        


class EMMeanVariance(nn.Module):
    def __init__(self, se_dim = 256, pe_dim = 256, n_components=32):
        super(EMMeanVariance, self).__init__()
        self.se_dim = se_dim
        self.pe_dim = pe_dim
        self.n_components = n_components
        self.means = np.random.uniform(-100, 100, n_components)
        self.variances = np.random.uniform(0.1, 10, n_components) #np.random.uniform(0.1, 10, n_components) 

        # for point embeddings se
        self.conv1d_1 = nn.Conv1d(se_dim, 128, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # for point positional embeddings pe
        self.conv1 = nn.Conv2d(pe_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Adaptive pooling to ensure consistent feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fully connected layers
        self.fc1 = nn.Linear(32*2*2 + 64, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_components * 2) # for mean an variance
        
        
    def forward(self, point_embeddings, point_positional_embeddings):
        # Point embeddings se: [batch_size, 2, 256]
        # Point positional embeddings pe: [1, 256, 16, 16]
        # EM weights: [32]

        # Process point embeddings
        batch_size = point_embeddings.size(0)
        point_embeddings = point_embeddings.transpose(1, 2)  # Transpose to [batch_size, 256, (n_pt+1)]
        point_embeddings = F.relu(self.conv1d_1(point_embeddings))  # -> [batch_size, 128, N]
        point_embeddings = F.relu(self.conv1d_2(point_embeddings))  # -> [batch_size, 64, N]
        point_embeddings = self.global_pool(point_embeddings)  # -> [batch_size, 64, 1]
        point_embeddings = point_embeddings.view(batch_size, -1)  # Flatten -> [2, 64]

        # Process point positional embeddings
        point_positional_embeddings = (F.relu(self.conv1(point_positional_embeddings)))  # -> [1, 128, H/2, W/2]
        point_positional_embeddings = (F.relu(self.conv2(point_positional_embeddings)))  # -> [1, 64, H/2, W/2]
        point_positional_embeddings = (F.relu(self.conv3(point_positional_embeddings)))  # -> [1, 32, H/2, W/2]
        point_positional_embeddings = self.adaptive_pool(point_positional_embeddings)  # -> [1, 32, 2, 2]
        point_positional_embeddings = point_positional_embeddings.view(1, -1)  # Flatten -> [1, 32*2*2]

        # Combine processed embeddings
        point_positional_embeddings = point_positional_embeddings.repeat(batch_size, 1)  # Repeat to match batch size
        combined_embeddings = torch.cat([point_embeddings, point_positional_embeddings], dim=1)  # now [batch_size, 64+32*2*2]
        
        # Fully connected layers
        combined_embeddings = F.relu(self.fc1(combined_embeddings))  # [batch_size, 128]
        combined_embeddings = F.relu(self.fc2(combined_embeddings))  # [batch_size, 64]
        combined_embeddings = self.fc3(combined_embeddings)  # [batch_size, n_components * 2]
                
        # Split into means and variances
        means, variances = torch.chunk(combined_embeddings, 2, dim=1)  # -> ([batch_size, n_components], [batch_size, n_components])

        # Ensure variances are positive
        variances = F.relu(variances) + 1e-6  # [32], ensure positive variance

        # Aggregate the batch dimension by averaging
        means = means.mean(dim=0)  # -> [n_components]
        variances = variances.mean(dim=0)  # -> [n_components]

        return means, variances


class Fcomb(nn.Module):
    """
    Combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, output_channels=256, latent_dim=64, no_convs_fcomb=4,   #latent_dim is the hidden sampling space
                 initializers={'w':'orthogonal', 'b':'normal'}, use_tile=True):
        super(Fcomb, self).__init__()
        self.output_channels = output_channels
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.latent_dim = latent_dim
        self.no_convs_fcomb = no_convs_fcomb
        self.use_tile = use_tile

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.output_channels+self.latent_dim, self.output_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(self.no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.output_channels, self.output_channels, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)


    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (image embedding) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return output

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)