import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, minmax_scale
import plotly.express as px

def normal_weight_init(layer):
    if isinstance(layer, nn.Linear):
        in_fan = layer.in_features
        torch.nn.init.normal_(layer.weight, 0.0, 1/np.sqrt(in_fan))
        torch.nn.init.constant(layer.bias, 0.0)

def uniform_weight_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.xavier_uniform(layer.weight) #glorot init
        torch.nn.init.constant(layer.bias, 0.0)


class Linear_layer(nn.Module):
    def __init__(self, activation, in_dim, out_dim, bias):
        super(Linear_layer, self).__init__()
        self._linear_layer = nn.Linear(in_dim, out_dim, bias)
        self._activation = True

        if activation == None:
            self._activation = False
        else:
            self._activation = activation

    def forward(self, x):
        x = self._linear_layer(x)
        if self._activation:
            x = self._activation.forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, depth, reduction_ratio, activation, initialization):
        super(Encoder, self).__init__()
        self._layers = nn.ModuleList([])

        in_shape = input_shape

        ## Hidden layers of encoder
        for x in range(depth-1):
            out_shape = int(np.ceil(in_shape * reduction_ratio))
            if x == 0:
                self._layers.append(Linear_layer(activation=None, in_dim=in_shape, out_dim=out_shape, bias=True))
            else:
                self._layers.append(Linear_layer(activation=activation, in_dim=in_shape, out_dim=out_shape, bias=True))
            in_shape = out_shape

        ## Latent space layer
        self._layers.append(Linear_layer(activation=activation, in_dim=in_shape, out_dim=output_shape, bias=True))

        ## Initialize layers
        if initialization == 'uniform':
            self.apply(uniform_weight_init)
        else:
            self.apply(normal_weight_init)


    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, depth, reduction_ratio, activation, initialization):
        super(Decoder, self).__init__()
        self._layers = nn.ModuleList([])

        in_shape = input_shape

        ## Hidden layers of decoder
        for x in range(depth-1):
            out_shape = int(np.ceil(in_shape * (1/reduction_ratio)))
            self._layers.append(Linear_layer(activation=activation, in_dim=in_shape, out_dim=out_shape, bias=True))
            in_shape = out_shape

        ## Output layer
        self._layers.append(Linear_layer(activation=None, in_dim=in_shape, out_dim=output_shape, bias=True))

        ## Initialize layers
        if initialization == 'uniform':
            self.apply(uniform_weight_init)
        else:
            self.apply(normal_weight_init)

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x


class AutoencoderDataset(Dataset):
    def __init__(self, df:pd.DataFrame, normalize=False):
        super(AutoencoderDataset, self).__init__()
        self.data = torch.from_numpy(df.to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :] ## x = y in an autoencoder

class AutoEncoder(nn.Module):
    """

    """
    def __init__(self,
                 input_shape,
                 latent_space_shape,
                 depth,
                 reduction_ratio,
                 activation,
                 loss_function,
                 initializer,
                 device):
        super(AutoEncoder, self).__init__()

        self._encoder = Encoder(input_shape=input_shape,
                                output_shape=latent_space_shape,
                                depth=depth,
                                reduction_ratio=reduction_ratio,
                                activation=activation,
                                initialization=initializer)

        self._decoder = Decoder(input_shape=latent_space_shape,
                                output_shape=input_shape,
                                depth=depth,
                                reduction_ratio=reduction_ratio,
                                activation=activation,
                                initialization=initializer)

        self.loss = loss_function
        self._device = device
        self._optimizer = None

    def latent_representation(self, batch):
        return self._encoder.forward(batch)

    def encode_decode(self, batch):
        x = self.latent_representation(batch)
        x = self._decoder.forward(x)
        return x

    def train_model(self, train_dataset, test_dataset, optimizer, lr_sched, batch_size, epochs, workers=1):
        self._optimizer = optimizer
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        training_statistics = dict({'Epoch': list(),
                                    'Train Loss': list(),
                                    'Validation Loss':list()})

        for i in range(epochs):
            batch_loss = 0.0
            val_loss = 0.0

            self.train(True)
            ## Training ##
            for idx, batch in enumerate(train_dataloader):
                batch = batch.float().to(self._device)
                out = self.encode_decode(batch)
                loss = self.loss(batch, out)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                batch_loss += loss.item()

            batch_loss = batch_loss / (idx + 1)

            if lr_sched != None:
                lr_sched.step(batch_loss)

            ## Validation ##
            self.train(False)
            for idx, batch in enumerate(test_dataloader):
                batch = batch.float().to(self._device)
                out = self.encode_decode(batch)
                loss = self.loss(batch, out)
                val_loss += loss.item()

            val_loss = val_loss / (idx + 1)

            training_statistics['Epoch'].append(i)
            training_statistics['Train Loss'].append(batch_loss)
            training_statistics['Validation Loss'].append(val_loss)

            print(f"Epoch {i} train loss: {batch_loss}, val loss: {val_loss}")

        return training_statistics

def main():

    ### Generate Data ###
    mean_1 = np.array([-1.0, -2.5, -3.4, 0.25, 2.3, 3.5, 2.0, 1.0, -3.0, 0.1])
    mean_2 = np.array([3.0, -0.5, -1.4, 2.25, -2.3, 0.11, .21, -3.0, 13.0, 0.45])
    mean_3 = np.array([-2.0, -5.5, 1.4, 3.25, .3, .5, 14.0, .24, 6.0, 0.21])
    clust_1 = np.random.multivariate_normal(mean=mean_1, cov=np.eye(mean_1.shape[0])*3.0, size=500000, check_valid='warn', tol=1e-8)
    clust_2 = np.random.multivariate_normal(mean=mean_2, cov=np.eye(mean_2.shape[0])*4.0, size=500000, check_valid='warn', tol=1e-8)
    clust_3 = np.random.multivariate_normal(mean=mean_3, cov=np.eye(mean_3.shape[0])*5.0, size=500000, check_valid='warn', tol=1e-8)

    combined = np.vstack([clust_1, clust_2, clust_3])
    df = pd.DataFrame(combined)
    train, test = train_test_split(df, test_size=0.2)

    ### Set Device ###
    device = torch.device('cuda:0')
    train_dataset = AutoencoderDataset(df=train, normalize=False)
    test_dataset = AutoencoderDataset(df=test, normalize=False)

    auto = AutoEncoder(input_shape=mean_1.shape[0],
                       latent_space_shape=3,
                       depth=3,
                       reduction_ratio=0.75,
                       activation=nn.Tanh(),
                       loss_function=nn.MSELoss(),
                       initializer='normal',
                       device=device
                       )
    auto.to(device=device)

    #optimizer = torch.optim.SGD(auto.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.RMSprop(auto.parameters(), lr=1e-2, momentum=0.9)
    optimizer = torch.optim.Adam(auto.parameters(), lr=0.01)
    #sched_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    sched_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True, factor=0.8)

    with_scheduler = auto.train_model(train_dataset=train_dataset,
                     test_dataset=test_dataset,
                     optimizer=optimizer,
                     lr_sched=sched_2,
                     batch_size=512,
                     epochs=2,
                     workers=1)

    a =12313



if __name__ == '__main__':
    main()
