import argparse
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from visualize import visualize_2d, visualize_1d

parser = argparse.ArgumentParser(description='VAE Model on Minist')
parser.add_argument('--epoch', type=int, default=80,
                    help='num of epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size')
parser.add_argument('--l1', type=float, default=1.0,
                    help='lambda * KL_loss')
parser.add_argument('--z_dim', type=int, default=2,
                    help='dim for mu and logvar')
parser.add_argument('--h1', type=int, default=512,
                    help='dim for hidden the first layer')
parser.add_argument('--h2', type=int, default=256,
                    help='dim for hidden the second layer')
parser.add_argument('--add_noise', action='store_true',
                    help='add_noise at the training data')
parser.add_argument('--add_BN', action='store_true',
                    help='add BN and then scaling after mu')
parser.add_argument('--noise_factor', type=float, default=0.1,
                    help='add_noise at the training data')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='0',
                    help='image save dir')
args = parser.parse_args()

# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, args.h1),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(args.h1, args.h2),
            # nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.e_mu = nn.Linear(args.h2, args.z_dim)
        self.e_var = nn.Linear(args.h2, args.z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(args.z_dim, args.h2),
            # nn.LeakyReLU(0.1),
            nn.ReLU(),
            nn.Linear(args.h2, args.h1),
            # nn.LeakyReLU(0.1),
            nn.ReLU(),
            nn.Linear(args.h1, 784),
            nn.Sigmoid()
        )

        self.mu_bn = nn.BatchNorm1d(args.z_dim)
        self.mu_bn.weight.requires_grad = False
        # self.mu_bn.weight.fill_(self.args.gamma)  # gamma = 0.5

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # [50, 2]
        # torch.randn_like -> N(0,1) torch.rand_like -> U(0,1)
        return mu + std * eps

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, log_var = self.e_mu(h), self.e_var(h)  # mu, log_var
        # mu [50, 2]
        if args.add_BN:
            mu = self.mu_bn(mu)

        z = self.sampling(mu, log_var)
        # print('z', z)
        pred = self.decoder(z)
        return pred, mu, log_var


# build model
device = torch.device("cuda" if args.cuda else "cpu")
vae = VAE(args=args).to(device)
optimizer = optim.Adam(vae.parameters(), lr=args.lr)


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # BCE1 = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    LOSS = args.l1*KLD + BCE
    return BCE, KLD, LOSS


# Denoising VAE，add noise
def add_white_noise(x, mean=0., std=1e-6):
    # x [50, 1, 28, 28]
    rand = torch.rand([x.shape[0], 1, 1, 1])
    rand = torch.where(rand > 0.5, 1., 0.).to(x.device)
    # rand [50, 1, 1, 1]
    white_noise = torch.normal(mean, std, size=x.shape, device=x.device)
    # white_noise [50, 1, 28, 28]
    noise_x = x + white_noise * rand
    noise_x = torch.clip(noise_x, 0., 1.)
    return noise_x


def train(epoch, device='cpu'):
    vae.train()
    train_loss = 0
    train_kld_loss = 0
    train_recon_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # data [50, 1, 28, 28], in [0,1], continous!
        # add noise: DVAE
        data_noise = add_white_noise(data) if args.add_noise else data
        data_noise = data_noise.to(device)
        # Forward
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data_noise)
        recon_loss, kld_loss, loss = loss_function(recon_batch, data_noise, mu, log_var)
        # Backward
        loss.backward()
        train_loss += loss.item()
        train_kld_loss += kld_loss.item()
        train_recon_loss += recon_loss.item()
        optimizer.step()

        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tReLoss: {:.6f}\tKLDLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_noise), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / args.batch_size,
                recon_loss.item() / args.batch_size, kld_loss.item() / args.batch_size))
    num = len(train_loader.dataset)
    print('===> Epoch: {} Average loss: {:.4f}, Kld loss: {:.4f}, Recon loss: {:.4f}'
          .format(epoch, train_loss / num, train_kld_loss / num, train_recon_loss / num))


def test(device='cpu'):
    vae.eval()
    test_loss = 0
    test_kld_loss = 0
    test_recon_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)
            # sum up batch loss
            recon_loss, kld_loss, loss = loss_function(recon, data, mu, log_var)
            test_loss += loss.item()
            test_kld_loss += kld_loss.item()
            test_recon_loss += recon_loss.item()
    num = len(test_loader.dataset)
    print('===> Test loss: {:.4f}, Kld loss: {:.4f}, Recon loss: {:.4f}'
          .format(test_loss/num, test_kld_loss/num, test_recon_loss/num))


# Training
for epoch in range(0, args.epoch):
    train(epoch+1, device=device)
    test(device=device)

# Visualize
if args.z_dim == 2:
    visualize_2d(vae, args.save, domain=5, num=20, device=device)
if args.z_dim == 1:
    visualize_1d(vae, args.save, num=20, left=-5, right=5, device=device)


