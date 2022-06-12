import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image


def visualize_2d(model, path, domain=5, num=20, device='cpu'):
    # 对z_dim=2的情况可视化
    x = torch.linspace(-domain, domain, steps=num)
    y = torch.linspace(-domain, domain, steps=num)
    x1, y1 = torch.meshgrid(x, y)
    coord = torch.stack((x1, y1), 2).view(-1, 2)
    # torch.Size([5, 5, 2])->[25,2]
    model.eval()
    img = model.decoder(coord.to(device))
    save_image(img.view(num**2, 1, 28, 28), './images/2d/' + path + '.png', nrow=num)


def visualize_1d(model, path, num, left, right, device='cpu'):
    # 对z_dim=1的情况可视化
    z = torch.linspace(left, right, steps=num*num).unsqueeze(dim=1)
    model.eval()
    img = model.decoder(z.to(device))
    save_image(img.view(num**2, 1, 28, 28), './images/1d/' + path + '.png', nrow=num)
