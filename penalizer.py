import torch
import torch.nn.functional as F
from Model import device

scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype = torch.float32)
scharr_y = scharr_x.t()

scharr_x = scharr_x.reshape(1, 3, 3).expand(3, -1, -1)
scharr_y = scharr_y.reshape(1, 3, 3).expand(3, -1, -1)

scharr = torch.stack([scharr_x, scharr_y])

conv = torch.nn.Conv2d(3, 2, 3)
conv.weight = torch.nn.Parameter(scharr)
conv = conv.to(device)

def loss(x, colorfulness = 0.0, sharp_edges = 0.0):

    return (
        (x ** 2.0).to("cpu").mean() * (10.0 ** -colorfulness) + 
        (conv(x) ** 2.0).to("cpu").mean() * (10.0 ** -sharp_edges)
    )