from cmd_args import args

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import tqdm
from PIL import Image

from Model import Model, device
import penalizer

step_size = 3.0 * args.ss
decay = 0.8
steps_per_search = 70
num_searches = 8
learning_momentum = 0.66
plot_loss = False

if device == "cpu": print("Warning: CUDA not in use")

mean_normalize = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
std_normalize = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

if plot_loss: import matplotlib.pyplot as plt
 
model = Model().to(device)

num_modules = 5
module_num_features = [64, 64, 128, 256, 512]

image_dim = (args.size, args.size)
colorfulness = args.colorfulness
sharp_edges = args.sharpness
pattern_intensity = args.intensity
num_octaves = args.octaves
octave_scaling = args.scaling
module = args.layer
feature = args.feature

def save_image(title = None): 
    
    img = x.detach() * std_normalize + mean_normalize
    img = (img[0].clamp(0.0, 1.0) * 255.0).to("cpu").to(torch.uint8).numpy().transpose((1, 2, 0))
    Image.fromarray(img, mode = "RGB").save("figs/" + title + ".png")
    
def mul(tpl, v): return (int(float(tpl[0]) * v), int(float(tpl[1]) * v))

if args.image is not None:

    try:
    
        with Image.open(args.image) as img:
            
            x = transforms.PILToTensor()(img).to(torch.float32).to(device) / 255.0
            x = (x - mean_normalize) / std_normalize
            
    except:
    
        print("Failed to open image: " + args.image)
        exit()
        
    if x.shape[1] != 3:
    
        print("Image " + args.image + " is not in RGB format")
        exit()
        
    image_dim = (x.shape[2], x.shape[3])

octaves = [image_dim]

if num_octaves > 1:
        
    for octave in range(num_octaves - 1):

        image_dim = mul(image_dim, 1.0 / octave_scaling)
        if image_dim[0] < 32 or image_dim[1] < 32: 
        
            print("Base image size is smaller than minimum 32x32, try decreasing number of octaves or using larger image")
            exit()
            
        octaves.append(image_dim)
        
    octaves.reverse()

    if args.image is not None: x = F.interpolate(x, (image_dim[0], image_dim[1]), mode = "bicubic")
    
if args.image is None: x = torch.rand(1, 3, image_dim[0], image_dim[1]).to(device) - 0.5
x.requires_grad = True

for octave in range(num_octaves):

    running_grad = torch.zeros_like(x)
    step = step_size
    
    if plot_loss: loss_history = []

    for i in tqdm.trange(steps_per_search * num_searches, ascii = "-123456789#", desc = "Octave " + str(octave + 1) + "/" + str(num_octaves)):
        
        x_condensed = x if args.magnify == 1 else F.avg_pool2d(x, args.magnify, args.magnify - 1)
        feature_map = model(x_condensed, module, feature).to("cpu")
        
        loss = penalizer.loss(x_condensed, colorfulness, sharp_edges) - feature_map.mean() * (10.0 ** pattern_intensity)
        loss.backward(inputs = [x])
        
        cur_grad = x.grad / (torch.max(torch.abs(x.grad)) + 0.0001)
        running_grad = running_grad * learning_momentum + cur_grad * (1.0 - learning_momentum)
        x = x - step_size * running_grad
        
        if (i + 1) % steps_per_search == 0: step *= decay
        if plot_loss: loss_history.append(loss.item())
    
    if num_octaves > 1 and octave < num_octaves - 1:
        
        image_dim = octaves[octave + 1]
        step_size /= octave_scaling * octave_scaling
        
        x = F.interpolate(x, (image_dim[0], image_dim[1]), mode = "bicubic")
    
    if plot_loss:
        
        plt.figure()
        plt.plot(loss_history)
        plt.title("Iteration vs. Loss")
        plt.show(block = True)
        
    if num_octaves == 1: break
        
if args.outscale != 1.0:

    image_dim = mul(image_dim, args.outscale)
    x = F.interpolate(x, (image_dim[0], image_dim[1]), mode = "bicubic")

img_name = "" if args.image is None else (".".join(args.image.split(".")[:-1]) + "_")
save_image(img_name + "module_" + str(module) + "_feature_" + str(feature) + "_" + str(image_dim[1]) + "x" + str(image_dim[0]))