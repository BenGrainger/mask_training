
import numpy as np
from napari_convpaint import conv_paint, conv_paint_param
import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import TiffDataset, compute_iou, compute_fpr, ScriptSetup, check_folder_exists
import os
import pandas as pd
import matplotlib.pyplot as plt

print('..........loading config')
config_path = r'configs/test_config.json'

script = ScriptSetup(config_path)
script.load_script()
config = script.return_config()
logging = script.return_logger()
out_dir = script.return_out_dir()
root = script.return_root()



directory = os.path.join(root, config['data_dir'])

print('..........Loading inputs')
batch_size = config["batch_size"]
n = config["slice_n"]
depth = config["depth"]
downsample_factor = config["downsample"]
layers_path = r'layers.txt'
with open("layers.txt", "r") as file:
    layer_list = [line.strip() for line in file.readlines()]


data_list = config["data_list"]
data_list = [i for i in data_list.values()]
annotations_path = data_list[0]
data_path = data_list[1]


data_dir = os.path.join(directory, data_path)
annotations_dir = os.path.join(directory, annotations_path)

print(annotations_dir)


print('..........Creating dataloaders')
# Define transformations (convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = TiffDataset(data_dir, annotations_dir, transform=transform)

train_size = int(0.8 * len(dataset))  
test_size = int(0.2 * len(dataset))    

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

for images, labels in train_dataloader:
    print("Images shape:", images.shape)  # (batch_size, channels, height, width)
    print("Labels shape:", labels.shape)  # (batch_size,)
    print(labels.sum())
    print(torch.unique(labels).numel() > 1)
    print(torch.unique(images).numel() > 1)
    print(images[0])
    print(labels[0])
