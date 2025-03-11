import numpy as np
from napari_convpaint import conv_paint, conv_paint_param
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch 
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
annotations_path = data_list[1]
data_path = data_list[0]


data_dir = os.path.join(directory, data_path)
annotations_dir = os.path.join(directory, annotations_path)
print('data dir', data_dir)
print('annotations dir', annotations_dir)

print('..........Creating dataloaders')


# Define separate transforms for image & annotation
image_transform = transforms.Compose([
    transforms.ToTensor(),  # This normalizes image (0-255 -> 0-1)
])

annotation_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8))  
# Keeps annotation as uint8 (0 or 1) without normalization


# Create dataset and dataloader
dataset = TiffDataset(data_dir, annotations_dir, image_transform=image_transform, annotation_transform=annotation_transform)

train_size = int(0.8 * len(dataset))  
test_size = int(0.2 * len(dataset))    

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


print('..........initiate model')
param_vgg = conv_paint_param.Param()
param_vgg.fe_name = "vgg16"
param_vgg.fe_scalings = [1, 2, 4]
param_vgg.fe_order = 0
param_vgg.fe_layers = layer_list[0:depth]
param_vgg.image_downsample = downsample_factor

# create model
model_vgg = conv_paint.create_model(param_vgg)

model = model_vgg
param = param_vgg



print('..........create embeddings')
all_features, all_targets = [], []
for i, (image, annotation) in enumerate(train_dataloader):

    print("Images shape:", image.squeeze().shape)  # (batch_size, channels, height, width)
    print("Labels shape:", annotation.squeeze().shape)  # (batch_size,)
    print("labels squeezed", annotation.squeeze())
    print("images squeezed", image.squeeze())
    features, targets = conv_paint.get_features_current_layers(
        image.squeeze(),
        annotation.squeeze(),
        model=model,
        param=param,
    )
    all_features.append(features)
    all_targets.append(targets)
    print('targets', targets)

features_array = np.concatenate(all_features, axis=0)
targets_array = np.concatenate(all_targets, axis=0)

print(features_array)
print(targets_array)
print(i.shape() for i in targets_array)
print(np.unique(targets_array))
