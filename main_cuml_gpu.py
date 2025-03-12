import numpy as np
from napari_convpaint import conv_paint, conv_paint_param
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
from utils import TiffDataset, compute_iou, compute_fpr, ScriptSetup, check_folder_exists
import os
import pandas as pd
import matplotlib.pyplot as plt
from cuml.ensemble import RandomForestClassifier
import cudf  # GPU dataframe
import skimage


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


print('..........Creating dataloaders')

image_transform = transforms.Compose([
    transforms.ToTensor(),  # This normalizes image (0-255 -> 0-1)
])

annotation_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.uint8))  
# Keeps annotation as uint8 (0 or 1) without normalization


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
    print(i)
    features, targets = conv_paint.get_features_current_layers(
        image.squeeze(),
        annotation.squeeze(),
        model=model,
        param=param,
    )
    all_features.append(features)
    all_targets.append(targets)

features_array = np.concatenate(all_features, axis=0)
targets_array = np.concatenate(all_targets, axis=0)


print('..........GPU train random forest')

X_gpu = cudf.DataFrame(features_array)
y_gpu = cudf.Series(targets_array)

# Train GPU-based Random Forest
rf_gpu = RandomForestClassifier(n_estimators=100, max_depth=10, n_bins=16)
rf_gpu.fit(X_gpu, y_gpu)

print('..........GPU predict')

iou_scores = []
fpr_scores = []

check_folder_exists(out_dir + "/images")

for i, (image, annotation) in enumerate(test_dataloader):
    print('.......... predicting test image', i)
    dummy_annotation = np.empty_like(image.squeeze())  

    # Extract features from the model
    features, targets = conv_paint.get_features_current_layers(
        image.squeeze(),
        dummy_annotation,
        model=model,
        param=param,
    )

    # Convert features to cuDF DataFrame for cuML processing
    X_gpu = cudf.DataFrame(features)

    # Use trained classifier to make predictions
    y_pred_gpu = rf_gpu.predict(X_gpu)

    # Convert predictions from cuDF to NumPy array
    y_pred_numpy = y_pred_gpu.to_numpy()

    rows = np.ceil(image.shape[-2] / downsample_factor).astype(int)
    cols = np.ceil(image.shape[-1] / downsample_factor).astype(int)

    predicted_image = np.reshape(y_pred_numpy, [rows, cols])

    predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=1).astype(np.uint8)

    # Read ground truth mask
    ground_truth = annotation.squeeze()

    # Compute IoU
    iou_scores.append(compute_iou(np.array(predicted_image), np.array(ground_truth)))
    fpr_scores.append(compute_fpr(np.array(predicted_image)-1, np.array(ground_truth)-1))

    plt.figure()
    plt.imshow(image.squeeze(), cmap="gray")
    plt.imshow(predicted_image, alpha=0.5, cmap="viridis", interpolation="nearest")
    filename = f"output{i:04d}.png"
    plt.savefig(os.path.join(out_dir + "/images", filename), bbox_inches='tight')

print('the IOU score is:', np.mean(np.array(iou_scores)), 'the FPR score is:', np.mean(np.array(fpr_scores)))

scores = {
    "score_type": ["IOU", "FPR"],
    "values": [np.mean(np.array(iou_scores)), np.mean(np.array(fpr_scores))]
}

df = pd.DataFrame(scores)
df.to_csv(out_dir + '/score.csv', index=False)

