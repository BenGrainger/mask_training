import numpy as np
import os
import tifffile
from torch.utils.data import Dataset 
from sklearn.metrics import jaccard_score
import logging
import json
import datetime 
from pathlib import Path


class TiffDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_transform=None, annotation_transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing TIFF images.
            annotation_dir (str): Path to the folder containing corresponding annotation masks.
            threshold (float): Minimum percentage of nonzero pixels to keep an image.
            transform: Optional transformations (e.g., ToTensor).
        """
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform

        # Get sorted image files
        self.image_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.tif')])

        # Filter valid image-mask pairs
        self.image_files = [f for f in self.image_files if self.is_valid_pair(f)]

    def is_valid_pair(self, image_name):
        """Checks if both the image and annotation are valid."""
        annotation_path = os.path.join(self.annotation_dir, image_name)
        image_path = os.path.join(self.image_dir, image_name)

        # Ensure both image & annotation exist
        if not (os.path.isfile(annotation_path) and os.path.isfile(image_path)):
            return False  

        # Try to load the annotation
        try:
            annotation = tifffile.imread(annotation_path)

            # Check if annotation is completely 0 or 1
            unique_values = np.unique(annotation)
            if len(unique_values) == 1:  # Only one unique value (either all 0s or all 1s)
                return False  

        except Exception as e:
            print(f"Error reading {image_name}: {e}")
            return False  
        print(f"valid pair {image_name}: contains {unique_values}")
        return True  # Valid image-mask pair

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        image = tifffile.imread(image_path).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)  # Normalize to [0,1]

        # Load annotation
        annotation_path = os.path.join(self.annotation_dir, image_name)
        annotation = tifffile.imread(annotation_path).astype(np.uint8)
        annotation = annotation + 1  # Maintain convention

        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.annotation_transform:
            annotation = self.annotation_transform(annotation)

        return image, annotation  # Return image-mask pair

    

def compute_iou(predicted_mask, ground_truth):
    """Computes the Intersection over Union (IoU) score."""
    predicted_mask = predicted_mask.flatten()
    ground_truth = ground_truth.flatten()
    return jaccard_score(ground_truth, predicted_mask, average='binary')

def compute_fpr(predicted_mask, ground_truth):
    """Computes IoU, Precision, Recall, and False Positive Rate."""
    predicted_mask = predicted_mask.flatten()
    ground_truth = ground_truth.flatten()
    
    # False Positive Rate: FP / (FP + TN)
    fp = ((predicted_mask == 1) & (ground_truth == 0)).sum()
    tn = ((predicted_mask == 0) & (ground_truth == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr


class ScriptSetup():
    def __init__(self, config, logger=None, out_dir=None, root=None):

        self.config = config
        self.logger = logger
        self.out_dir = out_dir
        self.root = root
    
    def load_script(self):
        
        self.root = get_project_root()

        with open(self.config, 'r') as config_file:
            self.config = json.load(config_file)

        self.out_dir = self.config["output_dir"]

        date = datetime.datetime.now()
        date_string = date.strftime("%G%m%d")

        self.out_dir = os.path.join(self.out_dir, date_string)

        check_folder_exists(self.out_dir)

    def return_logger(self):

        logging.basicConfig(filename=self.out_dir+ "/" + "logging.log",
                            format='%(asctime)s %(message)s',
                            filemode='w',
                            level=logging.INFO)
        
        self.logger = logging.getLogger()
        return self.logger
    
    def return_config(self):
        return self.config
    
    def return_out_dir(self):
        return self.out_dir
    
    def return_root(self):
        return self.root
    

def get_project_root(root=None):
    if root == None:
        root = Path(__file__).parent

    root_split = os.path.split(root)

    if root_split[1] == "users":

        return root.replace("\\", "/")
    else:
        
        return get_project_root(root_split[0])

def check_folder_exists(directory):

    if not os.path.exists(directory):

        try:
            # Create the folder if it doesn't exist
            os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")

        except OSError as e:
            print(f"Error creating folder '{directory}': {e}")
    else:
        print(f"Folder '{directory}' already exists.")