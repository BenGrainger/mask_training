import numpy as np
import skimage
import cudf 
import torch   

class CustomParam:
    """Object storing relevant information regarding the processing,
    e.g. the window size (padding), the analyzed data, the type of segmentation used.

    Parameters saved in a Param object:
    ----------------
        classifier: str
            path to the classifier model
        multi_channel_img: bool = None
            if the image dimensions allow, use multichannel NOTE: Needs overthinking
        normalize: int
            normalization mode
            1: no normalization, 2: normalize stack, 3: normalize each image
        image_downsample: int
            factor for downscaling the image right after input
            (predicted classes are upsampled accordingly for output)
        tile_annotations: bool
            if True, extract only features of bounding boxes around annotated areas
        tile_image: bool
            if True, extract features in tiles (for large images)
        fe_name: str
            name of the feature extractor model
        fe_layers: list[str]
            list of layers (names) to extract features from
        fe_padding : int
            padding for the feature extractor NOTE: Needs overthinking
        fe_scalings: list[int]
            list of scaling factors for the feature extractor, creating a pyramid of features
            (features are upscaled accordingly before input to classifier)
        fe_order: int
            interpolation order used for the upscaling of features for the pyramid
        fe_use_min_features: bool
            if True, use the minimum number of features among all layers
        fe_use_cuda: bool
            whether to use cuda (GPU) for feature extraction
        clf_iterations: int
            number of iterations for the classifier
        clf_learning_rate: float
            learning rate for the classifier
        clf_depth: int = None
            depth of the classifier
    """

    classifier: str = None
    # Image processing parameters
    multi_channel_img: bool = None
    normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image
    # Acceleration parameters
    image_downsample: int = None
    tile_annotations: bool = False
    tile_image: bool = False
    # Feature Extractor parameters
    fe_name: str = None
    fe_layers: list[str] = None
    fe_padding : int = 0
    fe_scalings: list[int] = None
    fe_order: int = None
    fe_use_min_features: bool = None
    fe_use_cuda: bool = None
    # Classifier parameters
    clf_iterations: int = None
    clf_learning_rate: float = None
    clf_depth: int = None


    def __post_init__(self):
        self.fe_scalings = [1, 2]


    def convert_path(self, dict, path):
        """Convert a path to a str.

        Parameters
        ----------
        dict : dict
            dictionary containing the path.
        path : str
            path to convert.

        Returns
        -------
        dict: dict
            dict with converted path.
        """

        if dict[path] is not None:
            if not isinstance(dict[path], str):
                dict[path] = dict[path].as_posix()
        
        return dict
    
def preprocess_image(self, image):
    '''Normalizes input image to image net stats, return to 1x3xHxW tensor.
    Expects image to be 3xHxW'''

    assert len(image.shape) == 3
    assert image.shape[0] == 3

    # for uint8 or uint16 images, get divide by max value
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image = image / 255
    elif image.dtype == np.uint16:
        image = image.astype(np.float32)
        image = image / 65535
    # else just min max normalize to 0-1.
    else:
        image = image.astype(np.float32)
        divisor = np.max(image) - np.min(image)
        if divisor == 0:
            divisor = 1e-6
        image = (image - np.min(image)) / divisor

    # normalize to imagenet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean[:, None, None]) / std[:, None, None]

    #       # make sure image is divisible by patch size
    h, w = image.shape[-2:]
    new_h = h - (h % self.patchsize)
    new_w = w - (w % self.patchsize)
    if new_h != h or new_w != w:
        image = image[:, :new_h, :new_w]

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    # convert to tensor
    image_tensor = torch.tensor(image, dtype=torch.float32,device=self.device)
    return image_tensor

def extract_features_rgb(image, model, patchsize=14, use_cuda=True):
    '''Extract features from image, return features as np.array with dimensions  H x W x nfeatures.
    Input image has to be multiple of patch size'''
    assert image.shape[-2] % patchsize == 0
    assert image.shape[-1] % patchsize == 0
    assert image.shape[0] == 3

    image_tensor = preprocess_image(image)
    with torch.no_grad():
        features_dict = model.forward_features(image_tensor)
    features = features_dict['x_norm_patchtokens']
    if use_cuda:
        features = features.cpu()
    features = features.numpy()[0]
    features_shape = (int(image.shape[-2] / patchsize), int(image.shape[-1] / patchsize), features.shape[-1])
    features = np.reshape(features, features_shape)

    assert features.shape[0] == image.shape[-2] / patchsize
    assert features.shape[1] == image.shape[-1] / patchsize
    return features

def extract_features(image, model):
    '''Helper function to extract features from image with arbitrary number of color channels, 
    return features as np.array with dimensions  H x W x nfeatures'''
    assert len(image.shape) == 3
    if image.shape[0] == 3:
        features = extract_features_rgb(image, model)
    else:
        features = []
        for channel_nb in range(image.shape[0]):
            channel = np.expand_dims(image[channel_nb], axis=0)
            channel = np.repeat(channel, 3, axis=0)
            features_rgb = extract_features_rgb(channel, model)
            features.append(features_rgb)
        features = np.concatenate(features, axis=-1)
    return features

def get_features(image, model, return_patches=False, patchsize=14):
    '''Given an CxWxH image, extract features.
    Returns features with dimensions nb_features x H x W'''

    #make sure image is divisible by patch size
    h, w = image.shape[-2:]
    new_h = (h // patchsize) * patchsize
    new_w = (w // patchsize) * patchsize
    h_pad_top = (h - new_h)//2
    w_pad_left = (w - new_w)//2
    h_pad_bottom = h - new_h - h_pad_top
    w_pad_right = w - new_w - w_pad_left

    if h_pad_top > 0 or w_pad_left > 0 or h_pad_bottom > 0 or w_pad_right > 0:
        image = image[:, h_pad_top:-h_pad_bottom if h_pad_bottom != 0 else None, 
                            w_pad_left:-w_pad_right if w_pad_right != 0 else None]


    features = extract_features(image, model) #[H, W, nfeatures]

    if not return_patches:
        #upsample features to original size
        features = np.repeat(features, patchsize, axis=0)
        features = np.repeat(features, patchsize, axis=1)

        #replace with padding where there are no annotations
        pad_width = ((h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right), (0,0))

        features = np.pad(features, pad_width=pad_width, mode= 'edge')
        features = np.moveaxis(features, -1, 0) #[nb_features, H, W]

        assert features.shape[1] == h and features.shape[2] == w
        return features
    
    else:
        #return patches
        features = np.moveaxis(features, -1, 0) #[nb_features, H, W]
        assert features.shape[1] == (h // patchsize) and features.shape[2] == (w // patchsize)
        return features

def get_features_scaled(model, image, param, return_patches = False):
        """
        Overwrite the get_features_scaled function, as we don't want to extract features at different scales for DINO.

        Parameters
        ----------
        image: 2d array
            image to segment
        order: int
            interpolation order for low scale resizing
        image_downsample: int, optional
            downsample image by this factor before extracting features, by default 1

        Returns
        -------
        features: [nb_features x width x height]
            return extracted features

        """

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if param.image_downsample > 1:
            image = image[:, ::param.image_downsample, ::param.image_downsample]
        
        features = get_features(image, model, order=param.fe_order, return_patches= return_patches) #features have shape [nb_features, width, height]
        nb_features = features.shape[0]

        if not return_patches:
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=param.fe_order)            
        return features

def get_features_current_layers(image, model, param: CustomParam):
    """Extract multiscale features from an image (without annotations) for use with cuML RandomForest.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D Image to extract features from.
    model : feature extraction model
        Model to extract features from the image.
    param : Param
        Parameters for feature extraction, such as scalings and padding.

    Returns
    -------
    features : cudf.DataFrame
        Extracted features (rows are pixels, columns are features).
    """
    if model is None:
        raise ValueError('Model must be provided')

    # Ensure image is 2D or 3D
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be either 2D or 3D")

    all_values = []

    # Find maximal padding necessary
    padding = param.fe_padding * np.max(param.fe_scalings)

    # If image is 3D, get slices along the first dimension (e.g., time or depth)
    if image.ndim == 3:
        non_empty = np.unique(np.where(image > 0)[0])  # Assuming the image contains non-zero data
        if len(non_empty) == 0:
            raise Warning('No valid pixels found in the image.')
    else:
        non_empty = [0]  # For 2D images, process the whole image

    # Iterating over the non-empty slices (if it's a 3D image)
    for t in non_empty:
        if image.ndim == 3:
            current_image = np.pad(image[t], padding, mode='reflect')  # Apply padding to each slice
        else:
            current_image = np.pad(image, padding, mode='reflect')

        # Extract features using the provided model
        extracted_features = get_features_scaled(model, image=current_image, param=param)

        # Apply downsampling if needed
        if param.image_downsample > 1:
            current_image = current_image[::param.image_downsample, ::param.image_downsample]

        # Flatten the extracted features into a 2D array (pixels, features)
        extracted_features = np.moveaxis(extracted_features, 0, -1)  # Move [features] to the last axis
        flattened_features = extracted_features.reshape(-1, extracted_features.shape[-1])  # Flatten to (pixels, features)

        all_values.append(flattened_features)

    # Concatenate features from different slices or images
    all_values = np.concatenate(all_values, axis=0)

    # Convert the features to a cuDF DataFrame for cuML processing
    features = cudf.DataFrame.from_records(all_values)  # Each row is a pixel, each column is a feature

    return features


    
def predict_image_with_rf_gpu(input_image, classifier, param):
    """
    Predicts a segmentation map using a GPU-based random forest classifier.

    Args:
        input_image (torch.Tensor): The input image tensor to predict on.
        classifier (cuml.ensemble.RandomForestClassifier): The trained random forest model.
        param: Additional parameters for the model (e.g., pre-trained weights).

    Returns:
        predicted_image (torch.Tensor): The predicted segmentation mask or classification map.
    """
    # Ensure the input image is on the correct device (GPU)
    input_image = input_image.cuda()
    # 1. Extract features using a suitable feature extraction method
    features = get_features_current_layers(
        input_image,  # Your image tensor
        param=param    # Any parameters for feature extraction
    )
    
    # Assuming `features` is a NumPy array (or convert it to a numpy array if it's not)
    features_array = features.cpu().numpy()  # Convert to NumPy array for cuDF processing
    
    # 2. Convert features to a cuDF DataFrame for cuML processing
    X_gpu = cudf.DataFrame.from_records(features_array)

    # 3. Use the trained classifier to make predictions
    y_pred_gpu = classifier.predict(X_gpu)

    # 4. Convert the predictions from cuDF back to a NumPy array or a PyTorch tensor
    predicted_image = y_pred_gpu.to_array()  # Convert cuDF Series to NumPy array
    
    return predicted_image