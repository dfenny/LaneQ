import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ..architectures import get_model
from .preprocessing import apply_img_preprocessing
from .common import generate_connected_components, expand_bbox, box_coco_to_corner


def load_saved_model(model_name, saved_weight_path, **kwargs):
    """
    Initialize a model architecture and load pretrained weights.

    Parameters
    ----------
    model_name : str
        Identifier for the model architecture, as accepted by `get_model`.
    saved_weight_path : str
        Filesystem path to the saved weights file.
    **kwargs
        Additional keyword arguments forwarded to `get_model`.

    Returns
    -------
    torch.nn.Module
        The model instance with weights loaded from `saved_weight_path`.
    """
    model = get_model(model_name, **kwargs)                                     # initialize model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model.load_state_dict((torch.load(saved_weight_path, map_location=DEVICE, weights_only=True)))   # load weights
    return model


def generate_saliency_map(model, image, target_class=None, device="cpu"):
    """
    Generate the saliency map for a given image using the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to compute the saliency map for.
    image : PIL.Image or ndarray
        The input image, to be converted to a tensor of shape (C, H, W).
    target_class : int or None
        The class index to compute the saliency map for. If None, uses the
        class with the highest predicted score.
    device : str or torch.device
        Device on which to perform computation (e.g., "cpu" or "cuda").

    Returns
    -------
    numpy.ndarray
        Saliency map array of shape (H, W), with values normalized to [0, 1].
    """
    # Define image transformation
    img_transform = transforms.Compose([transforms.ToTensor()])

    # Apply transformation and move to device
    input_tensor = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension
    input_tensor.requires_grad_(True)  # Enable gradient computation for the input

    # Ensure the model is in evaluation mode
    model.eval()
    output = model(input_tensor)  # Forward pass

    # If no target class is specified, use the class with the highest output score
    if target_class is None:
        target_class = output.argmax(dim=1).item()
        print("Predicted Class", target_class)

    model.zero_grad()   # Zero all existing gradients

    # Compute the gradient of the output with respect to the input image for the target class
    output[0, target_class].backward()

    # Get the absolute value of the gradients
    saliency_map, _ = torch.max(input_tensor.grad.data.abs(), dim=1)

    # Normalize the saliency map to [0, 1]
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()
    saliency_map = np.maximum(saliency_map, 0)
    saliency_map = saliency_map / (saliency_map.max() + 1e-6)

    return saliency_map


def pred_segmentation_mask(model, test_img, img_transform=None, add_batch_dim=False,
                           pos_threshold=0.5, device="cpu"):
    """
    Perform segmentation inference on a single image and return pixel-wise labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.
    test_img : PIL.Image or ndarray
        Input image to segment.
    img_transform : callable, optional
        Preprocessing transform to apply to the image. If None, assumes
        `apply_img_preprocessing` handles necessary steps.
    add_batch_dim : bool, default=False
        If True, adds a batch dimension before inference.
    pos_threshold : float, default=0.5
        Threshold for converting sigmoid outputs to binary labels when
        performing binary segmentation (single-channel output).
    device : str or torch.device, default="cpu"
        Device for model inference.

    Returns
    -------
    numpy.ndarray
        Predicted label map(s). Shape is (H, W) for a single image or
        (1, H, W) if `add_batch_dim=True`.
    """
    model = model.to(device)    # ensure model is on same device as test data

    # apply image transformations
    test_batch = apply_img_preprocessing(test_img, transform=img_transform)
    if add_batch_dim:
        test_batch = test_batch.unsqueeze(0)       # (b=1, 3, h, w)

    # Perform inference
    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        logits = model(test_batch)            # (b, c, h, w)

        if logits.shape[1] == 1:              # if only 1 class  binary segmentation
            # Apply sigmoid to logits to get probabilities, then threshold to get binary class labels
            pred = torch.sigmoid(logits)                  # Sigmoid for binary classification      # (b, c, h, w)
            pred_labels = (pred > pos_threshold).float()  # Convert to 0 or 1 based on threshold    # (b, 1, h, w)
            pred_labels = pred_labels.squeeze(dim=1)      # (b, h, w)

        # multi-class segmentation
        else:
            prob = F.softmax(logits, dim=1)          # convert to probs   (b, c, h, w)
            pred_labels = torch.argmax(prob, dim=1)  # convert to labels  (b, h, w)

    return pred_labels.cpu().numpy()


def pred_degradation_value(model, test_img, img_transform=None, add_batch_dim=False,
                           device="cpu", precision=4):
    """
    Predict a continuous degradation value from an input image.

    Parameters
    ----------
    model : torch.nn.Module
        Trained regression model.
    test_img : PIL.Image or ndarray
        Input image for prediction.
    img_transform : callable, optional
        Preprocessing transform to apply to the image.
    add_batch_dim : bool, default=False
        If True, adds a batch dimension before inference.
    device : str or torch.device, default="cpu"
        Device for model inference.
    precision : int, default=4
        Number of decimal places to round the output value.

    Returns
    -------
    float
        Predicted degradation value, rounded to `precision` decimals.
    """
    model = model.to(device)   # ensure model is on same device as test data

    # apply image transformations
    test_batch = apply_img_preprocessing(test_img, transform=img_transform)
    if add_batch_dim:
        test_batch = test_batch.unsqueeze(0)       # (b, 3, h, w)

    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        output = model(test_batch)
        pred_value = output.squeeze().cpu().item()

    return round(pred_value, precision)


def pred_degradation_category(model, test_img, img_transform=None,
                              add_batch_dim=False, device="cpu"):
    """
    Predict a discrete degradation category index from an input image.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    test_img : PIL.Image or ndarray
        Input image for prediction.
    img_transform : callable, optional
        Preprocessing transform to apply to the image.
    add_batch_dim : bool, default=False
        If True, adds a batch dimension before inference.
    device : str or torch.device, default="cpu"
        Device for model inference.

    Returns
    -------
    int
        Predicted class index for the degradation category.
    """
    model = model.to(device)

    test_batch = apply_img_preprocessing(test_img, transform=img_transform)
    if add_batch_dim:
        test_batch = test_batch.unsqueeze(0)       # (b, 3, h, w)

    model.eval()
    with torch.no_grad():
        test_batch = test_batch.to(device)
        output = model(test_batch)
        pred_value = output.argmax().item()

    return pred_value


def generate_individual_segments_n_annotations(img, mask, annot_prefix=None):
    """
    Extract individual connected components from a mask and prepare annotations.

    Parameters
    ----------
    img : ndarray
        Original image array of shape (H, W, C).
    mask : ndarray
        2D binary or label mask of shape (H, W).
    annot_prefix : str, optional
        Prefix to prepend to each annotation's 'id' field. Defaults to None.

    Returns
    -------
    dict
        Dictionary with keys:
          - 'image': str identifier for the image (prefix or '_image')
          - 'annotations': list of dicts, each containing:
              * 'id': unique segment identifier
              * 'bounding_box': original COCO-format [x, y, w, h]
              * 'degradation': placeholder (-1)
              * 'segment_crop': cropped image array for that segment
    """
    # generate connected components
    num_labels, label_mask, bboxes = generate_connected_components(mask, connectivity=8)

    # generate crops of each individual segments & its annotation (skip label 0, which is the background)
    annot_results = []
    for idx in range(1, num_labels):
        coco_bbox = bboxes[idx].tolist()  # get coco format bbox

        # expand bbox to add additional context
        expanded_coco_bbox = expand_bbox(coco_bbox=coco_bbox, image_width=img.shape[1], image_height=img.shape[0],
                                         padding=10)
        xmin, ymin, xmax, ymax = box_coco_to_corner(
            expanded_coco_bbox)  # convert to corner format required for cropping segments

        orig_mask = (label_mask == idx).astype(np.uint8)   # get only the current segment in original mask

        # # dilate the mask to include surrounding road region near lane marking
        dilate_kernel = 16
        kernel = np.ones((dilate_kernel, dilate_kernel))
        dilated_mask = cv2.dilate(orig_mask, kernel, iterations=1)

        # # use this dilated region to get neighboring road color info
        segment = cv2.bitwise_and(img, img, mask=dilated_mask)
        segment = segment[ymin:ymax + 1, xmin:xmax + 1].copy()  # get specific segment crop

        # replaced the masked out pixels (black colored) in the segment with a color that would rarely
        # appear on raod to help the model avoid confusion with dark colors at night
        target_color = [0, 0, 0]
        mask = np.all(segment == target_color, axis=-1)  # Find pixels that match the target color
        segment[mask] = [84, 245, 66]  # bright green color

        prefix = "" if annot_prefix is None else f"{annot_prefix}_"
        segment_id = f"{prefix}object_{idx}"

        # store annotations
        segment_n_annotation = {
            'id': segment_id,
            'bounding_box': coco_bbox,
            'degradation': -1,
            "segment_crop": segment.copy()
        }
        annot_results.append(segment_n_annotation)

    filename = "_image" if annot_prefix is None else f"{annot_prefix}"
    annotations_dict = {
        'image': filename,
        'annotations': annot_results
    }

    return annotations_dict