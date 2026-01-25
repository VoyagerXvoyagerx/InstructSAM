import pycocotools.mask as mask_util
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


TLM_COLOR_DICT = {'no labels': (0.0, 0.0, 0.0),
                  'agricultural areas': (0.0, 0.7843137254901961, 0.3254901960784314),
                  'vineyards': (0.36470588235294116, 0.09411764705882353, 0.5333333333333333),
                  'rocks': (0.3764705882352941, 0.49019607843137253, 0.5450980392156862),
                  'roads': (0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
                  'buildings': (0.8980392156862745, 0.2235294117647059, 0.20784313725490197),
                  'bush forest': (0.5803921568627451, 0.8784313725490196, 0.23921568627450981),
                  'forest': (0.10588235294117647, 0.3686274509803922, 0.12549019607843137),
                  'open forest': (0.4980392156862745, 0.42745098039215684, 0.0),
                  'river': (0.09411764705882353, 1.0, 1.0),
                  'lake': (0.0, 0.7372549019607844, 0.8313725490196079),
                  'wetlands': (0.14901960784313725, 0.5450980392156862, 0.5058823529411764),
                  'railways': (0.3803921568627451, 0.3803921568627451, 0.3803921568627451),
                  'rocks with grass': (1.0, 0.8784313725490196, 0.5098039215686274)}

def extract_results_from_coco(coco_preds, coco_ann, img_name, score_threshold=0):
    """
    Extract detection results for a specific image from standard COCO format predictions.

    Args:
        coco_preds (list): List of COCO format prediction dictionaries.
        coco_ann (dict): COCO format annotation dictionary containing image information.
        img_name (str): Name of the target image file.

    Returns:
        tuple: A tuple containing:
            - boxes (list): List of bounding boxes in format [x, y, width, height].
            - labels (list): List of class labels as strings.
            - scores (list): List of confidence scores.
            - segmentations (list): List of segmentation masks (either polygon or RLE format), or None if not available.
    """
    # Create mapping from image filename to image id
    image_name_to_id = {img_info['file_name']: img_info['id']
                        for img_info in coco_ann['images']}

    # Create mapping from category id to category name
    category_id_to_name = {cat['id']: cat['name']
                           for cat in coco_ann['categories']}

    # If coco_preds is None, return ground truth annotations
    if coco_preds is None:
        # Check if image exists
        if img_name not in image_name_to_id:
            return [], [], [], []
        image_id = image_name_to_id[img_name]
        # Gather GT annotations for this image
        gt_anns = [ann for ann in coco_ann['annotations']
                   if ann['image_id'] == image_id]
        # Extract boxes, labels, default scores, and segmentations
        boxes = [ann['bbox'] for ann in gt_anns]
        labels = [category_id_to_name.get(
            ann['category_id'], f"unknown_{ann['category_id']}") for ann in gt_anns]
        scores = [1.0] * len(gt_anns)
        segmentations = [ann.get('segmentation', None) for ann in gt_anns]
        # Normalize segmentations if all None
        if all(seg is None for seg in segmentations):
            segmentations = None
        return boxes, labels, scores, segmentations

    # Get the image id for the given image name
    if img_name not in image_name_to_id:
        return [], [], [], []  # Return empty lists if image not found

    image_id = image_name_to_id[img_name]

    # Filter predictions for the target image
    image_preds = [pred for pred in coco_preds if pred['image_id'] == image_id]

    # Extract boxes, labels, scores, and segmentations from the filtered predictions
    boxes = []
    labels = []
    scores = []
    segmentations = []

    for pred in image_preds:
        if 'score' in pred:
            if pred['score'] > score_threshold:
                scores.append(pred['score'])
            else:
                continue
        else:
            scores.append(1.0)  # Default score if not provided
        # Extract bounding box
        if 'bbox' in pred:
            boxes.append(pred['bbox'])

        # Extract category name from category id
        if 'category_id' in pred and 'label' not in pred:
            category_id = pred['category_id']
            labels.append(category_id_to_name.get(
                category_id, f"unknown_{category_id}"))
        elif 'label' in pred:
            labels.append(pred['label'])
        # Extract segmentation if available
        if 'segmentation' in pred:
            segmentations.append(pred['segmentation'])
        else:
            segmentations.append(None)

    # If no segmentations are found, return None instead of a list of Nones
    if all(seg is None for seg in segmentations):
        segmentations = None

    return boxes, labels, scores, segmentations


def visualize_prediction(image, bboxes=None, labels=None, segmentations=None, scores=None, color_dict=None,
                         dpi=100, show_gt=False, ann_data=None, img_path=None, alpha_mask=0.3, title='Predictions',
                         save_path=None, draw_bbox=True):
    """
    Visualize object detection predictions and optional Ground Truth.

    Parameters:
        image (numpy.ndarray): Input image.
        bboxes (list): COCO format bounding boxes, each as [x, y, w, h].
        labels (list): Class labels for each bounding box, as strings.
        segmentations (list): COCO format segmentation masks, either as polygon coordinates or RLE format.
        scores (list): Confidence scores for each bounding box (optional).
        dpi (int): Image resolution.
        show_gt (bool): Whether to show Ground Truth alongside predictions.
        ann_data (dict): COCO format annotation data (required if show_gt=True).
        img_path (str): Image path, used to extract annotation information.
        title (str): Image title.
    """
    # Create figure layout based on whether GT should be displayed
    if show_gt and (bboxes is not None or segmentations is not None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
        ax1.imshow(image)
        ax2.imshow(image)
        if title is not None:
            ax1.set_title(title)
            ax2.set_title("Ground Truth")
    else:
        plt.figure(figsize=(6, 6), dpi=dpi)
        ax1 = plt.gca()
        ax1.imshow(image)
        if title is not None:
            ax1.set_title(title)

    # Helper function to visualize RLE format segmentation
    def visualize_rle_mask(rle_data, ax, color):
        """
        Visualize a Run-Length Encoding (RLE) segmentation mask.

        Parameters:
            rle_data (dict): RLE segmentation data with 'size' and 'counts' fields.
            ax (matplotlib.axes.Axes): Matplotlib axes to draw on.
            color (tuple): RGB color for the mask.
        """
        # Decode RLE to binary mask
        if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
            binary_mask = mask_util.decode(rle_data)
            # Convert to RGBA with transparency
            mask_rgba = np.zeros(
                (binary_mask.shape[0], binary_mask.shape[1], 4))
            mask_rgba[binary_mask > 0] = (*color, alpha_mask)  # RGB + alpha
            ax.imshow(mask_rgba, interpolation='nearest')

    # Process prediction results
    if bboxes is not None and labels is not None:
        color_map = {}  # Store color for each class if color_dict is None
        for i in range(len(bboxes)):
            label = labels[i]
            if color_dict is not None:
                color = color_dict.get(label, np.random.rand(3))
            else:
                if label not in color_map:
                    # Generate random color for each class
                    color_map[label] = np.random.rand(3)
                color = color_map[label]
            # print("color", color)
            x, y, w, h = bboxes[i]
            
            if draw_bbox:
            # Draw bounding box
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
                ax1.add_patch(rect)
            else:
                x, y = x + w / 2, y + h / 2  # Center point for label

            # Display label and score
            if scores is not None:
                score = scores[i]
                text = f"{label} {score:.2f}"
            else:
                text = f"{label}"
            ax1.text(x, y, text, color='white', fontsize=10,
                     bbox=dict(facecolor=color, alpha=0.5, edgecolor='none'))

            # Draw segmentation
            if segmentations is not None and i < len(segmentations):
                seg = segmentations[i]

                # Handle RLE format
                if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                    visualize_rle_mask(seg, ax1, color)
                # Handle polygon format
                elif isinstance(seg, (list, tuple, np.ndarray)):
                    if len(seg) > 0:
                        if isinstance(seg[0], (list, tuple, np.ndarray)):
                            for polygon in seg:
                                points = np.array(polygon).reshape(-1, 2)
                                poly_patch = patches.Polygon(points, closed=True, fill=True,
                                                             color=color, alpha=alpha_mask, edgecolor=None)
                                ax1.add_patch(poly_patch)

    # Process Ground Truth
    if show_gt or (bboxes is None and segmentations is None and ann_data is not None and img_path is not None):
        if ann_data is None or img_path is None:
            raise ValueError("ann_data and img_path are required to show GT")

        # Find matching image_id
        image_id = next((img_info['id'] for img_info in ann_data['images']
                        if img_info['file_name'] == os.path.basename(img_path)), None)

        if image_id is None:
            raise ValueError(f"Image not found: {img_path}")

        # Get annotations for this image
        gt_annotations = [ann for ann in ann_data['annotations']
                          if ann['image_id'] == image_id]

        # Draw each GT annotation
        color_map = {}  # Store color for each class if color_dict is None
        for ann in gt_annotations:
            category_id = ann['category_id']
            label = next((cat['name'] for cat in ann_data['categories']
                         if cat['id'] == category_id), 'unknown')

            if color_dict is not None:
                color = color_dict.get(label, np.random.rand(3))
            else:
                if label not in color_map:
                    # Generate random color for each class
                    color_map[label] = np.random.rand(3)
                color = color_map[label]

            # Determine which axis to draw on
            target_ax = ax2 if show_gt and (
                bboxes is not None or segmentations is not None) else ax1

            # Draw segmentation (if available)
            if 'segmentation' in ann:
                seg = ann['segmentation']

                # Handle RLE format segmentation
                if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                    visualize_rle_mask(seg, target_ax, color)
                # Handle polygon format segmentation
                elif isinstance(seg, list):
                    for polygon in seg:
                        if isinstance(polygon, list):
                            points = np.array(polygon).reshape(-1, 2)
                            poly_patch = patches.Polygon(points, closed=True, fill=True,
                                                         color=color, alpha=alpha_mask, edgecolor=None)
                            target_ax.add_patch(poly_patch)

            # Draw GT bounding box
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            target_ax.add_patch(rect)

            # Draw label
            text = f"{label}"

            target_ax.text(x, y, text, color='white', fontsize=10,
                           bbox=dict(facecolor=color, alpha=0.5, edgecolor='none'))

    # Hide axes and adjust layout
    if show_gt and (bboxes is not None or segmentations is not None):
        ax1.axis('off')
        ax2.axis('off')
    else:
        ax1.axis('off')
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def show_sam_masks(image, anns, alpha=0.35):
    """
    Display the segmentation masks from SAM (Segment Anything Model) on the input image.
    Supports both binary mask format and RLE (Run Length Encoding) format.

    Args:
        image (numpy.ndarray): Input image as a numpy array (H, W, 3)
        anns (list): List of annotation dictionaries, each containing a 'segmentation' field
                    that can be either a binary mask or an RLE encoded mask
        alpha (float): Transparency level for the masks (0.0 to 1.0). Default is 0.35.
    """
    import pycocotools.mask as mask_util

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    if len(anns) == 0:
        plt.axis('off')
        plt.show()
        return

    # Sort masks by area for better visualization (larger masks behind smaller ones)
    if 'area' in anns[0]:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    else:
        sorted_anns = anns

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create an empty RGBA image with transparent background
    h, w = image.shape[:2]
    mask_overlay = np.zeros((h, w, 4))

    for ann in sorted_anns:
        # Get segmentation mask, handle both binary mask and RLE format
        if 'segmentation' in ann:
            seg = ann['segmentation']

            # Check if the segmentation is in RLE format
            if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                # Decode RLE to binary mask
                binary_mask = mask_util.decode(seg)
                mask = binary_mask.astype(bool)
            else:
                # Direct binary mask
                mask = seg

            # Generate a random color for this mask
            color_mask = np.concatenate([np.random.random(3), [alpha]])

            # Apply the mask color to the overlay
            mask_overlay[mask] = color_mask

    # Show the mask overlay on the image
    ax.imshow(mask_overlay)
    plt.axis('off')
    plt.show()
