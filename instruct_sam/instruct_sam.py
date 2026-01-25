import numpy as np
import torch
import open_clip
from PIL import Image
import os
import json
from instruct_sam.counting import encode_image_as_base64, predict, count_gt, extract_dict_from_string
from instruct_sam.visualize import show_sam_masks
from instruct_sam.matching import filter_boxes, match_boxes_and_counts
import cv2
from pycocotools import mask as mask_util
import copy


class InstructSAM:
    def __init__(self, annotations=None, img_dir=None, count_dir=None, rp_preds=None, device="cuda:0"):
        print("init from InstructSAM released")
        self.img_dir = img_dir
        self.count_dir = count_dir
        self.rp_preds = rp_preds
        self.device = device
        if annotations is not None:
            if isinstance(annotations, str):
                with open(annotations, 'r') as f:
                    self.anns = json.load(f)
            else:
                self.anns = annotations
            self.categories = [cat['name'] for cat in self.anns['categories']]
            self.category_name_to_id = {
                cat['name']: cat['id'] for cat in self.anns['categories']}
            self.img_name_to_id = {img['file_name']: img['id']
                                   for img in self.anns['images']}
        else:
            self.anns = None

    def set_image(self, img_path):
        self.img_path = img_path
        self.img_name = os.path.basename(img_path)
        self.img_jpeg = Image.open(self.img_path)
        self.img_array = np.array(self.img_jpeg.convert("RGB"))
        if self.anns is not None:
            self.gt_counts = count_gt(self.anns, self.img_name)
            self.img_id = self.img_name_to_id[self.img_name]

    def load_rps_and_cnts(self, thr=0.0):
        if self.count_dir is not None:
            img_suffix = os.path.splitext(self.img_name)[1]
            img_cnt_path = os.path.join(
                self.count_dir, self.img_name.replace(img_suffix, '.json'))
            try:
                with open(img_cnt_path, 'r') as f:
                    object_cnt = json.load(f)
                self.pred_counts = {k: v for k,
                                    v in object_cnt.items() if v != 0}
            except:
                print(
                    f"Error loading {img_cnt_path}, check if the file exists")
                self.pred_counts = {}

        if self.rp_preds is not None:
            pred = self.rp_preds[self.img_name]
            boxes, labels, scores = pred["bboxes"], pred["labels"], pred["scores"],
            segmentations = pred["segmentations"] if "segmentations" in pred else None
            areas = pred["areas"] if "areas" in pred else None
            # boxes, scores, labels, segmentations = filter_boxes(
            #     boxes, segmentations, scores, labels, thr)
            try:
                labels = [self.categories[int(label)] for label in labels]
            except:
                labels = ['RP'] * len(labels)
            self.bboxes = boxes
            self.segmentations = segmentations
            self.labels = labels
            self.scores = scores
            self.areas = areas

    def count_objects(self, prompt, gpt_model="gpt-4o-2024-11-20", json_output=False):
        base64_image = encode_image_as_base64(self.img_path)
        completion = predict(prompt, base64_image, top_p=1, temperature=0.01,
                             model=gpt_model, json_output=json_output)
        response = completion.choices[0].message.content
        pred_counts = extract_dict_from_string(response)
        self.response = response
        self.pred_counts = pred_counts

    @staticmethod
    def fill_holes(mask):
        """
        Fill holes in a binary mask.

        Args:
            mask (np.ndarray): Binary mask as a uint8 numpy array (values 0 or 255).

        Returns:
            np.ndarray: Binary mask with holes filled.

        The algorithm uses a flood fill from the border to detect the background and then
        inverts the flood filled image to obtain the holes, which are then filled.
        """
        # Copy the mask to avoid modifying the original
        im_floodfill = mask.copy()

        # Create a mask that is 2 pixels larger than the original mask.
        h, w = mask.shape[:2]
        floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from the top-left corner (assumed to be background)
        cv2.floodFill(im_floodfill, floodfill_mask, (0, 0), 255)

        # Invert floodfilled image to get the holes
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the original mask with the holes filled (logical OR)
        filled_mask = mask | im_floodfill_inv
        return filled_mask

    @staticmethod
    def boolean_mask_to_coco_polygons(boolean_mask, epsilon=1.0):
        """
        Convert a boolean binary mask to COCO polygon segmentation format.

        Args:
            boolean_mask (np.ndarray): 2D boolean numpy array representing the mask 
                                    (True for foreground, False for background).
            epsilon (float): Approximation accuracy parameter for cv2.approxPolyDP.
                            Typical values range from 1.0 to 5.0 depending on image scale.

        Returns:
            list: A list of polygons, where each polygon is represented as a list of 
                [x1, y1, x2, y2, ..., xN, yN]. If the mask contains multiple connected 
                components, each component will be represented as a separate polygon 
                after hole filling.

        The function first fills any holes in the mask, then extracts contours for each 
        connected component using cv2.findContours with RETR_EXTERNAL. For each contour, 
        cv2.approxPolyDP is applied to simplify the curve, and if the resulting approximation 
        has at least three points, it is added to the list of polygons.
        """
        # Convert boolean mask to uint8 (0, 255)
        mask_uint8 = (boolean_mask.astype(np.uint8)) * 255

        # Fill holes in the mask
        mask_filled = InstructSAM.fill_holes(mask_uint8)

        # Find contours on the filled mask.
        # Use RETR_EXTERNAL since holes are filled and we only need outer boundaries.
        contours, _ = cv2.findContours(
            mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            # Filter out small contours based on area to remove noise.
            if cv2.contourArea(contour) < 10:
                continue

            # Simplify contour using polygonal approximation.
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Skip contour if it does not form a valid polygon (must have at least 3 points).
            if approx.shape[0] < 3:
                continue

            # Flatten the coordinates to [x1, y1, x2, y2, ..., xN, yN]
            polygon = approx.flatten().tolist()
            polygons.append(polygon)

        return polygons

    @staticmethod
    def boolean_mask_to_rle(boolean_mask, fill_holes=True):
        """
        Convert a boolean binary mask to COCO RLE (Run Length Encoding) format.

        Args:
            boolean_mask (np.ndarray): 2D boolean numpy array representing the mask
                                    (True for foreground, False for background).
            fill_holes (bool): Whether to fill holes in the mask before encoding.
                            Defaults to True.

        Returns:
            dict: A dictionary containing RLE encoding with keys:
                - 'counts': RLE counts in binary format or UTF-8 string
                - 'size': list [height, width] representing the shape of the mask

        Using pycocotools.mask.encode to convert mask to RLE format
        """
        # Convert boolean mask to uint8 (0, 1)
        mask_uint8 = boolean_mask.astype(np.uint8)

        # Optionally fill holes in the mask
        if fill_holes:
            # Convert to 0-255 range for fill_holes function
            mask_255 = mask_uint8 * 255
            mask_255_filled = InstructSAM.fill_holes(mask_255)
            # Convert back to 0-1 range
            mask_uint8 = (mask_255_filled > 0).astype(np.uint8)

        # Make sure mask is in Fortran contiguous order for efficient processing
        mask_fortran = np.asfortranarray(mask_uint8)

        # Use pycocotools to encode the mask to RLE format
        rle = mask_util.encode(mask_fortran)

        # Convert binary 'counts' field to unicode string for JSON serialization if needed
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')

        return rle

    def segment_anything(self, mask_generator, max_masks=200, min_mask_region_area=0, rle=True):
        masks = mask_generator.generate(self.img_array)

        # Filter masks by area
        masks = [mask for mask in masks if mask['area'] > min_mask_region_area]

        # Filter masks outside the image
        img_width, img_height = self.img_jpeg.size
        len_mask_all = len(masks)
        masks = [mask for mask in masks if mask['bbox'][0]
                 < img_width and mask['bbox'][1] < img_height]
        if len(masks) < len_mask_all:
            print(
                f"Filtered {len_mask_all - len(masks)} masks outside the image")

        # Select max `max_masks` masks
        if len(masks) > max_masks:
            for mask in masks:
                mask['average_score'] = (
                    mask['predicted_iou'] + mask['stability_score']) / 2
            masks = sorted(
                masks, key=lambda x: x['average_score'], reverse=True)
            masks = masks[:max_masks]

        self.num_masks = len(masks)
        self.bboxes = [mask['bbox'] for mask in masks]

        if rle:
            self.segmentations = [self.boolean_mask_to_rle(
                mask['segmentation']) for mask in masks]
        else:
            self.segmentations = [self.boolean_mask_to_coco_polygons(
                mask['segmentation'], epsilon=1.0) for mask in masks]

        self.scores = [(mask['stability_score'] +
                        mask['predicted_iou'])/2 for mask in masks]
        self.labels = ['region_preposal'] * self.num_masks
        self.masks = masks

    def show_masks(self):
        show_sam_masks(self.img_array, self.masks)

    @staticmethod
    def enhance_clip_prompt(cat_list, prefix='an image of', single=True):
        """
        Generates a list of prompts for each category in the input list.
        Args:
            cat_list (list of str): A list of category names,
            e.g., ['people', 'skis'].
        Returns:
            list of str: A list of prompts,
            e.g., ['an image of a person', 'an image of skis'].
        """
        vowels = 'aeiou'
        prompts = []

        for category in cat_list:
            article = 'an' if category[0].lower() in vowels else 'a'
            if single:
                prompts.append(f'{prefix} {article} {category}')
            else:
                prompts.append(f'{prefix} {category}')

        return prompts

    def calculate_vocab_text_features(self, clip_model, tokenizer, prefix='a satellite image of', single=True):
        """calculate text feature cache of dataset categories

        Args:
            clip_model (torch.nn.Module): OpenCLIP model
            tokenizer (callable): OpenCLIP tokenizer
        """
        enhanced_prompt = self.enhance_clip_prompt(self.categories, prefix=prefix, single=single)
        print(f'Example prompt: {enhanced_prompt[0]}')
        TARGET_CATEGORY_tokens = tokenizer(enhanced_prompt).to(
            next(clip_model.parameters()).device)
        with torch.no_grad():
            if isinstance(clip_model, open_clip.model.CLIP):
                TARGET_FEATURES = clip_model.encode_text(
                    TARGET_CATEGORY_tokens)
            else:
                TARGET_FEATURES = clip_model.get_text_features(
                    TARGET_CATEGORY_tokens)
            TARGET_FEATURES /= TARGET_FEATURES.norm(dim=-1, keepdim=True)
        self.vocab_features = TARGET_FEATURES

    def calculate_pred_text_features(self, clip_model, tokenizer, use_vocab=False, prefix='a satellite image of', single=True):
        """calculate text features of predicted categories

        Args:
            clip_model (torch.nn.Module): OpenCLIP model
            tokenizer (callable): OpenCLIP tokenizer
            use_vocab (bool, optional): Whether to use vocabulary feature cache. Defaults to False.

        Returns:
            torch.Tensor: Text features of predicted categories
        """
        # if self has vocab features, use it
        if use_vocab:
            assert hasattr(
                self, 'vocab_features') == True, "Please calculate vocab features first"

            # remove predicted classes that are not in dataset categories
            pred_counts_copy = copy.deepcopy(self.pred_counts)
            for cls, _ in pred_counts_copy.items():
                if cls not in self.categories:
                    print(f"Category {cls} not in categories")
                    self.pred_counts.pop(cls)

            # remove predicted classes that have count 0
            gpt_predicted_classes = [
                cls for cls, count in self.pred_counts.items() if count > 0]

            if not gpt_predicted_classes:
                print(f"No categories predicted for {self.img_name}")
                self.pred_text_features = None
                return None
            self.pred_text_features = torch.stack(
                [self.vocab_features[self.categories.index(
                    cls)] for cls in gpt_predicted_classes]
            )
            return self.pred_text_features
        else:
            # categories = self.pred_counts.keys()
            gpt_predicted_classes = [
                cls for cls, count in self.pred_counts.items() if count > 0]
            if not gpt_predicted_classes:
                self.pred_text_features = None
                return None
            cat_prompt = self.enhance_clip_prompt(gpt_predicted_classes, prefix=prefix, single=single)
            text_tokens = tokenizer(cat_prompt).to(
                next(clip_model.parameters()).device)
            with torch.no_grad():
                if isinstance(clip_model, open_clip.model.CLIP):
                    text_features = clip_model.encode_text(text_tokens)
                else:
                    text_features = clip_model.get_text_features(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.pred_text_features = text_features
            return self.pred_text_features

    def match_boxes_and_labels(self,
                               clip_model, preprocess,
                               crop_scale=1.2, batch_size=200, min_crop_width=0,
                               show_similarities=False, zero_count_warning=True,
                               landcover=False, lambda_area=1.0, crop_mask=False):
        """
        Match predicted boxes with predicted counts

        Args:
            clip_model (torch.nn.Module): OpenCLIP model
            preprocess (torchvision.transforms): OpenCLIP preprocess
            crop_scale (float, optional): Crop scale. Defaults to 1.2.
            min_crop_width (int, optional): Minimum crop width. Defaults to 0.
            show_similarities (bool, optional): Show similarities. Defaults to False.

        Returns:
            list: Final boxes and labels (category names)
        """
        boxes_final, labels_final, segmentations_final, scores_final = match_boxes_and_counts(
            self.img_jpeg, self.bboxes, self.pred_counts, self.pred_text_features,
            clip_model, preprocess, self.segmentations, self.areas,
            crop_scale=crop_scale, batch_size=batch_size, min_crop_width=min_crop_width,
            show_similarities=show_similarities, zero_count_warning=zero_count_warning,
            landcover=landcover, lambda_area=lambda_area, crop_mask=crop_mask)

        self.boxes_final = boxes_final
        self.labels_final = labels_final
        self.segmentations_final = segmentations_final
        self.scores_final = scores_final
        return boxes_final, labels_final, segmentations_final, scores_final
