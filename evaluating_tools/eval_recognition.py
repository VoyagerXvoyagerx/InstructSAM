import os
import json
from instruct_sam.matching import init_clip_model
from instruct_sam.metrics import get_cat_map, evaluate_ap_f1, get_all_counts, evaluate_mIoU, calculate_f1_scores
from instruct_sam.instruct_sam import InstructSAM
from instruct_sam.metrics import convert_coco_raw_to_coco_pred
import argparse
import yaml
import copy
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='The path to the predictions in coco format.')
    parser.add_argument('--dataset_name', type=str, default='dior_mini')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
    parser.add_argument('--setting', type=str, default='open_ended',
                        help='open_vocabulary, open_ended, open_subclass')
    parser.add_argument('--checkpoint_config', type=str,
                        default='checkpoints/config.yaml')
    parser.add_argument('--map_model', type=str, default='georsclip',
                        help=('The CLIP model for mapping predicted categories to dataset categories.'
                              'Work for open-ended and open-subclass settings.'))
    parser.add_argument('--extra_classes', type=str,
                        choices=['unseen_classes',
                                 'means_of_transport', 'sports_field'],
                        default=None, help='Specify the extra classes to calculate metrics for.')
    parser.add_argument('--eval_mIoU', type=bool, default=False)
    parser.add_argument('--score_sweeping', action='store_true', default=False,
                        help='Perform score sweeping to find the best threshold for conventional detectors.')
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()

    return args


def score_sweeping(coco_predictions, anns_coco, unseen=None):
    # Thresholds to evaluate
    thresholds = [round(x, 4) for x in list(np.arange(0.18, 0.28, 0.002))]
    # thresholds = [round(x, 2) for x in list(np.arange(0, 1, 0.02))]
    mf1_values = []
    # Evaluate mAP50 for each threshold
    coco_predictions_copy = copy.deepcopy(coco_predictions)
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        coco_predictions_filtered = [
            pred for pred in coco_predictions_copy if pred['score'] >= threshold]
        print(len(coco_predictions_filtered))
        _, mean_f1_score, mean_f1_unseen = calculate_f1_scores(
            anns_coco, coco_predictions_filtered, unseen=unseen)
        f1 = mean_f1_score if args.setting != 'open_subclass' else mean_f1_unseen
        mf1_values.append(f1)
        print(f'threshold: {threshold}, mF1: {f1}')

    # calculate the best threshold
    best_threshold = thresholds[np.argmax(mf1_values)]
    print(
        f'best threshold: {best_threshold}, mF1: {mf1_values[np.argmax(mf1_values)]}')
    return best_threshold


if __name__ == '__main__':
    args = parse_args()

    # dataset info
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[args.dataset_name]
    ann_path = dataset_info['ann_path']
    with open(ann_path, 'r') as f:
        anns_coco = json.load(f)
    extra_classes = args.extra_classes
    if extra_classes is not None:
        extra_classes = dataset_info[extra_classes]

    # predictions
    coco_preds_path = args.predictions
    with open(coco_preds_path, 'r') as f:
        coco_preds_raw = json.load(f)

    pred_filename_prefix = os.path.basename(coco_preds_path)[:-5]
    save = f'./results/{args.dataset_name}/{args.setting}/{pred_filename_prefix}'

    save_results_path_bbox = f'./results/{args.dataset_name}/{args.setting}/{pred_filename_prefix}_bbox.txt'
    save_results_path_segm = f'./results/{args.dataset_name}/{args.setting}/{pred_filename_prefix}_segm.txt'
    save_results_path_mIoU = f'./results/{args.dataset_name}/{args.setting}/{pred_filename_prefix}_mIoU.txt'
    print(
        f'Saving predictions to ./results/{args.dataset_name}/{args.setting}/')

    predictor = InstructSAM(anns_coco)

    if args.setting != 'open_vocabulary':
        # Load checkpoint config
        with open(args.checkpoint_config, 'r') as f:
            checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
        ckpt_path = checkpoint_config[args.map_model]
        print(
            f'Loading {args.map_model} checkpoint from {ckpt_path} to {args.device}.')
        model, tokenizer, preprocess = init_clip_model(
            args.map_model, args.device, ckpt_path)

        all_counts = get_all_counts(coco_preds_raw)
        pred2cat = get_cat_map(
            model, tokenizer, predictor.categories, all_counts, threshold=0.95)
        print(pred2cat)
        print(f"Total unique classes: {len(all_counts)}")
        print(f"Examples of counts: {list(all_counts.items())[:5]}")

        coco_predictions = convert_coco_raw_to_coco_pred(
            coco_preds_raw, pred2cat, anns_coco)
        print(
            f"Total {len(coco_preds_raw)} raw predictions, {len(coco_predictions)} predictions after mapping\n")

    else:
        coco_predictions = coco_preds_raw

    if args.score_sweeping:
        best_threshold = score_sweeping(coco_predictions, anns_coco, extra_classes)
        coco_predictions = [
            pred for pred in coco_predictions if pred['score'] >= best_threshold]

    for pred in coco_predictions:
        pred['score'] = 1.0

    if 'segmentation' in coco_predictions[0]:
        mask_eval = True
    else:
        mask_eval = False

    print('## Evaluating box AP and F1 ##')
    box_metrics = evaluate_ap_f1(anns_coco, coco_predictions, iouType='bbox',
                                 unseen=extra_classes, save_path=save_results_path_bbox, print_result=True)
    # print(
    #     f"mF1_all: {box_metrics['mF1']:.1f}, mF1_{args.extra_classes}: {box_metrics['mF1_unseen']:.1f}")

    if mask_eval:
        print('\n\n## Evaluating mask AP and F1 ##')
        mask_metrics = evaluate_ap_f1(anns_coco, coco_predictions, iouType='segm',
                                      unseen=extra_classes, save_path=save_results_path_segm, print_result=True)
        # print(
        #     f"mF1_all: {mask_metrics['mF1']:.1f}, mF1_{args.extra_classes}: {mask_metrics['mF1_unseen']:.1f}")

        if args.eval_mIoU:
            print('\n\n## Evaluating mIoU ##')
            evaluate_mIoU(anns_coco, coco_predictions,
                          save_path=save_results_path_mIoU)
