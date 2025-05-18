import json
import numpy as np
from collections import defaultdict

def convert_coco_to_result_structure(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Map image_id to file_name
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Create a dict to collect bboxes per image
    image_bboxes_map = {}  # 12 classes

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        class_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        box = [x1, y1, x2, y2]  # format to match predicted boxes
        if image_id not in list(image_bboxes_map.keys()):
            image_bboxes_map[image_id] = [[] for i in range(12)]
        try:
            image_bboxes_map[image_id][class_id].append(box)
        except:
            # print(image_id,class_id,image_bboxes_map[image_id])
            print("")

    results = []
    for image_id, bboxes_per_class in image_bboxes_map.items():
        # Convert lists to np arrays
        structured_bboxes = [np.array(b) if b else np.empty((0, 4)) for b in bboxes_per_class]
        filename = image_id_to_filename[image_id]
        structured_bboxes.append(f"images/val/{filename}")  # append filename for compatibility
        results.append(structured_bboxes)
    # print(results)
    return results


import numpy as np
import os

def iou(boxA, boxB):
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def calculate_precision_recall(results, ground_truths, iou_threshold=0.5, score_threshold=0.3):
    num_classes = 12

    # Initialize counters for each class
    tp_per_class = [0] * num_classes
    fp_per_class = [0] * num_classes
    fn_per_class = [0] * num_classes

    # Map from image filename (e.g., 'img.jpg') to GT
    gt_map = {os.path.basename(gt[-1]): gt for gt in ground_truths}

    for pred_entry in results:
        filename = os.path.basename(pred_entry[-1])

        if filename not in gt_map:
            # print(f"[Warning] Ground truth missing for: {filename}")
            continue

        gt_bboxes_per_class = gt_map[filename][:num_classes]
        pred_bboxes_per_class = pred_entry[:num_classes]

        for class_idx in range(num_classes):
            preds = pred_bboxes_per_class[class_idx]
            gts = gt_bboxes_per_class[class_idx]
            matched_gt = set()

            if isinstance(preds, np.ndarray) and preds.shape[0] > 0:
                for pred in preds:
                    if len(pred) < 5:
                        continue
                    score = pred[4]
                    if score < score_threshold:
                        continue

                    pred_box = pred[:4]
                    match_found = False

                    for i, gt_box in enumerate(gts):
                        if i in matched_gt:
                            continue
                        if iou(pred_box, gt_box) >= iou_threshold:
                            tp_per_class[class_idx] += 1
                            matched_gt.add(i)
                            match_found = True
                            break

                    if not match_found:
                        fp_per_class[class_idx] += 1

            if isinstance(gts, np.ndarray):
                fn_per_class[class_idx] += len(gts) - len(matched_gt)

    # Compute class-wise precision & recall
    print("\nClass-wise Precision & Recall:")
    total_tp = total_fp = total_fn = 0
    for class_idx in range(num_classes):
        tp = tp_per_class[class_idx]
        fp = fp_per_class[class_idx]
        fn = fn_per_class[class_idx]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Class {class_idx + 1:2d}: Precision = {precision:.4f}, Recall = {recall:.4f}, TP = {tp}, FP = {fp}, FN = {fn}")

    # Compute overall
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    print(f"\nOverall Precision: {overall_precision:.4f}")
    print(f"Overall Recall:    {overall_recall:.4f}")

    return overall_precision, overall_recall, tp_per_class, fp_per_class, fn_per_class



import pickle
gt_results = convert_coco_to_result_structure('.\\Data\\Det\\annotations\\test.json')
with open('.\\results\\bboxes5.pkl', 'rb') as f:
    pred_results = pickle.load(f)

calculate_precision_recall(pred_results, gt_results)  # use previous precision/recall function
