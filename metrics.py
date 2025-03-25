import numpy as np

def compute_iou(box1, box2):
    """
    computes and returns IoU between two bounding boxes
    
    :param box1: list of 4 elements [x1, y1, x2, y2] in order of top-left and bottom-right coordinates
    :param box2: list of 4 elements [x1, y1, x2, y2] in order of top-left and bottom-right coordinates
    :return: IoU value
    """
    
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Returns the number of True Positives, False Positives, and False Negatives between two sets of bounding boxes; predictions and ground truths
    
    :param pred_boxes: list of predicted bounding boxes in the form of list of 6 elements [x1, y1, x2, y2, class, confidence] in order of top-left and bottom-right coordinates
    :param gt_boxes: list of ground truth bounding boxes in the form of list of 5 elements [x1, y1, x2, y2, class] in order of top-left and bottom-right coordinates 
    :param iou_threshold: IoU threshold for a prediction to be considered a True Positive
    :return: tuple of True Positives, False Positives, and False Negatives
    """
    TP, FP, FN = 0, 0, len(gt_boxes)
    matched_gt = set()

    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)

    # Loop to match each prediction with GT
    for pred in pred_boxes:
        
        pred_box, pred_class, pred_conf = pred[:4], pred[4], pred[5]  # Extract bbox, class, confidence
        best_iou, best_gt_idx = 0, -1

        for j, gt in enumerate(gt_boxes):
            gt_box, gt_class = gt[:4], gt[4]

            iou = compute_iou(pred_box, gt_box)
            
            # If IoU is greater than the best IoU so far and the class is the same and the GT box is not matched
            # then update the best IoU and the best GT index
            if iou > best_iou and gt_class == pred_class and j not in matched_gt:
                best_iou, best_gt_idx = iou, j

        # If the best IoU is greater than the threshold, then it's a TP
        # Otherwise, it's a FP
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt and pred_conf >= confidence_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)  # Mark GT as matched
            
        elif pred_conf >= confidence_threshold and best_iou < iou_threshold:
            FP += 1  # No valid match found

    FN = len(gt_boxes) - len(matched_gt)  # Remaining unmatched GT boxes are FN
    return TP, FP, FN

def compute_precision(tp, fp):
    """
    Computes and returns the precision value
    
    :param tp: number of True Positives
    :param fp: number of False Positives
    :return: precision value
    """
    return tp / (tp + fp) if tp + fp != 0 else 0

def compute_recall(tp, fn):
    """
    Computes and returns the recall value
    
    :param tp: number of True Positives
    :param fn: number of False Negatives
    :return: recall value
    """
    return tp / (tp + fn) if tp + fn != 0 else 0

def compute_f1_score(precision, recall):
    """
    Computes and returns the F1 score
    
    :param precision: precision value
    :param recall: recall value
    :return: F1 score
    """
    return 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

def compute_average_precision(precisions, recalls):
    """
    Computes the Average Precision (AP) by numerical integration of Precision-Recall curve.
    
    :param precisions: List of precision values at different thresholds
    :param recalls: List of recall values at different thresholds
    :return: AP value
    """
    
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    ap = 0
    for i in range(1, len(sorted_recalls)):
        ap += (sorted_recalls[i] - sorted_recalls[i - 1]) * sorted_precisions[i]
    
    return ap

# image + prev code as prompt
def compute_ap_at_gamma(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Computes Average Precision at IoU threshold γ (AP@γ).
    
    :param pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2, class, confidence]
    :param gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2, class]
    :param iou_threshold: IoU threshold (γ)
    :return: AP@γ value
    """
    precisions, recalls = [], []
    for confidence_threshold in np.linspace(0, 1, 101):
        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold, confidence_threshold)
        precision = compute_precision(tp, fp)
        recall = compute_recall(tp, fn)
        precisions.append(precision)
        recalls.append(recall)
        
    return compute_average_precision(precisions, recalls)

def mean_average_precision(pred_boxes, gt_boxes, iou_thresholds):
    """
    Computes the Mean Average Precision (mAP) at different IoU thresholds.
    
    :param pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2, class, confidence]
    :param gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2, class]
    :param iou_thresholds: List of IoU thresholds
    :return: mAP value
    """
    mAP = 0
    for threshold in iou_thresholds:
        mAP += compute_ap_at_gamma(pred_boxes, gt_boxes, threshold)
    return mAP / len(iou_thresholds)

