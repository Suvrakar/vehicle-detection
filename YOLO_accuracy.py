import numpy as np

# Example ground truth bounding boxes (4 boxes with [x, y, width, height] format)
ground_truth_boxes = np.array([
    [100, 150, 50, 50],
    [200, 250, 60, 60],
    [300, 350, 70, 70],
    [400, 450, 80, 80]
])

# Example predicted bounding boxes (4 boxes with [x, y, width, height] format)
predicted_bounding_boxes = np.array([
    [110, 160, 45, 45],
    [220, 260, 58, 58],
    [310, 355, 72, 72],
    [410, 460, 85, 85]
])

# Calculate the intersection over union (IoU) between ground truth and predicted bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] * box1[3]) + (box2[2] * box2[3]) - intersection
    return intersection / union

# Set a threshold for IoU to consider a detection as correct
iou_threshold = 0.5

# Calculate precision and recall
true_positives = 0
false_positives = 0
false_negatives = 0

for ground_truth_box in ground_truth_boxes:
    detected = False
    for predicted_box in predicted_bounding_boxes:
        iou = calculate_iou(ground_truth_box, predicted_box)
        if iou >= iou_threshold:
            true_positives += 1
            detected = True
            break
    if not detected:
        false_negatives += 1

false_positives = len(predicted_bounding_boxes) - true_positives

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
accuracy = true_positives / len(ground_truth_boxes)

print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

# Calculate Average Precision (AP) at IoU threshold 0.5
# This code assumes you have multiple predicted bounding boxes for each ground truth box
average_precision = true_positives / (true_positives + false_positives)
print('Average Precision (AP@0.5):', average_precision)
