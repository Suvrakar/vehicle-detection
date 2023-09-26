import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 1.143  # meters (approximately 45 inches)
VEHICLE_WIDTH = 70     # pixels

# Object detector constants
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.1

# Colors for object detection
HIGHLIGHT_COLOR = (0, 255, 0)
DISTANCE_BOX_COLOR = (0, 0, 0)
ALERT_COLOR = (0, 0, 255)

# Load pre-trained YOLOv4-tiny model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set preferable backend and target for the neural network
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Set up the object detection model
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load test video
cap = cv.VideoCapture('./Data/vid5.mp4')

# Define the desired video frame size (464x464)
frame_width = 464
frame_height = 464

# Create an empty list to store predicted bounding boxes
predicted_boxes = []

# Load ground truth bounding boxes (replace with actual ground truth data)
ground_truth_boxes = np.array([
    [40, 64, 424, 194],
    [38, 70, 426, 182],
    [39, 70, 425, 183],
    [41, 67, 423, 188],
    [44, 67, 420, 187],
    [44, 67, 420, 187],
    [46, 67, 418, 186],
    [45, 67, 419, 187],
    [43, 62, 421, 194],
    [44, 58, 420, 207],
    [42, 55, 422, 210],
    [45, 58, 419, 207],
    [40, 58, 424, 206],
    [38, 60, 426, 200],
    [0, 53, 407, 212],
    [0, 55, 410, 208],
    [0, 52, 419, 215],
    [0, 56, 424, 207],
    [38, 64, 426, 195],
    [36, 62, 428, 197],
    [38, 65, 426, 189],
    [37, 65, 427, 189],
    [17, 58, 426, 201],
    [25, 112, 439, 161],
    [34, 116, 430, 158],
    [37, 113, 427, 166],
    [35, 119, 429, 156],
    [22, 81, 418, 222],
    [33, 101, 431, 185],
    [26, 76, 415, 232],
    [26, 103, 438, 186],
    [26, 82, 410, 234],
    [30, 117, 407, 216],
    [29, 111, 411, 230],
    [49, 102, 415, 260],
    [4, 94, 381, 272],
    [81, 100, 375, 266],
    [86, 91, 356, 288],
    [24, 87, 355, 287],
    [25, 89, 353, 284]
], dtype=np.int32)


while True:
    ret, frame = cap.read()

    if not ret:
        # Break the loop if the video has ended
        break

    # Resize the frame to the desired size
    frame = cv.resize(frame, (frame_width, frame_height))

    # Detect objects on the road
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        for i in range(len(classes)):
            classid = int(classes[i])
            score = float(scores[i])
            box = boxes[i]

            if classid == 2 or classid == 7 or classid == 3:  # Vehicle and car class IDs in the YOLO model
                # Calculate distance based on object width
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                # Set default color for the vehicle rectangle and header
                vehicle_color = HIGHLIGHT_COLOR
                text_color = (0, 0, 0)  # Black

                # Trigger an alert if the object is too close
                if distance < 1:  # Distance threshold for alert (1 meter)
                    vehicle_color = ALERT_COLOR
                    text_color = (0, 0, 255)  # Red

                # Draw a rectangle around the vehicle or car
                cv.rectangle(frame, tuple(box), vehicle_color, 1)

                # Calculate Intersection over Union (IoU) for each predicted box with ground truth
                iou = 0.0
                for gt_box in ground_truth_boxes:
                    intersection = np.maximum(0, np.minimum(box[2], gt_box[2]) - np.maximum(box[0], gt_box[0])) * \
                        np.maximum(0, np.minimum(
                            box[3], gt_box[3]) - np.maximum(box[1], gt_box[1]))
                    union = (box[2] - box[0]) * (box[3] - box[1]) + (gt_box[2] -
                                                                     gt_box[0]) * (gt_box[3] - gt_box[1]) - intersection
                    iou = max(iou, intersection / union)

                predicted_boxes.append((box, iou))

    # Display the frame with detected objects and distances
    cv.imshow('Road Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()

# Convert the list of predicted boxes and IoU scores to a structured NumPy array
dtype = [('box', int, (4,)), ('iou', float)]
predicted_boxes = np.array(predicted_boxes, dtype=dtype)

# Sort predicted boxes by IoU scores in descending order
predicted_boxes = np.sort(predicted_boxes, order='iou')[::-1]

# Set IoU threshold for positive detection
iou_threshold = 0.5

# Initialize variables for calculating AP, recall, and detection rate
true_positives = 0
false_negatives = len(ground_truth_boxes)

precision = []
recall = []
detection_rate = []

for i, (box, iou) in enumerate(predicted_boxes):
    if iou >= iou_threshold:
        true_positives += 1
        false_negatives -= 1

    current_precision = true_positives / (i + 1)
    current_recall = true_positives / len(ground_truth_boxes)
    current_detection_rate = true_positives / \
        (true_positives + false_negatives)

    precision.append(current_precision)
    recall.append(current_recall)
    detection_rate.append(current_detection_rate)

# Calculate Average Precision (AP) at IoU threshold 0.5
ap_05 = np.trapz(precision, recall)

# Print AP at IoU threshold 0.5, recall, and detection rate
print(f"AP@0.5: {ap_05}")
print(f"Recall: {recall[-1]}")
print(f"Detection Rate: {detection_rate[-1]}")
