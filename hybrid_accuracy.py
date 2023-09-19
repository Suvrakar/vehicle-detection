import cv2 as cv
import os
import time

# Function to calculate Intersection over Union (IoU)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# YOLO Constants
KNOWN_DISTANCE = 1.143  # meters (approximately 45 inches)
VEHICLE_WIDTH = 70     # pixels
CONFIDENCE_THRESHOLD = 0.4  # Set a higher confidence threshold
NMS_THRESHOLD = 0.1

# Cascade Classifier
plate_detector = cv.CascadeClassifier("./BD_numberPlate_cascade_v1.xml")

# Video Capture
cap = cv.VideoCapture('./Data/vid5.mp4')

if not cap.isOpened():
    print('Error Reading Video')

# Specify the directory for saving the captured images
home_dir = os.path.expanduser("~")
save_dir = os.path.join(
    home_dir, "personalSuv/BUET-thesis/YOLO-Tiny(final)/detected-images")

# Initialize grayscale conversion flag outside the loop
convert_gray = True

# Specify the scaling factor for the captured plate image
scaling_factor = 1.1

# Load pre-trained YOLOv4-tiny model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set preferable backend and target for neural network
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Set up object detection model
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Initialize variables for evaluation metrics
true_positives = 0
false_positives = 0
false_negatives = 0

# Load ground truth data (replace with your actual ground truth data loading code)
# Each entry should include the frame number, class, and bounding box coordinates
# Example format: ground_truth_data = [(frame_number, class, (x, y, w, h)), ...]
ground_truth_data = []  # Replace with your actual ground truth data

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        # Break the loop if the video has ended
        break

    # Detect objects on the road using YOLO with confidence threshold
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        high_confidence_indices = [i for i, score in enumerate(
            scores) if score >= CONFIDENCE_THRESHOLD]

        for i in high_confidence_indices:
            classid = int(classes[i])
            score = float(scores[i])
            box = boxes[i]

            if classid == 2 or classid == 7 or classid == 3:  # Vehicle and car class IDs in the YOLO model
                # Calculate distance based on object width
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                # Trigger alert if the object is too close
                if distance < 1:  # Distance threshold for alert (1 meter)
                    vehicle_color = (0, 0, 255)  # Red
                    text_color = (0, 0, 255)  # Red
                else:
                    vehicle_color = (0, 255, 0)  # Green
                    text_color = (0, 0, 0)  # Black

                # Draw rectangle around the vehicle or car
                cv.rectangle(frame, box, vehicle_color, 1)

                # Check if the detection matches ground truth
                matched = False
                for (frame_number, gt_class, gt_box) in ground_truth_data:
                    if frame_count == frame_number and classid == gt_class:
                        iou = calculate_iou(box, gt_box)
                        if iou > 0.5:  # Adjust the IoU threshold as needed
                            matched = True
                            ground_truth_data.remove(
                                (frame_number, gt_class, gt_box))  # Remove matched ground truth entry
                            break

                if matched:
                    true_positives += 1
                else:
                    false_positives += 1

    # ... (Rest of the code)

    # Increment frame count
    frame_count += 1

    # Calculate FPS
    if frame_count % 10 == 0:  # Calculate FPS every 10 frames
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

    # Exit loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Calculate false negatives
false_negatives = len(ground_truth_data)

# Calculate accuracy, precision, recall, and F1 Score
total_objects = true_positives + false_positives + false_negatives
accuracy = (true_positives / total_objects) * 100 if total_objects > 0 else 0
precision = (true_positives / (true_positives + false_positives)
             ) if (true_positives + false_positives) > 0 else 0
recall = (true_positives / (true_positives + false_negatives)
          ) if (true_positives + false_negatives) > 0 else 0
f1_score = (2 * precision * recall) / (precision +
                                       recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()
