import cv2 as cv
import time

# Distance constants
KNOWN_DISTANCE = 1.143  # meters (approximately 45 inches)
VEHICLE_WIDTH = 70     # pixels

# Object detector constants
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.1

# Load pre-trained YOLOv4-tiny model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set preferable backend and target for neural network
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Set up object detection model
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(320, 320), scale=1/255, swapRB=True)

# Load test video
cap = cv.VideoCapture('./Data/vid2.mp4')

# Initialize variables for counting true positives, false positives, and false negatives
true_positives = 0
false_positives = 0
false_negatives = 0

# Initialize variables for measuring FPS
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        # Break the loop if the video has ended
        break

    # Detect objects on the road
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        for i in range(len(classes)):
            classid = int(classes[i])
            score = float(scores[i])
            box = boxes[i]
            
            if classid == 2 or classid == 7 or classid == 3:  # Vehicle and car class IDs in the YOLO model
                # Calculate distance based on object width
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                # Check if the object is a true positive based on your criteria (e.g., distance threshold)
                if distance < 1:  # Distance threshold for true positives (1 meter)
                    true_positives += 1
                else:
                    false_positives += 1
                    false_negatives += 1  # Count false negatives for objects that were not detected

    # Increment frame count
    frame_count += 1
    
    # Calculate and display FPS every 100 frames
    if frame_count % 100 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"Frame: {frame_count}, FPS: {fps:.2f}")
    
    # Exit loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Calculate and print accuracy, recall, precision, and F1-score
accuracy = true_positives / (true_positives + false_negatives)
recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()
