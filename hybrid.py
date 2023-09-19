import cv2 as cv
import os
import time

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
save_dir = os.path.join(home_dir, "personalSuv/BUET-thesis/YOLO-Tiny(final)/detected-images")

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

while True:
    ret, frame = cap.read()
    
    if not ret:
        # Break the loop if the video has ended
        break

    # Resize the frame to the desired size
    frame = cv.resize(frame, (464, 464))

    # Detect objects on the road using YOLO with confidence threshold
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        high_confidence_indices = [i for i, score in enumerate(scores) if score >= CONFIDENCE_THRESHOLD]

        for i in high_confidence_indices:
            classid = int(classes[i])
            score = float(scores[i])
            box = boxes[i]
            
            if classid == 2 or classid == 7 or classid == 3:  # Vehicle and car class IDs in the YOLO model
                # Calculate distance based on object width
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                # Set default color for the vehicle rectangle and header
                vehicle_color = (0, 255, 0)  # Green
                text_color = (0, 0, 0)  # Black

                # Trigger alert if the object is too close
                if distance < 1:  # Distance threshold for alert (1 meter)
                    vehicle_color = (0, 0, 255)  # Red
                    text_color = (0, 0, 255)  # Red

                # Draw rectangle around the vehicle or car
                cv.rectangle(frame, box, vehicle_color, 1)

                # Display the distance within a black box
                distance_text = f"{round(distance, 2)} m"
                text_size, _ = cv.getTextSize(distance_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = box[0] + int((box[2] - text_size[0]) / 2)
                text_y = box[1] - 10
                cv.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                             (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
                cv.putText(frame, distance_text, (text_x, text_y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add "alert" tag when the vehicle or car is too close
                if distance < 1:
                    alert_text = "Vehicle Detected!"
                    alert_size, _ = cv.getTextSize(alert_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    alert_x = box[0] + int((box[2] - alert_size[0]) / 2)
                    alert_y = box[1] + box[3] + 20
                    cv.rectangle(frame, (alert_x - 2, alert_y - alert_size[1] - 2),
                                 (alert_x + alert_size[0] + 2, alert_y + 2), (0, 0, 0), -1)
                    cv.putText(frame, alert_text, (alert_x, alert_y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv.LINE_AA)

                # Detect license plate within the vehicle region
                plate = plate_detector.detectMultiScale(
                    frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], scaleFactor=1.1, minNeighbors=10, minSize=(45, 45))

                for (x_plate, y_plate, w_plate, h_plate) in plate:
                    # Draw rectangle around the detected license plate
                    cv.rectangle(frame, (x_plate + box[0], y_plate + box[1]), (x_plate + w_plate + box[0], y_plate + h_plate + box[1]), (0, 0, 255), 2)
                    cv.putText(frame, text='BRTA-Approved License Plate', org=(x_plate + box[0] - 3, y_plate + box[1] - 3), fontFace=cv.FONT_HERSHEY_COMPLEX,
                                color=(0, 0, 255), thickness=1, fontScale=0.6)

                    # Capture and save the image of the license plate with a unique filename
                    # Generate a unique timestamp for each image
                    timestamp = int(time.time() * 1000)
                    filename = f"plate_{timestamp}.jpg"
                    save_path = os.path.join(save_dir, filename)

                    # Check if the save directory exists
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Extract the plate region from the frame
                    plate_region = frame[y_plate + box[1]:y_plate + h_plate + box[1], x_plate + box[0]:x_plate + w_plate + box[0]]

                    # Apply resizing to improve image clarity
                    resized_plate = cv.resize(plate_region, None, fx=scaling_factor, fy=scaling_factor)

                    # Apply denoising and contrast enhancement if necessary
                    # ...

                    # Save the image
                    cv.imwrite(save_path, resized_plate)

    # Display the frame with detected objects and distances
    cv.imshow('Road Object Detection', frame)
    
    # Exit loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()
