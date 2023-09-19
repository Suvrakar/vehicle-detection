import cv2
import os
import time

plate_detector = cv2.CascadeClassifier("./BD_numberPlate_cascade_v1.xml")
video = cv2.VideoCapture('./Data/vid2.mp4')

if (video.isOpened() == False):
    print('Error Reading Video')

# Specify the directory for saving the captured images
home_dir = os.path.expanduser("~")
save_dir = os.path.join(
    home_dir, "personalSuv/BUET-thesis/Car-Number-Plate-Recognition-Sysytem/detected-images")

# Initialize grayscale conversion flag outside the loop
convert_gray = True

# Specify the scaling factor for the captured plate image
scaling_factor = 1.1

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (300, 300))
    if ret == True:
        if convert_gray:
            gray_frame = cv2.resize(frame, (300, 300))
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = cv2.resize(frame, (300, 300))

        # plate = plate_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(45, 45))
        plate = plate_detector.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=10, minSize=(45, 45))

        for (x, y, w, h) in plate:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text='License Plate', org=(x-3, y-3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(0, 0, 255), thickness=1, fontScale=0.6)

            # Capture and save the image of the license plate with a unique filename
            # Generate a unique timestamp for each image
            timestamp = int(time.time() * 1000)
            filename = f"plate_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            print("Save path:", save_path)  # Debug statement

            # Check if the save directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Extract the plate region from the frame
            plate_region = frame[y:y+h, x:x+w]

            # Apply resizing to improve image clarity
            resized_plate = cv2.resize(
                plate_region, None, fx=scaling_factor, fy=scaling_factor)

            # Apply denoising and contrast enhancement if necessary
            # ...

            # Save the image
            cv2.imwrite(save_path, resized_plate)
            print("Image saved successfully")  # Debug statement

        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
