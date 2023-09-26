import cv2
import pytesseract
import tkinter as tk

# Function to display recognized text in a pop-up window
def display_text_in_popup(text):
    popup = tk.Tk()
    popup.title("Recognized Text")

    label = tk.Label(popup, text=text, font=("Arial", 12))
    label.pack(padx=10, pady=10)

    popup.mainloop()

# Load the image using OpenCV
img = cv2.imread("./detected-images/plate_1695629641049.jpg")

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply noise reduction using Gaussian blur
img_blurred = cv2.GaussianBlur(img_threshold, (5, 5), 0)

# Perform OCR on the processed image
text = pytesseract.image_to_string(img_blurred, lang='ben')

# Print the recognized text
print(text)

# Display recognized text in a pop-up window
display_text_in_popup(text)
