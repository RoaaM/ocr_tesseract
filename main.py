import cv2
import pytesseract
import csv
import os

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set path to tessdata directory
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Set path to input folder
input_folder = r'C:\Users\roaas\Downloads\images_for_experiments\images_for_experiments'

# Set path to output CSV file
output_file = "output.csv"

# Preprocess image function
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

# Extract text from image
def extract_text_from_image(img_path):
    # Load image
    img = cv2.imread(img_path)

    # Preprocess image
    processed_img = preprocess(img)

    # Extract text from the image using Tesseract
    text = pytesseract.image_to_string(processed_img)

    return text

# Iterate over all image files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Extract text from image
        text = extract_text_from_image(file_path)

    else:
        # Skip files that are not images
        continue

    # Save extracted text to CSV file
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename, text])
