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

# Set path to original text CSV file
original_text_file = r"C:\Users\roaas\Documents\roaa_workspace\ocr_tesseract\original_text.csv"

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

    return text.strip()

# Read original text from CSV file
original_text = {}
with open(original_text_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        original_text[row[0]] = row[1].strip()

# Process images in input folder
filenames = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
predicted_text = []
for filename in filenames:
    file_path = os.path.join(input_folder, filename)
    
    # Extract text from image
    text = extract_text_from_image(file_path)
    
    # Save predicted text
    predicted_text.append(text)

    # Save extracted text to output CSV file
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename, text])

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy

y_true = [original_text[filename] for filename in filenames]
accuracy = calculate_accuracy(y_true, predicted_text)

# Print accuracy
print("Accuracy:", accuracy)