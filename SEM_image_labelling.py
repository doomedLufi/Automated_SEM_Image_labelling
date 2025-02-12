import os
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# Function for loading the image and converting to greyscale if necessary
def load_image(file_path):
    try:
        pil_image = Image.open(file_path)
        image = np.array(pil_image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    except Exception as e:
        print(f"Error loading the image: {e}")
        return None

# Function for selecting a main folder and checking for "PlainImages"
def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the main folder")
    if not folder_path:
        print("No folder selected.")
        return None
    plain_images_path = os.path.join(folder_path, "PlainImages")
    if not os.path.isdir(plain_images_path):
        print("The subfolder 'PlainImages' was not found.")
        return None
    return plain_images_path

# Function for loading an image with file selection from the parent folder
def select_image_files(folder_path):
    # Liste der Dateien aus `PlainImages`
    plain_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not plain_images:
        print("No image files found in the 'PlainImages' folder.")
        return None, None

    # User selects a file
    print("Select a file for scale detection:")
    for i, file in enumerate(plain_images, 1):
        print(f"{i}: {file}")
    
    selected_index = int(input("Enter the number of the desired file: ")) - 1
    selected_file = plain_images[selected_index]

    # Paths for the image from the parent folder and `PlainImages`.
    over_folder_path = os.path.dirname(folder_path)
    over_file_path = os.path.join(over_folder_path, selected_file)
    plain_file_path = os.path.join(folder_path, selected_file)
    
    return over_file_path, plain_file_path

# Function for selecting a destination folder
def select_output_folder():
    root = Tk()
    root.withdraw()
    output_folder = filedialog.askdirectory(title="Select the output folder for saving the images")
    if not output_folder:
        print("No output folder selected.")
        return None
    return output_folder

# Function for recognising the scale based on fixed start coordinates
def detect_scale_fixed_position(image, start_x=None, start_y=None):
    """
    Detects the length of a scale line in an image, with starting positions
    dynamically calculated based on image size.
    Returns:
        int: The length of the scale line.
    """
    height, width = image.shape[:2]
    
    start_x = width // 64
    start_y = height - ((height // 32) + (width // 512))
    
    # Initialize scale length
    scale_length = 0
    
    # Loop to determine scale length
    for x in range(start_x, width):
        if image[start_y, x] > 200:  # Assuming white has a value > 200
            scale_length = x - start_x
            break
    
   
    return scale_length


def extract_text_from_roi(image_path):
    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Define ROI (Region of Interest)
    x1, x2 = int(width / 64), int(7 / 64 * width)
    y1, y2 = int(119 / 128 * height), int(243 / 256 * height)

    # Crop ROI
    roi = image[y1:y2, x1:x2]

    # Convert to grayscale (optional, improves OCR accuracy)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # 'en' for English

    # Perform OCR
    results = reader.readtext(gray)

    # Extract text
    extracted_text = " ".join([res[1] for res in results])

    return extracted_text

def correct_text(text):
    # Replace misrecognized "u" with "µ"
    corrected_text = text.replace("p", "µ").replace("u", "µ")

    # If OCR only detects "µm", assume it should be "1 µm"
    if corrected_text.strip() in ["µm", "um"]:
        corrected_text = "1 µm"

    return corrected_text


def add_new_scale_with_pillow(image, scale_length_pixels, scale_length_real, image_path, font_path="arial.ttf"):
    """
    Adds a new scale in the lower right area of the image and uses Pillow for texts.
    :param image: Greyscale image without banner
    :param scale_length_pixels: Length of the recognised scale in pixels
    :param scale_length_real: Actual length of the recognised scale in µm
    :param image_path: Path to the image from which scale text should be extracted
    :param font_path: Path to the TrueType font (e.g., Arial)
    :return: Image with new scale
    """
    # Convert to RGB to work with Pillow
    new_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(new_image)
    draw = ImageDraw.Draw(pil_image)

    # Extract text automatically
    extracted_text = extract_text_from_roi(image_path)
    scale_text = correct_text(extracted_text)

    # Scale bar calculation
    scale_bar_length = int(scale_length_pixels * 1)  # Assuming 1 µm = scale_length_pixels

    # Image dimensions
    height, width, _ = new_image.shape
    fontsize = width // 32

    # Load font
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except IOError:
        print("Font could not be loaded. Using default font.")
        font = ImageFont.load_default()

    # Calculate text size
    text_bbox = draw.textbbox((0, 0), scale_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Coordinates for backgrounds and scale
    x_sw_st = width - (width // 8)
    y_sw_st = height - (height // 8)

    x_bg_start = width - (width // 8) + (width // 512)
    x_bg_end = width - (width // 512)
    y_bg_start = height - (height // 8) + (height // 384)
    y_bg_end = height - (height // 384)

    # Center of white background
    x_center = x_bg_start + (x_bg_end - x_bg_start) // 2
    y_center = y_bg_start + (y_bg_end - y_bg_start) // 2

    # Scale position
    x_sc_start = x_center - (scale_bar_length // 2)
    x_sc_end = x_center + (scale_bar_length // 2)
    y_sc_start = y_center - (height // 32)
    y_sc_end = y_center - (height // 64)

    # Text position
    x_text = x_center - (text_width // 2)
    y_text = y_center + (text_height // 2) - (height // 48)

    # Draw elements
    draw.rectangle([(x_sw_st, y_sw_st), (width, height)], fill=(0, 0, 0))  # Black background   
    draw.rectangle([(x_bg_start, y_bg_start), (x_bg_end, y_bg_end)], fill=(255, 255, 255))  # White background
    draw.rectangle([(x_sc_start, y_sc_start), (x_sc_end, y_sc_end)], fill=(0, 0, 0))  # Black scale

    # Draw scale text
    draw.text((x_text, y_text), scale_text, fill=(0, 0, 0), font=font)

    # Convert back to OpenCV format
    return np.array(pil_image)



# Function for saving the image
def save_image(image, output_folder, output_name):
    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(f"{output_path}.png", image)
    cv2.imwrite(f"{output_path}.jpg", image)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(f"{output_path}.svg", format='svg')
    plt.close()

# Main function
def process_sem_image(detection_file_path, plain_file_path, output_folder):
    # Load image from the main folder for scale recognition
    detection_image = load_image(detection_file_path)
    if detection_image is None:
        print("Image for scale recognition could not be loaded.")
        return

    # Recognise scale in main folder image
    scale_length_pixels = detect_scale_fixed_position(detection_image)
    if scale_length_pixels is None:
        print("Scale could not be recognised.")
        return

    # Load image from `PlainImages` for editing
    plain_image = load_image(plain_file_path)
    if plain_image is None:
        print("Image for editing could not be loaded.")
        return

    # Extract text from the image (ROI)
    extracted_text = extract_text_from_roi(plain_file_path)
    
    # Add new scale
    final_image = add_new_scale_with_pillow(plain_image, scale_length_pixels, scale_length_real=1, image_path=detection_file_path)


    # Saving the image
    output_name = os.path.splitext(os.path.basename(plain_file_path))[0] + "_processed"
    save_image(final_image, output_folder, output_name)
    print(f"Images are successfully saved in {output_folder}.")

# Execution
folder_path = select_folder()
if folder_path:
    detection_file_path, plain_file_path = select_image_files(folder_path)
    output_folder = select_output_folder()
    if output_folder:
        process_sem_image(detection_file_path, plain_file_path, output_folder)
