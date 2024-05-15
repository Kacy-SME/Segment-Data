import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import torch
import numpy as np
import argparse
import logging
import imageio
import os

print(cv2.__version__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Image loaded and processing started...")

# Initialize device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_images(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def main(image_path):
    img_rgb = load_images(image_path)
    background = np.zeros_like(img_rgb)  # Create a black canvas

    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(img_rgb)
    mask = result[0]['segmentation'].astype(np.uint8)

    # Debug: Print unique mask values and display the mask
    print("Unique mask values:", np.unique(mask))
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.show()
    
    # Invert the mask: change 0s to 1s and 1s to 0s
    mask = 1 - mask  # This flips the mask

    background[mask == 1] = img_rgb[mask == 1]

    logging.info("Saving images...")
    save_images(background)
    logging.info("Images saved successfully!")

def save_images(background):
    output_dir = 'Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Display the result
    plt.imshow(background)
    plt.title('Final Result')
    plt.axis('off')  # Hide axis
    plt.show()

    # Save using OpenCV
    background_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/opencv_output.jpg', background_bgr)
    logging.info("Images saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from images.')
    parser.add_argument('image_path', type=str, help='Path to the foreground image.')
    args = parser.parse_args()
    main(args.image_path)


