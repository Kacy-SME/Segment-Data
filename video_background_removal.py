import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import torch
import numpy as np
import argparse
import logging
import os
import collections
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


print(cv2.__version__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Video loaded and processing started...")

# Initialize device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize a deque to store masks of previous frames for temporal smoothing
mask_history = collections.deque(maxlen=5)


def process_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    background = np.zeros_like(img_rgb)  # Create a black canvas

    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(img_rgb)
    mask = result[0]['segmentation'].astype(np.float32)

   # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Threshold the mask
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

    # Add current mask to history for temporal smoothing
    mask_history.append(mask)

    # Temporal smoothing by averaging masks from the history
    if len(mask_history) > 1:
        mask_avg = np.mean(np.stack(mask_history, axis=0), axis=0)
        mask = (mask_avg > 0.5).astype(np.uint8)


    # Invert the mask: change 0s to 1s and 1s to 0s
    mask = 1 - mask  # This flips the mask
    background[mask == 1] = img_rgb[mask == 1]

    # Convert back to BGR for video writing
    background_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    return background_bgr

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info("Video loaded and processing started...")

    output_dir = 'Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{output_dir}/processed_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    progress = tqdm(total=frame_count, desc="Processing video", unit="frame")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        out.write(processed_frame)
        progress.update(1)

    # Release everything when job is finished
    progress.close()
    cap.release()
    out.release()
    logging.info("Video saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    args = parser.parse_args()
    main(args.video_path)

