import cv2
import os
import torch
import numpy as np


def detect_and_remove_person(img, model):
    """
    Detect and remove people from the image.
    """
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    # If no person is detected, return None
    if len(detections) == 0 or all(int(det[5]) != 0 for det in detections):
        return None

    # Create a mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for det in detections:
        # Process only detections of class 'person'
        if int(det[5]) == 0:  # The index for the 'person' class is 0
            x1, y1, x2, y2 = map(int, det[:4])
            mask[y1:y2, x1:x2] = 255

    # Use OpenCV's inpaint method to restore the image
    inpainted_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img


def save_image_with_incremental_name(img, save_dir, start_index=945):
    """
    Save the image with an incremental name starting from the specified index.
    """
    index = start_index
    while True:
        file_name = os.path.join(save_dir, f"image ({index}).jpg")
        if not os.path.exists(file_name):
            cv2.imwrite(file_name, img)
            print(f"Saved: {file_name}")
            break
        index += 1


def main():
    # 1. Load the pretrained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # 2. Set paths
    input_dir = './data/Val/Val/JPEGImages'
    output_dir = './data/Val/Val/ProcessedImages'
    os.makedirs(output_dir, exist_ok=True)

    # 3. Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # 4. Process each image
    current_index = 161
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        # Detect and remove people
        processed_img = detect_and_remove_person(img, model)

        # Save the image only if people were detected and processed
        if processed_img is not None:
            save_image_with_incremental_name(processed_img, output_dir, current_index)
            current_index += 1


if __name__ == "__main__":
    main()
