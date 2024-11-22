import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Train the YOLOv5 model
# This code defines a function to train the YOLOv5 model using a specific dataset and training configuration.
def train_model():
    # Load the YOLOv5 training script
    from yolov5 import train

    def train_model():
        from yolov5 import train

        train.run(
            data='dataPedestrian/data.yaml',  # Path to dataset configuration file
            imgsz=640,  # Image size
            batch_size=16,  # Batch size
            epochs=10,  # Number of epochs
            weights='yolov5s.pt',  # Pretrained weights
            device='cpu',  # Device to run the training
            project='runs/train',  # Directory to save training results
            name='person_detection',  # Training run name
            save_period=1  # Save model after each epoch
        )

# Step 2: Load the trained YOLOv5 model
# This function loads the trained YOLOv5 model from the specified path and prepares it for inference.
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/person_detection/weights/best.pt',
                           force_reload=True)
    model.to(device)
    return model

# Step 3: Perform predictions and draw green bounding boxes
# This function performs predictions on a validation dataset and visualizes the results with bounding boxes.
def predict_and_display(model, val_images_dir):
    # Get the first few images from the validation set
    val_images = list(Path(val_images_dir).glob('*.jpg'))[:5]

    for img_path in val_images:
        img = cv2.imread(str(img_path))
        results = model(img)

        # Get detection results
        detections = results.xyxy[0].cpu().numpy()

        # Draw bounding boxes
        for x1, y1, x2, y2, conf, cls in detections:
            # Draw green bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Main execution
# This section executes the training, model loading, and prediction steps.
if __name__ == "__main__":
    print("Starting model training...")
    train_model()

    print("Loading the trained model...")
    model = load_model()

    print("Performing predictions on the validation set...")
    predict_and_display(model, 'dataPedestrian/images/Val')
