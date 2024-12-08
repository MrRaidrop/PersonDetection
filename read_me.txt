Project Overview: CV_Project
This repository contains various scripts and models aimed at training, detecting, and enhancing person detection systems using different datasets and methodologies. Below is a detailed description of each script in the repository:

Script Descriptions
PersonDetectionOriginal.py

This script uses the original dataset to train a person detection model with two classes: person and person-like.
Performance was poor regardless of parameter tuning, as the dataset lacked images with no-person scenarios.

VideoPersonDetection.py

This script captures frames from the webcam and detects people in the images.
Initially used the PersonDetectionOriginal.py model but demonstrated poor performance. The script has since evolved to test better models.
VOCtoYOLO.py

Converts annotation files from the Pascal VOC format to the YOLO format.
Essential for preparing datasets for YOLO-based training.
PedestrianDetectionYolo.py

Trains the YOLO pre-trained model using the data dataset for pedestrian detection.
Focused on leveraging YOLO's strengths for person detection tasks.
RGBImageGenerator.py

Generates RGB block images with visually similar color patterns.
These RGB images were used to train a model (person2.h5) on the data dataset. However, the resulting model learned to classify based on visual patterns rather than actual person detection.
xmlGenerator.py

Automatically generates Pascal VOC XML annotation files for no-person scenarios.
Useful for augmenting datasets by adding annotations for images without any people.
Txt_changer.py

Updates and manages text file lists (txt files) that correspond to image datasets in the data folder.
Helps synchronize image datasets with corresponding text annotations.
PicGeneratorYolo.py

Identifies a critical flaw in the RGB-trained model (person2.h5), which misclassified based on image center color differences rather than actual human features.
Uses YOLO to blur or pixelate all human figures in the data dataset, labeling them as no-person. This augmented dataset was used to train a better model (person3.h5).
TrainModelV1.py

A training script specifically designed to train models on the data dataset.
Focuses on improving person detection performance within this dataset.
TrainModelV2.py

An improved training script designed for the dataV2 dataset, which includes a significantly larger collection of 35,000 images.
Aims to train higher-quality models with a more comprehensive dataset.
Face_Mark_Script.py

Detects and marks faces in the dataV2 dataset.
Automatically generates Pascal VOC XML annotations for all detected faces.
Key Models in the Repository
person.h5: The model trained using the original dataset with poor performance.
person2.h5: The model trained using RGB-generated images but failed to generalize due to pattern-based learning.
person3.h5: A refined model trained using an augmented dataset where human figures were pixelated, significantly improving performance.
Person_Model.h5: Additional model trained as part of the project evolution.
Person_V2.h5: Model trained using the improved dataV2 dataset.
How to Use This Repository
Prepare the Environment:

Ensure OpenCV, TensorFlow, and other required libraries are installed.
Place datasets (data, dataV2) in the correct directory structure as shown in the project.
Dataset Preparation:

Use xmlGenerator.py to generate XML annotations for no-person scenarios.
Use VOCtoYOLO.py to convert annotations to YOLO format when necessary.
Use Face_Mark_Script.py to mark faces and generate annotations.
Training Models:

Use TrainModelV1.py or TrainModelV2.py to train models on their respective datasets.
Testing Models:

Use VideoPersonDetection.py to test the performance of trained models on webcam video input.
Enhancing Models:

Use tools like RGBImageGenerator.py and PicGeneratorYolo.py to augment datasets and refine models further.
Project Structure
Datasets:
data: Original dataset with limited images.
dataV2: Improved dataset with 35,000 images for better training.
Annotations:
Generated using scripts like xmlGenerator.py and Face_Mark_Script.py.
Models:
Pre-trained and custom-trained models (person.h5, person2.h5, etc.).
Scripts:
For dataset preparation, training, and testing.