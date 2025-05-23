# PersonDetection

## Introduction
Project for person detection using custom-trained models and YOLO. 


## Table of Contents
- [Introduction](#introduction)
- [How to Use This Repository](#how-to-use-this-repository)
- [File Descriptions](#file-descriptions)
- [Work Remain](#work-remain)

## how-to-use-this-repository

### Prerequisites
- Python 3.10 or higher
- TensorFlow 2.11 (if you don't want to use cuda, just use lastest version)
- OpenCV 4.10
- CUDA 11.8 (if you don't want to plan cuda, just ignore it)
- cuDNN 8.8 (if you don't want to plan cuda, just ignore it)

### Prepare the Environment:
opencv：

`pip install opencv-python`

`pip install opencv-python-headless`



yolov5:

`git clone https://github.com/ultralytics/yolov5.git`

`cd yolov5`

`pip install -r requirements.txt`

- NumPy, Pandas, and other standard libraries



Ensure OpenCV, TensorFlow, and other required libraries are installed.

Place datasets (data, dataV2) in the correct directory structure as shown in the project.

download https://www.kaggle.com/code/wldzia/pedestrian-detection-using-cnn
and place the dataset in 'data' folder
download https://www.kaggle.com/datasets/kazushiadachi/human-portrait-or-not-128128-binary-and-rgb/data
and place the dataset in 'dataV2' folder

## file-descriptions

**PersonDetectionOriginal.py**

This script uses the original dataset to train a person detection model with two classes: person and person-like.
Performance was poor regardless of parameter tuning, as the dataset lacked images with no-person scenarios.

**VideoPersonDetection.py**

This script captures frames from the webcam and detects people in the images.
Initially used the PersonDetectionOriginal.py model but demonstrated poor performance. The script has since evolved to test better models.

**VOCtoYOLO.py**

Converts annotation files from the Pascal VOC format to the YOLO format.
Essential for preparing datasets for YOLO-based training.
Look at How to Use This Repository to get yolo first.

**PedestrianDetectionYolo.py**

Trains the YOLO pre-trained model using the data dataset for pedestrian detection. YOLO performance is way better than our model.
It can even draw around person, but it's still a pre-trained model. **We planed to train our own**.

**RGBImageGenerator.py**

Generates RGB block images with visually similar color patterns.
These RGB images were used to train a model (person2.h5) on the data dataset. However, the resulting model learned to classify based on visual patterns rather than actual person detection.

**xmlGenerator.py**

Automatically generates XML annotation files for no-person scenarios.

**Txt_changer.py**

Updates and manages text file lists (txt files) that correspond to image datasets in the '**data**' folder.
Helps synchronize image datasets with corresponding text annotations.

**PicGeneratorYolo.py**

Identifies a critical flaw in the RGB-trained model (person2.h5), which misclassified based on image center color differences rather than actual human features.
Uses YOLO to blur or pixelate all human figures in the data dataset, labeling them as no-person. This augmented dataset was used to train a better model (person3.h5).

**TrainModelV1.py**

A training script specifically designed to train models on the '**data**' dataset.
Focuses on improving person detection performance within this dataset.

**TrainModelV2.py**

An improved training script designed for the '**dataV2**' dataset, which includes a significantly larger collection of 35,000 images.
Aims to train higher-quality models with a more comprehensive dataset.

**Face_Mark_Script.py**

Detects and marks faces in the '**dataV2**' dataset.
Automatically generates Pascal VOC XML annotations for all detected faces.

**Face_Mark_Script.py**

Detects and marks faces in the '**dataV2**' dataset using **haar_cascade**.
Automatically generates Pascal VOC XML annotations for all detected faces.

**TrainFaceModel.py**

use the detected and marked faces in the '**dataV2**' dataset to generate face_detection_model.h5
Need a long process same as the TrainV2.

**testFaceMark.py**

Test if haar_cascade mark the dataset right. A set of marked picture will show up.

**testFaceModel.py**

Test if the face_detection_model.h5 mark the pictures right. A set of marked picture will show up.

**VideoV2Face.py**

Will open your camera and use two models to detect if there is person and if where is the face in real-tiem.

**Models in the Repository**

person.h5: The model trained using the original dataset with poor performance.

person2.h5: The model trained using RGB-generated images but failed to generalize due to pattern-based learning.

person3.h5: A refined model trained using an augmented dataset where human figures were pixelated, significantly improving performance.

Person_V2.h5: Model trained using the improved dataV2 dataset. Acceptable result.

face_detection_model.h5: The model trained using images of datasetV2. The outcome is acceptable, can somehow track faces, still need fine-tune, can not deal with edge cases.

## work-remain

**Integrate Face Detection into Video Processing:** (finished, updated)

The Face_Mark_Script.py has been used to annotate faces in the dataV2 dataset with Pascal VOC XML format.
The next step is to train a model specifically for face detection using these annotations and integrate it into the VideoPersonDetection.py script.


**Edge Cases:**
The model still can not detect face in the edge of the screen, probabily can be fix by moving some proportioon faces of the dataset to the edge and rebuild the model.

**Optimize and Fine-Tune:**

The har_cascade detect two eyes at first and then the face, that maybe a better approach, maybe use that method.





