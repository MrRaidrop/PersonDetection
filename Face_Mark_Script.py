import os
import cv2
import xml.etree.ElementTree as ET


def create_annotation_file(image_name, annotations, output_dir, width, height):
    """
    Create an XML annotation file in Pascal VOC format.
    """
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "VOC2012"
    ET.SubElement(annotation, "filename").text = image_name
    ET.SubElement(annotation, "path").text = os.path.join(output_dir, image_name)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "The VOC2008 Database"
    ET.SubElement(source, "annotation").text = "PASCAL VOC2008"
    ET.SubElement(source, "image").text = "flickr"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "1"  # Assuming grayscale images for binary
    ET.SubElement(annotation, "segmented").text = "0"

    for (xmin, ymin, xmax, ymax) in annotations:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "face"
        ET.SubElement(obj, "pose").text = "Frontal"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
        ET.SubElement(obj, "difficult").text = "0"

    # Save the XML to file with the same name as the image
    xml_filename = os.path.splitext(image_name)[0] + ".xml"
    xml_path = os.path.join(output_dir, xml_filename)
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Annotation saved: {xml_path}")


def detect_faces_and_generate_annotations(image_dir, annotation_dir, haar_cascade_path):
    """
    Detect faces in images using Haar cascade and generate Pascal VOC annotations.
    """
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    for root, _, files in os.walk(image_dir):
        for file in files:
            # Process PNG images (you can add support for other formats if needed)
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    print(f"No faces detected in {file_path}")
                    continue

                annotations = []
                for (x, y, w, h) in faces:
                    annotations.append((x, y, x + w, y + h))

                # Save annotations in XML format, using the image name
                output_dir = os.path.join(annotation_dir, os.path.basename(root))  # Maintain folder structure
                create_annotation_file(file, annotations, output_dir, gray.shape[1], gray.shape[0])


def process_dataset(base_dir, haar_cascade_path):
    """
    Process the RGB and Binary datasets.
    """
    datasets = ["rgb/portrait", "binary/portrait"]

    for dataset in datasets:
        image_dir = os.path.join(base_dir, dataset)
        annotation_dir = os.path.join(base_dir, dataset.replace("portrait", "Annotations"))
        detect_faces_and_generate_annotations(image_dir, annotation_dir, haar_cascade_path)


if __name__ == "__main__":
    # Base directory of your dataset
    base_dir = "dataV2"

    # Path to OpenCV's pre-trained Haar cascade for face detection
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Process the dataset
    process_dataset(base_dir, haar_cascade_path)
