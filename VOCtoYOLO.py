import os
import xml.etree.ElementTree as ET
import shutil


def convert_voc_to_yolo(xml_file, output_dir, image_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image dimensions
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        # Check if width or height is 0
        if width == 0 or height == 0:
            print(f"Warning: Width or height of {xml_file} is 0. Skipping this file.")
            return None

        # Get the image file name
        filename = root.find("filename").text
        filename = os.path.splitext(filename)[0] + ".jpg"

        # Create a YOLO format label file
        yolo_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")

        with open(yolo_filename, "w") as yolo_file:
            # Iterate over all objects
            for obj in root.findall("object"):
                class_name = obj.find("name").text

                # Assume class_id for "person" is 0
                if class_name != "person":
                    continue  # Skip if it's not "person"

                class_id = 0

                # Get bounding box coordinates
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)

                # Calculate YOLO format coordinates
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                # Write to YOLO format
                yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        return filename
    except Exception as e:
        print(f"Error: An error occurred while processing file {xml_file}. Skipping. Error details: {e}")
        return None


def process_dataset(dataset_dir, output_dir):
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    sets = ['Train', 'Val', 'Test']
    for subset in sets:
        subset_dir = os.path.join(dataset_dir, subset, subset)
        annotation_dir = os.path.join(subset_dir, "Annotations")
        image_dir = os.path.join(subset_dir, "JPEGImages")

        output_images_dir = os.path.join(output_dir, "images", subset)
        output_labels_dir = os.path.join(output_dir, "labels", subset)
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        for xml_file in os.listdir(annotation_dir):
            if xml_file.endswith(".xml"):
                xml_path = os.path.join(annotation_dir, xml_file)

                # Convert XML file to YOLO format
                image_filename = convert_voc_to_yolo(xml_path, output_labels_dir, image_dir)

                # If conversion was successful, copy the corresponding image
                if image_filename:
                    original_image_path = os.path.join(image_dir, image_filename)
                    new_image_path = os.path.join(output_images_dir, image_filename)
                    if os.path.exists(original_image_path):
                        shutil.copy(original_image_path, new_image_path)

        print(f"Completed processing of {subset} dataset.")

# Run the script
dataset_dir = "data"
output_dir = "dataPedestrian"
process_dataset(dataset_dir, output_dir)

print("Dataset has been successfully converted to YOLO format!")
