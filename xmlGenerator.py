import os


def generate_xml_file(index, output_dir):
    """Generate XML file"""
    xml_content = f"""<annotation>
    <folder>VOC2012</folder>
    <filename>image ({index}).jpg</filename>
    <source>
        <database>The VOC2008 Database</database>
        <annotation>PASCAL VOC2008</annotation>
        <image>flickr</image>
    </source>
    <size>
        <width>500</width>
        <height>375</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>no-person</name>
        <pose>Frontal</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <bndbox>
            <xmin>0</xmin>
            <ymin>0</ymin>
            <xmax>0</xmax>
            <ymax>0</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
</annotation>"""

    # Create XML file
    file_name = os.path.join(output_dir, f"image ({index}).xml")
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(xml_content)
    print(f"Generated: {file_name}")


def process_directory(image_dir, annotation_dir):
    """Process a directory to ensure all images have corresponding XML files."""
    # Count the number of images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    num_images = len(image_files)

    # Count the number of existing XML files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    num_annotations = len(annotation_files)

    print(f"Directory: {image_dir}")
    print(f"Total images: {num_images}, Existing annotations: {num_annotations}")

    # Generate missing XML files
    if num_annotations < num_images:
        for index in range(num_annotations + 1, num_images + 1):
            generate_xml_file(index, annotation_dir)


def main():
    # Define the directories for Train, Test, and Val
    base_dirs = [
        'data/Train/Train',
        'data/Test/Test',
        'data/Val/Val'
    ]

    for base_dir in base_dirs:
        image_dir = os.path.join(base_dir, "JPEGImages")
        annotation_dir = os.path.join(base_dir, "Annotations")

        # Ensure the Annotations directory exists
        os.makedirs(annotation_dir, exist_ok=True)

        # Process the directory
        process_directory(image_dir, annotation_dir)


if __name__ == "__main__":
    main()
