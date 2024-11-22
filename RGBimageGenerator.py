import os
import numpy as np
import cv2


def generate_similar_color_image(base_color, size=(200, 200)):
    """
    Generate an RGB image with colors similar to the given base color.
    """
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for i in range(size[0]):
        for j in range(size[1]):
            # Generate a color similar to the base color with random perturbations
            perturbation = np.random.randint(-30, 30, 3)  # Variation range for each channel
            color = np.clip(base_color + perturbation, 0, 255)
            image[i, j] = color

    return image


def generate_random_images(save_dir, annotation_dir, size=(200, 200)):
    """
    Generate RGB images based on the number of .xml files in the Annotations folder.
    The starting index will be the existing number of files + 1.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Count the number of existing .jpg files in the save directory
    existing_images_count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])

    # Count the number of .xml files in the annotation directory
    num_annotations = len([f for f in os.listdir(annotation_dir) if f.endswith('.xml')])

    # Set the starting index for new images
    starting_index = existing_images_count + 1

    # Generate images starting from the calculated index
    for i in range(num_annotations):
        # Randomly select a base color (RGB)
        base_color = np.random.randint(0, 256, 3)
        image = generate_similar_color_image(base_color, size)

        # Define the filename for the new image
        image_filename = f'image ({starting_index + i}).jpg'

        # Save the generated image
        cv2.imwrite(os.path.join(save_dir, image_filename), image)
        print(f"Generated image: {image_filename}")


def process_dataset(root_dir):
    """
    Generate RGB images with similar colors for datasets based on .xml file counts.
    The starting index will account for the number of existing files.
    """
    datasets = ['Test/Test/']
    for dataset in datasets:
        jpeg_dir = os.path.join(root_dir, dataset, 'JPEGImages')
        annotation_dir = os.path.join(root_dir, dataset, 'Annotations')

        print(f"Processing the {dataset} dataset...")
        # Generate RGB images based on .xml file count
        generate_random_images(jpeg_dir, annotation_dir)
        print(f"RGB images for the {dataset} dataset have been generated!")


if __name__ == '__main__':
    root_directory = 'Person'
    process_dataset(root_directory)
    print("All datasets have their RGB images generated!")
