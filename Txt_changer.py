import os

def update_and_sort_text_file(image_dir, output_file):
    """
    Count the number of images in the specified directory, add their names
    to the text file, and sort the contents.
    :param image_dir: Directory containing image files
    :param output_file: Text file to update with image names
    """
    # 1. List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_indices = [int(f.split('(')[1].split(')')[0]) for f in image_files]

    # 2. Ensure the text file exists
    if not os.path.exists(output_file):
        open(output_file, 'w').close()

    # 3. Read the contents of the existing text file
    with open(output_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 4. Remove newline characters
    lines = [line.strip() for line in lines]

    # 5. Parse existing indices in the text file
    existing_indices = [int(line.split('(')[1].split(')')[0]) for line in lines if line.startswith("image")]

    # 6. Find missing indices and generate new entries
    missing_indices = set(image_indices) - set(existing_indices)
    new_lines = [f"image ({i}).jpg" for i in sorted(missing_indices)]

    # 7. Combine old and new entries, then sort them
    all_lines = lines + new_lines
    sorted_lines = sorted(all_lines, key=lambda x: int(x.split('(')[1].split(')')[0]))

    # 8. Write the sorted entries back to the text file
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in sorted_lines:
            file.write(f"{line}\n")

    print(f"Update and sorting complete for {output_file}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total entries in the text file: {len(sorted_lines)}")


def process_directories(base_dirs):
    """
    Process Train, Val, and Test directories to update their respective text files.
    :param base_dirs: List of base directories for Train, Val, and Test
    """
    for base_dir in base_dirs:
        image_dir = os.path.join(base_dir, "JPEGImages")
        output_file = os.path.join(base_dir, f"{os.path.basename(base_dir).lower()}.txt")

        # Process the directory and update the corresponding text file
        update_and_sort_text_file(image_dir, output_file)


if __name__ == "__main__":
    # Define the base directories for Train, Val, and Test
    base_dirs = [
        'data/Train/Train',
        'data/Val/Val',
        'data/Test/Test'
    ]
    process_directories(base_dirs)
