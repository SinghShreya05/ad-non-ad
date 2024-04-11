import os
import shutil

def copy_images(source_dir, target_dir):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

    count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                source_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)
                # target_subdir = os.path.join(target_dir, relative_path)

                # if not os.path.exists(target_subdir):
                #     os.makedirs(target_subdir)

                target_file_path = os.path.join(target_dir, f"{count}.png")

                shutil.copy(source_file_path, target_file_path)
                count += 1

    print(
        f"Total {count} images copied to {target_dir} preserving subdirectory structure."
    )

source_directory = "Ads"
target_directory = "dataset/train/class1"
copy_images(source_directory, target_directory)
