from PIL import Image
import os

# Set the input and output folder paths and the desired size
input_folder = ""
output_folder = ""
desired_size = (256, 256)

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
input_files = os.listdir(input_folder)

# Check if any files are found
if not input_files:
    print(f"No files found in the input folder: {input_folder}")
else:
    print(f"Found {len(input_files)} files in the input folder.")

for filename in input_files:
    # Check if the file is an image (you can add more file extensions as needed)
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        try:
            # Open the image using Pillow
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Resize the image to the desired size
            image = image.resize(desired_size, Image.Resampling.LANCZOS)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, filename)
            image.save(output_path)

            print(f"Resized and saved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Image resizing complete.")
