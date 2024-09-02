from PIL import Image
import os

# Set the input and output folder paths and the desired size
input_folder = ""
output_folder = ""
desired_size = (256, 256)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_files = os.listdir(input_folder)

if not input_files:
    print(f"No files found in the input folder: {input_folder}")
else:
    print(f"Found {len(input_files)} files in the input folder.")

for filename in input_files:
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        try:
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            image = image.resize(desired_size, Image.Resampling.LANCZOS)
            output_path = os.path.join(output_folder, filename)
            image.save(output_path)
            print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
print("Image resizing complete.")
