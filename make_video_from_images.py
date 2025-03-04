import cv2
import os
import glob
import re

# NOTE: Must load module opencv. using 'module load gcc opencv/4.11.0'
image_path = '12m_gpt2-large_decoding_images'

# Set the path to the folder containing images
image_folder = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/{image_path}"
output_video = f"{image_path}_video.mp4"
frame_rate = 1  # Adjust frame rate as needed
image_files = glob.glob(os.path.join(image_folder, '*.png'))
image_files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

print("Images are in this order: ", image_files)

# Read the first image to get dimensions
if len(image_files) == 0:
    raise ValueError("No images found in the specified folder.")

first_image = cv2.imread(image_files[0])
height, width, layers = first_image.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Add images to video
for img_file in image_files:
    img = cv2.imread(img_file)
    video.write(img)

# Release resources
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_video}")