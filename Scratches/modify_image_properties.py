import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the saved image
image_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/9m_gpt2-large_decoding_images/2025_Mar_9m-gpt2-large_to_vectors_fixed_seed_layer_1.png"  # Change this
output_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/9m_gpt2-large_decoding_images/test.png"

img = mpimg.imread(image_path)

# Create a new figure and overlay text
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
ax.axis("off")  # Hide axis if not needed

# Add new title (caption)
new_title = "Updated Caption Here"
plt.title(new_title, fontsize=14, fontweight="bold")

# Save the updated image
plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2)

print(f"Updated image saved as {output_path}")
