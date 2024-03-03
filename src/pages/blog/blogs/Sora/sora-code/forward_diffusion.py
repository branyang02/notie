import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Forward diffusion process

def forward_diffusion(image, beta_schedule):
T = len(beta_schedule)
diffused_images = [image]
for t in range(T):
beta = beta_schedule[t]
noise = np.random.normal(0, np.sqrt(beta), image.shape)
image = np.sqrt(1 - beta) \* image + noise
diffused_images.append(image)
return diffused_images

# Function to plot all diffused images

def plot*diffused_images(diffused_images, images_per_row=10):
n = len(diffused_images)
rows = (n + images_per_row - 1) // images_per_row
fig, axes = plt.subplots(
rows, images_per_row, figsize=(images_per_row * 3, rows \_ 3)
)
for i, image in enumerate(diffused_images):
r, c = divmod(i, images_per_row)
if n <= images_per_row:
ax = axes[c]
else:
ax = axes[r, c]
ax.imshow(image, cmap="gray")
ax.axis("off")
plt.tight_layout()
return fig

# Initialize

image_9 = np.array(
[
[0, 0, 0, 255, 255, 255, 255, 255, 0, 0],
[0, 0, 255, 0, 0, 0, 0, 255, 0, 0],
[0, 255, 0, 0, 0, 0, 0, 255, 0, 0],
[0, 255, 0, 0, 0, 0, 0, 255, 0, 0],
[0, 0, 255, 255, 255, 255, 255, 255, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
[0, 255, 255, 255, 255, 255, 255, 255, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
)

if **name** == "**main**":
T = 109
beta_schedule = np.linspace(0.1, 0.5, T)

    # Diffuse
    diffused_images = forward_diffusion(image_9, beta_schedule)

    fig = plot_diffused_images(diffused_images)

    # get_image is a predefined function to display a matplotlib figure
    get_image(fig)
