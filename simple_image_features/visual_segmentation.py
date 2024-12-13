import numpy as np
from skimage import segmentation, color
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

image_directory = os.getenv('IMAGE_DIR')

def segment_and_view_image(image_path, n_segments=100):
    """
    Segment an image using SLIC superpixels and display both original and segmented images.
    
    Args:
        image_path (str): Path to the input image
        n_segments (int): Number of segments to create (default: 100)
    """
    # Read the image
    img = plt.imread(image_path)
    
    # Convert to float if needed
    if img.dtype == np.uint8:
        img = img.astype('float') / 255

    # Apply SLIC segmentation
    segments = segmentation.slic(img, n_segments=n_segments, compactness=10, sigma=1)
    
    # Create segmented image
    segmented_img = color.label2rgb(segments, img, kind='avg')
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmented_img)
    ax2.set_title(f'Segmented Image ({n_segments} segments)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "cross_jane.jpg"
    image_path = f"{image_directory}/{image_path}"
    segment_and_view_image(image_path, n_segments=80)
