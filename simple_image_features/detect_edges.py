import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

image_directory = os.getenv('IMAGE_DIR')

def edge_detection(blurred_image:any, detection_method:str = "canny") -> np.ndarray:
    if detection_method == "canny":
        edges = cv2.Canny(blurred_image, 50, 150)
    elif detection_method == "sobel":
        sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)
    elif detection_method == "laplacian":
        edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
        edges = np.uint8(edges)
    elif detection_method == "prewitt":
        prewittx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        prewitty = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(prewittx, prewitty)
        edges = np.uint8(edges)
    elif detection_method == "roberts":
        robertsx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        robertsy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(robertsx, robertsy)
        edges = np.uint8(edges)
    elif detection_method == "scharr":
        scharrx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        scharry = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(scharrx, scharry)
        edges = np.uint8(edges)
    else:
        raise ValueError(f"Unsupported edge detection method: {detection_method}")
    return edges

def edge_trace(image_path, detection_method:str = "canny"):
    """
    Performs edge detection and line tracing on an image from local storage.

    Args:
        image_path: Path to the local image file.

    Returns:
        A NumPy array representing the traced image or None if an error occurs.
    """
    try:
        # Read the image directly from local storage in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # 2. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Perform Canny edge detection
        edges = edge_detection(blurred, detection_method)

        # 4. Find contours (lines) in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Draw the contours on a blank image for visualization
        traced_image = np.zeros_like(img)
        cv2.drawContours(traced_image, contours, -1, (255), 1)

        return traced_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

image_path = "mountains.jpg"
image_path = f"{image_directory}/{image_path}"
detection_method = "canny"
traced_img = edge_trace(image_path, detection_method)

if traced_img is not None:
    # Ensure the directory exists
    output_directory = "traced_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the image
    traced_image_path = f"{output_directory}/trace_{detection_method}_{os.path.basename(image_path)}"
    cv2.imwrite(traced_image_path, traced_img)
    cv2.imshow('Traced Image', traced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
