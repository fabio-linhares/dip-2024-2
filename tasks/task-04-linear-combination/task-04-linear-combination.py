import cv2
import numpy as np

def linear_combination(i1: np.ndarray, i2: np.ndarray, a1: float, a2: float) -> np.ndarray:
    """
    Compute the linear combination of two images using OpenCV: 
    i_out = a1 * i1 + a2 * i2.

    Args:
        i1 (np.ndarray): First input image.
        i2 (np.ndarray): Second input image.
        a1 (float): Scalar weight for the first image.
        a2 (float): Scalar weight for the second image.

    Returns:
        np.ndarray: The resulting image with the same dtype as the input images.
    """
    # Ensure images have the same dimensions
    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions.")

    ### START CODE HERE ###
    ### TODO
    ### END CODE HERE ###

    return None

# Example Usage
if __name__ == "__main__":
    # Load images
    i1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
    i2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

    if i1 is None or i2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

    # Define scalars
    a1, a2 = 0.6, 0.4

    # Compute the linear combination
    output = linear_combination(i1, i2, a1, a2)