# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Translation: shifting right by 10 pixels and down by 5 pixels
    def translate(img):
        translated = np.zeros_like(img)
        shift_x, shift_y = 10, 5
        translated[shift_y:, shift_x:] = img[:img.shape[0] - shift_y, :img.shape[1] - shift_x]
        return translated

    # Rotation 90 degrees clockwise
    def rotate_90_clockwise(img):
        return np.rot90(img, k=-1)

    # Horizontal stretching (scaling width by 1.5)
    def stretch(img, scale_x=1.5):
        stretched = np.zeros((img.shape[0], int(img.shape[1] * scale_x)))
        for i in range(img.shape[0]):
            x = np.arange(img.shape[1])
            xp = np.linspace(0, img.shape[1] - 1, stretched.shape[1])
            stretched[i, :] = np.interp(xp, x, img[i, :])
        return stretched

    # Horizontal mirror
    def mirror(img):
        return np.flip(img, axis=1)

    # Simple barrel distortion using a radial function
    def barrel_distort(img):
        # Simple radial distortion
        def map_pixel(x, y, cX, cY, k):
            deltaX = x - cX
            deltaY = y - cY
            dist = np.sqrt(deltaX**2 + deltaY**2)
            factor = 1 + k * (dist / cX)**2
            newX = int(cX + deltaX * factor)
            newY = int(cY + deltaY * factor)
            return newX, newY

        cX, cY = img.shape[1] / 2, img.shape[0] / 2
        k = 0.0005  # Distortion coefficient
        distorted = np.zeros_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                newX, newY = map_pixel(x, y, cX, cY, k)
                if 0 <= newX < img.shape[1] and 0 <= newY < img.shape[0]:
                    distorted[y, x] = img[newY, newX]
        return distorted

    # Applying transformations
    return {
        "translated": translate(img),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch(img),
        "mirrored": mirror(img),
        "distorted": barrel_distort(img)
    }

# Example usage:
# Suppose `img` is your input NumPy array representing a grayscale image.
#img = np.array([
#    [0.0, 0.1, 0.2, 0.3, 0.4],
#    [0.5, 0.6, 0.7, 0.8, 0.9],
#    [1.0, 0.9, 0.8, 0.7, 0.6],
#    [0.5, 0.4, 0.3, 0.2, 0.1],
#    [0.0, 0.1, 0.2, 0.3, 0.4]
#])
# You can call the function as follows:
#transformations = apply_geometric_transformations(img)
#print(transformations) 