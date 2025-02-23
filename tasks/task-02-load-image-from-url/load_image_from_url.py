import argparse
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """

    ### START CODE HERE ###
    image_link = cv.VideoCapture(url)

    if not image_link.isOpened():
        image_link.release() 
        raise Exception(f"Failed to open URL: {url}. Ensure the URL is valid and the resource is accessible.")

    ret, image = image_link.read()
    image_link.release()

    if not ret or image is None:
        raise Exception(f"Failed to load image from URL: {url}. Check the URL, network connection, and image format.")

    img_format = kwargs.get('img_format', '.jpg')
    
    if 'flags' in kwargs:
        try:
            _, img_encoded = cv.imencode(img_format, image)
            decoded_image = cv.imdecode(img_encoded, kwargs['flags'])
            
            if decoded_image is None:
                raise Exception("Failed to decode image with the specified flags. Ensure the flags are compatible with the image format.")

            image = decoded_image 
        except cv.error as e:
            raise Exception(f"OpenCV error during image decoding: {str(e)}. Review the OpenCV documentation for flag compatibility and image format support.")

    if kwargs.get('save_image', False):
        save_path = kwargs.get('save_path', 'output_image.jpg')
        try:
            cv.imwrite(save_path, image)
            print(f"Image saved successfully as {save_path}")
        except Exception as e:
            print(f"Failed to save image: {str(e)}")
            
    ### END CODE HERE ###
        
    return image

load_image_from_url()