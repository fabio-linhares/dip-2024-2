🧪 Exercise: Histogram Matching with OpenCV and Scikit-Image
Objective:
Implement a function that performs histogram matching to transfer the visual appearance of a reference image to a source image.
You will output the transformed image and (opptionally) plot the histograms of its RGB channels.

📂 Provided Files:
- ```source.jpg``` – Example image to be transformed.
- ```reference.jpg``` – Example image whose histogram we want to match.
- ```output.jpg``` – Example of expected output.

📌 Instructions:
1. Load the source and reference images.
1. Convert them to the ```RGB``` color space (Obs: If you open an image file using OpenCV, the standard is ```BGR```).
1. Perform histogram matching.
1. (Not included in the test, but recommended); Plot histograms of the original and matched images.