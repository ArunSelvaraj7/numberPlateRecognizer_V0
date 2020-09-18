NumberPlateRecognizer

Vehicle number plate recognition system with darknet-yolo config and weights. After finding the number plates from the image, a image segmentation algorithm is applied to segment the characters on the plate. The segmented character images are sent through a CNN classifier to find the exact character on the segmented image. (The CNN was trained with a simple character dataset) The output of the CNN will be a single character, which is appended together to form the entire vehicle number.


Run Instructions

1) Clone the repo using git clone (Do not download the zip file for usage)
2) pip install -r requirements.txt       #To install the required dependecies
3) python app.py                         #To run the app