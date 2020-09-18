import cv2

from utility.utilityFunctions import postprocess
from utility.utilityFunctions import getOutputsNames
from utility.utilityFunctions import find_contours
from utility.utilityFunctions import segment_characters
from utility.utilityFunctions import fix_dimension
import os

import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def extract_plate(img): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()

    #Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier(r'indian_license_plate.xml')

    # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)
    plate = None
    
    for (x,y,w,h) in plate_rect:
        plate = plate_img[y:y+h, x:x+w, :]
        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
#     print(plate)
    return plate_img, plate

def detect_plate(net , charModel, filename):
    UPLOAD_FOLDER = r'static/images'
    custom_config = r'-c tessedit_char_whitelist=QWERTYUIOPLKJHGFDSAZXCVBNM1234567890 --psm 6'
    inpWidth = 416  # 608     # Width of network's input image
    inpHeight = 416  # 608     # Height of network's input image
    img = os.path.join(UPLOAD_FOLDER,filename)
    frame = cv2.imread(img)

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    plate = postprocess(frame, outs)

    
    # Write the frame with the detection boxes
    if plate is None:
        frame, plate = extract_plate(frame)
    if plate is not None:
        cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename),frame)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        # Perform text extraction
        data = pytesseract.image_to_string(gray, lang='eng', config=custom_config)
        data = re.sub(r'\W+', '', data)
        output = data.replace('_', "")

        if (len(output)< 8):
        
            char = segment_characters(plate)
            dic = {}
            characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i,c in enumerate(characters):
                dic[i] = c

            output = []
            for i,ch in enumerate(char): #iterating over the characters
                img_ = cv2.resize(ch, (28,28))
                img = fix_dimension(img_)
                img = img.reshape(1,28,28,3) #preparing image for the model
                y_ = charModel.predict_classes(img)[0] #predicting the class

                character = dic[y_] 
                output.append(character) #storing the result in a list
        
        c = ''.join(output)
        
        if(len(c)>1):
            plate_number= []
            for i,char in enumerate(c):
                if (char.isalpha()):
                    plate_number.append(char)
                else:
                    break
            # c = ''.join(c)
            if len(plate_number)>0:
                if plate_number[-1] == 'O':
                    plate_number = plate_number[-3:]
                else:
                    plate_number = plate_number[-2:]

            for char in c[i:]:
                plate_number.append(char)
            plate_number = ''.join(plate_number)

        else:
            plate_number = ''

        print(plate_number)
        return(plate_number)
    else:
        return ''