#PRECODING
#pip install tensorflow
#pip install keras
#pip install googletrans
#pip install opencv-python

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from googletrans import Translator

#packages for GUI
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image

#varaibles
fileName = ""
image_address = ""
content = ""



#function for change language
def change_option(*args):

    # selected element
    select_lang = LANGCODES[var.get()]
    print(LANGCODES[var.get()])
    #translator code
    translator = Translator()

    txt = translator.translate(content, dest = select_lang)
    print(txt)
    out = tk.Label(root,
              justify=tk.CENTER,
              padx = 10,
              text=txt.text).pack()

#file dialog f for image accusation

def OpenFile():
        name = askopenfilename(initialdir = "/",
                                   title = "Select file",
                                   filetypes = (("jpeg files","*.jpg *.png"),("all files","*.*"))
                                   )
        
        global fileName
        fileName = name
        global image_address 
        image_address = fileName
        try:
                img = ImageTk.PhotoImage(Image.open(fileName))
                p2 = tk.Label(root, image = img).pack()
        except Exception :
                pass
                
        
        #detection code start from here

	#constant variables for text detection
        min_confidence = 0.5
        width = 320
        height = 320
        
        # load the input image
        image = cv2.imread(image_address)
        orig = image.copy()
        (H, W) = image.shape[:2]

        # set new width and height
        (newW, newH) = (width, height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        #the first layer is the output probabilities
        #the second layer is used to derive the bounding box coordinates of text
        layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet('frozen_east_text_detection.pb')

        # construct a blob from the image
        # create the model with the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # get rows and coloums from score
        #intialize rectangles gor bounding boxs and confidence score
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over rows
        for y in range(0, numRows):
                # get the score data and dimensions of the rectangle
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over columns
                for x in range(0, numCols):
                        # score not confident
                        if scoresData[x] < min_confidence:
                                continue

                        # compute the offset factor
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)

                        # get rotation angle and make sin and cosine
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)

                        # width and height of the bounding box
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]

                        # compute x and y coordinates of the bounding box
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)

                        # add the rectangles and score made
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scoresData[x])

        # apply non_max_suppression
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        #sorting of text arrangement
        def func(val):
            return val[1]

        boxes = sorted(boxes, key=func)

	#function to find if r1 rectangle contains r2 rectangle
        def contains(r1, r2):
           return (r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2]) and (r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3])
	
	#list to contain all text in character form
    #eg crop_imag = [[H,E,L,L,O][W,O,R,L,D]]
        crop_img = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            
            #croping of text from original image
            unit = orig[startY:endY, startX:endX]
            
            #changing cropped image to grayscale image
            gray_img = cv2.cvtColor(unit, cv2.COLOR_BGR2GRAY)
                
            #finding threshold value from the grayscale average value
            threshold = np.mean(gray_img)
            #getting the binary image with the help of thresholding
            _, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
            #diliation and erosion
#            thresh = cv2.dilate(thresh, kernel, iterations=1) 
#            thresh = cv2.erode(thresh, kernel, iterations=1) 
            
            #Finding contours in the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #list of contour polygon and bounding rectangle
            contours_poly = [None]*len(contours)
            boundRect = [None]*len(contours)
            
            #looping in the contour list
            for i, c in enumerate(contours):
                #making polygons of the contours
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                #making rectangles from that polygon
                # returns x,y,h,w of rectangle in a list
                boundRect[i] = cv2.boundingRect(contours_poly[i])
            
            #sorting the rectangle boxes into ascending order of x
            boundRect = sorted(boundRect, key= lambda x:x[0])
            #list to crop character image from text
            char_crop = []
            
            #cropping the rectangles that are not in another rectangle
            for i in range(len(boundRect)):
                count = 1
                for j in range(len(boundRect)):
                    if not i==j:
                        if not contains(boundRect[j], boundRect[i]):
                            count +=1
                            if count == len(boundRect):
                                char_crop.append(unit[boundRect[i][1]:boundRect[i][1]+boundRect[i][3], boundRect[i][0]:boundRect[i][0]+boundRect[i][2]])
                                
            #adding the cropped characters to the final list
            crop_img.append(char_crop)

        #importing keras files to load trained model
        from keras.models import model_from_json

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
         
        # Compile model
        loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        #funtion to return the predicition location that is equal to 1
        def result(array):
            rt = 26
            for i in range(0, len(array[0])):
                if array[0][i] == 1:
                    rt = i
            return rt
        
        #prediction list as the model was trained
        predict_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','']
        global content
        content = ''
        
        #predicting every character in the text in the crop_img
        for text_img in crop_img:
            for image in text_img:
                image = cv2.resize(image,(32,32))
                image = np.expand_dims(image, axis = 0)
                rslt = result(loaded_model.predict(image))
                content += predict_list[rslt]
            content += ' '
                
        print(content)

        #language codes
        LANGUAGES = {
            'af': 'afrikaans',
            'sq': 'albanian',
            'am': 'amharic',
            'ar': 'arabic',
            'hy': 'armenian',
            'az': 'azerbaijani',
            'eu': 'basque',
            'be': 'belarusian',
            'bn': 'bengali',
            'bs': 'bosnian',
            'bg': 'bulgarian',
            'ca': 'catalan',
            'ceb': 'cebuano',
            'ny': 'chichewa',
            'zh-cn': 'chinese (simplified)',
            'zh-tw': 'chinese (traditional)',
            'co': 'corsican',
            'hr': 'croatian',
            'cs': 'czech',
            'da': 'danish',
            'nl': 'dutch',
            'en': 'english',
            'eo': 'esperanto',
            'et': 'estonian',
            'tl': 'filipino',
            'fi': 'finnish',
            'fr': 'french',
            'fy': 'frisian',
            'gl': 'galician',
            'ka': 'georgian',
            'de': 'german',
            'el': 'greek',
            'gu': 'gujarati',
            'ht': 'haitian creole',
            'ha': 'hausa',
            'haw': 'hawaiian',
            'iw': 'hebrew',
            'hi': 'hindi',
            'hmn': 'hmong',
            'hu': 'hungarian',
            'is': 'icelandic',
            'ig': 'igbo',
            'id': 'indonesian',
            'ga': 'irish',
            'it': 'italian',
            'ja': 'japanese',
            'jw': 'javanese',
            'kn': 'kannada',
            'kk': 'kazakh',
            'km': 'khmer',
            'ko': 'korean',
            'ku': 'kurdish (kurmanji)',
            'ky': 'kyrgyz',
            'lo': 'lao',
            'la': 'latin',
            'lv': 'latvian',
            'lt': 'lithuanian',
            'lb': 'luxembourgish',
            'mk': 'macedonian',
            'mg': 'malagasy',
            'ms': 'malay',
            'ml': 'malayalam',
            'mt': 'maltese',
            'mi': 'maori',
            'mr': 'marathi',
            'mn': 'mongolian',
            'my': 'myanmar (burmese)',
            'ne': 'nepali',
            'no': 'norwegian',
            'ps': 'pashto',
            'fa': 'persian',
            'pl': 'polish',
            'pt': 'portuguese',
            'pa': 'punjabi',
            'ro': 'romanian',
            'ru': 'russian',
            'sm': 'samoan',
            'gd': 'scots gaelic',
            'sr': 'serbian',
            'st': 'sesotho',
            'sn': 'shona',
            'sd': 'sindhi',
            'si': 'sinhala',
            'sk': 'slovak',
            'sl': 'slovenian',
            'so': 'somali',
            'es': 'spanish',
            'su': 'sundanese',
            'sw': 'swahili',
            'sv': 'swedish',
            'tg': 'tajik',
            'ta': 'tamil',
            'te': 'telugu',
            'th': 'thai',
            'tr': 'turkish',
            'uk': 'ukrainian',
            'ur': 'urdu',
            'uz': 'uzbek',
            'vi': 'vietnamese',
            'cy': 'welsh',
            'xh': 'xhosa',
            'yi': 'yiddish',
            'yo': 'yoruba',
            'zu': 'zulu',
            'fil': 'Filipino',
            'he': 'Hebrew'
        }
        global LANGCODES
        LANGCODES = dict(map(reversed, LANGUAGES.items()))

        keys = sorted(LANGCODES)
        global var
        var = StringVar(root)
        var.set('Choose an option')

        option = OptionMenu(root, var, *keys, command=change_option)
        option.pack()
                
                

        
        
    

#root 
root = tk.Tk()
root.title("Translator")
root.geometry("500x500")
root.configure(background='white')


text = """Upload an Image for you want
to translate the text"""
w2 = tk.Label(root,
              justify=tk.CENTER,
              padx = 10,
              text=text).pack()

slogan = tk.Button(root,
                   text="Upload Image",
                   command=OpenFile).pack()

root.mainloop()

