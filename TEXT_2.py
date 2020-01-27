# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

min_confidence = 0.5
width = 320
height = 320

image_address = input('Enter image address: \n')

# load the input image and grab the image dimensions
image = cv2.imread(image_address)
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (width, height)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < min_confidence:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

#sorting of text arrangement
def func(val):
    return val[1]

boxes = sorted(boxes, key=func)

def contains(r1, r2):
   return (r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2]) and (r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3])

crop_img = []

average_colr = [0,0,0]

#blank canvas of image size
#drawing = np.zeros((orig.shape[0], orig.shape[1], 3), np.uint8)
#drawing.fill(255)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    
    unit = orig[startY:endY, startX:endX]
    
    avrg_colr = (np.mean(unit[0]),np.mean(unit[0]),np.mean(unit[0]))
    
    if average_colr == (0,0,0):
        average_colr = avrg_colr
    else:
        average_colr = (np.mean([average_colr[0],avrg_colr[0]]),np.mean([average_colr[1],avrg_colr[1]]),np.mean([average_colr[1],avrg_colr[1]]))
    
#    gray_img = cv2.cvtColor(unit, cv2.COLOR_BGR2GRAY)
#    threshold = np.mean(gray_img)
#    _, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)   
#
#    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    
#    contours_poly = [None]*len(contours)
#    boundRect = [None]*len(contours)
#    for i, c in enumerate(contours):
#        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#        boundRect[i] = cv2.boundingRect(contours_poly[i])
#    
#    boundRect = sorted(boundRect, key= lambda x:x[0])
#    char_crop = []
#    for i in range(len(boundRect)):
#        count = 1
#        for j in range(len(boundRect)):
#            if not i==j:
#                if not contains(boundRect[j], boundRect[i]):
#                    count +=1
#                    if count == len(boundRect):
#                        char_crop.append(unit[boundRect[i][1]:boundRect[i][1]+boundRect[i][3], boundRect[i][0]:boundRect[i][0]+boundRect[i][2]])
    crop_img.append(unit)                    
    
drawing = np.zeros((orig.shape[0], orig.shape[1], 3), np.uint8)
drawing = average_colr

i=0
for (startX, startY, endX, endY) in boxes:
    drawing[startY:endY, startX:endX] = crop_img[i]
    i+=1
cv2.imshow('image', drawing)
cv2.waitKey(0)
#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
#
#text = pytesseract.image_to_string(drawing)
#print(text)
#i=0
#for text in crop_img:
#        cv2.imshow('img'+str(i), text)
#        i+=1
#        cv2.waitKey(0)

#from keras.models import model_from_json
#
## load json and create model
#start = time.time()
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#end = time.time()
#print("[INFO] Loaded model from disk in {:.6f} seconds".format(end - start))
# 
## Compile model
#loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
#def result(array):
#    rt = 26
#    for i in range(0, len(array[0])):
#        if array[0][i] == 1:
#            rt = i
#    return rt
#
#predict_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
#                'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','']
#content = ''
#for text_img in crop_img:
#    for image in text_img:
#        image = cv2.resize(image,(32,32))
#        image = np.expand_dims(image, axis = 0)
#        rslt = result(loaded_model.predict(image))
#        content += predict_list[rslt]
#    content += ' '
#        
#print(content)
#from googletrans import Translator
#translator = Translator()
#text = translator.translate(content, dest = 'hi')
#print(text.text)
