import time
import argparse
import cv2
import pickle
import tensorflow
#import keras
import numpy as np
import gtts
import matplotlib.pyplot as plt
import os
from gtts import gTTS
from playsound import playsound
x_npy = np.load("C:/Users/Padmashree/Documents/ML_projects/DSP/X.npy")
y_npy = np.load("C:/Users/Padmashree/Documents/ML_projects/DSP/Y.npy")

#plt.imshow(x_npy[0].reshape(64, 64))
#plt.show()

class YOLO:

    def _init_(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size

        self.labels = labels
        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
        except:
            raise ValueError("Couldn't find the models!\nDid you forget to download them manually (and keep in the correct directory, models/) or run the shell script?")

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image):
        ih, iw = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        inference_time = end - start

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

        return iw, ih, inference_time, results


from yolo import YOLO
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="prn", help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()
model = tensorflow.keras.models.load_model('cnn_tensor_800')

args.network == "prn"
yolo = YOLO("cross-hands-tiny-prn.cfg", "cross-hands-tiny-prn.weights", ["hand"])


yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    time.sleep(0.5)
    backup = frame
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        try:
            # draw a bounding box rectangle and label on the image
            #color = (0, 255, 255)

            #cv2.rectangle(frame, (x-5, y-30), (x + w + 10, y + h + 10), 2)
            #text = "%s (%s)" % (name, round(confidence, 2))
            hand = backup[y - 50:y + h + 50, x - 20:x + w + 25]
            #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, color, 2)
            hand = cv2.resize(hand,(64,64))
            time.sleep(0.5)
            plt.imshow(hand)
            plt.show()
            hand_gray = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
            hand_gray = 1 - np.array(hand_gray).astype('float32') / 255.
            hand_gray = hand_gray.reshape(-1, 64, 64, 1)
            print(hand_gray.shape)

            prediction = model.predict(hand_gray)
            print(prediction)
            if (max(prediction[0])>0.57):
                digit = np.where(prediction[0]==max(prediction[0]))
                print(digit[0][0])
                myobj = gTTS(text=str(digit[0][0]), lang='en', slow=True)
                myobj.save("digit.mp3")
                os.system('digit.mp3')
        except:
            pass
    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()