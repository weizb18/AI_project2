import torch
from torchvision import transforms
import numpy as np
from cv2 import cv2
from net import *

Dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

def process_image(cv2Model, deepModel, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    location = cv2Model.detectMultiScale(gray_image, 1.1, 3)
    for (x, y, w, h) in location:
        face = gray_image[y:y+h, x:x+w]
        face = cv2.resize(image, (48,48))
        face = transforms.ToTensor()(face)
        face = torch.unsqueeze(face, 0)
        output = deepModel(face)
        predict = torch.argmax(output, 1)
        predict_label = Dict[predict.item()]
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, predict_label, (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    return image

if __name__ == '__main__':
    
    cv2Model = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    deepModel = torch.load('./checkpoint_epoch_44.pth', map_location=torch.device('cpu'))
    
    image = cv2.imread('./image.jpg')
    image = process_image(cv2Model, deepModel, image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # capture = cv2.VideoCapture('./video.mp4')
    # while capture.isOpened():
    #     ret, frame = capture.read()
    #     image = process_image(cv2Model, deepModel, frame)
    #     cv2.imshow('video', image)
    #     if cv2.waitKey(30) > 0:
    #         break
    # capture.release()
    # cv2.destroyAllWindows()
    