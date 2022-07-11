import random
import json
import torch
import cv2
import numpy as np
import os
from Brain import neuralnetwork
from Neural import bag_of_words ,tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("emotions.json","r") as json_data:
    intents = json.load(json_data)

FILE = "TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = neuralnetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#########################-->Neuro starts here
Name = "Neuro"

from Listen import Listen
from Speak import Say
from task import noninput
from task import inputfun 
import pyautogui as p
def Main():
    p.press('esc')
    Say("verification successful")
    Say("welcome back Likhith sir")
    sentence = Listen()
    result = str(sentence)

    if sentence == "bye":
            Say("thanks for using me sir, have a good day.")
            Say("Neuro, powering off")
            exit()
    
    sentence = tokenize(sentence)
    x = bag_of_words(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)

    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
            
                if "time" in reply:
                  noninput(reply)

                elif "date" in reply:
                  noninput(reply)
            
                elif "day" in reply:
                  noninput(reply)
                
                elif "volume up" in reply:
                  noninput(reply)
                
                elif "volume down" in reply:
                  noninput(reply)
                
                elif "volume mute" in reply:
                  noninput(reply)
                  
                elif "wikipedia" in reply:
                  inputfun(reply,sentence)

                elif "google" in reply:
                  inputfun(reply,result)
                
                elif "youtube" in reply:
                    inputfun(reply,result)
                
                elif "news" in reply:
                    inputfun(reply,result)
                
                elif "game" in reply:
                    inputfun(reply,result)
                
                elif "weather" in reply:
                    inputfun(reply,result)
                
                elif "system info" in reply:
                    inputfun(reply,result)
                    
                else:
                  Say(reply)


#######################################
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadepath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadepath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 2

names = ['','Likhith']

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


while True:

    ret, img = cam.read()

    converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        converted_image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w])

        if (accuracy < 100):
          id = names[id]
          accuracy="  {0}%".format(round(100 - accuracy))
          Main()
          
        else:
           id ="unknown"
           accuracy="  {0}%".format(round(100  - accuracy))
    
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("Thanks for using this program, have a good day")
cam.release()
cv2.destroyAllWindows()