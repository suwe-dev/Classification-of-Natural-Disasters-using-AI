import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, url_for
import requests
import json

app=Flask(__name__)


model=load_model("D:/IBM/Natural Disaster Intensity Analysis and Classification/Flask/disaster.h5")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/intro')
def intro():
    return render_template("intro.html")

@app.route('/predict')
def predict():
    video = cv2.VideoCapture(0)
    index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

    while(1):
        success,frame = video.read()
        cv2.imwrite("1.jpg", frame)
        img = image.load_img("1.jpg", target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x),axis=1)
        p = index[pred[0]]
        cv2.putText(frame,"Predicted disaster is " + str(p), (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.imshow("Prediction window ('Press 'Q' to quit')", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detected disaster is: " + str(p))
            break
    video.release()
    cv2.destroyAllWindows()
    return render_template("upload.html", disaster = str(p))


if __name__ == '__main__':
    app.run(debug = True, port=8001)
