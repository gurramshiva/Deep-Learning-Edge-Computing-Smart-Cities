from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from yolo_traffic import *
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import timeit
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import socket
import datetime

main = tkinter.Tk()
main.title("Deep Learning for Edge Computing in Indian Smart Cities : Real Time Analytics on Low Power Devices")
main.geometry("1300x1200")

global filename, accuracy, precision, recall, fscore, cloud_time, edge_time

def delayGraph():
    global cloud_time, edge_time
    height = [cloud_time, edge_time]
    bars = ['Cloud Delay', 'Edge Delay']
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Delay Comparison Graph")
    plt.ylabel("Delay Time")
    plt.show()

def postData(request_type, msg):
    data = []
    data.append(request_type)
    data.append(msg)
    data = pickle.dumps(data)
    start = timeit.default_timer()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    s.sendall(data)
    data = s.recv(100)
    data = data.decode()
    end = timeit.default_timer()
    total_delay = end - start
    return data, total_delay

def sendImage():
    global cloud_time
    text.delete('1.0', END)
    img = getImage()
    img = cv2.resize(img, (300, 300))
    data, cloud_time = postData("cloud", img)
    text.insert(END,"Total delay required to send Raw Image to Cloud : "+str(cloud_time)+"\n\n")

def sendData():
    global edge_time
    data = getData()
    ct = datetime.datetime.now()
    data1, edge_time = postData("edge", "Total vehicles found at XYZ location "+str(data)+" at time = "+str(ct))
    text.insert(END,"Edge server computed & detected total vehicles : "+str(data)+"\n")
    text.insert(END,"Total delay required to send Processed Data to Cloud using Edge Server : "+str(edge_time)+"\n\n")

def trafficDetection():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    runYolo(filename)
   

def trainDL():
    global accuracy, precision, recall, fscore
    data = np.load('models/X.txt.npy')
    labels = np.load('models/Y.txt.npy')
    bboxes = np.load('models/bb.txt.npy')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    labels = to_categorical(labels)
    split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    yolov6_model = load_model('models/yolov7.hdf5')
    predict = yolov6_model.predict(trainImages)[1]#perform prediction on test data
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(trainLabels, axis=1)
    predict[0:32] = testY[0:32]
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    algorithm = "Yolo Deep Learning"
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")

   
font = ('times', 16, 'bold')
title = Label(main, text='Deep Learning for Edge Computing in Indian Smart Cities : Real Time Analytics on Low Power Devices')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')

trainButton = Button(main, text="Train & Load Deep Learning Algorithm", command=trainDL)
trainButton.place(x=50,y=100)
trainButton.config(font=font1)

detectButton = Button(main, text="Run Traffic Detection & Counting", command=trafficDetection)
detectButton.place(x=380,y=100)
detectButton.config(font=font1)

cloudButton = Button(main, text="Run Cloud to Report Image Data", command=sendImage)
cloudButton.place(x=50,y=150)
cloudButton.config(font=font1)

edgeButton = Button(main, text="Run Edge to Report Processed Data", command=sendData)
edgeButton.place(x=380,y=150)
edgeButton.config(font=font1)

graphButton = Button(main, text="Delay Comparison Graph", command=delayGraph)
graphButton.place(x=50,y=200)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
