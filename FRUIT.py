from ultralytics import YOLO
import pygetwindow
import pyautogui as pt
import cv2 as cv
import numpy as np
import torch
from PIL import ImageGrab
device = torch.device("cuda")
model = YOLO('best.pt') #模型选用
window_title = "Fruit Ninja" #填写要获取的窗口名字
window = pygetwindow.getWindowsWithTitle(window_title)[0]
model.to(device)
while True:
    if window:
        x,y,w,h = window.left,window.top,window.width,window.height #窗口坐标
        screenshot = ImageGrab.grab(bbox=[x,y,x + w,y + h]) #截图

        img_scr = cv.cvtColor(np.array(screenshot),cv.COLOR_BGR2RGB)
        size_x,size_y = img_scr.shape[1],img_scr.shape[0]
        img_det = cv.resize(img_scr,(640,640)) #将截图转换成YOLO可以处理的格式

        with torch.no_grad():
            predictions = model(img_det) #对截图进行识别
        for box in predictions[0].boxes:
            boxes = box.xywhn[0].cpu().numpy()
            label = int(box.cls[0].cpu().numpy())

            cv.rectangle(img_scr,(int((boxes[0] - boxes[2]/2) * size_x),int((boxes[1]- boxes[3]/2) * size_y)),(int((boxes[0] + boxes[2]/2) * size_x),int((boxes[1] + boxes[3]/2) * size_y)),color = (255,255,0),thickness = 2)
            #对识别结果进行处理，同时给识别出的物体画框
            if label == 0:
                pt.click(x = x + boxes[0] * size_x,y = y + boxes[1] * size_y)#如果是水果 鼠标点击相应区域
        cv.imshow('frame',img_scr)#展示过程
        if cv.waitKey(1) == ord('q'):#按Q退出程序
            break
        pass

