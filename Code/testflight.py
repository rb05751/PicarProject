import picar
from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from tflite_runtime.interpreter import Interpreter
#...


model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_1228_new.h5')
cam = cv2.VideoCapture(-1)
cv2.namedWindow("test")


def img_preprocess(img):
  img = np.array(img, dtype = np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = np.array(cv2.resize(img, (320,120)) / 255)
  img = np.expand_dims(img, axis = 0)
  return img


def cam_prep(preprocessed_img):
    img = np.squeeze(np.array(preprocessed_img))
    img = img * 255
    img = img.astype(int)
    return img
    

def remote():
    i = 1
    j = 201
    while i != 0:
        for i in range(100):
            _,_ = cam.read()
        
        #Start Program
        sleep(1)
        angles = []
        picar.setup()
        bw = back_wheels.Back_Wheels()
        fw = front_wheels.Front_Wheels()
        motor_speed = 40
        bw.speed = motor_speed
        bw.backward()
        angle = int(input("angle:"))
        if angle == 7:
            _, img = cam.read()
            angle = 70
            fw.turn(angle)
            cv2.imwrite(f"/home/pi/PicarProject/Code/Images/NewImages/image{j}.jpg", img)
            angles.append(angle)
            j += 1
        elif angle == 8:
            _, img = cam.read()
            angle = 80
            fw.turn(angle)
            cv2.imwrite(f"/home/pi/PicarProject/Code/Images/NewImages/image{j}.jpg", img)
            angles.append(angle)
            j += 1
        elif angle == 9:
            angle = 100
            fw.turn(angle)
            _, img = cam.read()
            cv2.imwrite(f"/home/pi/PicarProject/Code/Images/NewImages/image{j}.jpg", img)
            j += 1
        elif angle == 0:
            motor_speed = 0
            bw.stop()
            stop()
            i = 0
            

def main():
    picar.setup()
    #global model
    model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_1228_new.h5')
    
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    angles = np.array(list(range(55,105,5)))
    
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(cam.get(3),cam.get(4))
    
    i = 0
    while cam.isOpened():
        for i in range(100):
            _,_ = cam.read()
            #sleep(1)
        motor_speed = 33
        start = time.time()
        _, img_off_cam = cam.read()
        b,g,r = cv2.split(img_off_cam)
        img = cv2.merge([r,g,b])
        img = img_preprocess(img)
        #cp_img = cam_prep(img)
        print(img.shape)
        steering_angle = model.predict(img)[0]
        steering_angle = int(np.dot(angles,steering_angle))
        if steering_angle > 90:
            steering_angle = steering_angle + (steering_angle - 89)
        #cv2.imwrite(f"testimage{i}_angle={steering_angle}.jpg", cp_img)
        print(steering_angle)
        bw.speed = motor_speed
        bw.backward()
        fw.turn(steering_angle)
        print(f"Execution time: {time.time() - start} seconds")
        i += 1

def stop():
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    bw.stop()

def save_and_take_image():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cam.isOpened():
    #cv2.namedWindow("test")
        for i in range(100):
            _,_ = cam.read()
            
        _,img_off_cam = cam.read()
        print(img_off_cam.shape)
        cv2.imwrite('ObjectDetectionImage1.jpg', img_off_cam)


def test_cam():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(cam.get(3),cam.get(4))
    if cam.isOpened():
    #cv2.namedWindow("test")
        for i in range(100):
            _,_ = cam.read()
            
        _,img_off_cam = cam.read()
        print(img_off_cam.shape)
        cv2.imwrite('testimage1.jpg', img_off_cam)
        b,g,r = cv2.split(img_off_cam)
        img_off_cam = cv2.merge([r,g,b])
        img = img_preprocess(img_off_cam)
        img = cam_prep(img)
        cv2.imwrite('testimage2.jpg', img)
        print(img.shape)#This gets returned as BGR, I think this is okay since OG video was taken with openCV
        #print(img)
        img_mod = img_preprocess(img_off_cam)
        angle = model.predict(img_mod)[0]
        angle_list = np.array(list(range(55,105,5)))
        angle = np.dot(angle, angle_list)
        print(angle)
    else:
        print("Could not run")
        
        
def testmodel():


    img = cv2.imread('/home/pi/PicarProject/Code/testimage1.jpg')
    img = img_preprocess(img)
    print(img.shape)
    angles = np.array(list(range(55,105,5)))
    
 
    start = time.time()
    steering_angle = model.predict(img)[0]
    print(type(img))
    print(time.time()-start)
    steering_angle = int(np.dot(angles,steering_angle))
    print(steering_angle)
    
def turn():
    picar.setup()
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    for i in range(10):
        sleep(2)
        motor_speed = 28
        bw.speed = motor_speed
        bw.backward()
        if i%2 == 0:
            fw.turn(110)
        else:
            fw.turn(60)
            
def save_and_take_image():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cam.isOpened():
    #cv2.namedWindow("test")
        for i in range(100):
            _,_ = cam.read()
            
        _,img_off_cam = cam.read()
        print(img_off_cam.shape)
        cv2.imwrite('ObjectDetectionImage50.jpg', img_off_cam)
        
if __name__ == '__main__':
    try:
        save_and_take_image()
    except KeyboardInterrupt:
        destroy()
