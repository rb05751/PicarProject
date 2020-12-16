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
#...


model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_1215_new.h5')
cam = cv2.VideoCapture(-1)
cv2.namedWindow("test")

def img_preprocess(img):
  img = np.array(img, dtype = np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = cv2.resize(img, (320,240)) / 255
  img = img[125:,:,:]
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
    global model
    
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print(cam.get(3),cam.get(4))
    i = 0
    while cam.isOpened():
        for i in range(100):
            _,_ = cam.read()
            #sleep(1)
        motor_speed = 28
        _, img = cam.read()#This gets returned as BGR, I think this is okay since OG video was taken with openCV
            #img = img_preprocess(img)
        img = img_preprocess(img)
        cp_img = cam_prep(img)
        print(img.shape)
        steering_angle = int(model.predict(img)[0][0])
        cv2.imwrite(f"testimage{i}_angle={steering_angle}.jpg", cp_img)
            #steering_angle = int((steering_angle * .001) + 40)
        print(steering_angle)
        bw.speed = motor_speed
        bw.backward()
        fw.turn(steering_angle)
        i += 1

        #bw.speed = motor_speed
        #bw.forward()
        #bw.backward()

def stop():
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    bw.stop()


def test_cam():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print(cam.get(3),cam.get(4))
    if cam.isOpened():
    #cv2.namedWindow("test")
        for i in range(100):
            _,_ = cam.read()
            
        _,img_off_cam = cam.read()
        cv2.imwrite('testimage1.jpg', img_off_cam)
        img = img_preprocess(img_off_cam)
        img = cam_prep(img)
        cv2.imwrite('testimage2.jpg', img)
        print(img.shape)#This gets returned as BGR, I think this is okay since OG video was taken with openCV
        #print(img)
        img_mod = img_preprocess(img_off_cam)
        angle = int(model.predict(img_mod)[0][0])
        print(angle)
    else:
        print("Could not run")
        
def testmodel():
    img = cv2.imread('/home/pi/picarproject/frame207.jpg')
    img = cv2.resize(img[275:,:,:] / 255, (200,66))
    img = np.expand_dims(img, axis = 0)
    print(img.shape)
    steering_angle = int(model.predict(img)[0][0])
    print(f"Angle should be 60ish but model outputted: {steering_angle}")
    
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
        
if __name__ == '__main__':
    try:
        stop()
    except KeyboardInterrupt:
        destroy()
