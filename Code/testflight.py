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
            
def obj_locate():
    picar.setup()
    #global model
    model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_0129_obj_local.h5')
    
    bw = back_wheels.Back_Wheels()
    fw = front_wheels.Front_Wheels()
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(cam.get(3),cam.get(4))
    
    names = ['nothing','stop','go','25','55','toy']
    
    
    while cam.isOpened():
        for i in range(100):
            _,_ = cam.read()
            
        start = time.time()
        _, img_off_cam = cam.read()
        b,g,r = cv2.split(img_off_cam)
        img = cv2.merge([r,g,b])
        img = np.array(cv2.resize(np.array(img, dtype = np.uint8), (320,320))) / 255
        
        predictions = model(np.expand_dims(img, axis = 0).astype(np.float32))
        P_c = predictions[0]
        print(f"This is Pc: {P_c}")
        bb_dims = predictions[1]
        print(f"These are the bb dims: {bb_dims}")
        classes = predictions[2][0]
        best_guess = names[int(np.argmax(classes))]
        print(f"These are the classes: {classes} and this is the name: {best_guess}")
        cv2.imwrite(f"obj_img.jpg", img*255)
    
    
            

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
        cv2.imwrite('ObjDetect/ObjectDetectionImage1.jpg', img_off_cam)


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
        
def draw_box(image, xmin,xmax,ymin,ymax):
  for i in range(ymax-ymin):
    for j in range(3):
      if j == 0:
        image[ymin+i,xmin:xmin + 3, j] = 255
      else:
        image[ymin+i,xmin:xmin + 3, j] = 0

  for i in range(xmax-xmin):
    for j in range(3):
      if j == 0:
        image[ymin:ymin+3,xmin+i, j] = 255
      else:
        image[ymin:ymin+3,xmin+i, j] = 0

  for i in range(ymax-ymin):
    for j in range(3):
      if j == 0:
        image[ymin+i,xmax-3:xmax, j] = 255
      else:
        image[ymin+i,xmax-3:xmax, j] = 0

  for i in range(xmax-xmin):
    for j in range(3):
      if j == 0:
        image[ymax-3:ymax,xmin+i, j] = 255
      else:
        image[ymax-3:ymax,xmin+i, j] = 0

  return image



def draw_and_save(img,bb):
    dims = bb[0]
    xmin, xmax, ymin, ymax = dims[0]*320, dims[1]*320, dims[2]*120, dims[3]*120
    img = draw_box(img, xmin = int(xmin),xmax = int(xmax),ymin = int(ymin),ymax = int(ymax))
    cv2.imwrite(f"obj_img.jpg", img*255)
    
        
        
def testmodel():
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(cam.get(3),cam.get(4))
    if cam.isOpened():
    #cv2.namedWindow("test")
        for i in range(100):
            _,_ = cam.read()
            
        _,img_off_cam = cam.read()

    img = np.array(cv2.resize(np.array(img_off_cam, dtype = np.float32), (320,120))) / 255
    model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_0120_obj_local.h5')
    #img = cv2.imread('/home/pi/PicarProject/Code/obj_img.jpg')
    print(img.shape)
    
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    
    names = ['nothing','stop','go','25','55','toy']
 
    start = time.time()
    predictions = model.predict(np.expand_dims(img,axis = 0))
    print(type(img))
    print(time.time()-start)
    #steering_angle = int(np.dot(angles,steering_angle))
    P_c = predictions[0]
    print(f"This is Pc: {P_c}")
    bb_dims = predictions[1]
    print(f"These are the bb dims: {bb_dims}")
    classes = predictions[2][0]
    best_guess = names[int(np.argmax(classes))]
    print(f"These are the classes: {classes} and this is the name: {best_guess}")
    draw_and_save(img,bb = bb_dims)
    
    
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
        
        for i in range(1,79):
            _,img_off_cam = cam.read()
            cv2.imwrite(f"/home/pi/PicarProject/Code/ObjDetect/ObjectDetectionImage{369 + i}.jpg", img_off_cam)
            print(f"Took Image {i}")
            time.sleep(5)
        
        
def inspect_model():
    sums = 0
    means = []
    for i in range(22):
        model = tf.keras.models.load_model('/home/pi/PicarProject/Code/tf_model_0126_obj_local.h5')
        print(f"Output: {tf.reduce_sum(model.weights[i])}")
        sums += tf.reduce_sum(model.weights[i])
        means.append(tf.reduce_mean(model.weights[i]))
    print(sums)
    print(np.mean(means))
    
    
def inspect_camera():
    image = cv2.imread('ObjDetect/ObjectDetectionImage2.jpg')
    def img_preprocess(img):
      img = np.array(img, dtype = np.uint8)
      img = np.array(cv2.resize(img, (320,120))) / 255
      return img
    image = img_preprocess(image)
    print(f"Mean: {np.mean(image.flatten())}")
    print(f"Stdev: {np.std(image.flatten())}")
        
if __name__ == '__main__':
    try:
        save_and_take_image()
    except KeyboardInterrupt:
        destroy()
