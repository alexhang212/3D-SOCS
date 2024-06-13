'''

Author: Michael Chimento
This code uses a modified version of Gaussian Mixture Model background
subtraction method object detection written by Addison Sears-Collins
'''
# import libraries
from picamera2 import Picamera2
import time 
import cv2 
import numpy as np
import RPi.GPIO as GPIO

#setup gpio for sending start/stop recording signals to follower CM4
GPIO.setmode(GPIO.BCM)
output_pin = 5 #6th pin from top right
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(output_pin, GPIO.LOW)

#globals used for object detection algorithm
kernel = np.ones((20,20),np.uint8)
min_area = 150 #used to determine if object is bird or not, can be tuned (number of pixels)
accWeighted_alpha=0.6
thresh_val = 255
dilate_iter = 4
avg = None


video_length = 60 #number of seconds to record each video
vid_num=0 #running index of video number since script started
flag = False #flag to note if video is currently recording (TRUE) or idle (FALSE)

# create background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2(history=150,
  varThreshold=25, detectShadows=True)

# initialize the camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (1640, 1280)}))
camera.start()

#give time to start follower CM4 scripts
print("[INFO] sleeping for 30 seconds")
time.sleep(30)

lastCapture=time.time()
# Capture frames continuously from the camera
while True:
     
    # Grab the raw NumPy array representing the image
    image = camera.capture_array()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
 
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue# Convert to foreground mask
    
    cv2.accumulateWeighted(gray, avg, accWeighted_alpha)
    
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
       
    # threshold frame to black and white based on second argument
    _, thresh = cv2.threshold(frame_delta, 20, thresh_val, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=dilate_iter)
    
     # find contours of objects
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
    num_obj = len(areas)
    
    current_time=time.time()
    
    #no objects
    if num_obj<1:
        if not flag:
            pass
    
    #object detected
    else:
        #object too small
        if max(areas)<min_area:
            pass
            
        #object might be bird
        else:
            if vid_num==0 or (not flag and (current_time - lastCapture > (video_length + 5))):
                print("[INFO] large object detected {}".format(max(areas)))
                print("[INFO] Changing board pin {} to high for {}s".format(output_pin, video_length))
                GPIO.output(output_pin, GPIO.HIGH)
                flag=True
                vid_num += 1
                print("[INFO] video number {}".format(vid_num))
                lastCapture=time.time()
                # Find the largest moving object in the image
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    #stop recording if currently recording and video has exceeded video_length            
    if flag:   
        if current_time - lastCapture > video_length:
            print("[INFO] Changing board pin {} back to low...".format(output_pin))
            GPIO.output(output_pin, GPIO.LOW)
            time.sleep(0.2)
            flag=False

    cv2.imshow('Frame',image)
    
    #shutdown with q
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
      break

GPIO.cleanup()
print("Cleaned up")

