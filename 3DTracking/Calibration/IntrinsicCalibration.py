#!/usr/bin/env python3

"""
Run intrinsic calibration sequence automatically

"""

import configparser
import cv2
from cv2 import aruco
import numpy as np
import scipy.stats as st
from tqdm import tqdm
import pickle
import sys
import os

###Detect calibration board and calculate for blur
def CheckFrame(img, board, aruco_dict, counter,debug=False):
    """
    Processes image, determines whether image is good fit for calibration
    img: input image
    board: calibration board template

    """
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners,ids,_ = cv2.aruco.detectMarkers(gray,aruco_dict) #Detect aruco markers
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    if len(corners) <15:
        #less than XX corners detected, return false
        return False,False
    else:
        # import ipdb; ipdb.set_trace()
        
        #Get subpixel detection for corners
        SubPixCorners = []
        for corner in corners:
            corner = cv2.cornerSubPix(gray, corner, winSize = (3,3),zeroZone = (-1,-1),
                criteria = criteria)
            SubPixCorners.append(corner)

        #interpolate markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
        minMarkers=1)

        ###Temp, draw markers
        if debug:
            img = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)
            # cv2.imshow(img)
            # cv2.waitKey(1)
            # cv2.imwrite('Temp/BoardDebug/%i.jpg'%(counter), img)
            counter +=1

        #######

        return charuco_corners, charuco_ids
        
def CheckDist(Corner1, Corner2):
    """
    Check the distance between two calibration board corner lists
    to see if the corners are similar. Returns True if similar, returns False if not
    """
    Diff = Corner1[0] - Corner2[0]

    if sum(abs(Diff[0]))/2 < 2:
        #If the mean difference of x and y coordinate of first point is less than 2 pixels, return True
        return True
    else:
        return False





##
def calibrate_auto(InputVid,CalibBoardDict,Freq= 30,debug = False):
    """
    Calibrate single camera by choosing images automatically
    Freq: sample frequency
    
    """

    aruco_dict = CalibBoardDict["ArucoDict"]
    board = aruco.CharucoBoard_create(int(CalibBoardDict["widthNum"]),
                                        int(CalibBoardDict["lengthNum"]),
                                        float(CalibBoardDict["squareLen"]),
                                        float(CalibBoardDict["arucoLen"]),aruco_dict)


    #Read config file
    #Read Video
    allCorners = []
    allIds = []
    
    ###Get imsize
    cap = cv2.VideoCapture(InputVid)
    ret,frame = cap.read()
    imsize = (frame.shape[1],frame.shape[0]) 


    ###Read video
    cap = cv2.VideoCapture(InputVid)

    counter=0

    # while(cap.isOpened()):
    for x in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):#add progress bar
        ret,frame = cap.read()
        counter += 1

        if counter % Freq == 0: #If multiple of Freq, go second by second
                if ret==True:
                    # cv2.imshow("Frame",frame)
                    # cv2.waitKey(1)
                    charuco_corners, charuco_ids = CheckFrame(frame, board, aruco_dict,counter,debug)
                    if charuco_corners is not False:
                        #corners detected, check if corners are more or less the same as previous
                        if len(allCorners) == 0: #first loop
                            SameBool = False
                        else:
                            SameBool = CheckDist(allCorners[len(allCorners)-1], charuco_corners)

                        if SameBool == False: #corners are not the same, append
                            allCorners.append(charuco_corners)
                            allIds.append(charuco_ids)

                else:
                    print("Video cannot be read")
                    break
        else:
            continue
    cap.release()
    # cv2.destroyAllWindows()
    # import ipdb; ipdb.set_trace()
    if len(allCorners) == 0:
        ##Nothing detected
        return None,None,None,None,None,None,None,None

    ## initial calibration
    retCal, cameraMatrix, distCoeffs, rvecs, tvecs,_,_,pVErr = aruco.calibrateCameraCharucoExtended(
    charucoCorners=allCorners,
    charucoIds=allIds,
    board=board,
    imageSize=imsize,
    cameraMatrix=None,
    distCoeffs=None)

    MeanError = sum(pVErr)/len(pVErr) #Mean projection error across all images
    allpVErr = pVErr.copy() #pVErr list of original calibration

    if MeanError > 1:
        #Error is larger than 1
        #Remove 1 image iteratively, until error is below 1 or if its less than 15 images
        #if less than 15 images, will use the calibration of lowest error

        print("Running optimization algorithm to improve calibration")
        SubCorners = allCorners.copy()
        SubIds = allIds.copy()

        LowestErrorTerms= 0 # The number of terms removed that gives lowest error
        TermsRemoved =0
        LowestError = MeanError

        while MeanError >1:
            #Alternatively: remove images with highest error
            Index = [i for i in range(len(pVErr)) if pVErr[i]==max(pVErr)]
            # print(Index)

            #remove item with index
            SubCorners.pop(Index[0])
            SubIds.pop(Index[0])

            #recalibration
            retCal, cameraMatrix, distCoeffs, rvecs, tvecs,_,_,pVErr  = aruco.calibrateCameraCharucoExtended(
            charucoCorners=SubCorners,
            charucoIds=SubIds,
            board=board,
            imageSize=imsize,
            cameraMatrix=None,
            distCoeffs=None)

            MeanError = sum(pVErr)/len(pVErr)

            #print results
            TermsRemoved +=1
            print("Images Removed: %i" %TermsRemoved)
            print("New Error is %f" %MeanError)
            # import ipdb; ipdb.set_trace()


            if MeanError < LowestError: #update error if error is new low
                LowestErrorTerms=TermsRemoved
                LowestError = MeanError

            if len(SubCorners)< 8:#if less than 15 images, stop and get the best error
                # import ipdb; ipdb.set_trace()
                SubCorners = allCorners.copy()
                SubIds = allIds.copy()
                pVErr = allpVErr.copy()

                for x in range(LowestErrorTerms):
                    Index = [i for i in range(len(pVErr)) if pVErr[i]==max(pVErr)]
                    SubCorners.pop(Index[0])
                    SubIds.pop(Index[0])
                    pVErr = np.delete(pVErr,Index[0],0)

                    # import ipdb;ipdb.set_trace()

                
                retCal, cameraMatrix, distCoeffs, rvecs, tvecs,_,_,pVErr = aruco.calibrateCameraCharucoExtended(
                charucoCorners=SubCorners,
                charucoIds=SubIds,
                board=board,
                imageSize=imsize,
                cameraMatrix=None,
                distCoeffs=None)
                FinalMeanError = sum(pVErr)/len(pVErr)


                print("Less than 8 images remaining,chose lowest error of %f, with %i image removed" %(FinalMeanError,LowestErrorTerms))

                break
    print("Final Error: %f" %MeanError)
    #get optimized camera matrix
    h,w = imsize
    NewCamMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
    # import ipdb; ipdb.set_trace()
    return retCal, cameraMatrix, distCoeffs, rvecs, tvecs,pVErr,NewCamMat,roi
            

