"""Useful transformation functions for working with 3D data"""

import numpy as np
import math

def GetMagnitude(point):
    """Return magnitute of vector"""
    return math.sqrt((point[0]**2+point[1]**2+point[2]**2))

def GetMidPoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

def GetEucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""
    
    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")
    
    return EucDist


def transformPoints(pointMatrix, rotationMatrix, translationMatrix):
    """
    transforms points based on given rotation and translation matrix
    :param pointMatrix : Matrix 3xN
    :param rotationMatrix : rotation matrix 3x3
    :param translationMatrix : translation matrix 3x1
    :return : Matrix 3xN
    """

    assert( pointMatrix.shape[0] == 3), "Expected dimensions of points are 3XN"
    assert (rotationMatrix.shape == (3, 3) and
            translationMatrix.shape == (3, 1)) , "Invalid dimension of rotation or tranlsation matrix"

    # P_out (3xN) = R (3x3) . P (3xN) + T (3x1)
    rotatedPoints = np.dot(rotationMatrix, pointMatrix)
    transformedPoints = np.add(rotatedPoints, translationMatrix)

    return transformedPoints


def computeExtrinsic(cam1Rotation, cam1Translation, cam2Rotation, cam2Translation):
    """
    compute extrinsic matrix between the given cameras. Rotation and translation parameters supposed to bring points to
    a common coordinate space from camera space.
    Extrinsics can convert points from cam2 space to cam1 space
    Pv = R_cam1. Pc1 + t_cam1 , Pv = R_cam2 . Pc2 + t_cam2
    :param cam1Rotation: rotation ( C -> V) 3x3 matrix
    :param cam1Translation: translation (C -> V) 3x1 matrix
    :param cam2Rotation: rotation (C -> V) 3x3 matrix
    :param cam2Translation: (C -> V) 3x1 matrix
    :return: Rotation and translation ( C2 -> C1)
    """

    # Rotation
    # rotationCam2toCam1 = inverse(cam1Rotation) . cam2Rotation
    rotationCam2toCam1 = np.dot( np.linalg.inv(cam1Rotation), cam2Rotation)

    # translationCam2toCam1 = inverse(cam1Rotation) . (cam2Translation - cam1Translation)
    translationCam2toCam1 = np.dot ( np.linalg.inv(cam1Rotation), (cam2Translation-cam1Translation))

    return rotationCam2toCam1, translationCam2toCam1

