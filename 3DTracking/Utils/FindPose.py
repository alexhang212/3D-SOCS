"""Given 2 sets of points, find pose between them"""

#!/usr/bin/python

from numpy import *
from math import sqrt

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def findPoseFromPoints(pointsInCurrentFrameA, pointsInTargetFrameB,Name=None):
    """
    Function finds pose to transfer points from frame of reference A to frame of reference B.
    :param pointsInCurrentFrameA: 3xN Matrix of point set A
    :param pointsInTargetFrameB: 3xN Matrix of point set B
    :return: 3x3 Rotation Matrix and 3x1 Translation matrix,
    """

    pointsInCurrentFrameA = mat(pointsInCurrentFrameA)
    pointsInTargetFrameB = mat(pointsInTargetFrameB)

    assert len(pointsInCurrentFrameA) == len(pointsInTargetFrameB)

    num_rows, num_cols = pointsInCurrentFrameA.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = pointsInTargetFrameB.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(pointsInCurrentFrameA, axis=1)
    centroid_B = mean(pointsInTargetFrameB, axis=1)

    if centroid_A.shape != (3,1) and centroid_B.shape != (3,1):
        centroid_A = centroid_A.reshape(3,1)
        centroid_B = centroid_B.reshape(3,1)

    # subtract mean
    Am = pointsInCurrentFrameA - tile(centroid_A, (1, num_cols))
    Bm = pointsInTargetFrameB - tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * transpose(Bm)

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        # TEMP
        # print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t