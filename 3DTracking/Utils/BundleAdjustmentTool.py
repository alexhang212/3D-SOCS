# !/usr/bin/env python3

"""
Python class to do bundle adjustment, both to do triangulation and tune extrinsics

Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import scipy as sc
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
from tqdm import tqdm
import math
from natsort import natsorted
import random

class BundleAdjustmentTool_Triangulation:
    """Bundle adjustment Tool, for triangulation"""

    def __init__(self, CamNames, CamParams):
        """
        CamNames: list of camera names
        CamParams: Dictionary of camera parameters (intrinsic and extrinsics)
        Points2D: Detections in 2D space
        
        """
        self.CamNames = CamNames
        self.CamParamDict = CamParams
        self.PrepareCamParams(CamParams)
        
    def PrepareCamParams(self,CamParams):
        """Reads in Dictionary of camera prameters then put it in correct format"""
        
        CamRvecList = []
        CamTvecList = []
        CamMatList = []
        DistList = []
        for i in range(len(self.CamNames)):
            Cam = self.CamNames[i]
            #Extrsinics
            # if CamParams[Cam]
            if CamParams[Cam]["R"].shape == (3,3):
                rvec = CamParams[Cam]["R"]
            else:
                rvec = cv2.Rodrigues(CamParams[Cam]["R"])[0]

            # rvec = CamParams[Cam]["R"]
            tvec = np.float32(CamParams[Cam]["T"])
            CamRvecList.append(rvec)
            CamTvecList.append(tvec)
            
            ##Intrinsics:
            CamMatList.append(CamParams[Cam]["cameraMatrix"])
            DistList.append(CamParams[Cam]["distCoeffs"])

        self.CamMatList = CamMatList
        self.DistList = DistList
        self.CamRvecList = CamRvecList
        self.CamTvecList = CamTvecList
        

    def PrepareInputData(self,PointsDictList):
        """
        input:
        PointsDictList: Dictionary named by camera names, with inner dictionary of points 
        {PointName : [x,y]}
        
        output:
        Points3D (N_point,3): 3D coordinates of unique 3D points
        PointIndex (N_Observation,): index of which 3d points a given 2d point belongs
        CamIndex (N_Observation,) : index of which camera a given 2d point belongs
        Points2D (N_Observation,2) : 2D coordinates of given feature
        """
        ##Output Lists
        Points3DList = []
        PointIndexList = []
        CamIndexList = []
        Points2DList = []        
        
        #Get all unique frames
        Cameras = list(PointsDictList.keys())
        Point2DNameList = [list(val.keys()) for val in PointsDictList.values()]
        
        #get all unique 2Dpoints:
        AllUnqPoints = sorted(list(set().union(*Point2DNameList)))
        
        Point3DIndexCounter = 0 ##rolling count for 3d point index
        PointNameList = [] ##Kep tracks of point names
        # Initial3DPoints = []
        
        for point in AllUnqPoints:
            ##Exists 2D measurements in at least 2 views:
            ExistFrameData = {}
            for x in range(len(Cameras)):
                Cam = Cameras[x]
                if point in list(PointsDictList[Cam].keys()):
                    # PointIndex = PointsDictList[Cam]["ImgIDs"].index(point)
                    ExistFrameData.update({Cam:PointsDictList[Cam][point]})
                else:
                    continue
            if len(ExistFrameData) <2:
                ##Only 1 cam saw point
                continue
            
            #At least 2 views detected point:
            AllCamIndexes = list(ExistFrameData.keys())
            CamIndexes = (AllCamIndexes[0],AllCamIndexes[1]) #Just choose first 2 cameras
            #Initial 3D point estimate:
            PointDicts = (ExistFrameData[CamIndexes[0]],ExistFrameData[CamIndexes[1]] )
            # PointDicts = (PointsDictList[Cameras[CamIndexes[0]]][frame], PointsDictList[Cameras[CamIndexes[1]]][frame])
            Initial3D = self.TriangulatePoints(CamIndexes, PointDicts) #OverlapID is the checkboard object ids that were used in triangulation
            PointNameList.append(point)
            
            Points3DList.append(Initial3D.tolist()[0])
            
            for CamName in ExistFrameData.keys():
                Points2DList.append(ExistFrameData[CamName]) #Save 2D point
                CamIndexList.append(self.CamNames.index(CamName))
                PointIndexList.append(Point3DIndexCounter)
                
            Point3DIndexCounter += 1
        self.Points3DArr = np.array(Points3DList)            
        self.PointIndexArr = np.array(PointIndexList)
        self.CamIndexArr = np.array(CamIndexList)
        self.Points2DArr = np.array(Points2DList)
        self.PointsNameList = PointNameList
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]
        
        if Point3DIndexCounter == 0: ###no valid points
            return False
        else:
            return True
        

    def TriangulatePoints(self,CamIndexes, PointDicts):
        """Triangulate Points using 2 camera views"""
        CamNames1 = CamIndexes[0]
        CamNames2 = CamIndexes[1]
        
        Cam1ParamDict = self.CamParamDict[CamNames1]
        Cam2ParamDict = self.CamParamDict[CamNames2]
        
        # import ipdb;ipdb.set_trace()
        #Projection matrix:
        projectionMatrixCam1 = self.ComputeProjectionMatrix(Cam1ParamDict["R"],Cam1ParamDict["T"],Cam1ParamDict["cameraMatrix"])
        projectionMatrixCam2 = self.ComputeProjectionMatrix(Cam2ParamDict["R"],Cam2ParamDict["T"],Cam2ParamDict["cameraMatrix"])
        
        #Prepare 2D points
        Cam1PointsArr = np.array(PointDicts[0], dtype = np.float32).T
        Cam2PointsArr = np.array(PointDicts[1], dtype = np.float32).T
        
        triangulatedPointsHomogenous = cv2.triangulatePoints(projectionMatrixCam1,projectionMatrixCam2,Cam1PointsArr,Cam2PointsArr)
        triangulatedPointsArray = cv2.convertPointsFromHomogeneous(triangulatedPointsHomogenous.T)
        triangulatedPointsMatrix = np.matrix(triangulatedPointsArray)
        
        return triangulatedPointsMatrix
            
        
    def ComputeProjectionMatrix(self,rotationMatrix,translationMatrix,intrinsicMatrix):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x1 matrix
        :param translationMatrix: 3x1 Matrix
        :param intrinsicMatrix: 3x3 matrix
        :return: 3x4 projection matrix
        """
        # rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]

        if rotationMatrix.shape == (3,3):
            rotationMatrix = rotationMatrix
        else:
            rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]

        RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
        projectionMatrix = np.dot(intrinsicMatrix, RT)
        return projectionMatrix
    
    def GetResiduals(self,Params):
        """Given points and parameters, get residuals using reproject"""
        #Back transform to get back parameters and 3d points
        
        Points3D = Params.reshape((self.n_points, 3))
        points_proj = self.Reproject(Points3D[self.PointIndexArr])
        PointDiffs = points_proj - self.Points2DArr
        EucError = np.sqrt(PointDiffs[:,0]**2 + PointDiffs[:,1]**2)
        
        # import ipdb;ipdb.set_trace()
        return (points_proj - self.Points2DArr).ravel()
        # return EucError
        
    def Reproject(self, Points3DAll):
        """Given 3D points, do reprojection and get new 2d points"""
        AllPoint2D = []
        for i in range(len(Points3DAll)):
            rvec = self.CamRvecList[self.CamIndexArr[i]]
            tvec = self.CamTvecList[self.CamIndexArr[i]]
            FundMat = self.CamMatList[self.CamIndexArr[i]]
            Dist = self.DistList[self.CamIndexArr[i]]

            Point2D = cv2.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
            AllPoint2D.append(Point2D[0][0][0])

        AllPoint2DArr = np.array(AllPoint2D)
        return AllPoint2DArr
    
    def BundleAdjustmentSparsity(self, n,m):
        # m = self.CamIndexArr.size * 2
        # n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.CamIndexArr.size)
        # for s in range(6):
        #     A[2 * i, self.CamIndexArr * 6 + s] = 1
        #     A[2 * i + 1, self.CamIndexArr * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.PointIndexArr * 3 + s] = 1
            A[2 * i + 1, self.PointIndexArr * 3 + s] = 1

        return A
    
    def GetFinalParam(self,results):
        """Back transform arrays to get final param"""
        Points3D = results.reshape((self.n_points,3))
        
        FinalDict = {}
        for i in range(len(Points3D)):
            FinalDict.update({self.PointsNameList[i]:Points3D[i]})

        return FinalDict
    
    
    def run(self):
        """Run Bundle Adjustment"""
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 50,verbose=0, x_scale='jac', ftol=1e-8, method='trf')
                    # args=(self.n_cameras, self.n_points,self.CamIndexArr, self.PointIndexArr, self.Points2DArr,self.IntMat,self.Distortion))
        # plt.plot(res.fun)
        # plt.show()
        # import ipdb;ipdb.set_trace()

        FinalParamDict = self.GetFinalParam(res.x)

        return FinalParamDict
    

class BundleAdjustmentTool_Triangulation_Filter:
    """
    Bundle adjustment Tool, for triangulation
    New implementation: if high reprojection error, filter cam
    
    """

    def __init__(self, CamNames, CamParams):
        """
        CamNames: list of camera names
        CamParams: Dictionary of camera parameters (intrinsic and extrinsics)
        Points2D: Detections in 2D space
        
        """
        self.CamNames = CamNames
        self.CamParamDict = CamParams
        self.PrepareCamParams(CamParams)
        
    def PrepareCamParams(self,CamParams):
        """Reads in Dictionary of camera prameters then put it in correct format"""
        
        CamRvecList = []
        CamTvecList = []
        CamMatList = []
        DistList = []
        for i in range(len(self.CamNames)):
            Cam = self.CamNames[i]
            #Extrsinics
            rvec = cv2.Rodrigues(CamParams[Cam]["R"])[0]
            tvec = CamParams[Cam]["T"]
            CamRvecList.append(rvec)
            CamTvecList.append(tvec)
            
            ##Intrinsics:
            CamMatList.append(CamParams[Cam]["cameraMatrix"])
            DistList.append(CamParams[Cam]["distCoeffs"])

        self.CamMatList = CamMatList
        self.DistList = DistList
        self.CamRvecList = CamRvecList
        self.CamTvecList = CamTvecList
        

    def PrepareInputData(self,PointsDictList):
        """
        input:
        PointsDictList: Dictionary named by camera names, with inner dictionary of points 
        {PointName : [x,y]}
        
        output:
        Points3D (N_point,3): 3D coordinates of unique 3D points
        PointIndex (N_Observation,): index of which 3d points a given 2d point belongs
        CamIndex (N_Observation,) : index of which camera a given 2d point belongs
        Points2D (N_Observation,2) : 2D coordinates of given feature
        """
        ##Output Lists
        Points3DList = []
        PointIndexList = []
        CamIndexList = []
        Points2DList = []        
        
        #Get all unique frames
        Cameras = list(PointsDictList.keys())
        Point2DNameList = [list(val.keys()) for val in PointsDictList.values()]
        
        #get all unique 2Dpoints:
        AllUnqPoints = sorted(list(set().union(*Point2DNameList)))
        
        Point3DIndexCounter = 0 ##rolling count for 3d point index
        PointNameList = [] ##Kep tracks of point names
        # Initial3DPoints = []
        
        for point in AllUnqPoints:
            ##Exists 2D measurements in at least 2 views:
            ExistFrameData = {}
            for x in range(len(Cameras)):
                Cam = Cameras[x]
                if point in list(PointsDictList[Cam].keys()):
                    # PointIndex = PointsDictList[Cam]["ImgIDs"].index(point)
                    ExistFrameData.update({Cam:PointsDictList[Cam][point]})
                else:
                    continue
            if len(ExistFrameData) <2:
                ##Only 1 cam saw point
                continue
            
            #At least 2 views detected point:
            AllCamIndexes = list(ExistFrameData.keys())
            CamIndexes = (AllCamIndexes[0],AllCamIndexes[1]) #Just choose first 2 cameras
            #Initial 3D point estimate:
            PointDicts = (ExistFrameData[CamIndexes[0]],ExistFrameData[CamIndexes[1]] )
            # PointDicts = (PointsDictList[Cameras[CamIndexes[0]]][frame], PointsDictList[Cameras[CamIndexes[1]]][frame])
            Initial3D = self.TriangulatePoints(CamIndexes, PointDicts) #OverlapID is the checkboard object ids that were used in triangulation
            PointNameList.append(point)
            
            Points3DList.append(Initial3D.tolist()[0])
            
            for CamName in ExistFrameData.keys():
                Points2DList.append(ExistFrameData[CamName]) #Save 2D point
                CamIndexList.append(self.CamNames.index(CamName))
                PointIndexList.append(Point3DIndexCounter)
                
            Point3DIndexCounter += 1
        self.Points3DArr = np.array(Points3DList)            
        self.PointIndexArr = np.array(PointIndexList)
        self.CamIndexArr = np.array(CamIndexList)
        self.Points2DArr = np.array(Points2DList)
        self.PointsNameList = PointNameList
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]
        
        if Point3DIndexCounter == 0: ###no valid points
            return False
        else:
            return True
        

    def TriangulatePoints(self,CamIndexes, PointDicts):
        """Triangulate Points using 2 camera views"""
        CamNames1 = CamIndexes[0]
        CamNames2 = CamIndexes[1]
        
        Cam1ParamDict = self.CamParamDict[CamNames1]
        Cam2ParamDict = self.CamParamDict[CamNames2]
        
        # import ipdb;ipdb.set_trace()
        #Projection matrix:
        projectionMatrixCam1 = self.ComputeProjectionMatrix(Cam1ParamDict["R"],Cam1ParamDict["T"],Cam1ParamDict["cameraMatrix"])
        projectionMatrixCam2 = self.ComputeProjectionMatrix(Cam2ParamDict["R"],Cam2ParamDict["T"],Cam2ParamDict["cameraMatrix"])
        
        #Prepare 2D points
        Cam1PointsArr = np.array(PointDicts[0], dtype = np.float32).T
        Cam2PointsArr = np.array(PointDicts[1], dtype = np.float32).T
        
        triangulatedPointsHomogenous = cv2.triangulatePoints(projectionMatrixCam1,projectionMatrixCam2,Cam1PointsArr,Cam2PointsArr)
        triangulatedPointsArray = cv2.convertPointsFromHomogeneous(triangulatedPointsHomogenous.T)
        triangulatedPointsMatrix = np.matrix(triangulatedPointsArray)
        
        return triangulatedPointsMatrix
            
        
    def ComputeProjectionMatrix(self,rotationMatrix,translationMatrix,intrinsicMatrix):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x1 matrix
        :param translationMatrix: 3x1 Matrix
        :param intrinsicMatrix: 3x3 matrix
        :return: 3x4 projection matrix
        """

        if rotationMatrix.shape == (3,3):
            rotationMatrix = rotationMatrix
        else:
            rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]

        RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
        projectionMatrix = np.dot(intrinsicMatrix, RT)
        return projectionMatrix
    
    def GetResiduals(self,Params):
        """Given points and parameters, get residuals using reproject"""
        #Back transform to get back parameters and 3d points
        
        Points3D = Params.reshape((self.n_points, 3))
        points_proj = self.Reproject(Points3D[self.PointIndexArr])
        PointDiffs = points_proj - self.Points2DArr
        EucError = np.sqrt(PointDiffs[:,0]**2 + PointDiffs[:,1]**2)
        
        # import ipdb;ipdb.set_trace()
        return (points_proj - self.Points2DArr).ravel()
        # return EucError
        
    def Reproject(self, Points3DAll):
        """Given 3D points, do reprojection and get new 2d points"""
        AllPoint2D = []
        for i in range(len(Points3DAll)):
            rvec = self.CamRvecList[self.CamIndexArr[i]]
            tvec = self.CamTvecList[self.CamIndexArr[i]]
            FundMat = self.CamMatList[self.CamIndexArr[i]]
            Dist = self.DistList[self.CamIndexArr[i]]

            Point2D = cv2.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
            AllPoint2D.append(Point2D[0][0][0])

        AllPoint2DArr = np.array(AllPoint2D)
        return AllPoint2DArr
    
    def BundleAdjustmentSparsity(self, n,m):
        # m = self.CamIndexArr.size * 2
        # n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.CamIndexArr.size)
        # for s in range(6):
        #     A[2 * i, self.CamIndexArr * 6 + s] = 1
        #     A[2 * i + 1, self.CamIndexArr * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.PointIndexArr * 3 + s] = 1
            A[2 * i + 1, self.PointIndexArr * 3 + s] = 1

        return A
    
    def GetFinalParam(self,results):
        """Back transform arrays to get final param"""
        Points3D = results.reshape((self.n_points,3))
        
        FinalDict = {}
        for i in range(len(Points3D)):
            FinalDict.update({self.PointsNameList[i]:Points3D[i]})

        return FinalDict
    
    def Get_ReprojectError(self, results):
        """check reprojection error """
        Points3D = results.reshape((self.n_points,3))
        All3DPoints = Points3D[self.PointIndexArr]

        # import ipdb;ipdb.set_trace()
        Reproject2D = self.Reproject(All3DPoints)

        ErrorList = [self.GetEucDist(self.Points2DArr[x],Reproject2D[x]) for x in range(self.Points2DArr.shape[0])]
        ErrorArray = np.array(ErrorList)

        return ErrorArray

    def OptimizeAndFilter(self, results):
        """check reprojection error then filter out points that have high ones, then retriangulate"""
        Points3D = results.reshape((self.n_points,3))
        All3DPoints = Points3D[self.PointIndexArr]

        # import ipdb;ipdb.set_trace()
        Reproject2D = self.Reproject(All3DPoints)

        ErrorList = [self.GetEucDist(self.Points2DArr[x],Reproject2D[x]) for x in range(self.Points2DArr.shape[0])]
        ErrorArray = np.array(ErrorList)
        UpperBound  = ErrorArray.mean() + (1*np.std(ErrorArray)) #Upper bound: 1 sds
        # UpperBound = 10 #Hard Threshold????

        OutlierIndex = np.where(ErrorArray>UpperBound)[0]

        # plt.plot(ErrorArray)
        # plt.axhline(y=UpperBound, color='r', linestyle='-')
        # plt.show()



        if len(OutlierIndex) == 0:
            FinalParamDict = self.GetFinalParam(results)

            return FinalParamDict

        ###Delete indicies from all the arrays
        # import ipdb;ipdb.set_trace()
        self.PointIndexArr = np.delete(self.PointIndexArr,OutlierIndex)
        self.CamIndexArr =  np.delete(self.CamIndexArr,OutlierIndex)
        self.Points2DArr = np.delete(self.Points2DArr,OutlierIndex,0)

        if len(self.PointIndexArr) == 0:
            return {}

        # yo = self.Points2DArr.copy()

        ####Rerun bundle adjustment:

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 50,verbose=0, x_scale='jac', ftol=1e-8, method='trf')


        FinalParamDict = self.GetFinalParam(res.x)
        
        return FinalParamDict



    def GetEucDist(self,Point1,Point2):
        """Get euclidian error, both 2D and 3D"""
        
        if len(Point1) ==3 & len(Point2) ==3:
            EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
        elif len(Point1) ==2 & len(Point2) ==2:
            EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
        else:
            import ipdb;ipdb.set_trace()
            Exception("point input size error")
        
        return EucDist

    def run(self):
        """Run Bundle Adjustment"""
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]

        if self.n_cameras == 2:###Just 2 cameras, no need to optimize, use initial guess (which is cv2 stereo triangulation)
            FinalParamDict = self.GetFinalParam(self.Points3DArr.ravel())
            return FinalParamDict, np.zeros(self.Points2DArr.shape[0])

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 50,verbose=0, x_scale='jac', ftol=1e-8, method='trf')
                    # args=(self.n_cameras, self.n_points,self.CamIndexArr, self.PointIndexArr, self.Points2DArr,self.IntMat,self.Distortion))
        # plt.plot(res.fun)
        # plt.show()
        # import ipdb;ipdb.set_trace()

        # FinalParamDict = self.GetFinalParam(res.x)

        FinalParamDict = self.OptimizeAndFilter(res.x)
        ReprojectError = self.Get_ReprojectError(res.x)
        

        return FinalParamDict, ReprojectError
    




class BundleAdjustmentTool_Calibration:
    """Bundle adjustment Tool, for tuning extrinsic parameters"""

    def __init__(self, CamNames, CamParams):
        """
        CamNames: list of camera names
        CamParams: Dictionary of camera parameters (intrinsic and extrinsics)
        Points2D: Detections in 2D space
        
        """
        self.CamNames = CamNames
        self.CamParamDict = CamParams


        self.PrepareCamParams(CamParams)
        
    def PrepareCamParams(self,CamParams):
        """Reads in Dictionary of camera prameters then put it in correct format"""
        CamParamList = [] 
        for i in range(len(self.CamNames)):
            Cam = self.CamNames[i]
            #Extrsinics (to be optimized)
            rvec = CamParams[Cam]["R"]
            if rvec.shape == (3,3):
                rvec = cv2.Rodrigues(CamParams[Cam]["R"])[0]
            tvec = CamParams[Cam]["T"]
            
            
            paramArr = rvec.reshape(1,3)[0]
            paramArr = np.append(paramArr, tvec)
            
            ##Intrinsics:
            # CamMat = CamParams[Cam]["cameraMatrix"]
            # CamMatParams = np.array([CamMat[0,0],CamMat[1,1], CamMat[0,2],CamMat[1,2]]) #Exrract the 4 params that matter
            # paramArr = np.append(paramArr, CamMatParams)

            # distCoeffs  = CamParams[Cam]["distCoeffs"]
            # paramArr = np.append(paramArr, distCoeffs)

            CamParamList.append(paramArr)

        # import ipdb;ipdb.set_trace()
        self.CamParamArr = np.array(CamParamList)


    def PrepareInputData(self,PointsDictList):
        """
        Prepare input data, by taking in checkerboard detections and building out arrays for bundle
        input:
        PointsDictList: Dictionary named by camera names, with inner dictionary of points named by frame name,
                        of dict {objectpoint, pointID, 2d image point}
        
        output:
        Points3D (N_point,3): 3D coordinates of unique 3D points
        PointIndex (N_Observation,): index of which 3d points a given 2d point belongs
        CamIndex (N_Observation,) : index of which camera a given 2d point belongs
        Points2D (N_Observation,2) : 2D coordinates of given feature
        """
        self.Point2DList = PointsDictList

        ##Output Lists
        Points3DList = []
        PointIndexList = []
        CamIndexList = []
        Points2DList = []        
        
        #Get all unique frames
        KeyList = [list(camDict.keys()) for camDict in PointsDictList]
        
        #get all unique frames:
        AllUnqFrames = sorted(list(set().union(*KeyList)))
        
        Point3DIndexCounter = 0 ##rolling count for 3d point index
        
        for frame in AllUnqFrames:
            ##Exists 2D measurements:
            ExistFrameData = {i:PointsDictList[i][frame] for i in range(len(self.CamNames)) if frame in list(PointsDictList[i].keys())}
            
            if len(ExistFrameData) < 2: #if only 1 view saw the board
                continue
            else:
                #At least 2 views detected board:
                AllCamIndexes = list(ExistFrameData.keys())
                CamIndexes = (AllCamIndexes[0],AllCamIndexes[1]) #Just choose first 2 cameras
                #Initial 3D point estimate:
                PointDicts = (PointsDictList[CamIndexes[0]][frame], PointsDictList[CamIndexes[1]][frame])
                Initial3D, OverlapID= self.TriangulatePoints(CamIndexes, PointDicts) #OverlapID is the checkboard object ids that were used in triangulation
                

                for i in range(Initial3D.shape[0]):
                    Points3DList.append(Initial3D[i].tolist()[0])
                    for CamIndex in ExistFrameData.keys():
                        CameraDict = ExistFrameData[CamIndex]
                        
                        IndexofCorner = [idx for idx in range(len(CameraDict["IDs"])) if OverlapID[i] in CameraDict["IDs"]] #index of the id
                        if len(IndexofCorner) == 0:
                            continue
                        
                        Points2DList.append(CameraDict["ImgPoints"][IndexofCorner[0]]) #Save 2D point
                        CamIndexList.append(CamIndex)
                        PointIndexList.append(Point3DIndexCounter)
                    Point3DIndexCounter += 1
        
        self.Points3DArr = np.array(Points3DList)            
        self.PointIndexArr = np.array(PointIndexList)
        self.CamIndexArr = np.array(CamIndexList)
        self.Points2DArr = np.array(Points2DList)

        if Point3DIndexCounter == 0: ###no valid points
            return False
        else:
            return True
        
                
    def TriangulatePoints(self,CamIndexes, PointDicts):
        """Triangulate Points using 2 camera views"""
        
        CamNames1 = self.CamNames[CamIndexes[0]]
        CamNames2 = self.CamNames[CamIndexes[1]]
        
        Cam1ParamDict = self.CamParamDict[CamNames1]
        Cam2ParamDict = self.CamParamDict[CamNames2]
        
        # import ipdb;ipdb.set_trace()
        projectionMatrixCam1 = self.ComputeProjectionMatrix(Cam1ParamDict["R"],Cam1ParamDict["T"],Cam1ParamDict["cameraMatrix"])
        projectionMatrixCam2 = self.ComputeProjectionMatrix(Cam2ParamDict["R"],Cam2ParamDict["T"],Cam2ParamDict["cameraMatrix"])
        
        ##Find 2D points that both cameras can see:
        OverlapID = sorted(list(set(PointDicts[0]["IDs"]) & set(PointDicts[1]["IDs"])))
        
        Cam1Points2D = [Point2D for i, Point2D in enumerate(PointDicts[0]["ImgPoints"]) if PointDicts[0]["IDs"][i] in OverlapID ]
        Cam2Points2D = [Point2D for i, Point2D in enumerate(PointDicts[1]["ImgPoints"]) if PointDicts[1]["IDs"][i] in OverlapID ]
        
        Cam1PointsArr = np.array(Cam1Points2D, dtype = np.float32)
        Cam2PointsArr = np.array(Cam2Points2D, dtype = np.float32)
        
        # Cam1PointsArr = Cam1PointsArr.reshape(2,Cam1PointsArr.shape[0])
        # Cam2PointsArr = Cam2PointsArr.reshape(2,Cam2PointsArr.shape[0])

        Cam1PointsArr = Cam1PointsArr.T
        Cam2PointsArr = Cam2PointsArr.T

        triangulatedPointsHomogenous = cv2.triangulatePoints(projectionMatrixCam1,projectionMatrixCam2,Cam1PointsArr,Cam2PointsArr)
        triangulatedPointsArray = cv2.convertPointsFromHomogeneous(triangulatedPointsHomogenous.T)
        triangulatedPointsMatrix = np.matrix(triangulatedPointsArray)
        
        return triangulatedPointsMatrix, OverlapID
            
        
    def ComputeProjectionMatrix(self,rotationMatrix,translationMatrix,intrinsicMatrix):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x3 matrix
        :param translationMatrix: 3x3 Matrix
        :param intrinsicMatrix: 3x3 matrix
        :return: 3x4 projection matrix
        """
            
        RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
        projectionMatrix = np.dot(intrinsicMatrix, RT)
        return projectionMatrix
    
    def SummarizeParams(self,CamParams):
        """Convert params back to dictionary form"""
        ParamDict = {}
        for i in range(CamParams.shape[0]):
            CamDict  = {}

            # CamDict["R"] = cv2.Rodrigues(CamParams[i,0:3])[0]
            CamDict["R"] = CamParams[i,0:3]
            CamDict["T"] = CamParams[i,3:6].reshape(3,1)
            CamDict["cameraMatrix"] = self.CamParamDict[self.CamNames[i]]["cameraMatrix"]
            CamDict["distCoeffs"] = self.CamParamDict[self.CamNames[i]]["distCoeffs"]

            ParamDict[self.CamNames[i]] = CamDict

        return ParamDict
    
    def GetResiduals(self,Params):
        """Given points and parameters, get residuals using reproject"""
        #Back transform to getc back parameters and 3d points
        #new param dicts:
        CamParams = Params[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        CamParamDict = self.SummarizeParams(CamParams)


        ##Get back 3D points:
        ##Loops through each frame to avoid confusion
        # import ipdb;ipdb.set_trace()
        ###Get all unique frames from all cameras
        UnqFrames = []
        for x in range(len(self.Point2DList)):
            UnqFrames.extend(list(self.Point2DList[x].keys()))
        UnqFrames = natsorted(list(set(UnqFrames)))

        # RandomFrames = random.sample(UnqFrames, 20)
        # Lengths = [max(list(self.Point2DList[x].keys())) for x in range(len(self.Point2DList))]

        PointErrors = []

        for i in UnqFrames:
            Frame2Ddict = {}
            for x in range(len(self.Point2DList)):
                if i in self.Point2DList[x].keys():
                    Frame2Ddict[self.CamNames[x]] = {self.Point2DList[x][i]["IDs"][y]:self.Point2DList[x][i]["ImgPoints"][y] for y in range(len(self.Point2DList[x][i]["ImgPoints"])) }

            if len(Frame2Ddict) < 2:
                continue

            TriangTool = BundleAdjustmentTool_Triangulation(self.CamNames,CamParamDict)
            TriangTool.PrepareInputData(Frame2Ddict)
            Point3DDict = TriangTool.run()
            
            points_proj, points_detect = self.Reproject(Point3DDict,Frame2Ddict,CamParamDict )

            PointDiffs = (np.array(points_proj) - np.array(points_detect)).ravel()
            PointErrors.extend(PointDiffs.tolist())

        # import ipdb;ipdb.set_trace()
        return np.array(PointErrors, dtype=np.float64)
        # return EucError
        
    def Reproject(self, Point3DDict,Frame2Ddict,CamParamDict):
        """Given 3D points and extrinsics, do reprojection and get new 2d points"""

        points_proj = [] #projected points
        points_detect = [] #original 2d detections

        for cam in Frame2Ddict.keys():
            Point2D = cv2.projectPoints(np.array(list(Point3DDict.values())), 
                                        CamParamDict[cam]["R"],
                                        CamParamDict[cam]["T"],
                                        CamParamDict[cam]["cameraMatrix"],
                                        CamParamDict[cam]["distCoeffs"])[0]
            for i,kp in enumerate(Point3DDict.keys()):
                if kp in Frame2Ddict[cam]:
                    points_proj.append(Point2D[i][0])
                    points_detect.append(Frame2Ddict[cam][kp])


        return points_proj,points_detect
    
    def BundleAdjustmentSparsity(self, n,m):
        # m = self.CamIndexArr.size * 2
        # m = 976
        # or m = 976/2 *2
        # n = self.n_cameras * 15

        A = lil_matrix((m, n), dtype=int)
        NumPoints = int(m/2)

        A[:,:] = 1

        # i = np.arange(NumPoints)
        # for s in range(15):
        #     A[2 * i,NumPoints * 15 + s] = 1
        #     A[2 * i + 1, NumPoints * 15 + s] = 1

        # # for s in range(3):
        #     A[2 * i, self.n_cameras * 6 + self.PointIndexArr * 3 + s] = 1
        #     A[2 * i + 1, self.n_cameras * 6 + self.PointIndexArr * 3 + s] = 1

        return A
    
    def GetFinalParam(self,results):
        """Back transform arrays to get final param"""
        camera_params = results[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        Points3D = results[self.n_cameras * 6:].reshape((self.n_points,3))
        
        FinalDict = {}
        for i in range(len(camera_params)):
            FinalDict.update({
                self.CamNames[i]:{
                "rvec":camera_params[i][0:3],
                "tvec":camera_params[i][3:6]
                }
            })

        return FinalDict
    
    
    def run(self):
        """Run Bundle Adjustment"""
        # import ipdb;ipdb.set_trace()
        self.n_cameras = self.CamParamArr.shape[0]
        self.n_points = self.Points3DArr.shape[0]

        n = 6 * self.n_cameras#total number of parameters to optimize
        # m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.CamParamArr.ravel()
        f0 = self.GetResiduals(Params)
        m = f0.shape[0]#total number of residuals
        # import ipdb;ipdb.set_trace()
        
        A = self.BundleAdjustmentSparsity(n,m)
        # import ipdb;ipdb.set_trace()


        res = least_squares(self.GetResiduals, Params, jac_sparsity=A, 
                verbose=2, f_scale = 1,max_nfev = 50,x_scale='jac', ftol=1e-8,method='trf',loss='linear')


        Results = res.x[:self.n_cameras * 6].reshape((self.n_cameras, 6))

        FinalParamDict = self.SummarizeParams(Results)

        return FinalParamDict