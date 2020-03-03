# !/usr/bin/python

# General import 
import numpy as np
from numpy.linalg import inv

class kalmanFilter():

    def __init__(self, x, P, F, H, Q, R):
        
        self.x = x
        self.x_ = np.zeros(6)
        self.P = P
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.I = np.eye(6)
    

    def prediction(self):
        # Prediction step
        self.x_ = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self,obs):
        # Update step
        y = obs - self.H.dot(self.x_)
        S = self.R + self.H.dot(self.P).dot(self.H.T)
        K = self.P.dot(self.H.T).dot(inv(S))

        self.x = self.x_ + K.dot(y)
        self.P = (self.I-K.dot(self.H)).dot(self.P).dot((self.I-K.dot(self.H)).T)+K.dot(self.R).dot(K.T)

class extendedKalmanFilter():

    def __init__(self, x, P, F, H, Q, R):
        
        self.x = x
        self.x_ = np.zeros(6)
        self.P = P
        self.F = F
        self.H = H
        #self.h = h
        self.Q = Q
        self.R = R
        self.I = np.eye(6)
    

    def prediction(self):
        # Prediction step
        self.x_ = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, obs, h_mat):

        x = self.x_[0]
        y = self.x_[1]
        z = self.x_[2]

        '''
        h = np.array([
            np.cos(roll)*np.cos(pitch)*x + (np.cos(roll)*np.sin(pitch)*np.sin(yaw)-np.cos(yaw)*np.sin(roll))*y + (np.sin(roll)*np.sin(yaw)+np.cos(roll)*np.cos(yaw)*np.sin(pitch))*z + cx,
            np.cos(pitch)*np.sin(roll)*x + (np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(pitch)*np.sin(yaw))*y + (np.cos(yaw)*np.sin(roll)*np.sin(pitch)-np.cos(roll)*np.sin(yaw))*z + cy,
            -np.sin(pitch)*x             + (np.cos(pitch)*np.sin(yaw))*y                                       + (np.cos(pitch)*np.cos(yaw))*z + cz])
        
        self.H = np.array([[np.cos(roll)*np.cos(pitch), np.cos(roll)*np.sin(pitch)*np.sin(yaw)-np.cos(yaw)*np.sin(roll), np.sin(roll)*np.sin(yaw)+np.cos(roll)*np.cos(yaw)*np.sin(pitch),   0, 0, 0 ],
                  [np.cos(pitch)*np.sin(roll),        np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(pitch)*np.sin(yaw),   np.cos(yaw)*np.sin(roll)*np.sin(pitch)-np.cos(roll)*np.sin(yaw), 0, 0, 0],
                  [-np.sin(pitch),                  np.cos(pitch)*np.sin(yaw),                                 np.cos(pitch)*np.cos(yaw),                                0, 0, 0,]])

        '''

        self.H[0:3,0:3] = h_mat[0:3,0:3]

        # Update step
        b = h_mat.dot(np.array([[x,y,z,1]]).T)[0:3,:]
        
        y = obs - np.array([b[0,0], b[1,0], b[2,0]])

        S = self.R + self.H.dot(self.P).dot(self.H.T)
        K = self.P.dot(self.H.T).dot(inv(S))

        self.x = self.x_ + K.dot(y)
        self.P = (self.I-K.dot(self.H)).dot(self.P)#.dot((self.I-K.dot(self.H)).T)+K.dot(self.R).dot(K.T)
