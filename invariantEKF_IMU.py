#!/usr/bin/env python

import numpy as np
import math
import scipy.io
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag, logm, expm
from matplotlib import pyplot as plt

# import data
data = scipy.io.loadmat('data.mat')
a = data['a'] # linear acceleration measurement  (1277, 3)
dt = data['dt'] # time difference at each step  (1277, 1)
euler_gt = data['euler_gt']  # ground truth orientation as ZYX Euler angles (1277, 3)
g = data['g'] # gravity vector (3, 1)
omega = data['omega']  # gyroscope reading  (1277, 3)


class Right_IEKF:
    def __init__(self, system):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        self.A = system['A']  # error dynamics matrix
        self.f = system['f']  # process model
        self.H = system['H']  # measurement error matrix
        self.Q = system['Q']  # input noise covariance
        self.N = system['N']  # measurement noise covariance
        self.X = np.eye(3)    # state vector, so(3)
        self.P = 0.1 * np.eye(3)  # state covariance, adjustable
        
    def Ad(self, X):
        # Adjoint in SO(3) = R
        # See http://ethaneade.com/lie.pdf for detail derivation
        return X

    def wedge(self, x):
        # wedge operation for se(3) to put an R^3 vector to the Lie algebra basis
        G1 = np.array([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])  # omega_1
        G2 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]])  # omega_2
        G3 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])  # omega_3
        xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
        return xhat
    
    def prediction(self, u, delta_t):
        # EKF propagation (prediction) step
        self.X = self.f(self.X, u, delta_t) # Motion Model
        phi = expm(self.A * delta_t) # Transition matrix
        self.P = np.dot(np.dot(phi, self.P), phi.T) + np.dot(np.dot(self.Ad(self.X), self.Q), self.Ad(self.X).T)        
    
    def correction(self, Y, b):
        # RI-EKF correction Step
        Y = Y.reshape(-1,1) # 3x1; b: 3x1
        H = self.H(b)
        N = np.dot(np.dot(self.X, self.N), self.X.T)
        S = np.dot(np.dot(H, self.P), H.T) + N
        L = np.dot(np.dot(self.P, H.T), np.linalg.inv(S)) 
        
        # Update state
        nu = self.X @ Y - b
        delta = self.wedge(np.dot(L, nu))  # innovation in the spatial frame
        self.X = np.dot(expm(delta), self.X)

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - np.dot(L, H)
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(L, N), L.T)
        state = Rotation.from_matrix(np.asmatrix(self.X))
        return np.array(state.as_euler('zyx', degrees=False))


def motion_model(x, u, delta_t):
    u_skew = posemat(u[0], u[1], u[2])
    return np.dot(x, expm(u_skew * delta_t)) 
    # expm is computing the matrix exponential

def measurement_Jacobian(m):
    # measurement error matrix
    # m.shape: 3x1
    m1 = m[0].item()
    m2 = m[1].item()
    m3 = m[2].item()
    H = np.array([[0, -m3, m2],
                  [m3, 0, -m1],
                  [-m2, m1, 0]])
    return H

def posemat(w1, w2, w3):
    # construct a so(3) matrix element
    H = np.array([[0, -1*w3, w2],
                  [w3, 0, -1*w1],
                  [-1*w2, w1, 0]], dtype=object)
    return H


# construct noise free motion trajectory (sanity check for the generated inputs)
path = {}
path['T'] = euler_gt[0]
path['x'] = euler_gt[1]
path['y'] = euler_gt[2]

sys = {}
sys['A'] = 0 * np.eye(3)
sys['f'] = motion_model
sys['H'] = measurement_Jacobian
sys['Q'] = np.diag(np.power([0.01, 0.01, 0.01], 2))
sys['N'] = np.diag(np.power([0.75, 0.75, 0.75], 2))

iekf_filter = Right_IEKF(sys)  # create an RI-EKF object

result = np.zeros_like(euler_gt)
u = omega
time = [0.]
cul_t = 0
for i in range(len(dt)):
    iekf_filter.prediction(u[i].reshape(-1, 1), dt[i].reshape(-1, 1))
    result[i] = iekf_filter.correction(a[i], g)
    cul_t += dt[i]
    time.append(i)
time /= cul_t
# result.shape: (1278, 3)
# time.shape: (1278)

# plotting
fig1 = plt.figure()
# yaw
plt.title('Yaw')
plt.xlabel('time')
plt.ylabel('angle')
line1, = plt.plot(time, result[:, 0], '-', color='blue', linewidth=2)
line2, = plt.plot(time, euler_gt[:, 0], '-', color='tomato', linewidth=2)
plt.legend([line1, line2], ['estimated angle', 'ground truth'], loc='best')
plt.grid(True)
# plt.show()

fig2 = plt.figure()
# pitch
plt.title('Pitch')
plt.xlabel('time')
plt.ylabel('angle')
line3, = plt.plot(time, result[:, 1], '-', color='blue', linewidth=2)
line4, = plt.plot(time, euler_gt[:, 1], '-', color='tomato', linewidth=2)
plt.legend([line3, line4], ['estimated angle', 'ground truth'], loc='best')
plt.grid(True)
# plt.show()

fig3 = plt.figure()
# roll
plt.title('Roll')
plt.xlabel('time')
plt.ylabel('angle')
line5, = plt.plot(time, result[:, 2], '-', color='blue', linewidth=2)
line6, = plt.plot(time, euler_gt[:, 2], '-', color='tomato', linewidth=2)
plt.legend([line5, line6], ['estimated angle', 'ground truth'], loc='best')
plt.grid(True)
plt.show()
