import numpy as np
import matplotlib.pyplot as plt
import scipy.io

np.random.seed(1)
m = 3
n = 5
N = 50
A = scipy.io.loadmat('A.mat')['A']
Q = scipy.io.loadmat('Q.mat')['Q']
Qhalf = scipy.io.loadmat('Qhalf.mat')['Qhalf']
Rhalf = scipy.io.loadmat('Rhalf.mat')['Rhalf']
R = scipy.io.loadmat('R.mat')['R']
C = scipy.io.loadmat('C.mat')['C']
P = np.eye(n,n)
x = scipy.io.loadmat('x_init.mat')['x']
xhat = np.zeros((n,1))

# Initialize arrays for plotting
norm_P = np.zeros(N)
norm_x = np.zeros(N)
norm_dx = np.zeros(N)

for i in range(N):
    
    # Project the error covariance ahead (use Q)
    P = A @ P @ A.T + Q  # Time Update for covariance

    # Project the state ahead (both prediction and measurement) (use Qhalf and Rhalf)
    wk = Qhalf @ np.random.randn(n, 1)  # Process noise
    x = A @ x + wk  # True state update
    vk = Rhalf @ np.random.randn(m, 1)  # Measurement noise
    z = C @ x + vk  # Measurement generation
    xhat_prior = A @ xhat  # Predict estimate (prior)

    # Measurement update (use R)
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)  # Kalman Gain
    xhat = xhat_prior + K @ (z - C @ xhat_prior)  # Correct estimate
    P = (np.eye(n) - K @ C) @ P  # Update covariance

    # Plotting
    # Inside the loop:    
    norm_P[i] = np.linalg.norm(P)
    norm_x[i] = np.linalg.norm(x)
    norm_dx[i] = np.linalg.norm(x-xhat)

print(norm_x)
x = np.linspace(0,N,N)
fig, axs = plt.subplots(3, 1)
axs[0].plot(x, norm_P)
axs[0].set_title('Norm P')
axs[1].plot(x, norm_x, 'tab:orange')
axs[1].set_title('Norm x')
axs[2].plot(x, norm_dx, 'tab:green')
axs[2].set_title('Norm x - x_hat')
plt.show()
