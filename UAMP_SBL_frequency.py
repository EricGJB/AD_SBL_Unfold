import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary

# use partial dataset 
test_num = 200

data = io.loadmat('./data/data.mat')

H_list = data['H_list'][:test_num]
Y_list = data['Y_list'][:test_num]

H_list = np.transpose(H_list,(0,2,1))
print(H_list.shape)

print(Y_list.shape)

W = np.matrix(data['W'])
print(W.shape)

Nr = 32

SNR = 20 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 8

num_sc = 16 # number of subcarriers
fc = 28 * 1e9 # central frequency
fs = 4 * 1e9 # bandwidth
eta = fs / num_sc  # subcarrier spacing

G_angle = 64 # angular resolution 

A_R = dictionary(Nr, G_angle)
A_R = np.matrix(A_R) 

Phi = W.H.dot(A_R)
Phi = np.array(Phi)


#%%  UAMP-SBL
svd = 1 # whether to execute unitary pre-processing 
if svd:
    U,Sigma,V = np.linalg.svd(Phi)
    y_list = np.reshape(Y_list,(test_num*num_sc,Mr))
    y_list = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(y_list)))
    y_list = np.reshape(y_list,(test_num,num_sc,Mr,1))
    Phi = np.transpose(np.conjugate(U)).dot(Phi)
else:
    y_list = np.expand_dims(Y_list,axis=-1)

def AMP_SBL(R,Phi,G,Mr,sigma_2,num_iter):
    # initialization
    Tau_x = np.ones((G,1))
    X_hat = np.zeros((G,1))
    epsilon = 0.001 # the initial epsilon value 
    Gamma_hat = np.ones((G,1))
    # accurate noise precision is assumed to be known 
    beta = 1/sigma_2 
    S = np.zeros((Mr,1))
    for i in range(num_iter):
        Tau_p = (np.abs(Phi)**2).dot(Tau_x)
        P = Phi.dot(X_hat)-Tau_p*S
        Tau_s = 1/(Tau_p+1/beta)
        S = Tau_s*(R-P)
        Tau_q = 1/((np.abs(np.transpose(np.conjugate(Phi)))**2).dot(Tau_s))
        Q = X_hat + Tau_q*np.transpose(np.conjugate(Phi)).dot(S)
        Tau_x = Tau_q/(1+Tau_q*Gamma_hat)
        X_hat = Q/(1+Tau_q*Gamma_hat)
        Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+Tau_x)
        # adaptive epsilon method 
        epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
        # heuristic epsilon methods 
        # epsilon = 3*sigma_2 
        # epsilon = 0
    return X_hat

num_iter = 100

error = 0
error_nmse = 0
for i in range(test_num): 
    if i%10==0:
        print('Sample %d/%d'%(i,test_num))
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    prediction_X = np.zeros((G_angle,num_sc),dtype=np.complex64)
    for j in range(num_sc):
        r = y_list[i,j]
        x_hat = AMP_SBL(r,Phi,G_angle,Mr,sigma_2,num_iter)
        prediction_X[:,j] = np.squeeze(x_hat)
        prediction_h = A_R.dot(x_hat)
        prediction_H[:,j] = np.squeeze(prediction_h)
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2

mse_amp_sbl = error / (test_num * Nr * num_sc)
nmse_amp_sbl = error_nmse / test_num
print(mse_amp_sbl)
print(nmse_amp_sbl)

# print(beta_hat)
plt.figure()
plt.imshow(np.abs(prediction_X),cmap='gray_r')
plt.xlabel('Subcarriers')
plt.ylabel('Angular Grids')
plt.title('Angular-Frequency Channel Modulus')  


#%% 
# def AMP_SBL(R,Phi,G,sigma_2,num_iter):
#     # initialization
#     Lamda = np.expand_dims(Sigma**2,axis=-1)
#     tau_x = 1
#     X_hat = np.zeros((G,1))
#     epsilon = 0.001
#     Gamma_hat = np.ones((G,1))
#     # accurate noise precision
#     beta = 1/sigma_2 
#     S = np.zeros((Mr,1))
#     num_iter = 100
#     for i in range(num_iter):
#         Tau_p = tau_x*Lamda
#         P = Phi.dot(X_hat)-Tau_p*S
#         Tau_s = 1/(Tau_p+1/beta)
#         S = Tau_s*(R-P)
#         tau_q = G/(np.transpose(np.conjugate(Lamda)).dot(Tau_s))
#         Q = X_hat + tau_q*np.transpose(np.conjugate(Phi)).dot(S)
#         tau_x = tau_q/G*np.sum(1/(1+tau_q*Gamma_hat))
#         X_hat = Q/(1+tau_q*Gamma_hat)
#         Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+tau_x)
#         epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
        
#     return X_hat


# def AMP_SBL_estimate_noise(R,Phi,G):
#     # initialization
#     Lamda = np.expand_dims(Sigma**2,axis=-1)
#     tau_x = 1
#     X_hat = np.zeros((G,1))
#     epsilon = 0.001
#     Gamma_hat = np.ones((G,1))
#     # noise precision
#     beta_hat = 1
#     S = np.zeros((Mr,1))
#     num_iter = 100
#     for i in range(num_iter):
#         Tau_p = tau_x*Lamda
#         P = Phi.dot(X_hat)-Tau_p*S
#         V_h = Tau_p/(1+beta_hat*Tau_p)
#         H_hat = (beta_hat*Tau_p*R+P)/(1+beta_hat*Tau_p)
#         beta_hat = Mr/(np.linalg.norm(R-H_hat)**2+np.sum(V_h))
#         Tau_s = 1/(Tau_p+1/beta_hat)
#         S = Tau_s*(R-P)
#         tau_q = G/(np.transpose(np.conjugate(Lamda)).dot(Tau_s))
#         Q = X_hat + tau_q*np.transpose(np.conjugate(Phi)).dot(S)
#         tau_x = tau_q/G*np.sum(1/(1+tau_q*Gamma_hat))
#         X_hat = Q/(1+tau_q*Gamma_hat)
#         Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+tau_x)
#         epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
        
#     return X_hat,beta_hat


# def AMP_SBL_estimate_noise_v2(R,Phi,G):
#     # initialization
#     Tau_x = np.ones((G,1))
#     X_hat = np.zeros((G,1))
#     epsilon = 0.001
#     Gamma_hat = np.ones((G,1))
#     # noise precision
#     beta_hat = 1
#     S = np.zeros((Mr,1))
#     num_iter = 100
#     for i in range(num_iter):
#         Tau_p = (np.abs(Phi)**2).dot(Tau_x)
#         P = Phi.dot(X_hat)-Tau_p*S
#         V_h = Tau_p/(1+beta_hat*Tau_p)
#         H_hat = (beta_hat*Tau_p*R+P)/(1+beta_hat*Tau_p)
#         beta_hat = Mr/(np.linalg.norm(R-H_hat)**2+np.sum(V_h))
#         Tau_s = 1/(Tau_p+1/beta_hat)
#         S = Tau_s*(R-P)
#         Tau_q = 1/((np.abs(np.transpose(np.conjugate(Phi)))**2).dot(Tau_s))
#         Q = X_hat + Tau_q*np.transpose(np.conjugate(Phi)).dot(S)
#         Tau_x = Tau_q/(1+Tau_q*Gamma_hat)
#         X_hat = Q/(1+Tau_q*Gamma_hat)
#         Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+Tau_x)
#         epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
        
#     return X_hat,beta_hat


## plot error curve
# def AMP_SBL_estimate_noise_v3(R,Phi):
#     # initialization
#     Tau_x = np.ones((G,1))
#     X_hat = np.zeros((G,1))
#     epsilon = 0.001
#     Gamma_hat = np.ones((G,1))
#     # noise precision
#     beta_hat = 1
#     S = np.zeros((Mr,1))
#     num_iter = 100
#     X_hat_list = []
#     for i in range(num_iter):
#         Tau_p = (np.abs(Phi)**2).dot(Tau_x)
#         P = Phi.dot(X_hat)-Tau_p*S
#         V_h = Tau_p/(1+beta_hat*Tau_p)
#         H_hat = (beta_hat*Tau_p*R+P)/(1+beta_hat*Tau_p)
#         beta_hat = Mr/(np.linalg.norm(R-H_hat)**2+np.sum(V_h))
#         Tau_s = 1/(Tau_p+1/beta_hat)
#         S = Tau_s*(R-P)
#         Tau_q = 1/((np.abs(np.transpose(np.conjugate(Phi)))**2).dot(Tau_s))
#         Q = X_hat + Tau_q*np.transpose(np.conjugate(Phi)).dot(S)
#         Tau_x = Tau_q/(1+Tau_q*Gamma_hat)
#         X_hat = Q/(1+Tau_q*Gamma_hat)
#         Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+Tau_x)
#         epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
#         X_hat_list.append(X_hat)
#     return X_hat_list,beta_hat

# error_list = np.zeros((test_num,num_sc,100))
# for i in range(test_num):
#     for j in range(num_sc):
#         R = r_list[i,j]
#         X_hat_list = AMP_SBL_estimate_noise_v3(R,Phi)
#         iter_count = 0
#         for X_hat in X_hat_list[0]:
#             prediction_h = A_R.dot(X_hat)
#             true_h = np.expand_dims(h_list[i,j],axis=-1)
#             error_list[i,j,iter_count] = np.linalg.norm(prediction_h - true_h) ** 2
#             iter_count = iter_count + 1
            
# mse_amp_sbl_list = np.mean(error_list,axis=(0,1)) / Nr

# plt.figure()
# plt.plot(mse_amp_sbl_list)