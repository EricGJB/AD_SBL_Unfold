import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary,update_mu_Sigma_delay,update_mu_Sigma_PC,update_alpha_PC,C2R,R2C#,dictionary_delay

# use partial dataset 
test_num = 10

Nr = 32

SNR = 20 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 16

num_sc = 32 # number of subcarriers
fc = 28 * 1e9 # central frequency
fs = 4 * 1e9 # bandwidth
eta = fs / num_sc  # subcarrier spacing

G_angle = 64 # angular resolution

channel_model = 'cluster'

data = io.loadmat('./data/data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))

H_list = data['H_list'][:test_num]

W = np.matrix(data['W'])
print(W.shape)

A_R = dictionary(Nr, G_angle)
A_R = np.matrix(A_R) 

# delay domain sparse transformation dictionary 
tau_max = 25*1e-9
# if the value is larger than 1, multiple path delays will correspond to the same delay grid 
max_normalized_delay_value = eta*tau_max 
print(max_normalized_delay_value)

G_delay = 64
A_K = dictionary(num_sc, G_delay)
# each path delay correspond to a unique delay grid, but the correlation among atoms is not in control like the oversampled DFT 
# A_K = dictionary_delay(num_sc,eta,tau_max,G_delay)
A_K = np.matrix(A_K)

Kron2 = np.kron(A_K,A_R)

Q = np.kron(np.eye(num_sc),W.H)

Phi = Q.dot(Kron2)
# another equivalent calculation formula of Phi
# Phi = np.kron(A_K,W.H.dot(A_R))

Phi = np.array(Phi)
Phi_list = np.reshape(Phi,(1,num_sc*Mr, G_angle*G_delay))
Phi_list = np.tile(Phi_list,(test_num,1,1))
Phi_real_imag_list = C2R(Phi_list)

print(Phi_real_imag_list.shape) # (test_num,num_sc*Mr,G_angle*G_delay,2)

# Y_list = data['Y_list'][:test_num]
# # notice that, the dimension num_sc has to be put before Nr/Mr to ensure correct vectorization orders
# y_list = np.reshape(Y_list,(test_num,num_sc*Mr))
# y_real_imag_list = C2R(y_list)

test_data = io.loadmat('./data/test_data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))
y_real_imag_list = test_data['y_real_imag_list_test'][:test_num]
U = test_data['U']
y_list = R2C(y_real_imag_list)
y_list = np.transpose(U.dot(np.transpose(y_list)))
y_real_imag_list = C2R(y_list)

print(y_real_imag_list.shape) # (test_num,num_sc*Mr,2)


#%%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)

num_layers = 100
batch_size = 32

plot_sample_index = 1

H_list = np.transpose(H_list,(0,2,1))


#%% SBL
def SBL_layer(Mr, num_sc, G_angle, G_delay, sigma_2):
    Phi_real_imag = Input(shape=(num_sc*Mr, G_angle*G_delay, 2))
    y_real_imag = Input(shape=(num_sc*Mr, 2))
    alpha_list = Input(shape=(G_angle*G_delay,))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_delay(x,sigma_2,num_sc,Mr))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    alpha_list_updated = Lambda(lambda x:x[0]+x[1])([mu_square,diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated[:,:,0],mu_real,mu_imag])
    model.compile(loss='mse')
    return model
SBL_single_layer = SBL_layer(Mr, num_sc, G_angle, G_delay, sigma_2)

alpha_list = np.ones((test_num,G_angle*G_delay)) #initialization

mse_sbl_list = []

for i in range(num_layers):
    if i%10==0:
        print('SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = SBL_single_layer.predict([Phi_real_imag_list, y_real_imag_list, alpha_list],batch_size=batch_size)
   
    predictions_X = mu_real+1j*mu_imag
    predictions_X = np.reshape(predictions_X,(test_num,G_delay,G_angle))
    # notice that, the inverse vectorization operation is also to re-stack column-wisely
    predictions_X = np.transpose(predictions_X,(0,2,1))
    
    error = 0
    for i in range(test_num):
        true_H = H_list[i]
        prediction_H = (A_R.dot(predictions_X[i])).dot(A_K.T)
        error = error + np.linalg.norm(prediction_H - true_H) ** 2
    mse_sbl_list.append(error / (test_num * Nr * num_sc))  

# loss curve 
plt.figure()
plt.plot(mse_sbl_list)  
plt.xlabel('SBL iteration')
plt.ylabel('MSE')
plt.show()

# final performance 
predictions_X = mu_real+1j*mu_imag
predictions_X = np.reshape(predictions_X,(test_num,G_delay,G_angle))
# notice that, the inverse vectorization operation is also to re-stack column-wisely
predictions_X = np.transpose(predictions_X,(0,2,1))

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = (A_R.dot(predictions_X[i])).dot(A_K.T)
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

plt.figure()
plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
plt.xlabel('$G_D$')
plt.ylabel('$G_A$')
plt.title('AD channel modulus')
plt.savefig('./figures/AD_channel_1.pdf')
plt.show()


#%% PC-SBL
#TODO: best hyperparameters to be searched
a = 0.5
b = 1e-6
beta = 0.5

G = G_angle*G_delay
def PCSBL_layer(Mr, G, num_sc, sigma_2, a,b,beta):
    # assume G_angle = G_delay
    Phi_real_imag = Input(shape=(num_sc*Mr, G, 2))
    y_real_imag_0 = Input(shape=(num_sc*Mr, 2))
    y_real_imag = tf.expand_dims(y_real_imag_0,axis=-2)
    alpha_list = Input(shape=(G,))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_PC(x, int(np.sqrt(G)), 1, sigma_2, Mr, num_sc, beta))\
        ([Phi_real_imag, y_real_imag, alpha_list])
    alpha_list_updated = update_alpha_PC([mu_real, mu_imag, diag_Sigma_real], int(np.sqrt(G)), 1, a, b, beta)
    model = Model(inputs=[Phi_real_imag, y_real_imag_0, alpha_list], outputs=[alpha_list_updated[:,:,0],mu_real,mu_imag])
    model.compile(loss='mse')
    return model
PCSBL_single_layer = PCSBL_layer(Mr,G,num_sc,sigma_2,a,b,beta)

alpha_list = np.ones((test_num,G)) #initialization
for i in range(num_layers):
    if i%10==0:
        print('PC-SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = PCSBL_single_layer.predict([Phi_real_imag_list, y_real_imag_list, alpha_list],batch_size=batch_size)

predictions_X = mu_real+1j*mu_imag
predictions_X = np.reshape(predictions_X,(test_num,G_delay,G_angle))
# notice that, the inverse vectorization operation is also to re-stack column-wisely
predictions_X = np.transpose(predictions_X,(0,2,1))

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = (A_R.dot(predictions_X[i])).dot(A_K.T)
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

# plt.figure()
# plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
# plt.xlabel('$G_D$')
# plt.ylabel('$G_A$')
# plt.show()


#%% With frequency-rotated dictionaries
# construct the new measurement matrix
def dictionary_angle(N, G, sin_value):
    A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G)))) / np.sqrt(N)
    return A

sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)

A_list = []
A_list_expanded = np.zeros((num_sc*Nr,num_sc*G_angle),dtype=np.complex64)
for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    sin_value_scn = sin_value_sc0*(fn/fc) 
    # sin_value_scn = sin_value_sc0 # frequency-independent measurement matrices
    A_n = dictionary_angle(Nr, G_angle, sin_value_scn)
    A_list.append(A_n)
    A_list_expanded[n*Nr:(n+1)*Nr,n*G_angle:(n+1)*G_angle] = A_n

new_Phi = (np.kron(np.eye(num_sc),W.H).dot(A_list_expanded)).dot(np.kron(A_K,np.eye(G_angle)))

new_Phi = np.array(new_Phi)
Phi_list = np.reshape(new_Phi,(1,num_sc*Mr, G_angle*G_delay))
Phi_list = np.tile(Phi_list,(test_num,1,1))
Phi_real_imag_list = C2R(Phi_list)

alpha_list = np.ones((test_num,G_angle*G_delay)) #initialization
for i in range(num_layers):
    if i%10==0:
        print('FR SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = SBL_single_layer.predict([Phi_real_imag_list, y_real_imag_list, alpha_list],batch_size=batch_size)

predictions_X = mu_real+1j*mu_imag
predictions_X = np.reshape(predictions_X,(test_num,G_delay,G_angle))
# notice that, the inverse vectorization operation is also to re-stack column-wisely
predictions_X = np.transpose(predictions_X,(0,2,1))

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_Q = predictions_X[i].dot(A_K.T) # angular-frequency channel prediction
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(prediction_Q[:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

plt.figure()
plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
plt.xlabel('$G_D$')
plt.ylabel('$G_A$')
plt.title('AD channel modulus')
plt.savefig('./figures/AD_channel_2.pdf')
plt.show()


#%% PC-SBL with frequency rotation
alpha_list = np.ones((test_num,G_angle*G_delay)) #initialization
for i in range(num_layers):
    if i%10==0:
        print('FR PC-SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = PCSBL_single_layer.predict([Phi_real_imag_list, y_real_imag_list, alpha_list],batch_size=batch_size)

predictions_X = mu_real+1j*mu_imag
predictions_X = np.reshape(predictions_X,(test_num,G_delay,G_angle))
# notice that, the inverse vectorization operation is also to re-stack column-wisely
predictions_X = np.transpose(predictions_X,(0,2,1))

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_Q = predictions_X[i].dot(A_K.T) # angular-frequency channel prediction
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(prediction_Q[:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

# plt.figure()
# plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
# plt.xlabel('$G_D$')
# plt.ylabel('$G_A$')
# plt.show()


# compare the good list performance with UAMP SBL
# good_set = io.loadmat('./results/UAMP_SBL_good_test_samples_%dBeams_%dSNR_path.mat'%(Mr,SNR))
# index_list = np.squeeze(good_set['index_list'])
# nmse_list_UAMP = np.squeeze(good_set['nmse_list'])
# nmse_list_SBL = []
# for i in index_list:
#     true_H = H_list[i]
#     prediction_Q = predictions_X[i].dot(A_K.T) # angular-frequency channel prediction
#     prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
#     for j in range(num_sc):
#         prediction_h = A_list[j].dot(prediction_Q[:,j:j+1])
#         prediction_H[:,j:j+1] = prediction_h
#     sample_nmse = (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
#     nmse_list_SBL.append(sample_nmse)
#
# nmse_SBL = np.mean(nmse_list_SBL)
# nmse_UAMP = np.mean(nmse_list_UAMP)
#
# print('NMSE of %d/%d good testing samples:'%(len(index_list),test_num))
# print('SBL FR:%.4f'%nmse_SBL)
# print('UAMP-SBL FR:%.4f'%nmse_UAMP)
