import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary,update_mu_Sigma,update_mu_Sigma_MSBL,C2R,R2C

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

W = np.matrix(data['W'])
print(W.shape)

H_list = data['H_list'][:test_num]
H_list = np.transpose(H_list,(0,2,1))
print(H_list.shape)

# Y_list = data['Y_list'][:test_num]
# Y_real_imag_list = C2R(Y_list)
# Y_real_imag_list = np.transpose(Y_real_imag_list,(0,2,1,3))

test_data = io.loadmat('./data/test_data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))
y_real_imag_list = test_data['y_real_imag_list_test'][:test_num]
U = test_data['U']
y_list = R2C(y_real_imag_list)
y_list = np.transpose(U.dot(np.transpose(y_list)))
Y_list = np.reshape(y_list,(test_num,num_sc,Mr))
Y_list = np.transpose(Y_list,(0,2,1)) # (test_num,Mr,num_sc)
Y_real_imag_list = C2R(Y_list)

print(Y_real_imag_list.shape)

A_R = dictionary(Nr, G_angle)
A_R = np.matrix(A_R) 

Phi = W.H.dot(A_R)
Phi = np.array(Phi)
Phi_list = np.reshape(Phi,(1,Mr,G_angle))
Phi_list = np.tile(Phi_list,(test_num,1,1))
Phi_real_imag_list = C2R(Phi_list)

print(Phi_real_imag_list.shape) # (test_num,Mr,G,2)


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

#%% SBL 
def SBL_layer(Mr, G, num_sc, sigma_2):
    Phi_real_imag = Input(shape=(Mr, G, 2))
    y_real_imag = Input(shape=(Mr, num_sc, 2))
    alpha_list = Input(shape=(G,num_sc))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x,num_sc,sigma_2,Mr))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    alpha_list_updated = Lambda(lambda x:x[0]+x[1])([mu_square,diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated,mu_real,mu_imag])
    return model
SBL_single_layer = SBL_layer(Mr,G_angle,num_sc,sigma_2)

alpha_list = np.ones((test_num,G_angle,num_sc)) #initialization

mse_sbl_list = []

for i in range(num_layers):
    if i%10==0:
        print('SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = SBL_single_layer.predict([Phi_real_imag_list, Y_real_imag_list, alpha_list],batch_size=batch_size)

    # record performance of every iteration
    predictions_X = mu_real+1j*mu_imag    
    error = 0
    for i in range(test_num):
        true_H = H_list[i]
        prediction_H = A_R.dot(predictions_X[i])
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

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = A_R.dot(predictions_X[i])
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

plt.figure()
plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
plt.xlabel('$K$')
plt.ylabel('$G_A$')
plt.title('AF channel modulus')
plt.savefig('./figures/AF_channel_1.pdf')
plt.show()

#%% SBL with early stop 
# epsilon = 0.03

# error = 0
# error_nmse = 0
# for i in range(test_num):
#     print('Sample %d'%i)
#     alpha_list = np.ones((1,G_angle,num_sc)) #initialization     
#     for j in range(num_layers):
#         [alpha_list_new,mu_real,mu_imag] = SBL_single_layer.predict([Phi_real_imag_list[i:i+1], Y_real_imag_list[i:i+1], alpha_list])
#         # print(np.linalg.norm(alpha_list_new-alpha_list)/np.linalg.norm(alpha_list))
#         if (np.linalg.norm(alpha_list_new-alpha_list)/np.linalg.norm(alpha_list))<epsilon:
#             print(j)
#             break
#         alpha_list = np.copy(alpha_list_new)    
                
#     # final performance 
#     prediction_X = mu_real+1j*mu_imag

#     true_H = H_list[i]
#     prediction_H = A_R.dot(prediction_X)
#     error = error + np.linalg.norm(prediction_H - true_H) ** 2/ (Nr * num_sc)
#     error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
    
# mse_sbl = error/test_num
# nmse_sbl = error_nmse/test_num

# print(mse_sbl)
# print(nmse_sbl)

# plt.figure()
# plt.imshow(np.abs(prediction_X[0]),cmap='gray_r')
# plt.xlabel('$K$')
# plt.xlabel('$G_A$')


#%% MSBL
# def MSBL_layer(Mr, num_sc, G, sigma_2):
#     Phi_real_imag = Input(shape=(Mr, G, 2))
#     y_real_imag = Input(shape=(Mr, num_sc, 2))
#     alpha_list = Input(shape=(G, num_sc))
#
#     # update mu and Sigma
#     mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x,num_sc,sigma_2,Mr))(
#         [Phi_real_imag, y_real_imag, alpha_list])
#     # update alpha_list
#     mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
#     mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
#     mu_square = tf.tile(mu_square_average,(1,1,num_sc))
#     alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
#     model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
#     return model
#
# MSBL_single_layer = MSBL_layer(Mr, num_sc, G_angle, sigma_2)

# low-complexity version
def MSBL_layer(Mr, num_sc, G, sigma_2):
    Phi_real_imag = Input(shape=(Mr, G, 2))
    y_real_imag = Input(shape=(Mr, num_sc, 2))
    alpha_list = Input(shape=(G, num_sc))

    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_MSBL(x,num_sc,sigma_2,Mr))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
    mu_square = tf.tile(mu_square_average,(1,1,num_sc))
    alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    return model

MSBL_single_layer = MSBL_layer(Mr, num_sc, G_angle, sigma_2)

alpha_list = np.ones((test_num, G_angle, num_sc))  # initialization

mse_sbl_list = []

for i in range(num_layers):
    if i % 10 == 0:
        print('MSBL iteration %d' % i)
    [alpha_list, mu_real, mu_imag] = MSBL_single_layer.predict([Phi_real_imag_list, Y_real_imag_list, alpha_list],
                                                              batch_size=batch_size)

    # record performance of every iteration
    predictions_X = mu_real + 1j * mu_imag
    error = 0
    for i in range(test_num):
        true_H = H_list[i]
        prediction_H = A_R.dot(predictions_X[i])
        error = error + np.linalg.norm(prediction_H - true_H) ** 2
    mse_sbl_list.append(error / (test_num * Nr * num_sc))

# loss curve
plt.figure()
plt.plot(mse_sbl_list)
plt.xlabel('MSBL iteration')
plt.ylabel('MSE')
plt.show()

# final performance
predictions_X = mu_real + 1j * mu_imag

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = A_R.dot(predictions_X[i])
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H - true_H) / np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

# plt.figure()
# plt.imshow(np.abs(predictions_X[plot_sample_index]), cmap='gray_r')
# plt.xlabel('$K$')
# plt.ylabel('$G_A$')
# plt.show()


#%% With frequency-rotated dictionaries
# construct the new measurement matrix
def dictionary_angle(N, G, sin_value):
    A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G)))) / np.sqrt(N)
    return A

sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)

A_list = []
for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    sin_value_scn = sin_value_sc0*(fn/fc) 
    # sin_value_scn = sin_value_sc0 # frequency-independent measurement matrices
    A_list.append(dictionary_angle(Nr, G_angle, sin_value_scn))
A_list = np.array(A_list)

Phi_list = np.zeros((num_sc, Mr, G_angle)) + 1j * np.zeros((num_sc, Mr, G_angle))
for i in range(num_sc):
    Phi_list[i] = W.H.dot(A_list[i])
Phi_list = np.tile(np.expand_dims(Phi_list,axis=0),(test_num,1,1,1))
Phi_real_imag_list = C2R(Phi_list)
# print(Phi_real_imag_list.shape)

def FR_SBL_layer(Mr, num_sc, G, sigma_2):
    Phi_real_imag = Input(shape=(num_sc, Mr, G, 2))
    y_real_imag = Input(shape=(Mr, num_sc, 2))
    alpha_list = Input(shape=(G, num_sc))
    # update mu and Sigma
    mu_real = []
    mu_imag = []
    diag_Sigma_real = []
    for i in range(num_sc):
        mu_real_sc, mu_imag_sc, diag_Sigma_real_sc = Lambda(lambda x: update_mu_Sigma(x,1,sigma_2,Mr))(
            [Phi_real_imag[:,i], y_real_imag[:,:,i:i+1], alpha_list[:,:,i:i+1]])
        mu_real.append(mu_real_sc)
        mu_imag.append(mu_imag_sc)
        diag_Sigma_real.append(diag_Sigma_real_sc)
    mu_real = tf.concat(mu_real,axis=-1)
    mu_imag = tf.concat(mu_imag,axis=-1)
    diag_Sigma_real = tf.concat(diag_Sigma_real,axis=-1)
    
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])

    alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    return model

FR_SBL_single_layer = FR_SBL_layer(Mr, num_sc, G_angle, sigma_2)

alpha_list = np.ones((test_num,G_angle,num_sc)) #initialization
for i in range(num_layers):
    if i%10==0:
        print('FR SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = FR_SBL_single_layer.predict([Phi_real_imag_list, Y_real_imag_list, alpha_list],batch_size=batch_size)

predictions_X = mu_real+1j*mu_imag

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(predictions_X[i,:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

plt.figure()
plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
plt.xlabel('$K$')
plt.ylabel('$G_A$')
plt.title('AF channel modulus')
plt.savefig('./figures/AF_channel_2.pdf')
plt.show()

#%% 
def FR_MSBL_layer(Mr, num_sc, G, sigma_2):
    Phi_real_imag = Input(shape=(num_sc, Mr, G, 2))
    y_real_imag = Input(shape=(Mr, num_sc, 2))
    alpha_list = Input(shape=(G, num_sc))
    # update mu and Sigma
    mu_real = []
    mu_imag = []
    diag_Sigma_real = []
    for i in range(num_sc):
        mu_real_sc, mu_imag_sc, diag_Sigma_real_sc = Lambda(lambda x: update_mu_Sigma(x,1,sigma_2,Mr))(
            [Phi_real_imag[:,i], y_real_imag[:,:,i:i+1], alpha_list[:,:,i:i+1]])
        mu_real.append(mu_real_sc)
        mu_imag.append(mu_imag_sc)
        diag_Sigma_real.append(diag_Sigma_real_sc)
    mu_real = tf.concat(mu_real,axis=-1)
    mu_imag = tf.concat(mu_imag,axis=-1)
    diag_Sigma_real = tf.concat(diag_Sigma_real,axis=-1)
    
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    
    # M-SBL
    mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
    mu_square = tf.tile(mu_square_average,(1,1,num_sc))
    alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    return model

FR_MSBL_single_layer = FR_MSBL_layer(Mr, num_sc, G_angle, sigma_2)


alpha_list = np.ones((test_num,G_angle,num_sc)) #initialization

mse_sbl_list = []

for i in range(num_layers):
    if i%10==0:
        print('FR MSBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = FR_MSBL_single_layer.predict([Phi_real_imag_list, Y_real_imag_list, alpha_list],batch_size=batch_size)

    # record performance of every iteration
    predictions_X = mu_real+1j*mu_imag    
    error = 0
    for i in range(test_num):
        true_H = H_list[i]
        prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
        for j in range(num_sc):
            prediction_h = A_list[j].dot(predictions_X[i,:,j:j+1])
            prediction_H[:,j:j+1] = prediction_h
        error = error + np.linalg.norm(prediction_H - true_H) ** 2
    mse_sbl_list.append(error / (test_num * Nr * num_sc))  
    
# loss curve 
plt.figure()
plt.plot(mse_sbl_list)  
plt.xlabel('FR MSBL iteration')
plt.ylabel('MSE')
plt.show()

# final performance 
predictions_X = mu_real+1j*mu_imag

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(predictions_X[i,:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

# plt.figure()
# plt.imshow(np.abs(predictions_X[plot_sample_index]),cmap='gray_r')
# plt.xlabel('$K$')
# plt.ylabel('$G_A$')
# plt.show()

#%% SOMP with FR dicts 
# def SOMP_CE(y_s,A_list,Phi_list,num_sc,num_antenna_bs,num_beams,G_angle,max_iter_count):
#     residual = np.copy(y_s) # (num_sc, num_beams, 1)
#
#     change_of_residual = 1e4
#
#     iter_count = 0
#
#     max_angle_indexes = []
#
#     while (change_of_residual > 1e-2) & (iter_count < max_iter_count):
#         # compute the direction with largest average response energy
#         responses = 0
#         for n in range(num_sc):
#             responses = responses + np.linalg.norm(np.matrix(Phi_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle, 1)
#
#         max_angle_index = np.argmax(responses)
#         max_angle_indexes.append(max_angle_index)
#
#         # print(np.sort(responses))
#         # plt.figure()
#         # plt.plot(responses)
#
#         # update F_RF_n matrices with one vector added
#         if iter_count == 0:
#             Phi_n_list = Phi_list[:,:,max_angle_index:max_angle_index+1]
#         else:
#             Phi_n_list = np.concatenate([Phi_n_list, Phi_list[:,:,max_angle_index:max_angle_index+1]],axis=-1)
#
#         residual_new = np.copy(residual)
#         X_hats_tmp = []
#         for n in range(num_sc):
#             Phi_n = Phi_n_list[n]
#             x_hat_n = np.linalg.pinv(Phi_n).dot(y_s[n])
#             residual_new[n] = y_s[n] - Phi_n.dot(x_hat_n)
#             X_hats_tmp.append(x_hat_n)
#
#         change_of_residual = np.linalg.norm(residual_new-residual)
#         # print(change_of_residual)
#         residual = residual_new
#
#         iter_count = iter_count + 1
#
#     X_hats = np.zeros((num_sc,G_angle,1),dtype=np.complex64)
#     # 如果不是因为迭代次数到了而跳出循环，或者存在重复index
#     if (iter_count < max_iter_count) or (len(max_angle_indexes)!=len(np.unique(max_angle_indexes))):
#         max_angle_indexes = max_angle_indexes[:-1]
#         X_hats[:,max_angle_indexes] = np.array(X_hats_tmp)[:,:-1]
#     else:
#         X_hats[:,max_angle_indexes] = np.array(X_hats_tmp)
#
#     assert len(max_angle_indexes)==len(np.unique(max_angle_indexes))
#
#     H_hats = np.zeros((num_sc,num_antenna_bs,1),dtype=np.complex64)
#     for n in range(num_sc):
#         x_hat_n = X_hats[n]
#         h_hat_n = A_list[n].dot(x_hat_n)
#         H_hats[n] = h_hat_n
#
#     return H_hats,X_hats
#
# num_clusters = 3
# if channel_model == 'path':
#     max_iter_count = num_clusters
# else:
#     max_iter_count = num_clusters*2
#
# # comment this line if use the dataset from generate_data.py
# Y_list = np.transpose(Y_list,(0,2,1)) # (test_num,num_sc,Mr)
#
# error = 0
# error_nmse = 0
# for i in range(test_num):
#     if i%10==0:
#         print('Sample %d'%i)
#     true_H = H_list[i]
#     prediction_H,prediction_X = SOMP_CE(np.expand_dims(Y_list[i],axis=-1), A_list, Phi_list[0], num_sc, Nr, Mr, G_angle, max_iter_count)
#     prediction_H = np.transpose(np.squeeze(prediction_H))
#     prediction_X = np.transpose(np.squeeze(prediction_X))
#     if i==0:
#         prediction_X_sample0 = prediction_X
#     error = error + np.linalg.norm(prediction_H - true_H) ** 2
#     error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
# mse_omp = error / (test_num * Nr * num_sc)
# nmse_omp = error_nmse / test_num
# print(mse_omp)
# print(nmse_omp)
#
# plt.figure()
# plt.imshow(np.abs(prediction_X_sample0),cmap='gray_r')
# plt.xlabel('Subcarriers')
# plt.ylabel('Angular Grids')
# plt.title('Angular-Frequency Channel Modulus')

# plt.show()
