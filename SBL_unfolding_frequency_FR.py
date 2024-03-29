import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from scipy import io
from functions import update_mu_Sigma_FR,A_R_Layer_FR,Fixed_Phi_Layer_FR,C2R,R2C#,circular_padding_single_sc,dictionary

# use partial dataset 
data_num = 10000
test_num = 200

data = io.loadmat('./data/data.mat')

H_list = data['H_list'][:data_num]
Y_list = data['Y_list'][:data_num]

H_list = np.transpose(H_list,(0,2,1))
H_real_imag_list = C2R(H_list)
# split the testing set at the head part
H_list_test = H_list[:test_num]
H_real_imag_list = H_real_imag_list[test_num:]
print(H_real_imag_list.shape)

Y_real_imag_list = C2R(Y_list)
Y_real_imag_list = np.transpose(Y_real_imag_list,(0,2,1,3))
# split the testing set at the head part
Y_real_imag_list_test = Y_real_imag_list[:test_num]
Y_real_imag_list = Y_real_imag_list[test_num:]
print(Y_real_imag_list.shape)

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
A_list_real_imag = C2R(A_list)

Phi_list = np.zeros((num_sc, Mr, G_angle)) + 1j * np.zeros((num_sc, Mr, G_angle))
for i in range(num_sc):
    Phi_list[i] = W.H.dot(A_list[i])
Phi_real_imag_list = C2R(Phi_list)


#%%
import tensorflow as tf
tf.random.set_seed(2023)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv1D,Lambda#,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)


#%% construct the network
def SBL_net(Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size):
    y_real_imag = Input(shape=(Mr, num_sc, 2))

    # of shape (?,G,num_sc)
    alpha_list_init = tf.tile(tf.ones_like(y_real_imag[:, 0, 0:1, 0:1]), (1, G, num_sc))

    batch_zeros = tf.tile(tf.zeros_like(y_real_imag[:, :, 0:1, :]), (1, 1, G, 1))
    batch_zeros = tf.tile(tf.expand_dims(batch_zeros,axis=1),(1,num_sc,1,1,1))
    Phi_real_imag_list = Fixed_Phi_Layer_FR(num_sc, Mr, G)(batch_zeros)

    ## the first update of mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_FR(x, num_sc, sigma_2, Mr))(
        [Phi_real_imag_list, y_real_imag, alpha_list_init])

    for i in range(num_layers):
        # if nn_type == 1:        
        mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])

        # feature tensor of dim (?,G,num_sc,2)
        temp = Lambda(lambda x: tf.concat(x, axis=-1))([tf.expand_dims(mu_square,axis=-1),\
                            tf.expand_dims(diag_Sigma_real,axis=-1)])
        # 2D Convolution Layers
        # notice that, "padding" should be set to "valid" if circular padding is used
        conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,padding='same',activation='relu')
        conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,padding='same',activation='relu')

        ## update alpha 
        # temp = circular_padding_single_sc(temp, kernel_size=kernel_size, strides=1)
        temp = conv_layer1(temp)
        # temp = circular_padding_single_sc(temp, kernel_size=kernel_size, strides=1)
        alpha_list = conv_layer2(temp)

        # if nn_type == 2:
        #     mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
        #     mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
        #     mu_square = tf.tile(mu_square_average,(1,1,num_sc))
        
        #     # feature tensor of dim (?,G,num_sc,2)
        #     temp = Lambda(lambda x: tf.concat(x, axis=-1))([tf.expand_dims(mu_square,axis=-1),\
        #                         tf.expand_dims(diag_Sigma_real,axis=-1)])
        #     temp = tf.transpose(temp,(0,2,1,3))
            
        #     conv_layer1 = Conv1D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,padding='same',activation='relu')
        #     conv_layer2 = Conv1D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,padding='same',activation='relu')
        #     temp = conv_layer1(temp)
        #     alpha_list = conv_layer2(temp)
        #     alpha_list = tf.transpose(alpha_list,(0,2,1,3))
            
        ## update mu and Sigma
        mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_FR(x, num_sc, sigma_2, Mr)) \
            ([Phi_real_imag_list, y_real_imag, alpha_list[:,:,:,0]])

    X_hat = Lambda(lambda x: tf.concat([tf.expand_dims(x[0], axis=-1), tf.expand_dims(x[1], axis=-1)], axis=-1),name='X_hat')\
        ([mu_real, mu_imag])

    H_hat = A_R_Layer_FR(Nr, G, num_sc)(X_hat)

    model = Model(inputs=y_real_imag, outputs=H_hat)
    
    return model

num_layers = 3
num_filters = 8
kernel_size = 5

model = SBL_net(Mr, Nr, G_angle, sigma_2, num_sc, num_layers, num_filters, kernel_size)

# model.summary()

epochs = 1000
batch_size = 128
best_model_path = './models/SBL_unfolding_frequency_FR.h5'

# weight initialization
for layer in model.layers:
    if 'a_r_' in layer.name:
        print('Set A_R weights')
        layer.set_weights([A_list_real_imag])
    if 'phi_' in layer.name:
        print('Set Phi weights')
        layer.set_weights([Phi_real_imag_list])

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
                              min_delta=1e-5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)

model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))

model.fit(Y_real_imag_list, H_real_imag_list, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])
    
    
#%% test the final performance, using the same (untrained) testing set as other algorithms for fairness
model.load_weights(best_model_path)
predictions_H = model.predict(Y_real_imag_list_test,verbose=1)
predictions_H = R2C(predictions_H)

error = 0
error_nmse = 0
for i in range(test_num):
    true_H = H_list_test[i]
    prediction_H = predictions_H[i]
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
mse = error / (test_num * Nr * num_sc)
nmse = error_nmse / test_num
print(mse)
print(nmse)

