import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from scipy import io
from functions import dictionary,update_mu_Sigma_delay,A_R_Layer_FR,A_T_Layer,Fixed_Phi_Layer,C2R,R2C#,circular_padding_single_sc,

# import time
# time.sleep(10*3600)

# use partial dataset
data_num = 10000
test_num = 200

Nr = 32

SNR = 10 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Q = 5
N_RF = 4
Mr = Q*N_RF

channel_model = 'cluster'

data = io.loadmat('./data/data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))

H_list = data['H_list'][:data_num]
Y_list = data['Y_list'][:data_num]

H_list = np.transpose(H_list,(0,2,1))
H_real_imag_list = C2R(H_list)
# split the testing set at the head part
H_list_test = H_list[:test_num]
H_real_imag_list = H_real_imag_list[test_num:]
print(H_real_imag_list.shape)

# notice that, the dimension num_sc has to be put before Nr/Mr to ensure correct vectorization orders
y_list = np.reshape(Y_list,(data_num,-1))
y_real_imag_list = C2R(y_list)
# split the testing set at the head part
y_real_imag_list_test = y_real_imag_list[:test_num]
y_real_imag_list = y_real_imag_list[test_num:]

print(y_real_imag_list.shape) # (data_num,num_sc*Mr,2)

W = np.matrix(data['W'])
print(W.shape)

num_sc = 32 # number of subcarriers
fc = 28 * 1e9 # central frequency
fs = 4 * 1e9 # bandwidth
eta = fs / num_sc  # subcarrier spacing

G_angle = 64 # angular resolution

G_delay = 64
A_K = dictionary(num_sc, G_delay)
A_K = np.matrix(A_K)

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

A_list_real_imag = C2R(A_list)

Phi = (np.kron(np.eye(num_sc),W.H).dot(A_list_expanded)).dot(np.kron(A_K,np.eye(G_angle)))

Phi = np.array(Phi)

Phi_real_imag = C2R(Phi)


#%%
import tensorflow as tf
tf.random.set_seed(2023)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv1D,Lambda,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)


#%% construct the network
def SBL_net(Mr, Nr, G_angle, G_delay, sigma_2, num_sc, num_layers, num_filters, kernel_size):
    y_real_imag = Input(shape=(Mr*num_sc, 2))

    batch_zeros = tf.tile(tf.zeros_like(y_real_imag[:, 0:1, :]), (1, G_angle*G_delay, 1))
    batch_zeros = tf.tile(tf.expand_dims(batch_zeros,axis=1),(1,Mr*num_sc,1,1))
    Phi_real_imag = Fixed_Phi_Layer(Mr*num_sc, G_angle*G_delay)(batch_zeros)

    # of shape (?,G_angle*G_delay)
    alpha_list_init = tf.ones_like(Phi_real_imag[:, 0, :, 0])

    ## the first update of mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_delay(x,sigma_2,num_sc,Mr))(
        [Phi_real_imag, y_real_imag, alpha_list_init])

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

        alpha_list = tf.reshape(alpha_list,(-1,G_angle*G_delay))

        ## update mu and Sigma
        mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_delay(x, sigma_2, num_sc, Mr)) \
            ([Phi_real_imag, y_real_imag, alpha_list])

    X_hat = Lambda(lambda x: tf.concat(x,axis=-1))([mu_real, mu_imag])
    X_hat = Reshape((G_delay, G_angle, 2))(X_hat)
    # remove the effect of vectorization
    # (?,G_angle,G_delay,2)
    X_hat = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)),name='X_hat')(X_hat)

    H_hat = A_T_Layer(num_sc, G_delay)(X_hat)# (?,G_angle,num_sc,2)
    H_hat = A_R_Layer_FR(Nr, G_angle, num_sc)(H_hat) # (?,Nr,num_sc,2)

    model = Model(inputs=y_real_imag, outputs=H_hat)

    return model

num_layers = 3
num_filters = 8
kernel_size = 5

model = SBL_net(Mr, Nr, G_angle, G_delay, sigma_2, num_sc, num_layers, num_filters, kernel_size)

# model.summary()

epochs = 1000
batch_size = 64
best_model_path = './models/SBL_unfolding_delay_FR_%dBeams_%dSNR_%s.h5'%(Mr, SNR, channel_model)

# weight initialization
init_weights_R = A_list_real_imag
init_weights_T = C2R(A_K.T)
init_weights_Phi = Phi_real_imag
for layer in model.layers:
    if 'a_r_' in layer.name:
        print('Set A_R weights')
        layer.set_weights([init_weights_R])
    if 'phi_' in layer.name:
        print('Set Phi weights')
        layer.set_weights([init_weights_Phi])
    if 'a_t_' in layer.name:
        print('Set A_K weights')
        layer.set_weights([init_weights_T])

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
                              min_delta=1e-5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)

model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))

model.fit(y_real_imag_list, H_real_imag_list, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])


#%% test the final performance, using the same (untrained) testing set as other algorithms for fairness
# use UAMP SBL unfolding generator test data
test_data = io.loadmat('./data/test_data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))
y_real_imag_list = test_data['y_real_imag_list_test']
U = test_data['U']
y_list = R2C(y_real_imag_list)
y_list = np.transpose(U.dot(np.transpose(y_list)))
y_real_imag_list_test = C2R(y_list)

model.load_weights(best_model_path)
predictions_H = model.predict(y_real_imag_list_test,verbose=1)
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

#save conv layer weights of the final model
# weight_dict = {}
# for layer in model.layers:
#     if 'SBL_' in layer.name:
#         print('Save Conv weights')
#         weight_dict[layer.name] = layer.get_weights()
#
# io.savemat('./results/weight_SBL_unfolding_delay_FR.mat',weight_dict)
# print('Weight dict saved!')
