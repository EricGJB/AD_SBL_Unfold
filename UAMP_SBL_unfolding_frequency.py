import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from scipy import io
from functions import complex_matrix_multiplication,dictionary,A_R_Layer,Fixed_Phi_Layer,C2R,R2C#,circular_padding_single_sc

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

svd = 1 # whether to execute unitary pre-processing
if svd:
    U,Sigma,V = np.linalg.svd(Phi)
    y_list = np.reshape(Y_list,(data_num*num_sc,Mr))
    y_list = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(y_list)))
    Y_list = np.reshape(y_list,(data_num,num_sc,Mr))
    Phi = np.transpose(np.conjugate(U)).dot(Phi)

Y_real_imag_list = C2R(Y_list)
Y_real_imag_list = np.transpose(Y_real_imag_list,(0,2,1,3))
# split the testing set at the head part
Y_real_imag_list_test = Y_real_imag_list[:test_num]
Y_real_imag_list = Y_real_imag_list[test_num:]
print(Y_real_imag_list.shape)

Phi_real_imag = C2R(Phi)


#%%
import tensorflow as tf
tf.random.set_seed(2023)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Lambda#,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)


#%% 
def update_UAMP_SBL(Phi_real_imag, y_real_imag_list, Tau_x_list, X_hat_real_imag_list, alpha_hat_list, S_real_imag_list, beta):
    Tau_x_list_new = []
    X_hat_real_imag_list_new = []
    S_real_imag_list_new = []
    
    Phi_real_imag_H = tf.concat([tf.transpose(Phi_real_imag[:,:,:,0:1],(0,2,1,3)),-tf.transpose(Phi_real_imag[:,:,:,1:2],(0,2,1,3))],axis=-1)
    term1 = Phi_real_imag[:,:,:,0]**2+Phi_real_imag[:,:,:,1]**2
    
    for i in range(num_sc):
        Tau_p = tf.matmul(term1,Tau_x_list[:,i])
        P_real_imag = complex_matrix_multiplication(Phi_real_imag, X_hat_real_imag_list[:,i]) - tf.expand_dims(Tau_p,axis=-1) * S_real_imag_list[:,i]
        Tau_s = 1 / (Tau_p + 1 / beta)
        S_real_imag = tf.expand_dims(Tau_s,axis=-1) * (y_real_imag_list[:,i] - P_real_imag)
        Tau_q = 1 / tf.matmul(tf.transpose(term1,(0,2,1)), Tau_s)
        Q_real_imag = X_hat_real_imag_list[:,i] + tf.expand_dims(Tau_q,axis=-1) * complex_matrix_multiplication(Phi_real_imag_H, S_real_imag)
        Tau_x = Tau_q * alpha_hat_list[:,i] / (alpha_hat_list[:,i] + Tau_q)
        X_hat_real_imag = Q_real_imag * tf.expand_dims(alpha_hat_list[:,i],axis=-1) / tf.expand_dims(alpha_hat_list[:,i] + Tau_q, axis=-1)

        Tau_x_list_new.append(tf.expand_dims(Tau_x,axis=1))
        X_hat_real_imag_list_new.append(tf.expand_dims(X_hat_real_imag,axis=1))
        S_real_imag_list_new.append(tf.expand_dims(S_real_imag,axis=1))
        
    Tau_x_list_new = tf.concat(Tau_x_list_new,axis=1)
    X_hat_real_imag_list_new = tf.concat(X_hat_real_imag_list_new,axis=1)
    S_real_imag_list_new = tf.concat(S_real_imag_list_new,axis=1)
    
    return Tau_x_list_new, X_hat_real_imag_list_new, S_real_imag_list_new


#%% construct the network
def SBL_net(block_length, Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size):
    y_real_imag_0 = Input(shape=(Mr, num_sc, 2))

    y_real_imag_list = tf.expand_dims(y_real_imag_0,axis=-2)
    # (?,num_sc,Mr,1,2)
    y_real_imag_list = tf.transpose(y_real_imag_list,(0,2,1,3,4))
    
    batch_zeros = tf.tile(tf.zeros_like(y_real_imag_0[:, :, 0:1, :]), (1, 1, G, 1))
    Phi_real_imag = Fixed_Phi_Layer(Mr, G)(batch_zeros)
    beta = 1 / sigma_2

    # Initialization
    # (?,num_sc,G,1)
    Tau_x_list = tf.tile(tf.ones_like(y_real_imag_0[:, 0:1, 0:1, 0:1]), (1, num_sc, G, 1))
    # (?,num_sc,G,1,2)
    X_hat_real_imag_list = tf.tile(tf.zeros_like(y_real_imag_list[:, 0:1, 0:1, 0:1, 0:1]), (1, num_sc, G, 1, 2)) # complex
    # (?,num_sc,G,1)
    alpha_hat_list = tf.tile(tf.ones_like(y_real_imag_0[:, 0:1, 0:1, 0:1]), (1, num_sc, G, 1))
    # (?,num_sc,Mr,1,2)
    S_real_imag_list = tf.tile(tf.zeros_like(y_real_imag_list[:, 0:1, 0:1, 0:1, 0:1]), (1, num_sc, Mr, 1, 2)) # complex

    # update mu and Sigma
    Tau_x_list, X_hat_real_imag_list, S_real_imag_list = update_UAMP_SBL(Phi_real_imag, y_real_imag_list, Tau_x_list, X_hat_real_imag_list, alpha_hat_list, S_real_imag_list, beta)

    model_list = []

    for i in range(num_layers):
        mu_square_list = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([X_hat_real_imag_list[:,:,:,:,0], X_hat_real_imag_list[:,:,:,:,1]])

        # feature tensor of dim (?,num_sc,G,2)
        temp = Lambda(lambda x: tf.concat(x, axis=-1))([mu_square_list,Tau_x_list])

        conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')
        conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='same',activation='relu')

        temp = conv_layer1(temp)

        alpha_hat_list = conv_layer2(temp)
        
        # update mu and Sigma
        Tau_x_list, X_hat_real_imag_list, S_real_imag_list = update_UAMP_SBL(Phi_real_imag, y_real_imag_list, Tau_x_list, X_hat_real_imag_list, alpha_hat_list, S_real_imag_list, beta)

        if (i+1)%block_length==0:
            H_hat = A_R_Layer(Nr, G)(tf.transpose(X_hat_real_imag_list[:, :, :, 0, :], (0, 2, 1, 3)))
            model = Model(inputs=y_real_imag_0, outputs=H_hat)
            model_list.append(model)
            
    return model_list

num_layers = 12
num_filters = 8
kernel_size = 3
# trade off of training complexity and stablity
block_length = 2 #1 corresponds to layer wise training
assert num_layers%block_length == 0

model_list = SBL_net(block_length, Mr, Nr, G_angle, sigma_2, num_sc, num_layers, num_filters, kernel_size)

print('Totally %d models to be trained sequentially'%len(model_list))

epochs = 1000
batch_size = 128

# weight initialization
init_weights_R = C2R(A_R)
init_weights_Phi = Phi_real_imag

model_count = 0

for model in model_list:
    # different weight initialization for different models, block wise inherent
    if model_count == 0:
        for layer in model.layers:
            if 'a_r_' in layer.name:
                print('Set A_R weights')
                layer.set_weights([init_weights_R])
            if 'phi_' in layer.name:
                print('Set Phi weights')
                layer.set_weights([init_weights_Phi])

    else:
        conv_count = 0
        for layer in model.layers:
            if 'a_r_' in layer.name:
                print('Set A_R weights')
                layer.set_weights([init_weights_R])
            if 'phi_' in layer.name:
                print('Set Phi weights')
                layer.set_weights([init_weights_Phi])
            if 'SBL_' in layer.name:
                print('Set Conv weights')
                layer.set_weights(weight_list[conv_count])
                conv_count = conv_count + 1

    model_count = model_count + 1

    # Training
    # define callbacks
    best_model_path = './models/UAMP_SVD%d_SBL_unfolding_frequency_%dLayers.h5'%(svd,model_count*block_length)
    checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
                                  min_delta=1e-5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)

    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))

    # model.summary()

    model.fit(Y_real_imag_list, H_real_imag_list, epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=True, \
              validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])

    model.load_weights(best_model_path)

    # creat the initial conv weight list for the next model
    weight_list = []
    for layer in model.layers:
        if 'SBL_' in layer.name:
            print('Save Conv weights')
            weight_list.append(layer.get_weights())
    for j in range(block_length):
        weight_list = weight_list + weight_list[-2:]


#%% Testing
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