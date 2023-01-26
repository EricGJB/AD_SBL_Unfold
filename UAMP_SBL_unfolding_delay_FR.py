import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from scipy import io
from functions import dictionary,A_R_Layer_FR,A_T_Layer,Fixed_Phi_Layer,complex_matrix_multiplication,C2R,R2C#circular_padding_single_sc

# use partial dataset 
data_num = 10000
test_num = 200

Nr = 32

SNR = 20 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 8

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

W = np.matrix(data['W'])
print(W.shape)

num_sc = 32 # number of subcarriers
fc = 28 * 1e9 # central frequency
fs = 4 * 1e9 # bandwidth
eta = fs / num_sc  # subcarrier spacing

G_angle = 64 # angular resolution 

A_R = dictionary(Nr, G_angle)
A_R = np.matrix(A_R) 

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

svd = 1 # whether to execute unitary pre-processing
if svd:
    U,Sigma,V = np.linalg.svd(Phi)
    y_list = np.reshape(Y_list,(data_num,num_sc*Mr))
    y_list = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(y_list)))
    Phi = np.transpose(np.conjugate(U)).dot(Phi)
else:
    y_list = np.reshape(Y_list,(data_num,-1))

Phi_real_imag = C2R(Phi)

# notice that, the dimension num_sc has to be put before Nr/Mr to ensure correct vectorization orders
y_real_imag_list = C2R(y_list)
# split the testing set at the head part
y_real_imag_list_test = y_real_imag_list[:test_num]
y_real_imag_list = y_real_imag_list[test_num:]

print(y_real_imag_list.shape) # (data_num,num_sc*Mr,2)


#%%
import tensorflow as tf
tf.random.set_seed(2023)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Lambda,ReLU,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)
    
    
#%%
def update_UAMP_SBL(Phi_real_imag, y_real_imag, Tau_x, X_hat_real_imag, alpha_hat, S_real_imag, beta): 
    Phi_real_imag_H = tf.concat([tf.transpose(Phi_real_imag[:,:,:,0:1],(0,2,1,3)),-tf.transpose(Phi_real_imag[:,:,:,1:2],(0,2,1,3))],axis=-1)
    term1 = Phi_real_imag[:,:,:,0]**2+Phi_real_imag[:,:,:,1]**2
    
    Tau_p = tf.matmul(term1,Tau_x)
    P_real_imag = complex_matrix_multiplication(Phi_real_imag, X_hat_real_imag) - tf.expand_dims(Tau_p,axis=-1) * S_real_imag
    Tau_s = 1 / (Tau_p + 1 / beta)
    S_real_imag = tf.expand_dims(Tau_s,axis=-1) * (tf.expand_dims(y_real_imag,axis=-2) - P_real_imag)
    Tau_q = 1 / tf.matmul(tf.transpose(term1,(0,2,1)), Tau_s)
    Q_real_imag = X_hat_real_imag + tf.expand_dims(Tau_q,axis=-1) * complex_matrix_multiplication(Phi_real_imag_H, S_real_imag)
    Tau_x = Tau_q * alpha_hat / (alpha_hat + Tau_q)
    X_hat_real_imag = Q_real_imag * tf.expand_dims(alpha_hat,axis=-1) / tf.expand_dims(alpha_hat + Tau_q, axis=-1)
    
    return Tau_x, X_hat_real_imag, S_real_imag


#%% construct the network
def SBL_net(block_length, Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size):
    y_real_imag = Input(shape=(Mr*num_sc, 2))
    
    batch_zeros = tf.tile(tf.zeros_like(tf.expand_dims(y_real_imag,axis=-2)), (1, 1, G, 1))
    Phi_real_imag = Fixed_Phi_Layer(Mr*num_sc, G)(batch_zeros)
    beta = 1 / sigma_2

    # Initialization
    # (?,G,1)
    Tau_x = tf.tile(tf.ones_like(y_real_imag[:, 0:1, 0:1]), (1, G, 1))
    # (?,G,1,2)
    X_hat_real_imag = tf.tile(tf.expand_dims(tf.zeros_like(Tau_x),axis=-1), (1, 1, 1, 2)) # complex
    # (?,G,1)
    alpha_hat = tf.ones_like(Tau_x)
    # (?,Mr*num_sc,1,2)
    S_real_imag = tf.tile(tf.zeros_like(tf.expand_dims(y_real_imag[:, 0:1, 0:1],axis=-2)), (1, Mr*num_sc, 1, 2)) # complex

    # update mu and Sigma
    Tau_x, X_hat_real_imag, S_real_imag = update_UAMP_SBL(Phi_real_imag, y_real_imag, Tau_x, X_hat_real_imag, alpha_hat, S_real_imag, beta)

    model_list = []

    for i in range(num_layers):
        mu_square_list = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([X_hat_real_imag[:,:,:,0], X_hat_real_imag[:,:,:,1]])

        # feature tensor of dim (?,num_sc,G,2)
        temp = Lambda(lambda x: tf.concat(x, axis=-1))([mu_square_list,Tau_x])

        temp = tf.reshape(temp,(-1,G_delay,G_angle,2))
        
        temp = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')(temp)
        temp = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='same')(temp)

        if i==0:
            alpha_hat = ReLU()(temp)
        else: # residual structure
            alpha_hat = ReLU()(temp+alpha_hat)
        
        #alpha_hat = ReLU()(temp)
        
        # update mu and Sigma
        Tau_x, X_hat_real_imag, S_real_imag = update_UAMP_SBL(Phi_real_imag, y_real_imag, Tau_x, X_hat_real_imag, tf.reshape(alpha_hat,(-1,G_angle*G_delay,1)), S_real_imag, beta)
        
        if (i+1)%block_length==0:
            X = Reshape((G_delay, G_angle, 2))(X_hat_real_imag)
            # remove the effect of vectorization
            # (?,G_angle,G_delay,2)
            X = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(X)
            X = A_T_Layer(num_sc, G_delay)(X)
            H_hat_real_imag = A_R_Layer_FR(Nr, G_angle, num_sc)(X)
            model = Model(inputs=y_real_imag, outputs=H_hat_real_imag)
            model_list.append(model)

    return model_list

num_layers = 12
num_filters = 8
kernel_size = 3
# trade off of training complexity and stablity
block_length = 1 #1 corresponds to layer wise training
assert num_layers%block_length == 0

G = G_angle*G_delay

model_list = SBL_net(block_length, Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size)

training = False

if training:

    print('Totally %d models to be trained sequentially'%len(model_list))

    epochs = 1000
    batch_size = 128

    # weight initialization
    init_weights_R = A_list_real_imag
    init_weights_T = C2R(A_K.T)
    init_weights_Phi = Phi_real_imag

    # skip the first model, since the two Conv blocks have different functions
    model_count = 0

    #for model in model_list:
    for model in model_list[1:]: # start from at least 2 UAMP-SBL layers, if use the residual structure
        # different weight initialization for different models, block wise inherent

        if model_count == 0:
            for layer in model.layers:
                if 'a_r_' in layer.name:
                    print('Set A_R weights')
                    layer.set_weights([init_weights_R])
                if 'a_t_' in layer.name:
                    print('Set A_T weights')
                    layer.set_weights([init_weights_T])
                if 'phi_' in layer.name:
                    print('Set Phi weights')
                    layer.set_weights([init_weights_Phi])

        else:
            conv_count = 0
            for layer in model.layers:
                if 'a_r_' in layer.name:
                    print('Set A_R weights')
                    layer.set_weights([init_weights_R])
                if 'a_t_' in layer.name:
                    print('Set A_T weights')
                    layer.set_weights([init_weights_T])
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
        best_model_path = './models/UAMP_SBL_unfolding_delay_%dLayers_%dBeams_%dSNR_res_%s.h5'%(model_count*block_length+1,Mr,SNR,channel_model) # res, +1
        checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
                                      min_delta=1e-5, min_lr=1e-5)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)

        #if model_count*block_length < 6:
        #    init_learning_rate = 1e-3
        #else:
        #    init_learning_rate = 1e-4

        init_learning_rate = 1e-3

        model.compile(loss='mse', optimizer=Adam(learning_rate=init_learning_rate))

        # model.summary()

        model.fit(y_real_imag_list, H_real_imag_list, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=True, \
                  validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])

        # load the best model if exist
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
else:
    # test a certain layer
    layer_count = 9
    best_model_path = './models/UAMP_SBL_unfolding_delay_%dLayers_%dBeams_%dSNR_res_%s.h5'%(layer_count,Mr,SNR,channel_model)
    model = model_list[layer_count-1]

    model.load_weights(best_model_path)

    predictions_H = model.predict(y_real_imag_list_test,verbose=1)
    predictions_H = R2C(predictions_H)

    error = 0
    error_nmse = 0
    mse_list = []
    nmse_list = []
    for i in range(test_num):
        true_H = H_list_test[i]
        prediction_H = predictions_H[i]
        sample_mse = np.linalg.norm(prediction_H - true_H) ** 2/(Nr*num_sc)
        error = error + sample_mse
        sample_nmse = (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
        error_nmse = error_nmse + sample_nmse
        mse_list.append(sample_mse)
        nmse_list.append(sample_nmse)
    mse = error / test_num
    nmse = error_nmse / test_num
    print(mse)
    print(nmse)

# compare the good list performance with UAMP SBL
# good_set = io.loadmat('./results/UAMP_SBL_good_test_samples_%dBeams_%dSNR_path.mat'%(Mr,SNR))
# index_list = np.squeeze(good_set['index_list'])
# nmse_list_UAMP = np.squeeze(good_set['nmse_list'])
# nmse_list_SBL = []
# for i in index_list:
#     true_H = H_list[i]
#     prediction_H = predictions_H[i]
#     sample_nmse = (np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2
#     nmse_list_SBL.append(sample_nmse)
#
# nmse_SBL = np.mean(nmse_list_SBL)
# nmse_UAMP = np.mean(nmse_list_UAMP)
#
# print('NMSE of %d/%d good testing samples:'%(len(index_list),test_num))
# print('UAMP SBL unfolding FR:%.4f'%nmse_SBL)
# print('UAMP-SBL FR:%.4f'%nmse_UAMP)
