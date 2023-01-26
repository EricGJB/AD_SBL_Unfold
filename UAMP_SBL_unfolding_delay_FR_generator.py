import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from scipy import io
from functions import dictionary,A_R_Layer_FR,A_T_Layer,Fixed_Phi_Layer,complex_matrix_multiplication,C2R,R2C#circular_padding_single_sc

# import time
# time.sleep(6*3600)

# use partial dataset 
data_num = 10000
test_num = 200

Nr = 32

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-SNR')
# args = parser.parse_args()
# SNR = int(args.SNR)

SNR = 10

sigma_2 = 1/10**(SNR/10) # noise variance

Q = 5
N_RF = 4
Mr = Q*N_RF

channel_model = 'cluster'

data = io.loadmat('./data/data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model))

H_list = data['H_list'][:data_num]

H_list = np.transpose(H_list,(0,2,1))
H_real_imag_list = C2R(H_list)

# split the testing set at the head part
H_list_test = H_list[:test_num]

H_real_imag_list_test = H_real_imag_list[:test_num]

H_real_imag_list = H_real_imag_list[test_num:]

print(H_real_imag_list.shape) # (data_num, Nr, num_sc, 2)

W_original = np.matrix(data['W_original'])

W = np.matrix(data['W'])
print(W.shape)

pre_whiter = data['pre_whiter']
print(pre_whiter.shape)
pre_whiter_real_imag = C2R(pre_whiter)

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

# unitary transformation
U,Sigma,V = np.linalg.svd(Phi)
print(U.shape)

U_real_imag = C2R(U)

Phi = np.transpose(np.conjugate(U)).dot(Phi)
Phi_real_imag = C2R(Phi)

# Y_list = data['Y_list']
# y_list = np.reshape(Y_list, (data_num, num_sc * Mr))
# y_list = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(y_list)))
#
# # notice that, the dimension num_sc has to be put before Nr/Mr to ensure correct vectorization orders
# y_real_imag_list = C2R(y_list)
# # split the testing set at the head part
# y_real_imag_list_test = y_real_imag_list[:test_num]
# y_real_imag_list = y_real_imag_list[test_num:]


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
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
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


class Transmission_Layer(tf.keras.layers.Layer):
    def __init__(self, Nr, Q, N_RF, num_sc, sigma_2):
        super(Transmission_Layer, self).__init__()
        self.Q = Q
        self.N_RF = N_RF
        self.num_sc = num_sc
        self.sigma_2 = sigma_2
        self.Mr = self.Q*self.N_RF
        self.Nr = Nr

    def build(self, input_shape):
        self.pre_whiter_real_imag = self.add_weight("pre_whiter", shape=[self.Mr, self.Mr, 2], trainable=False)
        self.U_real_imag = self.add_weight("U", shape=[self.Mr * self.num_sc, self.Mr * self.num_sc, 2], trainable=False)
        self.W = self.add_weight("W", shape=[self.Nr, self.Mr], trainable=False) # real W is considered in the paper

    def call(self, input):
        H_real_imag = input  # (?, Nr, num_sc, 2), real
        H = tf.cast(H_real_imag[:, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, 1], tf.complex64) #  (?, Nr, num_sc), complex
        W = tf.cast(self.W,tf.complex64) # (Nr,Mr), real values but in the complex format 
        Y_q_list = []
        for q in range(self.Q):
            W_q = W[:,q * self.N_RF:(q + 1) * self.N_RF] # (Nr,N_RF)
            Noise_real_imag = tf.random.normal(shape=tf.shape(H_real_imag), dtype=np.float32) * np.sqrt(self.sigma_2 / 2)
            Noise = tf.cast(Noise_real_imag[:, :, :, 0], tf.complex64) + 1j * tf.cast(Noise_real_imag[:, :, :, 1], tf.complex64)
            Noisy_H = H+Noise # (?,Nr,num_sc)
            Y_q = tf.matmul(W_q,Noisy_H,adjoint_a=True) # (?,N_RF,num_sc)
            Y_q_list.append(Y_q)
        Y = tf.concat(Y_q_list,axis=1) # (?,Mr,num_sc)
        # (Mr,Mr), complex
        pre_whiter = tf.cast(self.pre_whiter_real_imag[:, :, 0], tf.complex64) + 1j * tf.cast(self.pre_whiter_real_imag[:, :, 1], tf.complex64)
        Y = tf.matmul(pre_whiter,Y) # (?,Mr,num_sc)
        Y = tf.transpose(Y,(0,2,1)) # (?,num_sc,Mr)
        y = tf.reshape(Y,(-1,self.Mr*self.num_sc,1)) # (?,num_sc*Mr,1)
        U = tf.cast(self.U_real_imag[:, :, 0], tf.complex64) + 1j * tf.cast(self.U_real_imag[:, :, 1], tf.complex64) # (num_sc*Mr,num_sc*Mr)
        y = tf.matmul(U,y,adjoint_a=True)
        y_real_imag = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1) # (?,Mr*num_sc,2)

        return y_real_imag


#%% construct the network
def SBL_net(block_length, Q, N_RF, Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size):
    # need H_list here
    H_real_imag = Input(shape=(Nr, num_sc, 2))

    # (?, Mr * num_sc, 2)
    y_real_imag = Transmission_Layer(Nr,Q,N_RF,num_sc,sigma_2)(H_real_imag)

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
            model = Model(inputs=H_real_imag, outputs=H_hat_real_imag)
            model_list.append(model)

    return model_list

num_layers = 12
num_filters = 8
kernel_size = 3
# trade off of training complexity and stablity
block_length = 1 #1 corresponds to layer wise training
assert num_layers%block_length == 0

G = G_angle*G_delay

model_list = SBL_net(block_length, Q, N_RF, Mr, Nr, G, sigma_2, num_sc, num_layers, num_filters, kernel_size)

training = False

if training:

    print('Totally %d models to be trained sequentially'%len(model_list))

    epochs = 1000
    batch_size = 128

    # weight initialization
    init_weights_R = A_list_real_imag
    init_weights_T = C2R(A_K.T)
    init_weights_Phi = Phi_real_imag
    init_weights_pre_whiter = pre_whiter_real_imag
    init_weights_U = U_real_imag
    init_weights_W = W_original 

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
                if 'transmission' in layer.name:
                    print('Set the weights of the transmission layer')
                    layer.set_weights([init_weights_pre_whiter,init_weights_U,init_weights_W])

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
                if 'transmission' in layer.name:
                    print('Set the weights of the transmission layer')
                    layer.set_weights([init_weights_pre_whiter,init_weights_U,init_weights_W])
                if 'SBL_' in layer.name:
                    print('Set Conv weights')
                    layer.set_weights(weight_list[conv_count])
                    conv_count = conv_count + 1

        model_count = model_count + 1

        # Training
        # define callbacks
        best_model_path = './models/UAMP_SBL_unfolding_delay_%dLayers_%dBeams_%dSNR_res_%s_generator.h5'%(model_count*block_length+1,Mr,SNR,channel_model) # res, +1
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

        model.fit(H_real_imag_list, H_real_imag_list, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=True, \
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
    layer_count = 8
    best_model_path = './models/UAMP_SBL_unfolding_delay_%dLayers_%dBeams_%dSNR_res_%s_generator.h5'%(layer_count,Mr,SNR,channel_model)
    model = model_list[layer_count-1]

    model.load_weights(best_model_path)

    predictions_H = model.predict(H_real_imag_list_test,verbose=1)
    predictions_H = R2C(predictions_H)

    import tensorflow.keras.backend as K
    get_y = K.function([model.input],[model.layers[1].output])
    y_real_imag_list_test = get_y(H_real_imag_list_test)[0]
    io.savemat('./data/test_data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model),{'y_real_imag_list_test':y_real_imag_list_test,'U':U})

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
