import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv3D,Reshape,Lambda,Cropping3D,Cropping2D,ZeroPadding3D,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


def C2R(complex_array):
    complex_array = np.expand_dims(complex_array,axis=-1)
    real_array = np.concatenate([np.real(complex_array),np.imag(complex_array)],axis=-1)
    return real_array

def R2C(real_array):
    ndim = np.ndim(real_array)-1
    if ndim==1:
        complex_array = real_array[:,0]+1j*real_array[:,1]
    if ndim==2:
        complex_array = real_array[:,:,0]+1j*real_array[:,:,1]
    if ndim==3:
        complex_array = real_array[:,:,:,0]+1j*real_array[:,:,:,1]
    if ndim==4:
        complex_array = real_array[:,:,:,:,0]+1j*real_array[:,:,:,:,1]
    return complex_array

def dictionary(N, G):
    A = np.zeros((N, G)) + 1j * np.zeros((N, G))
    # 矩阵各列将角度sin值在[-1,1]之间进行 M 等分
    count = 0
    for sin_value in np.linspace(-1 + 1 / G, 1 - 1 / G, G):
        A[:, count] = np.exp(-1j * np.pi * np.arange(N) * sin_value) / np.sqrt(N)
        count = count + 1
    return A

def dictionary_delay(num_sc,eta,tau_max,G_delay):
    # G_delay = int(np.ceil(eta*num_sc*Tau_max))
    A = np.zeros((num_sc, G_delay)) + 1j * np.zeros((num_sc, G_delay))
    count = 0
    for delay_value in np.linspace(0, tau_max, G_delay):
        A[:, count] = np.exp(-1j * 2 * np.pi * np.arange(num_sc) * eta * delay_value) / np.sqrt(num_sc)
        count = count + 1
    return A#,G_delay

def update_mu_Sigma(inputs,num_sc,sigma_2,Mr):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_mu_Sigma_MSBL(inputs,num_sc,sigma_2,Mr):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        if i==0:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
            inv = tf.linalg.inv(
                tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr, dtype=tf.complex64))
            z = tf.matmul(Rx_PhiH, inv)
        y = y_list[:, :, i:i + 1]
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_mu_Sigma_delay(inputs,sigma_2,num_sc,Mr):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y = tf.cast(inputs[1][:, :, 0:1], tf.complex64) + 1j * tf.cast(inputs[1][:,  :, 1:], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    Rx_PhiH = tf.multiply(tf.expand_dims(alpha_list,axis=-1), tf.transpose(Phi, (0, 2, 1), conjugate=True))
    inv = tf.linalg.inv(
        tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(num_sc*Mr, dtype=tf.complex64))
    z = tf.matmul(Rx_PhiH, inv)
    mu = tf.matmul(z, y)
    diag_Sigma = alpha_list - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
    # return the updated parameters
    mu_real_list.append(tf.math.real(mu))
    mu_imag_list.append(tf.math.imag(mu))
    diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list

## PC-SBL functions
def update_alpha_PC(input_list,G,num_sc,a,b,beta):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2
    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:1]),mu_square,tf.zeros_like(mu_square[:,:,0:1])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:1]),mu_square,tf.zeros_like(mu_square[:,0:1])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:1])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:1])],axis=1)
    w_list = (mu_square[:,1:-1,1:-1]+diag_Sigma_real[:,1:-1,1:-1])+beta*(mu_square[:,1:-1,:-2]+diag_Sigma_real[:,1:-1,:-2])\
                +beta*(mu_square[:,1:-1,2:]+diag_Sigma_real[:,1:-1,2:])+beta*(mu_square[:,2:,1:-1]+diag_Sigma_real[:,2:,1:-1])\
                +beta*(mu_square[:,:-2,1:-1]+diag_Sigma_real[:,:-2,1:-1])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list

def update_mu_Sigma_PC(inputs,G,num_sc,sigma_2,Mr,Mt,beta):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)

    alpha_list = tf.cast(inputs[2],tf.complex64)
    # 取倒数
    alpha_list = 1/alpha_list
    alpha_list = tf.reshape(alpha_list,(-1,G,G,num_sc))
    # expand the head and tail of two dimensions
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,:,0:1]),alpha_list,tf.zeros_like(alpha_list[:,:,0:1])],axis=2)
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,0:1]),alpha_list,tf.zeros_like(alpha_list[:,0:1])],axis=1)
    # 错位加权相加
    alpha_list = (alpha_list[:,1:-1,1:-1]+beta*(alpha_list[:,1:-1,:-2]+alpha_list[:,1:-1,2:]+\
                                                alpha_list[:,:-2,1:-1]+alpha_list[:,2:,1:-1]))
    alpha_list = tf.reshape(alpha_list,(-1,G*G,num_sc))
    alpha_list = 1 / alpha_list

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list

class Fixed_Phi_Layer(tf.keras.layers.Layer):
    def __init__(self, Nr, G):
        super(Fixed_Phi_Layer, self).__init__()
        self.Nr = Nr
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.Nr, self.G, 2], trainable=False)

    def call(self, input):
        batch_zeros = input
        Phi = self.kernel + batch_zeros

        return Phi
    
class A_R_Layer(tf.keras.layers.Layer):
    def __init__(self, Nr, G):
        super(A_R_Layer, self).__init__()
        self.Nr = Nr
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.Nr, self.G, 2], trainable=False)

    def call(self, input):
        input_real = input[:, :, :, 0]
        input_imag = input[:, :, :, 1]
        batch_zeros = input_real[:, 0:1, 0:1] - input_real[:, 0:1, 0:1]
        # obtain a (?,Nt,Nt*resolution) dimensional zero matrix
        batch_zeros = tf.tile(batch_zeros, (1, self.Nr, self.G))
        # the weights in the kernel is the B matrix
        B_real = self.kernel[:, :, 0] + batch_zeros
        B_imag = self.kernel[:, :, 1] + batch_zeros
        # obtain h = B.dot(x)
        RR = tf.matmul(B_real, input_real)
        RI = tf.matmul(B_real, input_imag)
        IR = tf.matmul(B_imag, input_real)
        II = tf.matmul(B_imag, input_imag)
        output_real = RR - II
        output_imag = RI + IR

        return tf.concat([tf.expand_dims(output_real, axis=-1), tf.expand_dims(output_imag, axis=-1)], axis=-1)

class Fixed_Phi_Layer_FR(tf.keras.layers.Layer):
    def __init__(self, num_sc, Nr, G):
        super(Fixed_Phi_Layer_FR, self).__init__()
        self.num_sc = num_sc
        self.Nr = Nr
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.num_sc, self.Nr, self.G, 2], trainable=False)

    def call(self, input):
        batch_zeros = input
        Phi = self.kernel + batch_zeros

        return Phi

def update_mu_Sigma_FR(inputs,num_sc,sigma_2,Mr):
    Phi_list = tf.cast(inputs[0][:, :, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        Phi = Phi_list[:,i]
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list

class A_R_Layer_FR(tf.keras.layers.Layer):
    def __init__(self, Nr, G, num_sc):
        super(A_R_Layer_FR, self).__init__()
        self.Nr = Nr
        self.G = G
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.num_sc, self.Nr, self.G, 2], trainable=False)

    def call(self, input):
        input_real = input[:, :, :, 0]
        input_imag = input[:, :, :, 1]
        batch_zeros = input_real[:, 0:1, 0:1] - input_real[:, 0:1, 0:1]
        # obtain a (?,Nt,Nt*resolution) dimensional zero matrix
        batch_zeros = tf.tile(batch_zeros, (1, self.Nr, self.G))
        output_real_list = []
        output_imag_list = []
        for i in range(self.num_sc):
            # the weights in the kernel is the B matrix
            B_real = self.kernel[i, :, :, 0] + batch_zeros
            B_imag = self.kernel[i, :, :, 1] + batch_zeros
            # obtain h = B.dot(x)
            RR = tf.matmul(B_real, input_real[:,:,i:i+1])
            RI = tf.matmul(B_real, input_imag[:,:,i:i+1])
            IR = tf.matmul(B_imag, input_real[:,:,i:i+1])
            II = tf.matmul(B_imag, input_imag[:,:,i:i+1])
            output_real_list.append(RR - II)
            output_imag_list.append(RI + IR)

        output_real = tf.concat(output_real_list,axis=-1)
        output_imag = tf.concat(output_imag_list,axis=-1)

        return tf.concat([tf.expand_dims(output_real, axis=-1), tf.expand_dims(output_imag, axis=-1)], axis=-1)
    
class A_T_Layer(tf.keras.layers.Layer):
    def __init__(self, Nt, G):
        super(A_T_Layer, self).__init__()
        self.Nt = Nt
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.G, self.Nt, 2], trainable=False)

    def call(self, input):
        input_real = input[:, :, :, 0]
        input_imag = input[:, :, :, 1]
        batch_zeros = input_real[:, 0:1, 0:1] - input_real[:, 0:1, 0:1]
        # obtain a (?,Nt,Nt*resolution) dimensional zero matrix
        batch_zeros = tf.tile(batch_zeros, (1, self.G, self.Nt))
        # the weights in the kernel is the B matrix
        B_real = self.kernel[:, :, 0] + batch_zeros
        B_imag = self.kernel[:, :, 1] + batch_zeros
        # obtain A = W_rf.dot(B)
        RR = tf.matmul(input_real, B_real)
        RI = tf.matmul(input_real, B_imag)
        IR = tf.matmul(input_imag, B_real)
        II = tf.matmul(input_imag, B_imag)
        output_real = RR - II
        output_imag = RI + IR

        return tf.concat([tf.expand_dims(output_real, axis=-1), tf.expand_dims(output_imag, axis=-1)], axis=-1)
        
def complex_matrix_multiplication(A_real_imag,B_real_imag):
    A_real = A_real_imag[:,:,:,0]
    A_imag = A_real_imag[:,:,:,1]
    B_real = B_real_imag[:,:,:,0]
    B_imag = B_real_imag[:,:,:,1]
    C_real = tf.matmul(A_real,B_real)-tf.matmul(A_imag,B_imag)
    C_imag = tf.matmul(A_real,B_imag)+tf.matmul(A_imag,B_real)
    C_real_imag = tf.concat([tf.expand_dims(C_real,axis=-1),tf.expand_dims(C_imag,axis=-1)],axis=-1)
    return C_real_imag

    
class Optimized_Phi_Layer(tf.keras.layers.Layer):
    def __init__(self, Mr, Nr, G, num_sc):
        super(Optimized_Phi_Layer, self).__init__()
        self.Mr = Mr
        self.Nr = Nr
        self.G = G
        self.num_sc = num_sc

    def build(self, input_shape):
        self.Q = self.add_weight("Q", shape=[self.Mr*self.num_sc, self.Nr*self.num_sc, 2], trainable=False)
        self.AR = self.add_weight("A_R", shape=[self.Nr, self.G], trainable=True)
        self.AK = self.add_weight("A_K", shape=[self.num_sc, self.G], trainable=True)

    def call(self, input):
        noise_real_imag = input[1]
        batch_zeros_Phi = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:,0:1,:]),axis=-2),(1,self.Mr*self.num_sc,self.G**2,1))
        batch_zeros_A_R = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:,0:1,:]),axis=-2),(1,self.Nr,self.G,1))
        batch_zeros_A_K = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:,0:1,:]),axis=-2),(1,self.num_sc,self.G,1))
        
        noise_real_imag = tf.cast(noise_real_imag,tf.complex64)
        noise = noise_real_imag[:,:,0:1]+1j*noise_real_imag[:,:,1:]

        A_R = self.AR
        A_K = self.AK
        Q = self.Q
        
        A_R_real = tf.cos(A_R)
        A_R_imag = tf.sin(A_R)
        A_K_real = tf.cos(A_K)
        A_K_imag = tf.sin(A_K)

        A_R = (tf.cast(A_R_real,tf.complex64)+1j*tf.cast(A_R_imag,tf.complex64))/tf.cast(tf.sqrt(self.Nr*1.0),tf.complex64)
        A_K = (tf.cast(A_K_real,tf.complex64)+1j*tf.cast(A_K_imag,tf.complex64))/tf.cast(tf.sqrt(self.num_sc*1.0),tf.complex64)
        Q = tf.cast(Q[:,:,0],tf.complex64)+1j*tf.cast(Q[:,:,1],tf.complex64)
        
        Kron = tf.linalg.LinearOperatorKronecker(\
        [tf.linalg.LinearOperatorFullMatrix(A_K),tf.linalg.LinearOperatorFullMatrix(A_R)]\
            ).to_dense()
    
        Phi = tf.matmul(Q,Kron)
        
        h_real_imag = tf.cast(input[0],tf.complex64)
        h = h_real_imag[:,:,0:1]+1j*h_real_imag[:,:,1:]
        
        y = tf.matmul(Q,h+noise)
        
        Phi = tf.expand_dims(Phi, -1)
        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)
        Phi_real_imag = Phi_real_imag + batch_zeros_Phi
        
        y_real_imag = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)        
        
        A_R = tf.expand_dims(A_R, -1)
        A_R_real_imag = tf.concat([tf.math.real(A_R), tf.math.imag(A_R)], axis=-1)
        A_R_real_imag = A_R_real_imag + batch_zeros_A_R       
        
        A_K = tf.expand_dims(A_K, -1)
        A_K_real_imag = tf.concat([tf.math.real(A_K), tf.math.imag(A_K)], axis=-1)  
        A_K_real_imag = A_K_real_imag + batch_zeros_A_K 
        
        return y_real_imag, Phi_real_imag, A_R_real_imag, A_K_real_imag

class Optimized_Phi_Layer_v2(tf.keras.layers.Layer):
    def __init__(self, Mr, Nr, G, num_sc, train_dict, train_W):
        super(Optimized_Phi_Layer_v2, self).__init__()
        self.Mr = Mr
        self.Nr = Nr
        self.G = G
        self.num_sc = num_sc
        self.train_dict = train_dict
        self.train_W = train_W

    def build(self, input_shape):
        self.W = self.add_weight("W", shape=[self.Nr, self.Mr], trainable=self.train_W)
                                 #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.AR = self.add_weight("A_R", shape=[self.Nr, self.G], trainable=self.train_dict)
        self.AK = self.add_weight("A_K", shape=[self.num_sc, self.G], trainable=self.train_dict)

    def call(self, input):
        noise_real_imag = input[1]
        batch_zeros_Phi = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.Mr * self.num_sc, self.G ** 2, 1))
        batch_zeros_A_R = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.Nr, self.G, 1))
        batch_zeros_A_K = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.num_sc, self.G, 1))

        noise_real_imag = tf.cast(noise_real_imag, tf.complex64)
        noise = noise_real_imag[:, :, 0:1] + 1j * noise_real_imag[:, :, 1:]

        A_R = tf.tanh(self.AR)
        A_K = tf.tanh(self.AK)
        W = self.W

        A_R_real = tf.cos(A_R)
        A_R_imag = tf.sin(A_R)
        A_K_real = tf.cos(A_K)
        A_K_imag = tf.sin(A_K)
        W_real = tf.cos(W)
        W_imag = tf.sin(W)

        A_R = (tf.cast(A_R_real, tf.complex64) + 1j * tf.cast(A_R_imag, tf.complex64)) / tf.cast(
            tf.sqrt(self.Nr * 1.0), tf.complex64)
        A_K = (tf.cast(A_K_real, tf.complex64) + 1j * tf.cast(A_K_imag, tf.complex64)) / tf.cast(
            tf.sqrt(self.num_sc * 1.0), tf.complex64)
        W = (tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)) / tf.cast(
            tf.sqrt(self.Nr*1.0),tf.complex64)
        Q = tf.linalg.LinearOperatorKronecker( \
            [tf.linalg.LinearOperatorFullMatrix(tf.eye(self.num_sc,dtype=tf.complex64)), tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))] \
            ).to_dense()

        Kron = tf.linalg.LinearOperatorKronecker( \
            [tf.linalg.LinearOperatorFullMatrix(A_K), tf.linalg.LinearOperatorFullMatrix(A_R)] \
            ).to_dense()

        Phi = tf.matmul(Q, Kron)

        h_real_imag = tf.cast(input[0], tf.complex64)
        h = h_real_imag[:, :, 0:1] + 1j * h_real_imag[:, :, 1:]

        y = tf.matmul(Q, h + noise)

        Phi = tf.expand_dims(Phi, -1)
        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)
        Phi_real_imag = Phi_real_imag + batch_zeros_Phi

        y_real_imag = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)

        A_R = tf.expand_dims(A_R, -1)
        A_R_real_imag = tf.concat([tf.math.real(A_R), tf.math.imag(A_R)], axis=-1)
        A_R_real_imag = A_R_real_imag + batch_zeros_A_R

        A_K = tf.expand_dims(A_K, -1)
        A_K_real_imag = tf.concat([tf.math.real(A_K), tf.math.imag(A_K)], axis=-1)
        A_K_real_imag = A_K_real_imag + batch_zeros_A_K

        return y_real_imag, Phi_real_imag, A_R_real_imag, A_K_real_imag


class Optimized_Phi_Layer_v3(tf.keras.layers.Layer):
    def __init__(self, Mr, Nr, G, num_sc, train_dict, train_W):
        super(Optimized_Phi_Layer_v3, self).__init__()
        self.Mr = Mr
        self.Nr = Nr
        self.G = G
        self.num_sc = num_sc
        self.train_dict = train_dict
        self.train_W = train_W

    def build(self, input_shape):
        self.W = self.add_weight("W", shape=[self.Nr, self.Mr], trainable=self.train_W)
                                 #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.AR = self.add_weight("A_R", shape=[1,self.G], trainable=self.train_dict)
                                    #initializer=tf.random_uniform_initializer(-1,1))
        self.AK = self.add_weight("A_K", shape=[1,self.G], trainable=self.train_dict)

    def call(self, input):
        noise_real_imag = input[1]
        batch_zeros_Phi = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.Mr * self.num_sc, self.G ** 2, 1))
        batch_zeros_A_R = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.Nr, self.G, 1))
        batch_zeros_A_K = tf.tile(tf.expand_dims(tf.zeros_like(noise_real_imag[:, 0:1, :]), axis=-2),
                                  (1, self.num_sc, self.G, 1))

        noise_real_imag = tf.cast(noise_real_imag, tf.complex64)
        noise = noise_real_imag[:, :, 0:1] + 1j * noise_real_imag[:, :, 1:]

        A_R_directions = self.AR
        A_K_directions = self.AK
        W = self.W

        A_R = tf.exp(-1j * math.pi * tf.cast(tf.matmul(tf.expand_dims(tf.range(self.Nr,dtype=tf.float32),axis=-1),A_R_directions),tf.complex64)) / tf.cast(tf.sqrt(self.Nr*1.0),tf.complex64)
        A_K = tf.exp(-1j * math.pi * tf.cast(tf.matmul(tf.expand_dims(tf.range(self.num_sc,dtype=tf.float32),axis=-1),A_K_directions),tf.complex64)) / tf.cast(tf.sqrt(self.num_sc*1.0),tf.complex64)

        W_real = tf.cos(W)
        W_imag = tf.sin(W)

        W = (tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)) / tf.cast(
            tf.sqrt(self.Nr*1.0),tf.complex64)
        Q = tf.linalg.LinearOperatorKronecker( \
            [tf.linalg.LinearOperatorFullMatrix(tf.eye(self.num_sc,dtype=tf.complex64)), tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))] \
            ).to_dense()

        Kron = tf.linalg.LinearOperatorKronecker( \
            [tf.linalg.LinearOperatorFullMatrix(A_K), tf.linalg.LinearOperatorFullMatrix(A_R)] \
            ).to_dense()

        Phi = tf.matmul(Q, Kron)

        h_real_imag = tf.cast(input[0], tf.complex64)
        h = h_real_imag[:, :, 0:1] + 1j * h_real_imag[:, :, 1:]

        y = tf.matmul(Q, h + noise)

        Phi = tf.expand_dims(Phi, -1)
        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)
        Phi_real_imag = Phi_real_imag + batch_zeros_Phi

        y_real_imag = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)

        A_R = tf.expand_dims(A_R, -1)
        A_R_real_imag = tf.concat([tf.math.real(A_R), tf.math.imag(A_R)], axis=-1)
        A_R_real_imag = A_R_real_imag + batch_zeros_A_R

        A_K = tf.expand_dims(A_K, -1)
        A_K_real_imag = tf.concat([tf.math.real(A_K), tf.math.imag(A_K)], axis=-1)
        A_K_real_imag = A_K_real_imag + batch_zeros_A_K

        return y_real_imag, Phi_real_imag, A_R_real_imag, A_K_real_imag


# circular padding function
def circular_padding_2d(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)
    pad_along_depth = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_shallow = pad_along_depth // 2
    pad_deep = pad_along_depth - pad_shallow

    # top and bottom side padding
    pad_top = Cropping3D(cropping=((in_height - pad_top, 0), (0, 0), (0, 0)))(x)
    pad_bottom = Cropping3D(cropping=((0, in_height - pad_bottom), (0, 0), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping3D(cropping=((0, 0), (in_width - pad_left, 0), (0, 0)))(conc)
    pad_right = Cropping3D(cropping=((0, 0), (0, in_width - pad_right), (0, 0)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    # zero padding for the third dimension, i.e., subcarrier
    conc = ZeroPadding3D(((0, 0), (0, 0), (pad_shallow, pad_deep)))(conc)

    return conc


def circular_padding_2D(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # top and bottom side padding
    pad_top = Cropping3D(cropping=((in_height - pad_top, 0), (0, 0), (0, 0)))(x)
    pad_bottom = Cropping3D(cropping=((0, in_height - pad_bottom), (0, 0), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping3D(cropping=((0, 0), (in_width - pad_left, 0), (0, 0)))(conc)
    pad_right = Cropping3D(cropping=((0, 0), (0, in_width - pad_right), (0, 0)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    return conc


def circular_padding_single_sc(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # top and bottom side padding
    pad_top = Cropping2D(cropping=((in_height - pad_top, 0), (0, 0)))(x)
    pad_bottom = Cropping2D(cropping=((0, in_height - pad_bottom), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping2D(cropping=((0, 0), (in_width - pad_left, 0)))(conc)
    pad_right = Cropping2D(cropping=((0, 0), (0, in_width - pad_right)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    return conc

def update_mu_Sigma_mixed_SNR(inputs,num_sc,Mr,Mt):
    if len(inputs[0].shape)==4:
        Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    if len(inputs[0].shape)==3:
        Phi = tf.cast(inputs[0][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)
    sigma_2 = tf.cast(inputs[3],tf.complex64)
    sigma_2 = tf.reshape(sigma_2,(-1,1,1))

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        if len(Phi.shape)==3:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        if len(Phi.shape)==2:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


class Phi_Layer(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G):
        super(Phi_Layer, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G

    def build(self, input_shape):
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt*self.Nr, (self.G)**2, 2], trainable=False)

    def call(self, input):
        W_real = input[0][:, :, :, 0]
        W_imag = input[0][:, :, :, 1]
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)

        F_real = input[1][:, :, :, 0]
        F_imag = input[1][:, :, :, 1]
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Kron_1 = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F, (0, 2, 1))), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W, (0, 2, 1), conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Kron_1, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        return tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)


class Phi_Layer_joint_opt(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_joint_opt, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=True)
                                        #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=True)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag



class Phi_Layer_joint_opt_fixed(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_joint_opt_fixed, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=False)
                                        #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=False)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag






import math
class Phi_Layer_joint_opt_single_sc(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF):
        super(Phi_Layer_joint_opt_single_sc, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=True,\
                                        initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=True, \
                                        initializer=tf.random_uniform_initializer(0, 2 * math.pi))

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise = tf.concat(noise_list, axis=1)

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, 1], tf.complex64)

        Y = tf.matmul(tf.matmul(W, H, adjoint_a=True),F)+ effective_noise
        Y_real_imag = tf.concat([tf.math.real(Y),tf.math.imag(Y)], axis=-1)

        return Y_real_imag



class Phi_Layer_multipleT(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_multipleT, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=False)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_real = input[0][:, :, :, 0]
        W_imag = input[0][:, :, :, 1]
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)

        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,(0,2,1),conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:,:,r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,(0,2,1),conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[2]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag



#%% symmetric convolution functions
def symmetric_pre(feature_map,kernel_size):
    # split into four feature_maps
    G = feature_map.shape[1]+1-kernel_size
    sub_feature_map1 = feature_map[:,:G//2+kernel_size-1,:G//2+kernel_size-1]
    sub_feature_map2 = feature_map[:,:G//2+kernel_size-1,G//2:]
    sub_feature_map2 = K.reverse(sub_feature_map2,axes=2)
    sub_feature_map3 = feature_map[:,G//2:,:G//2+kernel_size-1]
    sub_feature_map3 = K.reverse(sub_feature_map3,axes=1)
    sub_feature_map4 = feature_map[:,G//2:,G//2:]
    sub_feature_map4 = K.reverse(sub_feature_map4, axes=[1,2])

    return sub_feature_map1,sub_feature_map2,sub_feature_map3,sub_feature_map4


def symmetric_post(sub_output1,sub_output2,sub_output3,sub_output4):
    output_upper = tf.concat([sub_output1,K.reverse(sub_output2,axes=2)],axis=2)
    output_lower = tf.concat([K.reverse(sub_output3,axes=1),K.reverse(sub_output4,axes=[1,2])],axis=2)
    output = tf.concat([output_upper,output_lower],axis=1)

    return output



def update_alpha_PC_M(input_list,G,num_sc,a,b,beta):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2

    # averaging here is the only change
    mu_square = tf.reduce_mean(mu_square,axis=-1,keepdims=True)
    mu_square = tf.tile(mu_square,(1,1,1,num_sc))

    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:1]),mu_square,tf.zeros_like(mu_square[:,:,0:1])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:1]),mu_square,tf.zeros_like(mu_square[:,0:1])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:1])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:1])],axis=1)
    w_list = (mu_square[:,1:-1,1:-1]+diag_Sigma_real[:,1:-1,1:-1])+beta*(mu_square[:,1:-1,:-2]+diag_Sigma_real[:,1:-1,:-2])\
                +beta*(mu_square[:,1:-1,2:]+diag_Sigma_real[:,1:-1,2:])+beta*(mu_square[:,2:,1:-1]+diag_Sigma_real[:,2:,1:-1])\
                +beta*(mu_square[:,:-2,1:-1]+diag_Sigma_real[:,:-2,1:-1])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list


def update_alpha_PC_high_order(input_list,G,num_sc,a,b,beta1,beta2):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2
    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:2]),mu_square,tf.zeros_like(mu_square[:,:,0:2])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:2]),mu_square,tf.zeros_like(mu_square[:,0:2])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:2]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:2])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:2]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:2])],axis=1)
    w_list = (mu_square[:,2:-2,2:-2]+diag_Sigma_real[:,2:-2,2:-2])+beta1*(mu_square[:,2:-2,1:-3]+diag_Sigma_real[:,2:-2,1:-3])\
                +beta1*(mu_square[:,2:-2,3:-1]+diag_Sigma_real[:,2:-2,3:-1])+beta1*(mu_square[:,3:-1,2:-2]+diag_Sigma_real[:,3:-1,2:-2])\
                +beta1*(mu_square[:,1:-3,2:-2]+diag_Sigma_real[:,1:-3,2:-2])\
                +beta2*(mu_square[:,2:-2,:-4]+diag_Sigma_real[:,2:-2,:-4])\
                +beta2*(mu_square[:,2:-2,4:]+diag_Sigma_real[:,2:-2,4:])+beta2*(mu_square[:,4:,2:-2]+diag_Sigma_real[:,4:,2:-2])\
                +beta2*(mu_square[:,:-4,2:-2]+diag_Sigma_real[:,:-4,2:-2])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list


def update_mu_Sigma_PC_high_order(inputs,G,num_sc,sigma_2,Mr,Mt,beta1,beta2):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)

    alpha_list = tf.cast(inputs[2],tf.complex64)
    # 倒数
    alpha_list = 1/alpha_list
    alpha_list = tf.reshape(alpha_list,(-1,G,G,num_sc))
    # expand the head and tail of two dimensions
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,:,0:2]),alpha_list,tf.zeros_like(alpha_list[:,:,0:2])],axis=2)
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,0:2]),alpha_list,tf.zeros_like(alpha_list[:,0:2])],axis=1)
    # 错位加权相加
    alpha_list = (alpha_list[:,2:-2,2:-2]+beta1*(alpha_list[:,2:-2,1:-3]+alpha_list[:,2:-2,3:-1]+\
                                                alpha_list[:,1:-3,2:-2]+alpha_list[:,3:-1,2:-2]) \
                                        +beta2 * (alpha_list[:, 2:-2, :-4] + alpha_list[:, 2:-2, 4:] + \
                                                alpha_list[:, :-4, 2:-2] + alpha_list[:, 4:, 2:-2]))
    alpha_list = tf.reshape(alpha_list,(-1,G*G,num_sc))
    alpha_list = 1 / alpha_list

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_mu_Sigma_2D(inputs,sigma_2,Mr,Mt):
    if len(inputs[0].shape)==4:
        Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    if len(inputs[0].shape)==3:
        Phi = tf.cast(inputs[0][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, 1], tf.complex64)
    y = tf.cast(inputs[1][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    if len(Phi.shape)==3:
        Rx_PhiH = tf.multiply(alpha_list, tf.transpose(Phi, (0, 2, 1), conjugate=True))
    if len(Phi.shape)==2:
        Rx_PhiH = tf.multiply(alpha_list, tf.transpose(Phi, conjugate=True))
    inv = tf.linalg.inv(
        tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
    z = tf.matmul(Rx_PhiH, inv)
    mu = tf.matmul(z, tf.expand_dims(y,axis=-1))
    diag_Sigma = alpha_list[:,:,0] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)

    # return the updated parameters
    mu_real = tf.math.real(mu)
    mu_imag = tf.math.imag(mu)
    diag_Sigma_real = tf.math.real(tf.expand_dims(diag_Sigma,axis=-1))

    return mu_real, mu_imag, diag_Sigma_real