import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary

# use partial dataset 
test_num = 200

Nr = 32

SNR = 20 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 16

data = io.loadmat('./data/data_%dBeams_%dSNR_path_random_phaseW.mat'%(Mr,SNR))

H_list = data['H_list'][:test_num]
Y_list = data['Y_list'][:test_num]

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

Kron2 = np.kron(A_K,A_R)

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

Phi = (np.kron(np.eye(num_sc),W.H).dot(A_list_expanded)).dot(np.kron(A_K,np.eye(G_angle)))
Phi = np.array(Phi)

y_list = np.reshape(Y_list,(test_num,num_sc*Mr))

svd = 1 # whether to execute unitary pre-processing 
if svd:
    U,Sigma,V = np.linalg.svd(Phi)
    y_list = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(y_list)))
    Phi = np.transpose(np.conjugate(U)).dot(Phi)

H_list = np.transpose(H_list,(0,2,1))


#%%  UAMP-SBL
G = G_delay*G_angle

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
        #print(epsilon)
        # heuristic epsilon methods 
        # epsilon = 3*sigma_2 
        # epsilon = 0
    return X_hat

num_iter = 50

error = 0
error_nmse = 0
good_sample_list = []
good_performance_list = []
for i in range(test_num): 
    # if i%10==0:
    print('Sample %d/%d'%(i,test_num))
    true_H = H_list[i]

    r = y_list[i]
    r = np.expand_dims(r,axis=-1)
    
    x_hat = AMP_SBL(r,Phi,G,Mr*num_sc,sigma_2,num_iter)
        
    prediction_X = np.reshape(x_hat,(G_delay,G_angle))
    # notice that the inverse vectorization operation is also to re-stack column-wisely
    prediction_X = np.transpose(prediction_X)
    
    prediction_Q = prediction_X.dot(A_K.T) # angular-frequency channel prediction
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(prediction_Q[:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
    
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    sample_nmse = (np.linalg.norm(prediction_H - true_H) / np.linalg.norm(true_H)) ** 2
    error_nmse = error_nmse + sample_nmse

    if sample_nmse<0.05: # good samples, bad samples have inf nmse
        print(sample_nmse)
        good_sample_list.append(i)
        good_performance_list.append(sample_nmse)
        # plt.figure()
        # plt.imshow(np.abs(prediction_X),cmap='gray_r')
        # plt.xlabel('Delay Grids')
        # plt.ylabel('Angular Grids')
        # plt.title('Angular-Delay Channel Modulus')

mse_amp_sbl = error / (test_num * Nr * num_sc)
nmse_amp_sbl = error_nmse / test_num
print(mse_amp_sbl)
print(nmse_amp_sbl)

# io.savemat('./results/UAMP_SBL_good_test_samples_%dBeams_%dSNR_path.mat'%(Mr,SNR),{'index_list':np.array(good_sample_list),'nmse_list':np.array(good_performance_list)})

plt.figure()
plt.imshow(np.abs(prediction_X),cmap='gray_r')
plt.xlabel('Delay Grids')
plt.ylabel('Angular Grids')
plt.title('Angular-Delay Channel Modulus')
