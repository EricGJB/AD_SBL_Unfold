#%% generate channel
from matplotlib import pyplot as plt
import numpy as np
random_seed = 2023
np.random.seed(random_seed)

# fixed system parameters
fc, fs, tau_max, num_subpaths = 28 * 1e9, 4 * 1e9, 25 * 1e-9, 10

# varaibles 
Nr, num_sc, data_num, num_clusters = 32, 32, 10000, 3

channel_model = 'cluster'

if channel_model == 'path':
    AS = 0
    DS = 0
else:
    AS = 4
    DS = 0.06*1e-9

b_angle = np.sqrt(AS**2/2) # the variance of Laplace distribution is 2b^2
b_delay = np.sqrt(DS**2/2)

eta = fs / num_sc
Lp = num_clusters * num_subpaths

H_list = np.zeros((data_num, num_sc, Nr),dtype=np.complex64)

# normalization_vector = np.ones(Lp)/np.sqrt(num_subpaths)
normalization_vector = np.ones(Lp)/np.sqrt(Lp)

print('Generating channels')

for i in range(data_num):
    if i % 1000 == 0:
        print('Channel %d/%d' % (i, data_num)) 
    path_gains = np.sqrt(1 / 2) * (np.random.randn(Lp) + 1j * np.random.randn(Lp))
    taus = np.zeros(Lp)
    normalized_AoAs = np.zeros(Lp)

    for nc in range(num_clusters):
        # truncated laplacian distribution
        mean_AoA = np.random.uniform(0,360)    
        
        AoAs = np.random.laplace(loc=mean_AoA, scale=b_angle, size=num_subpaths) 
        AoAs = np.maximum(AoAs, mean_AoA-2*AS)
        AoAs = np.minimum(AoAs, mean_AoA + 2 * AS)
        AoAs = AoAs / 180 * np.pi
        
        normalized_AoAs[nc*num_subpaths:(nc+1)*num_subpaths] = np.sin(AoAs) / 2
        
        mean_tau = np.random.uniform(0, tau_max)
        taus_cluster = np.random.laplace(loc=mean_tau, scale=b_delay, size=num_subpaths)
        taus_cluster = np.maximum(taus_cluster,mean_tau-2*DS)
        taus_cluster = np.minimum(taus_cluster, mean_tau + 2 * DS)
        taus_cluster = np.maximum(taus_cluster, 0)
        taus_cluster = np.minimum(taus_cluster, tau_max)
        taus[nc*num_subpaths:(nc+1)*num_subpaths] = taus_cluster

    for n in range(num_sc):
        fn = fc + eta*(n-(num_sc-1)/2)
        # frequency dependent steering vectors with beam squint
        A_T = np.exp(-2j*np.pi*(fn/fc)*(np.expand_dims(normalized_AoAs,axis=-1).dot(np.expand_dims(np.arange(Nr),axis=0))))
        scaler_matrix = path_gains*np.exp(-2j*np.pi*fn*taus)*normalization_vector
        h_sample = np.squeeze(np.expand_dims(scaler_matrix,axis=0).dot(A_T))
        H_list[i,n] = h_sample

print(H_list.shape) # (data_num, num_sc, Nr)
print('\n')

# plt.plot(np.abs(np.fft.fft(h_sample)))


#%% generate data
SNR = 10 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 16
N_RF = 4
Q = Mr//N_RF # number of instances to traverse Mr beams using only N_RF (N_RF<Mr) RF chains 

# the receive combining matrix, consists of random binary phases  
W = np.random.choice([1,-1],Nr*Mr,replace=True)/np.sqrt(Nr)
W = np.reshape(W, (Nr, Mr))
W = np.matrix(W)

# random phases
# W = np.random.uniform(-np.pi,np.pi,Nr*Mr)
# W = np.reshape(W, (Nr, Mr))
# W = (np.cos(W)+1j*np.sin(W))/np.sqrt(Nr)
# W = np.matrix(W)

W_original = np.copy(W)

# partially connected HAD
# W = np.zeros((Nr,Mr))+1j*np.zeros((Nr,Mr))
# for j in range(Mr):
#     random_phases_R = np.random.uniform(0, 2 * np.pi, Nr//Mr)
#     W[j*2:(j+1)*2,j]=(np.cos(random_phases_R) + 1j * np.sin(random_phases_R)) / np.sqrt(Nr/Mr)
# W = np.matrix(W) 

Y_list = np.zeros((data_num, num_sc, Mr)) + 1j * np.zeros((data_num, num_sc, Mr))

print('Generating data samples')

C = np.zeros((Mr,Mr),dtype=np.complex64)
for i in range(data_num):
    if i % 1000 == 0:
        print('Sample %d/%d' % (i, data_num)) 

    for q in range(Q):
        W_q = W[:,q*N_RF:(q+1)*N_RF]
        C_q = W_q.H.dot(W_q)
        C[q*N_RF:(q+1)*N_RF,q*N_RF:(q+1)*N_RF] = C_q*sigma_2
        noise = np.sqrt(sigma_2/2)*(np.random.randn(num_sc,Nr)+1j*np.random.randn(num_sc,Nr))
        for j in range(num_sc):
            Y_list[i,j,q*N_RF:(q+1)*N_RF] = np.squeeze(W_q.H.dot(np.transpose(H_list[i,j:j+1]+noise[j:j+1])))
            # noiseless 
            # Y_list[i,j,q*N_RF:(q+1)*N_RF] = np.squeeze(W_q.H.dot(np.transpose(H_list[i,j:j+1])))
print(Y_list.shape) # (data_num, num_sc, Mr)

pre_white = 1

if pre_white:
    # calculate the cholesky decomposition of the equivalent noise covariance matrix
    D = np.linalg.cholesky(C)/np.sqrt(sigma_2) 
    D = np.matrix(D)
    
    plt.figure()
    plt.imshow(np.abs(C))
    plt.figure()
    plt.imshow(np.abs(D))
    
    assert np.linalg.norm(C-sigma_2*(D.dot(D.H)))<1e-6
    
    # apply D^(-1) to Y_list and the receive combining matrix W to pre-white the noise
    pre_whiter = np.linalg.inv(D)
    W = (pre_whiter.dot(W.H)).H
    
    Y_list_original = np.copy(Y_list)
    Y_list = np.zeros((data_num, num_sc, Mr)) + 1j * np.zeros((data_num, num_sc, Mr))
    for i in range(data_num):
        if i % 1000 == 0:
            print('Pre-whiten Sample %d/%d' % (i, data_num)) 
        for j in range(num_sc):
            Y_list[i,j] = np.squeeze(pre_whiter.dot(np.expand_dims(Y_list_original[i,j],axis=-1)))

from scipy import io
# save common dataset for fair performance comparison among various algorithms 
dataset_name = 'data_%dBeams_%dSNR_%s.mat'%(Mr,SNR,channel_model)
io.savemat('./data/'+dataset_name,{'H_list':H_list,'Y_list':Y_list,'W':W,'W_original':W_original,'pre_whiter':pre_whiter})
print(dataset_name)
print('Dataset generated!')
