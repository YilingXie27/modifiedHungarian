import numpy as np
from algo import track_Hungarian, track_revised_Hungarian
from keras.datasets import cifar10
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import linear_sum_assignment
import pandas as pd

##############################
#load and preprocess the data#
##############################
#%%
import os
#%%
os.chdir('/Users/yilingxie/Desktop/code/DOTmark_1.0/Data')

Resolution = [32, 64, 128, 256, 512]

Data_dir = []
for filename in os.listdir():
    if filename != '.DS_Store':
       Data_dir.append(filename)
    
X_train = []  
 
for resolution in Resolution:
    X = []
    for data_dir in Data_dir:
        os.chdir('/Users/yilingxie/Desktop/code/DOTmark_1.0/Data')
        for filename in os.listdir(data_dir):
           if str(resolution) in filename:
            #print(filename)
            os.chdir('/Users/yilingxie/Desktop/code/DOTmark_1.0/Data/'+data_dir)
            df = pd.read_csv(filename, sep=',', header=None)
            X.append(df.values.reshape(1,resolution*resolution)[0])
    X = np.array(X).astype('float32')
    X_train.append(X)


##########################################################
# Compare between Hungarian and modified Hungarian on dot#
##########################################################
#%%
r = 0
        
X = X_train[r]
X /= X.max() # Normalization of pixel values (to [0-1] range)

Xvar = X[0:50]
Yvar = X[50:100]


#%%intialize parameters before iterations (independent case
#m = np.arange(10,80,10)
m = np.arange(5,40,5)
sample = np.tile(np.repeat(m,10),2)
l = np.repeat(np.array([1,2]),70)
settings = list(zip(sample,l))

#%%create list to store the results
# results_hungarian_oper = []
i=1
results_hungarian_time = []
results_revised_hungarian_oper = []
results_revised_hungarian_time = []


#%% independent case comparison
for sample,l in settings:
 #random.seed(1)
     Xvarindex = np.random.randint(0,len(Xvar),sample)
     Yvarindex = np.random.randint(0,len(Yvar),sample)

     Xvarsample = Xvar[Xvarindex]
     Yvarsample = Yvar[Yvarindex]

     cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
     cost2=  cdist(Yvarsample, np.tile(Yvarsample,(sample,1)), 'minkowski', p=l)
     cost = cost1 + cost2
 
 
     costrep=np.repeat(cost,sample,axis=0) 

 
     time0=time.time()
     solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
     storerevisedHungariantime = time.time()-time0
     #print(storerevisedHungarianoper)
     # results_revised_hungarian_oper.append(storerevisedHungarianoper)
     results_revised_hungarian_time.append(storerevisedHungariantime)


     time0=time.time()
     row_ind, col_ind = linear_sum_assignment(costrep.max()-costrep)
     storeHungariantime = time.time()-time0
     results_hungarian_time.append(storeHungariantime)


      solution, storeHungarianoper = track_Hungarian(costrep.max()-costrep)
      results_hungarian_oper.append(storeHungarianoper)
 





     print("running {}/{} experiment".format(i,len(settings)))
     i += 1
#%%create list to store the results
# results_hungarian_oper_z = []
i=1
results_hungarian_time_z = []
# results_revised_hungarian_oper_z = []
results_revised_hungarian_time_z = []


#%% dependent case comparison
for sample,l in settings:
 Xvarindex = np.random.randint(0,len(Xvar),sample)
 Yvarindex = np.random.randint(0,len(Yvar),sample)

 Xvarsample = Xvar[Xvarindex]
 Yvarsample = Yvar[Yvarindex]
 Zvarsample = 0.5*Xvarsample + 0.5*Yvarsample


 cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
 cost2=  cdist(Zvarsample, np.tile(Zvarsample,(sample,1)), 'minkowski', p=l)
 cost = cost1 + cost2


 
 costrep=np.repeat(cost,sample,axis=0) 

 
 time0=time.time()
 solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
 storerevisedHungariantime = time.time()-time0
 # results_revised_hungarian_oper_z.append(storerevisedHungarianoper)
 #print(storerevisedHungarianoper)
 results_revised_hungarian_time_z.append(storerevisedHungariantime)




 # solution, storeHungarianoper = track_Hungarian(costrep.max()-costrep)
 # results_hungarian_oper_z.append(storeHungarianoper)
 time0=time.time()
 row_ind, col_ind = linear_sum_assignment(costrep.max()-costrep)
 storeHungariantime = time.time()-time0
 results_hungarian_time_z.append(storeHungariantime)





 print("running {}/{} experiment".format(i,len(settings)))
 i += 1

#####################
#plot of comparison#
####################
#%%
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 25
matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 25
matplotlib.rcParams['lines.markersize'] = 6
#%%plotting independent and dependent case wrt operation
l=[1,2]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
axes[0].set_ylabel(r'$ln( \# operation)$')
for i in [0,1]:
    results_revised_hungarian_oper_reshape = np.array(results_revised_hungarian_oper_z ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_oper_worst = results_revised_hungarian_oper_reshape.max(axis=1)
    results_revised_hungarian_oper_best = results_revised_hungarian_oper_reshape.min(axis=1)
    results_revised_hungarian_oper_average = results_revised_hungarian_oper_reshape.mean(axis=1)
    
    results_hungarian_oper_reshape = np.array(results_hungarian_oper_z ).reshape(2,70)[i].reshape(7,10)
    results_hungarian_oper_worst = results_hungarian_oper_reshape.max(axis=1)
    results_hungarian_oper_best = results_hungarian_oper_reshape.min(axis=1)
    results_hungarian_oper_average = results_hungarian_oper_reshape.mean(axis=1)

    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i].plot(np.log(m), np.log(results_hungarian_oper_average),label='Hungarian average', marker ='v',color='coral')
    axes[i].plot(np.log(m), np.log(results_hungarian_oper_best),label='Hungarian best', marker ='v',linestyle='-.',color='sandybrown')
    axes[i].plot(np.log(m), np.log(results_hungarian_oper_worst),label='Hungarian worst', marker ='v',linestyle=':',color='darkred')
    
    axes[i].set_xlabel(r'$ln(sample \ size)$')
    #axes[i].ylabel(r'$ln(oper)$',fontsize=18) 
    #plt.legend(fontsize=10,frameon=False)
    axes[i].set_title(r"dependent case, $p={}$".format(l[i]))
    axes[i].set_ylim(4,21)
 
l=[1,2]
for i in [0,1]:
    results_revised_hungarian_oper_reshape = np.array(results_revised_hungarian_oper ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_oper_worst = results_revised_hungarian_oper_reshape.max(axis=1)
    results_revised_hungarian_oper_best = results_revised_hungarian_oper_reshape.min(axis=1)
    results_revised_hungarian_oper_average = results_revised_hungarian_oper_reshape.mean(axis=1)
    
    results_hungarian_oper_reshape = np.array(results_hungarian_oper ).reshape(2,70)[i].reshape(7,10)
    results_hungarian_oper_worst = results_hungarian_oper_reshape.max(axis=1)
    results_hungarian_oper_best = results_hungarian_oper_reshape.min(axis=1)
    results_hungarian_oper_average = results_hungarian_oper_reshape.mean(axis=1)

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_average),label='modified_Hungarian_average', marker ='o',color='mediumslateblue')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_best),label='modified_Hungarian_best', marker ='o',linestyle='-.',color='plum')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_worst),label='modified_Hungarian_worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i+2].plot(np.log(m), np.log(results_hungarian_oper_average),label='Hungarian_average', marker ='v',color='coral')
    axes[i+2].plot(np.log(m), np.log(results_hungarian_oper_best),label='Hungarian_best', marker ='v',linestyle='-.',color='sandybrown')
    axes[i+2].plot(np.log(m), np.log(results_hungarian_oper_worst),label='Hungarian_worst', marker ='v',linestyle=':',color='darkred')
    
    axes[i+2].set_xlabel(r'$ln(sample \ size)$')
    #plt.ylabel(r'$ln(oper)$',fontsize=18) 
    #plt.legend(fontsize=10,frameon=False,loc=2)
    axes[i+2].set_title(r"independent case, $p={}$".format(l[i]))
    axes[i+2].set_ylim(4,21)    
 
    
fig.tight_layout(rect=[0, 0.08, 1, 1])  
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.02), ncol=6)    



os.chdir('/Users/yilingxie/Desktop/code')
plt.savefig("Stanfordoper" + str(Resolution[r]) + ".pdf".format(l[i]), format="pdf")
plt.figure()

#%%plotting independent and dependent case wrt time
l=[1,2]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
axes[0].set_ylabel(r'$ln(time)$')
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_z ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_hungarian_time_reshape = np.array(results_hungarian_time_z ).reshape(2,70)[i].reshape(7,10)
    results_hungarian_time_worst = results_hungarian_time_reshape.max(axis=1)
    results_hungarian_time_best = results_hungarian_time_reshape.min(axis=1)
    results_hungarian_time_average = results_hungarian_time_reshape.mean(axis=1)

    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i].plot(np.log(m), np.log(results_hungarian_time_average),label='Hungarian average', marker ='v',color='coral')
    axes[i].plot(np.log(m), np.log(results_hungarian_time_best),label='Hungarian best', marker ='v',linestyle='-.',color='sandybrown')
    axes[i].plot(np.log(m), np.log(results_hungarian_time_worst),label='Hungarian worst', marker ='v',linestyle=':',color='darkred')
    
    axes[i].set_xlabel(r'$ln(sample \ size)$')
    axes[i].set_title(r"dependent case, $p={}$".format(l[i]))
    axes[i].set_ylim(-11,2)

     
l=[1,2]
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_hungarian_time_reshape = np.array(results_hungarian_time ).reshape(2,70)[i].reshape(7,10)
    results_hungarian_time_worst = results_hungarian_time_reshape.max(axis=1)
    results_hungarian_time_best = results_hungarian_time_reshape.min(axis=1)
    results_hungarian_time_average = results_hungarian_time_reshape.mean(axis=1)

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified_Hungarian_average', marker ='o',color='mediumslateblue')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified_Hungarian_best', marker ='o',linestyle='-.',color='plum')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified_Hungarian_worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i+2].plot(np.log(m), np.log(results_hungarian_time_average),label='Hungarian_average', marker ='v',color='coral')
    axes[i+2].plot(np.log(m), np.log(results_hungarian_time_best),label='Hungarian_best', marker ='v',linestyle='-.',color='sandybrown')
    axes[i+2].plot(np.log(m), np.log(results_hungarian_time_worst),label='Hungarian_worst', marker ='v',linestyle=':',color='darkred')
    
    axes[i+2].set_xlabel(r'$ln(sample \ size)$')
    axes[i+2].set_title(r"independent case, $p={}$".format(l[i]))
    axes[i+2].set_ylim(-11,2)

        
 
    
fig.tight_layout(rect=[0, 0.08, 1, 1])  
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.02), ncol=6)    


os.chdir('/Users/yilingxie/Desktop/code')
plt.savefig("Stanfordtime" + str(Resolution[r]) + ".pdf".format(l[i]), format="pdf")
plt.figure()
