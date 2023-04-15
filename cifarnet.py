import numpy as np
from algo import track_Hungarian, track_revised_Hungarian
from keras.datasets import cifar10
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import linear_sum_assignment
import networkx as nx
import math

##############################
#load and preprocess the data#
##############################
#%%
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


X_train = np.reshape(X_train,(50000,3072)) # Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)
X_test = np.reshape(X_test,(10000,3072))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



X_train /= 255 # Normalization of pixel values (to [0-1] range)
X_test /= 255


X_total = np.vstack((X_train,X_test))
y_total = np.vstack((y_train,y_test))


Xvar = X_total[np.where(y_total<=4)[0]]
Yvar = X_total[np.where(y_total>4)[0]]
Zvar = Xvar*0.5 + Yvar*0.5


#####################################################################
# Compare between network simplex and modified Hungarian on cifar 10#
#####################################################################
#%%intialize parameters before iterations (independent case
m = np.arange(5,40,5)
sample = np.tile(np.repeat(m,10),2)
l = np.repeat(np.array([1,2]),70)
settings = list(zip(sample,l))

#%%create list to store the results
i=1
results_revised_hungarian_time = []
results_net_time = []


#%% independent case comparison
for sample,l in settings:
 #random.seed(1)
 Xvarindex = np.random.randint(0,len(Xvar),sample)
 Yvarindex = np.random.randint(0,len(Yvar),sample)

 Xvarsample = Xvar[Xvarindex]
 Yvarsample = Yvar[Yvarindex]

 cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
 cost2=  cdist(Yvarsample[:,0:1536], np.tile(Yvarsample[:,0:1536],(sample,1)), 'minkowski', p=l)
 cost = cost1 + cost2
 
 
 costrep=np.repeat(cost,sample,axis=0) 

 
 time0=time.time()
 solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
 storerevisedHungariantime = time.time()-time0
 #print(storerevisedHungarianoper)
 results_revised_hungarian_time.append(storerevisedHungariantime)



 time0=time.time()
 G = nx.DiGraph()
 left  = list(np.arange(sample))
 right =  list(np.arange(sample*sample)+sample)

 G.add_nodes_from(left,demand = -sample)
 G.add_nodes_from(right,demand= 1)

 source_nodes = list(np.repeat(left,sample*sample))
 dest_nodes = right*sample


 vfunc = np.vectorize(lambda t: math.floor(t))
 costround = vfunc(cost)

# Each element of this zip will be
# (source[i], dest[i], data[i]) 
 for u,v,d in zip(source_nodes, dest_nodes, costround.ravel()):
    G.add_edge(u, v, weight=d)
 
 flowCost, flowDict = nx.network_simplex(G)
 storenettime = time.time()-time0
 results_net_time.append(storenettime)





 print("running {}/{} experiment".format(i,len(settings)))
 i += 1


#%%intialize parameters before iterations (dependent case
sample = np.tile(np.repeat(m,10),2)
l = np.repeat(np.array([1,2]),70)
settings = list(zip(sample,l))

#%%create list to store the results
i=1
results_revised_hungarian_time_z = []
results_net_time_z = []

#%% dependent case comparison
for sample,l in settings:
 Xvarindex = np.random.randint(0,len(Xvar),sample)
 Yvarindex = np.random.randint(0,len(Yvar),sample)

 Xvarsample = Xvar[Xvarindex]
 Yvarsample = Yvar[Yvarindex]
 Zvarsample = 0.5*Yvarsample[:,1536:3073]+0.5*Yvarsample[:,0:1536]


 cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
 cost2=  cdist(Zvarsample, np.tile(Zvarsample,(sample,1)), 'minkowski', p=l)
 cost = cost1 + cost2


 
 costrep=np.repeat(cost,sample,axis=0) 

 
 time0=time.time()
 solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
 storerevisedHungariantime = time.time()-time0
 results_revised_hungarian_time_z.append(storerevisedHungariantime)




 time0=time.time()
 G = nx.DiGraph()
 left  = list(np.arange(sample))
 right =  list(np.arange(sample*sample)+sample)

 G.add_nodes_from(left,demand = -sample)
 G.add_nodes_from(right,demand= 1)

 source_nodes = list(np.repeat(left,sample*sample))
 dest_nodes = right*sample


 vfunc = np.vectorize(lambda t: math.floor(t))
 costround = vfunc(cost)

# Each element of this zip will be
# (source[i], dest[i], data[i]) 
 for u,v,d in zip(source_nodes, dest_nodes, costround.ravel()):
    G.add_edge(u, v, weight=d)
 
 flowCost, flowDict = nx.network_simplex(G)
 storenettime = time.time()-time0
 results_net_time_z.append(storenettime)





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

#%%plotting independent and dependent case wrt time
l=[1,2]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
axes[0].set_ylabel(r'$ln(time)$')
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_z ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_net_time_reshape = np.array(results_net_time_z ).reshape(2,70)[i].reshape(7,10)
    results_net_time_worst = results_net_time_reshape.max(axis=1)
    results_net_time_best = results_net_time_reshape.min(axis=1)
    results_net_time_average = results_net_time_reshape.mean(axis=1)

    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i].plot(np.log(m), np.log(results_net_time_average),label='Network simplex average', marker ='*',color='dodgerblue')
    axes[i].plot(np.log(m), np.log(results_net_time_best),label='Network simplex best', marker ='*',linestyle='-.',color='skyblue')
    axes[i].plot(np.log(m), np.log(results_net_time_worst),label='Network simplex worst', marker ='*',linestyle=':',color='steelblue')
    
    axes[i].set_xlabel(r'$ln(sample \ size)$')
    axes[i].set_title(r"dependent case, $p={}$".format(l[i]))
    axes[i].set_ylim(-10,2)

     
l=[1,2]
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time ).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_net_time_reshape = np.array(results_net_time ).reshape(2,70)[i].reshape(7,10)
    results_net_time_worst = results_net_time_reshape.max(axis=1)
    results_net_time_best = results_net_time_reshape.min(axis=1)
    results_net_time_average = results_net_time_reshape.mean(axis=1)

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified_Hungarian_average', marker ='o',color='mediumslateblue')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified_Hungarian_best', marker ='o',linestyle='-.',color='plum')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified_Hungarian_worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i+2].plot(np.log(m), np.log(results_net_time_average),label='Network simplex average', marker ='*',color='dodgerblue')
    axes[i+2].plot(np.log(m), np.log(results_net_time_best),label='Network simplex best', marker ='*',linestyle='-.',color='skyblue')
    axes[i+2].plot(np.log(m), np.log(results_net_time_worst),label='Network simplex worst', marker ='*',linestyle=':',color='steelblue')
    
    axes[i+2].set_xlabel(r'$ln(sample \ size)$')
    axes[i+2].set_title(r"independent case, $p={}$".format(l[i]))
    axes[i+2].set_ylim(-10,2)

        
 
    
fig.tight_layout(rect=[0, 0.08, 1, 1])  
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.02), ncol=6)    



plt.savefig("CIFARt10imenet.pdf".format(l[i]), format="pdf")
plt.figure()

