import numpy as np
from algo import track_Hungarian, track_revised_Hungarian,sinkhorn_log
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import linear_sum_assignment

#####################################################################
# Compare between Sinkhorn and modified Hungarian on synthetic data #
#####################################################################
#%%intialize parameters before iterations (independent case
m = np.arange(5,40,5)
sample = np.tile(np.repeat(m,10),2)
l = np.repeat(np.array([1,2]),70)
settings = list(zip(sample,l))

#%%create list to store the results  (independent case
i=1

results_revised_hungarian_oper_syn = []
results_revised_hungarian_time_syn = []
results_sinkhorn_oper_syn = []
results_sinkhorn_time_syn = []
results_greenkhorn_oper_syn = []
results_greenkhorn_time_syn = []

#%% independent case comparison
for sample,l in settings:
 sigma = np.eye(10)*30
 mu = np.ones(10)*5
 Xvarsample = np.random.multivariate_normal(mu, sigma, sample)
 Yvarsample = np.random.uniform(10,20,(sample,25))
 cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
 cost2=  cdist(Yvarsample, np.tile(Yvarsample,(sample,1)), 'minkowski', p=l)
 cost = cost1 + cost2
 
 
 costrep=np.repeat(cost,sample,axis=0) 

 
 time0=time.time()
 solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
 storerevisedHungariantime = time.time()-time0
 results_revised_hungarian_oper_syn.append(storerevisedHungarianoper)
 results_revised_hungarian_time_syn.append(storerevisedHungariantime)
 print(storerevisedHungariantime)
 
 a=[1/sample]*sample
 b=[1/(sample*sample)]*(sample*sample)
 
 time0=time.time()
 solution, storesinkhornoper = sinkhorn_log(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
 storesinkhorntime = time.time()-time0
 results_sinkhorn_oper_syn.append(storesinkhornoper)
 results_sinkhorn_time_syn.append(storesinkhorntime)
 print(storesinkhorntime)

 # time0=time.time()
 # solution, storegreenkhornoper = greenkhorn(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
 # storegreenkhorntime = time.time()-time0
 # results_greenkhorn_oper_syn.append(storegreenkhornoper)
 # results_greenkhorn_time_syn.append(storegreenkhorntime)
 # print(storegreenkhorntime)



 print("running {}/{} experiment".format(i,len(settings)))
 i += 1

#%%create list to store the results  (dependent case
i=1

results_revised_hungarian_oper_z_syn = []
results_revised_hungarian_time_z_syn = []
results_sinkhorn_oper_z_syn = []
results_sinkhorn_time_z_syn = []
results_greenkhorn_oper_z_syn = []
results_greenkhorn_time_z_syn = []



#%% dependent case comparison
for sample,l in settings:
 sigma = np.eye(10)*30
 mu = np.ones(10)*5
 Xvarsample = np.random.multivariate_normal(mu, sigma, sample)
 Yvarsample = np.random.uniform(10,20,(sample,25))
 cost1 = cdist(Xvarsample, np.repeat(Xvarsample,sample,axis=0), 'minkowski', p=l)
 Zvarsample = Yvarsample[:,0:5]+Xvarsample[:,0:5]
 cost2=  cdist(Zvarsample, np.tile(Zvarsample,(sample,1)), 'minkowski', p=l)
 cost = cost1 + cost2
 
 
 costrep=np.repeat(cost,sample,axis=0) 

 
 time0=time.time()
 solution, storerevisedHungarianoper = track_revised_Hungarian(cost.max()-cost)
 storerevisedHungariantime = time.time()-time0
 results_revised_hungarian_oper_z_syn.append(storerevisedHungarianoper)
 results_revised_hungarian_time_z_syn.append(storerevisedHungariantime)
 print(storerevisedHungariantime)



 a=[1/sample]*sample
 b=[1/(sample*sample)]*(sample*sample)
 
 time0=time.time()
 solution, storesinkhornoper = sinkhorn_log(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
 storesinkhorntime = time.time()-time0
 results_sinkhorn_oper_z_syn.append(storesinkhornoper)
 results_sinkhorn_time_z_syn.append(storesinkhorntime)
 print(storesinkhorntime)

 # time0=time.time()
 # solution, storegreenkhornoper = greenkhorn(np.array(a), np.array(b), cost, reg=0.1, acc=0.0001)
 # storegreenkhorntime = time.time()-time0
 # results_greenkhorn_oper_z_syn.append(storegreenkhornoper)
 # results_greenkhorn_time_z_syn.append(storegreenkhorntime)





 print("running {}/{} experiment".format(i,len(settings)))
 i += 1




####################
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
    results_revised_hungarian_oper_reshape = np.array(results_revised_hungarian_oper_z_syn).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_oper_worst = results_revised_hungarian_oper_reshape.max(axis=1)
    results_revised_hungarian_oper_best = results_revised_hungarian_oper_reshape.min(axis=1)
    results_revised_hungarian_oper_average = results_revised_hungarian_oper_reshape.mean(axis=1)
    
    results_sinkhorn_oper_reshape = np.array(results_sinkhorn_oper_z_syn).reshape(2,70)[i].reshape(7,10)
    results_sinkhorn_oper_worst = results_sinkhorn_oper_reshape.max(axis=1)
    results_sinkhorn_oper_best = results_sinkhorn_oper_reshape.min(axis=1)
    results_sinkhorn_oper_average = results_sinkhorn_oper_reshape.mean(axis=1)

    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_oper_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i].plot(np.log(m), np.log(results_sinkhorn_oper_average),label='sinkhorn average', marker ='v',color='teal')
    axes[i].plot(np.log(m), np.log(results_sinkhorn_oper_best),label='sinkhorn best', marker ='v',linestyle='-.',color='powderblue')
    axes[i].plot(np.log(m), np.log(results_sinkhorn_oper_worst),label='sinkhorn worst', marker ='v',linestyle=':',color='darkslategray')


    
    axes[i].set_xlabel(r'$ln(sample \ size)$')
    axes[i].set_title(r"dependent case, $p={}$".format(l[i]))
    axes[i].set_ylim(4,25)

l=[1,2]
for i in [0,1]:
    results_revised_hungarian_oper_reshape = np.array(results_revised_hungarian_oper_syn).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_oper_worst = results_revised_hungarian_oper_reshape.max(axis=1)
    results_revised_hungarian_oper_best = results_revised_hungarian_oper_reshape.min(axis=1)
    results_revised_hungarian_oper_average = results_revised_hungarian_oper_reshape.mean(axis=1)
    
    results_sinkhorn_oper_reshape = np.array(results_sinkhorn_oper_syn).reshape(2,70)[i].reshape(7,10)
    results_sinkhorn_oper_worst = results_sinkhorn_oper_reshape.max(axis=1)
    results_sinkhorn_oper_best = results_sinkhorn_oper_reshape.min(axis=1)
    results_sinkhorn_oper_average = results_sinkhorn_oper_reshape.mean(axis=1)

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_oper_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_oper_average),label='sinkhorn average', marker ='<',color='teal')
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_oper_best),label='sinkhorn best', marker ='<',linestyle='-.',color='powderblue')
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_oper_worst),label='sinkhorn worst', marker ='<',linestyle=':',color='darkslategray')
    
    axes[i+2].set_xlabel(r'$ln(sample \ size)$')
    axes[i+2].set_title(r"independent case, $p={}$".format(l[i]))
    axes[i+2].set_ylim(4,25)
    
 
    
fig.tight_layout(rect=[0, 0.08, 1, 1])  
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.02), ncol=6)    



#plt.savefig("synopersink.pdf".format(l[i]), format="pdf")
plt.figure()

#%%plotting independent and dependent case wrt time
l=[1,2]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
axes[0].set_ylabel(r'$ln(time)$')
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_z_syn).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_sinkhorn_time_reshape = np.array(results_sinkhorn_time_z_syn).reshape(2,70)[i].reshape(7,10)
    results_sinkhorn_time_worst = results_sinkhorn_time_reshape.max(axis=1)
    results_sinkhorn_time_best = results_sinkhorn_time_reshape.min(axis=1)
    results_sinkhorn_time_average = results_sinkhorn_time_reshape.mean(axis=1)

    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i].plot(np.log(m), np.log(results_sinkhorn_time_average),label='sinkhorn average', marker ='v',color='teal')
    axes[i].plot(np.log(m), np.log(results_sinkhorn_time_best),label='sinkhorn best', marker ='v',linestyle='-.',color='powderblue')
    axes[i].plot(np.log(m), np.log(results_sinkhorn_time_worst),label='sinkhorn worst', marker ='v',linestyle=':',color='darkslategray')
    
    axes[i].set_xlabel(r'$ln(sample \ size)$')
    #axes[i].ylabel(r'$ln(time)$',fontsize=18) 
    #plt.legend(fontsize=10,frameon=False)
    axes[i].set_title(r"dependent case, $p={}$".format(l[i]))
    axes[i].set_ylim(-10,6)

 
l=[1,2]
for i in [0,1]:
    results_revised_hungarian_time_reshape = np.array(results_revised_hungarian_time_syn).reshape(2,70)[i].reshape(7,10)
    results_revised_hungarian_time_worst = results_revised_hungarian_time_reshape.max(axis=1)
    results_revised_hungarian_time_best = results_revised_hungarian_time_reshape.min(axis=1)
    results_revised_hungarian_time_average = results_revised_hungarian_time_reshape.mean(axis=1)
    
    results_sinkhorn_time_reshape = np.array(results_sinkhorn_time_syn).reshape(2,70)[i].reshape(7,10)
    results_sinkhorn_time_worst = results_sinkhorn_time_reshape.max(axis=1)
    results_sinkhorn_time_best = results_sinkhorn_time_reshape.min(axis=1)
    results_sinkhorn_time_average = results_sinkhorn_time_reshape.mean(axis=1)

    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_average),label='modified Hungarian average', marker ='o',color='mediumslateblue')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_best),label='modified Hungarian best', marker ='o',linestyle='-.',color='plum')
    axes[i+2].plot(np.log(m), np.log(results_revised_hungarian_time_worst),label='modified Hungarian worst', marker ='o',linestyle=':',color='indigo')
    
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_time_average),label='sinkhorn average', marker ='<',color='teal')
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_time_best),label='sinkhorn best', marker ='<',linestyle='-.',color='powderblue')
    axes[i+2].plot(np.log(m), np.log(results_sinkhorn_time_worst),label='sinkhorn worst', marker ='<',linestyle=':',color='darkslategray')
    
    
    axes[i+2].set_xlabel(r'$ln(sample \ size)$')
    #plt.ylabel(r'$ln(time)$',fontsize=18) 
    #plt.legend(fontsize=10,frameon=False,loc=2)
    axes[i+2].set_title(r"independent case, $p={}$".format(l[i]))
    axes[i+2].set_ylim(-10,6)
    
 
    
fig.tight_layout(rect=[0, 0.08, 1, 1])  
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.02), ncol=6)    



#plt.savefig("syntimesink.pdf".format(l[i]), format="pdf")
plt.figure()

