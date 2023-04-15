import numpy as np
def Hungarian(Cost):
    ##generate initial labelling
    Size = np.shape(Cost)[0]
    x_label = Cost.max(axis = 1)[...,np.newaxis] #add dimension to faciliate calculating
    y_label = np.zeros(Size)[np.newaxis,]
    
        
    
    ##generate initial matching
    Cost_temp = x_label - Cost
    Matched = np.zeros((Size, Size))
    col_uncovered = np.ones(Size)
    row_uncovered = np.ones(Size)
    for i, j in zip(*np.where(Cost_temp == 0)):
        if col_uncovered[j] and row_uncovered[i]:
            Matched[i, j] = 1  #star Z
            col_uncovered[j] = False
            row_uncovered[i] = False
            
    if Matched.sum() == Size:
        return Matched
    free = np.where(~Matched.any(axis=1))[0][0]  #initializing; pick one free vertex 
    S = [free]
    T = []    
    Unmatched_path = []
    Matched_path = []

    while Matched.sum() != Size:
        #step 2: pick free vertex
        
        mark = 1 #ancillary variable, will be explained later
        ##compute the neighbor of S
        S_neighbor = []
        for s in S:
            s_neighbor = list(np.where(Cost_temp[s] == 0)[0])
            S_neighbor.extend(s_neighbor) 
        #step 3: update labelling 
        if set(S_neighbor)==set(T):
           Tcomp = list(set(range(Size))-set(T)) #get y\not in T
           Cost_temp = x_label + y_label - Cost
           alpha = Cost_temp[S,:][:,Tcomp].min()
           x_label[S] = x_label[S] - alpha
           y_label[0,T] = y_label[0,T] + alpha
           #update reduced cost function
           Cost_temp = x_label + y_label - Cost
           #compute neighbor
           S_neighbor = []
           for s in S:
             s_neighbor = list(np.where(Cost_temp[s] == 0)[0])
             S_neighbor.extend(s_neighbor) 
        
        #step 4
        if set(S_neighbor)!=set(T):
            y= list(set(S_neighbor)-set(T))[0]
            
            if not np.where(Matched[:,y])[0].size:#free
               ycorre = list(set(np.where(Cost_temp[:,y] == 0)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size))-set(S)) )[0]#ycorre should in S
               
               Unmatched_path.append([ycorre,y])
               
               Unmatched_augment_path=[]
               Matched_augment_path=[]
               Unmatched_augment_path.append([ycorre,y])
               terminate = ycorre
               #augmenting
               #first we trace the augmenting path backward
               while terminate != free:#when the initial point is free vertex, stop tracing
                  #if np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0].size:
                     Matched_augment_path.append( Matched_path[np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0][0]])
                     Unmatched_augment_path.append( Unmatched_path[np.where(np.array(Unmatched_path)[:,1]== Matched_augment_path[-1][1])[0][0]])
                     terminate= Unmatched_augment_path[-1][0]
                   
               for e in  Unmatched_augment_path:
                   Matched[e[0],e[1]]=1
               for e in  Matched_augment_path:
                   Matched[e[0],e[1]]=0    
               #   flag = np.where(Matched[ycorre,:]==1)[0][0]
               #   index2 = np.where(np.array(Unmatched_path)[:,1]==flag)[0]
               #   if index2.size:
               #     for i in range(index2[0]+1):
               #       e = Unmatched_path[i]
               #       Matched[e[0],e[1]]=1
               # Matched[ycorre,y]=1      
                     

               # index1 = np.where(np.array(Matched_path)[:,0]==ycorre)[0]
               # if index1.size:
               #    for i in range(index1[0]+1):
               #       e = Matched_path[i]
               #       Matched[e[0],e[1]]=0
                   
                   
               
               
               #back to step 2
               if Matched.sum() == Size:
                 return Matched
               free = np.where(~Matched.any(axis=1))[0][0]  
               S = [free]
               T = []
               Unmatched_path = []
               Matched_path = []
               mark = 0 ##tell the alg go back to step 2
            
            if (mark == 1) & (np.where(Matched[:,y])[0].size):#matched
               ycorre = list(set(np.where(Cost_temp[:,y] == 0)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size))-set(S)) )[0]
               
               #print('suc')
               
               #extending the alternating tree
               Unmatched_path.append([ycorre,y])
               z = np.where(Matched[:,y])[0][0]
               Matched_path.append([z,y])
               
               T.append(y)
               S.append(np.where(Matched[:,y])[0][0])



def revised_Hungarian(Cost):
    ##generate initial labelling
    Size1 = np.shape(Cost)[0]
    Size2 = np.shape(Cost)[1]
    x_label = np.zeros(Size1)[...,np.newaxis]
    y_label = Cost.max(axis = 0)[np.newaxis,] #add dimension to faciliate calculating

    
        
    
    ##generate initial psedo-matching
    Cost_temp = y_label - Cost
    Matched = np.zeros((Size1, Size2))
    col_uncovered = np.zeros(Size2)
    row_uncovered = np.zeros(Size1)
    for i, j in zip(*np.where(Cost_temp == 0)):
        if col_uncovered[j]==0 and row_uncovered[i] < Size1:
            Matched[i, j] = 1  #star Z
            col_uncovered[j] += 1
            row_uncovered[i] += 1
            
    if Matched.sum() == Size2 and all(Matched.sum(axis=1) == (Size1*np.ones(Size1))):#judge if it is a perfect psedo-matching
        return Matched
    free = np.where(Matched.sum(axis=1)<Size1)[0][0]  #initializing; pick one free vertex 
    S = [free]
    #print('free')
    T = []
    if np.where(Matched[free]==1)[0].size:
       T.extend(list(np.where(Matched[free]==1)[0]))
    Unmatched_path = []
    Matched_path = []

    while Matched.sum() != Size2 and any(Matched.sum(axis=1) != (Size1*np.ones(Size1))):
        #step 2: pick free vertex
        
        mark = 1 #ancillary variable, will be explained later
        ##compute the neighbor of S
        S_neighbor = []
        for s in S:
            s_neighbor = list(np.where(Cost_temp[s] <= 1e-8)[0])
            S_neighbor.extend(s_neighbor) 
        #step 3: update labelling 
        if not len(list(set(S_neighbor)-set(T))):
           #print("labeling") 
           Tcomp = list(set(range(Size2))-set(T)) #get y\not in T
           Cost_temp = x_label + y_label - Cost
           alpha = Cost_temp[S,:][:,Tcomp].min()
           x_label[S] = x_label[S] - alpha
           y_label[0,T] = y_label[0,T] + alpha
           #update reduced cost function
           Cost_temp = x_label + y_label - Cost
           #compute neighbor
           S_neighbor = []
           for s in S:
             s_neighbor = list(np.where(Cost_temp[s] <= 1e-8)[0])
             S_neighbor.extend(s_neighbor) 
        
        #step 4
        if len(list(set(S_neighbor)-set(T))):
            y= list(set(S_neighbor)-set(T))[0]
            
            if not np.where(Matched[:,y])[0].size:#free
               ycorre = list(set(np.where(Cost_temp[:,y] <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size1))-set(S)) )[0]#ycorre should in S
               
               Unmatched_path.append([ycorre,y])
               
               Unmatched_augment_path=[]
               Matched_augment_path=[]
               Unmatched_augment_path.append([ycorre,y])
               terminate = ycorre
               #augmenting
               #first we trace the augmenting path backward
               while terminate != free:#when the initial point is free vertex, stop tracing
                  #if np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0].size:
                     Matched_augment_path.append( Matched_path[np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0][0]])
                     Unmatched_augment_path.append( Unmatched_path[np.where(np.array(Unmatched_path)[:,1]== Matched_augment_path[-1][1])[0][0]])
                     terminate= Unmatched_augment_path[-1][0]
                   
               for e in  Unmatched_augment_path:
                   Matched[e[0],e[1]]=1
               for e in  Matched_augment_path:
                   Matched[e[0],e[1]]=0    
                                  
               
               #back to step 2
               if Matched.sum() == Size2 and all(Matched.sum(axis=1) == (Size1*np.ones(Size1))):
                 return Matched
               free = np.where(Matched.sum(axis=1)<Size1)[0][0] 
               S = [free]
               #print('new')
               #print('free')
               T = []
               if np.where(Matched[free]==1)[0].size:
                   T.extend(list(np.where(Matched[free]==1)[0]))
               Unmatched_path = []
               Matched_path = []
               mark = 0 ##tell the alg go back to step 2
            
            if (mark == 1) & (np.where(Matched[:,y])[0].size):#psedo-matched
               ycorre = list(set(np.where(Cost_temp[:,y] <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size1))-set(S)) )[0]
               
               #print('suc')
               
               #extending the alternating tree
               Unmatched_path.append([ycorre,y])
               z = np.where(Matched[:,y])[0][0]
               Matched_path.append([z,y])
               
               T.append(y)
               S.append(np.where(Matched[:,y])[0][0])
               
               #print(T)
               #print(len(T))
               
               if np.where(Matched[z]==1)[0].size:
                   T.extend(list(np.where(Matched[z]==1)[0]))
               
               #back to step3


def track_Hungarian(Cost):
    ##generate initial labelling  
    oper = 0
    Size = np.shape(Cost)[0]
        
    
    x_label = Cost.max(axis = 1)[...,np.newaxis] #add dimension to faciliate calculating
    y_label = np.zeros(Size)[np.newaxis,]
    

    
    ##generate initial matching
    Cost_temp = x_label - Cost
    oper += Size*Size
    
    Matched = np.zeros((Size, Size))
    col_uncovered = np.ones(Size)
    row_uncovered = np.ones(Size)
    for i, j in zip(*np.where(Cost_temp == 0)):
        if col_uncovered[j] and row_uncovered[i]:
            Matched[i, j] = 1  #star Z
            col_uncovered[j] = False
            row_uncovered[i] = False
            
    if Matched.sum() == Size:
        return Matched,oper
    free = np.where(~Matched.any(axis=1))[0][0]  #initializing; pick one free vertex 
    S = [free]
    T = []    
    Unmatched_path = []
    Matched_path = []
    
    Slack = (x_label[free] + y_label - Cost[free,:])[0] 
    oper += Size

    S_neighbor = list(np.where(Slack == 0)[0])#initialize S_neighbor
    
   
    while Matched.sum() != Size:
        #step 2: pick free vertex
        
        mark = 1 #ancillary variable, will be explained later
        ##compute the neighbor of S
        # S_neighbor = []
        # for s in S:
        #     s_neighbor = list(np.where(Cost_temp[s] == 0)[0])
        #     S_neighbor.extend(s_neighbor) 
            
            
        #step 3: update labelling 
        if set(S_neighbor)==set(T):
          # print("step3")
           #Tcomp = list(set(range(Size))-set(T)) #get y\not in T
           #Cost_temp = x_label + y_label - Cost
           #alpha = Cost_temp[S,:][:,Tcomp].min()
           alpha =  Slack[np.where(~np.isnan(Slack))[0]].min()
           oper += len(Slack[np.where(~np.isnan(Slack))[0]])
           
           s_neighbor = list(np.where(Slack==alpha)[0])
           S_neighbor.extend(s_neighbor) 

           x_label[S] = x_label[S] - alpha
           oper += len(set(S))
           
           y_label[0,T] = y_label[0,T] + alpha
           oper += len(set(T))
           
           #update reduced cost function
           
           #Cost_temp = x_label + y_label - Cost
           #compute neighbor
           Slack[np.where(~np.isnan(Slack))[0]] = Slack[np.where(~np.isnan(Slack))[0]]-alpha
           oper += len(Slack[np.where(~np.isnan(Slack))[0]])
           
           
           
        
        #step 4
        if set(S_neighbor)!=set(T):
         #   print("step4")
            y= list(set(S_neighbor)-set(T))[0]
            tempcost = x_label + y_label[0,y] - Cost[:,y].reshape(-1,1)
            oper += Size
            
            
            if not np.where(Matched[:,y])[0].size:#free
               
               ycorre = list(set(np.where(tempcost <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size))-set(S)) )[0]#ycorre should in S
               #ycorre = list(set(S) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size))-set(S)) )[0]#ycorre should in S

               
               Unmatched_path.append([ycorre,y])
               
               Unmatched_augment_path=[]
               Matched_augment_path=[]
               Unmatched_augment_path.append([ycorre,y])
               terminate = ycorre
               #augmenting
               #first we trace the augmenting path backward
               while terminate != free:#when the initial point is free vertex, stop tracing
                  #if np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0].size:
                     Matched_augment_path.append( Matched_path[np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0][0]])
                     Unmatched_augment_path.append( Unmatched_path[np.where(np.array(Unmatched_path)[:,1]== Matched_augment_path[-1][1])[0][0]])
                     terminate= Unmatched_augment_path[-1][0]
                   
               for e in  Unmatched_augment_path:
                   Matched[e[0],e[1]]=1
               for e in  Matched_augment_path:
                   Matched[e[0],e[1]]=0    
               #   flag = np.where(Matched[ycorre,:]==1)[0][0]
               #   index2 = np.where(np.array(Unmatched_path)[:,1]==flag)[0]
               #   if index2.size:
               #     for i in range(index2[0]+1):
               #       e = Unmatched_path[i]
               #       Matched[e[0],e[1]]=1
               # Matched[ycorre,y]=1      
                     

               # index1 = np.where(np.array(Matched_path)[:,0]==ycorre)[0]
               # if index1.size:
               #    for i in range(index1[0]+1):
               #       e = Matched_path[i]
               #       Matched[e[0],e[1]]=0
                   
                   
               
               
               #back to step 2
               if Matched.sum() == Size:
                 return Matched,oper
               free = np.where(~Matched.any(axis=1))[0][0]  
               S = [free]
               T = []
               
               
               Unmatched_path = []
               Matched_path = []

               Slack = (x_label[free] + y_label - Cost[free,:])[0] 
               oper += Size

               S_neighbor = list(np.where(Slack == 0)[0])#initialize S_neighbor               
               
               
               
               mark = 0 ##tell the alg go back to step 2
            
            if (mark == 1) & (np.where(Matched[:,y])[0].size):#matched
               ycorre = list( set(np.where(tempcost <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size))-set(S)) )[0]
               
               #print('suc')
               
               #extending the alternating tree
               Unmatched_path.append([ycorre,y])
               z = np.where(Matched[:,y])[0][0]
               Matched_path.append([z,y])
               
               T.append(y)
               
               Slack[y]= None
               
               s = np.where(Matched[:,y])[0][0]              
               S.append(s)
                
               temp = (x_label[s] + y_label - Cost[s,:])[0]
               oper += Size
               
               s_neighbor = list(np.where(temp == 0)[0])
               S_neighbor.extend(s_neighbor) 

               
                              
               temp2 = np.vstack((temp[np.where(~np.isnan(Slack))[0]],Slack[np.where(~np.isnan(Slack))[0]])).min(axis=0)           
               oper += len(temp[np.where(~np.isnan(Slack))[0]])
               
               Slack[np.where(~np.isnan(Slack))[0]]=temp2
              
     
def track_revised_Hungarian(Cost):
    ##generate initial labelling
    oper = 0
    Size1 = np.shape(Cost)[0]
    Size2 = np.shape(Cost)[1]
    x_label = np.zeros(Size1)[...,np.newaxis]
    y_label = Cost.max(axis = 0)[np.newaxis,] #add dimension to faciliate calculating

    
        
    
    ##generate initial psedo-matching
    Cost_temp = y_label - Cost
    oper += Size1*Size2
    
    Matched = np.zeros((Size1, Size2))
    col_uncovered = np.zeros(Size2)
    row_uncovered = np.zeros(Size1)
    for i, j in zip(*np.where(Cost_temp == 0)):
        if col_uncovered[j]==0 and row_uncovered[i] < Size1:
            Matched[i, j] = 1  #star Z
            col_uncovered[j] += 1
            row_uncovered[i] += 1
            
    if Matched.sum() == Size2 and all(Matched.sum(axis=1) == (Size1*np.ones(Size1))):#judge if it is a perfect psedo-matching
        return Matched, oper
    free = np.where(Matched.sum(axis=1)<Size1)[0][0]  #initializing; pick one free vertex 
    S = [free]
    #print('free')
    T = []
    
    Slack = (x_label[free] + y_label - Cost[free,:])[0] 
    oper += Size2
    
    S_neighbor = list(np.where(Slack == 0)[0])#initialize S_neighbor

    if np.where(Matched[free]==1)[0].size:
       t = list(np.where(Matched[free]==1)[0])
       T.extend(t)
       Slack[t] = None
    Unmatched_path = []
    Matched_path = []
    

    while Matched.sum() != Size2 and any(Matched.sum(axis=1) != (Size1*np.ones(Size1))):
        #step 2: pick free vertex
        
        mark = 1 #ancillary variable, will be explained later
        ##compute the neighbor of S
        #S_neighbor = []
        #for s in S:
        #   s_neighbor = list(np.where(Cost_temp[s] == 0)[0])
        #   S_neighbor.extend(s_neighbor) 
        #step 3: update labelling 
        if not len(list(set(S_neighbor)-set(T))):
         #  Tcomp = list(set(range(Size2))-set(T)) #get y\not in T
         #  Cost_temp = x_label + y_label - Cost
         #  alpha = Cost_temp[S,:][:,Tcomp].min()
         #print('update')
         alpha =  Slack[np.where(~np.isnan(Slack))[0]].min()
         oper += len(Slack[np.where(~np.isnan(Slack))[0]])
         
         s_neighbor = list(np.where(Slack==alpha)[0])
         S_neighbor.extend(s_neighbor) 

         x_label[S] = x_label[S] - alpha
         oper += len(set(S))
         
         y_label[0,T] = y_label[0,T] + alpha
         oper += len(set(T))
           
           #update reduced cost function
           
           #Cost_temp = x_label + y_label - Cost
           #compute neighbor
         Slack[np.where(~np.isnan(Slack))[0]] = Slack[np.where(~np.isnan(Slack))[0]]-alpha
         oper += len(Slack[np.where(~np.isnan(Slack))[0]])
         
        #step 4
        if len(list(set(S_neighbor)-set(T))):
            y= list(set(S_neighbor)-set(T))[0]
            tempcost = x_label + y_label[0,y] - Cost[:,y].reshape(-1,1)
            oper += Size1
            
            if not np.where(Matched[:,y])[0].size:#free
               ycorre = list(set(np.where(tempcost <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size1))-set(S)) )[0]#ycorre should in S
               
               Unmatched_path.append([ycorre,y])
               
               Unmatched_augment_path=[]
               Matched_augment_path=[]
               Unmatched_augment_path.append([ycorre,y])
               terminate = ycorre
               #augmenting
               #first we trace the augmenting path backward
               while terminate != free:#when the initial point is free vertex, stop tracing
                  #if np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0].size:
                     Matched_augment_path.append( Matched_path[np.where(np.array(Matched_path)[:,0]== Unmatched_augment_path[-1][0])[0][0]])
                     Unmatched_augment_path.append( Unmatched_path[np.where(np.array(Unmatched_path)[:,1]== Matched_augment_path[-1][1])[0][0]])
                     terminate= Unmatched_augment_path[-1][0]
                   
               for e in  Unmatched_augment_path:
                   Matched[e[0],e[1]]=1
               for e in  Matched_augment_path:
                   Matched[e[0],e[1]]=0    
                                  
               
               #back to step 2
               if Matched.sum() == Size2 and all(Matched.sum(axis=1) == (Size1*np.ones(Size1))):
                 return Matched, oper
               free = np.where(Matched.sum(axis=1)<Size1)[0][0] 
               S = [free]
               #print('free')
               Slack = (x_label[free] + y_label - Cost[free,:])[0] 
               oper += Size2
    
               S_neighbor = list(np.where(Slack == 0)[0])#initialize S_neighbor

               T = []
               
               if np.where(Matched[free]==1)[0].size:
                   t=list(np.where(Matched[free]==1)[0])
                   T.extend(t)
                   Slack[t] = None

                   
               Unmatched_path = []
               Matched_path = []
               mark = 0 ##tell the alg go back to step 2
               
               
                          
            if (mark == 1) & (np.where(Matched[:,y])[0].size):#psedo-matched
               ycorre = list(set(np.where(tempcost <= 1e-8)[0]) -set(np.where(Matched[:,y] == 1)[0]) - (set(range(Size1))-set(S)) )[0]
               
              
               
               #extending the alternating tree
               Unmatched_path.append([ycorre,y])
               z = np.where(Matched[:,y])[0][0]
               Matched_path.append([z,y])
               
               T.append(y)
               Slack[y]= None
               
               s = np.where(Matched[:,y])[0][0]
               S.append(np.where(Matched[:,y])[0][0])
               #print('S')
               tempcost = x_label[s] + y_label - Cost[s,:]
               oper += Size2
               
               s_neighbor = list(np.where(tempcost[0] == 0)[0])
               S_neighbor.extend(s_neighbor)
               
               if np.where(Matched[z]==1)[0].size:
                   tt=list(np.where(Matched[z]==1)[0])
                   T.extend(tt)
                   Slack[tt]=None
                   
                   
                          
 

               

               temp2 = np.vstack((tempcost[0][np.where(~np.isnan(Slack))[0]],Slack[np.where(~np.isnan(Slack))[0]])).min(axis=0)        
              
               #print(len(tempcost[0][np.where(~np.isnan(Slack))[0]]))
               oper += len(tempcost[0][np.where(~np.isnan(Slack))[0]])
               
               Slack[np.where(~np.isnan(Slack))[0]]=temp2
              
    
               
               #back to step3
               
               
#%%Sinkhorn
import numpy as np
import statistics 
from scipy.special import softmax
import scipy.special

def sinkhorn_log(a, b, M, reg, acc):
    dim_1 = len(a)
    dim_2 = b.shape[0]
    


    oper = 0

    Mr = - M / reg
    oper += dim_1 * dim_2

    u = np.zeros(dim_1)
    v = np.zeros(dim_2)

    def get_logT(u, v):

        return Mr + u[:, None] + v[None, :]

    loga = np.log(a)
    logb = np.log(b)

    error = 1
    while True:

        v = logb - scipy.special.logsumexp(Mr + u[:, None], 0)
        oper += dim_1 + dim_1 * dim_2 * 3

        u = loga - scipy.special.logsumexp(Mr + v[None, :], 1)
        oper += dim_2 + dim_1 * dim_2 * 3

        G = np.exp(get_logT(u, v))
        oper += 3 * dim_1 * dim_2




        error = abs(np.sum(G, axis=1) - a).sum() + abs(np.sum(G, axis=0) - b).sum()
        if error < acc:
               return G, oper




