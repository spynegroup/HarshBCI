import numpy as np
import pgmpy
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch
import pandas as pd
import scipy.io



## Load the .mat file
## Harsh 
# mat = scipy.io.loadmat("C:/Users/harsh/Desktop/BAYESIAN NETWORK/DATASET 1/BCICIV_calib_ds1a.mat",struct_as_record=False, squeeze_me=True)
## SP
mat = scipy.io.loadmat("C:\\Users\\sapta\\Documents\\GitHub\\HarshBCI\\assets\\BCIC_IV_ds1\\Hz100\\BCICIV_calib_ds1a", struct_as_record = False, squeeze_me = True)

## Extract sampling frequency and store it.
## Sampling frequency = 100 Hz.
sf = mat["nfo"].fs
print(sf)

## Stores the EEG data of shape (190594, 59).
## No. of samples = 1,90,594
## No. of channels = 59
cn = mat["cnt"]
print(mat["cnt"])

## Convert the values into uV i.e. micro volt as advised in:
## https://www.bbci.de/competition/iv/desc_1.html
cnt = cn.astype(float)
cnt = 0.1*cnt

##extract the channel names from the data and store it
##No of channel name = 59
clab = mat["nfo"].clab
print(clab)

##calculate the no of samples taken
print(type(cnt))
ns=cnt.shape[0]
print(ns)

# # Calculate time at which each sample is taken or sampling time
time=np.arange(ns)/sf
print(time)
print(type(time))


## extract the no of samples before each trial to elimate the data between the interleaved period
## Extract the sample IDs of the cues.
po = mat["mrk"].pos

##extract the y position of electrode to classify data according to motor imagery
yl = mat["mrk"].y

## How many class 1 cues are there
print(np.count_nonzero(yl == 1))

## How many class 2 cues are there
print(np.count_nonzero(yl == -1))

##extract the type of motor imagery chosen by the subject
cs = mat["nfo"].classes



## make an numpy array of required column names
## Reruried column name include 59 channels name, time and motor imagery
cl=np.insert(clab,[0,clab.shape[0]],["Time","MI"])
print(cl)



##access field names in mrk to understand its structure
print(mat["mrk"]._fieldnames)



##constructing motor imagery column in data frame to understand which task is performed 
mi=[]
poo=np.append(po, mat["cnt"].shape[0]+1)
i=0
j=0
z=0
while i < mat["cnt"].shape[0]:
    if i>=poo[j] :
        if yl[j] == -1:
              cls = cs[0]
        elif yl[j]==1:
              cls = cs[1]
        
        if cls == 'left':
            for k in range(0,(4*sf),1):
               mi.append(2)
        elif cls == 'right':
            for k in range(0,(4*sf),1):
               mi.append(3)
        elif cls=='foot':
            for k in range(0,(4*sf),1):
               mi.append(4)
        print(poo[j])
        j=j+1
        i=i+int(4*sf)
    else:
            mi.append(1)
            i=i+1
mic=np.array(mi)

#combine time , eeg data and motor imagery data together
dt=np.column_stack((time,cnt,mic))
print(dt)

#creating the data frame 
df=pd.DataFrame(dt,index=None,columns=cl)
print(df)

###.........................for left hand mi...........................

## extract the eeg data where motor imagery is left(2)
ndf2=df[df["MI"]==2]

## remove mi column to prevent it from becoming node in dag
ndf2 = ndf2.drop(["MI"], axis=1)

##apply  hillclimb search on left mi data
hc2 = HillClimbSearch(data=ndf2,use_cache=True)

# #apply estimation function to find the best network structure of left mi
bm2 = hc2.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
print(bm2.edges()) 
bm2e=np.array(bm2.edges())

# #drawing and saving the network structure
# # Create a NetworkX DiGraph from the edges of your model
G = nx.DiGraph(bm2.edges())
pos = nx.spring_layout(G,k=1.5)  
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig("network of calib 1a for left hand.png")
plt.savefig("network of calib 1a for left hand.svg")
plt.show()


##.....................for foot mi.................

## extract the eeg data where motor imagery is foot(4)
ndf4=df[df["MI"]==4]

## remove mi column to prevent it from becoming node in dag as it is constant
ndf4 = ndf4.drop(["MI"], axis=1)

##apply  hillclimb search on foot mi data
hc4 = HillClimbSearch(data=ndf4,use_cache=True)

# #apply estimation function to find the best network structure of foot mi
bm4 = hc4.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
print(bm4.edges()) 
bm4e=np.array(bm4.edges())

# #drawing and saving the network structure
# # Create a NetworkX DiGraph from the edges of your model
G = nx.DiGraph(bm4.edges())
pos = nx.spring_layout(G,k=1.5)  
# plt.figure(figsize=(10000, 10000)) 
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig("network of 1a for foot.png")
plt.savefig("networkof 1a for foot.svg")
plt.show()


##.....................for rest.................

## extract the eeg data where motor imagery is rest(1)
ndf1=df[df["MI"]==1]

## remove mi column to prevent it from becoming node in dag and error during the development of bayesian network
ndf1 = ndf1.drop(["MI"], axis=1)

##apply  hillclimb search on rest data
hc1 = HillClimbSearch(data=ndf1,use_cache=True)

# #apply estimation function to find the best network structure of rest
bm1 = hc1.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
print(bm1.edges()) 
bm1e=np.array(bm1.edges())

# #drawing and saving the network structure
# # Create a NetworkX DiGraph from the edges of your model
G = nx.DiGraph(bm1.edges())
pos = nx.spring_layout(G,k=1.5)  
# plt.figure(figsize=(10000, 10000)) 
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig("network of 1a for rest.png")
plt.savefig("networkof 1a for rest.svg")
plt.show()

##........................ unique edges in left hand imagery........................
## track that a edge is present in bm2e and bm1e but not in bm4e i.e it is present in two network but not in third network
f=0  
## track that a edge is present in bm2e and bm4e but not in bm1e i.e it is present in two network but not in third network
f1=0
ecul2=[]
for i in range (0,bm2e.shape[0],1):
    f=0
    f1=0
    ## loop checks  that a edge is present in bm2e and bm1e but not in bm4e 
    for j in range(0,bm4e.shape[0],1):
        if (bm2e[i][0]==bm4e[j][0] or  bm2e[i][0]==bm4e[j][1]) and (bm2e[i][1]==bm4e[j][0] or  bm2e[i][1]==bm4e[j][1]) :
            f+=1
    ## loop checks that a edge is present in bm2e and bm4e not in bm1e
    for k in range(0,bm1e.shape[0],1):
        if  (bm2e[i][0]==bm1e[k][0] or  bm2e[i][0]==bm1e[k][1]) and (bm2e[i][1]==bm1e[k][0] or  bm2e[i][1]==bm1e[k][1]):
                f1+=1
    ## checks that edge is only present in left mi network
    if f1==0 and f==0:
        ecul2.append(bm2e[i])
print("the edges which are only present or unique edges of left hand mi is : ",ecul2)

##........................ unique edges in rest........................
## track that a edge is present in bm1e and bm2e but not in bm4e i.e it is present in two network but not in third network
f=0  
## track that a edge is present in bm1e and bm4e but not in bm2e i.e it is present in two network but not in third network
f1=0
ecur1=[]
i,j,k=0,0,0
for i in range (0,bm1e.shape[0],1):
    f=0
    f1=0
    ## loop checks  that a edge is present in  bm1e and bm2e but not in bm4e
    for j in range(0,bm4e.shape[0],1):
        if (bm1e[i][0]==bm4e[j][0] or  bm1e[i][0]==bm4e[j][1]) and (bm1e[i][1]==bm4e[j][0] or  bm1e[i][1]==bm4e[j][1]) :
            f+=1
    ## loop checks that a edge is present in bm1e and bm4e but not in bm2e
    for k in range(0,bm2e.shape[0],1):
        if  (bm1e[i][0]==bm2e[k][0] or  bm1e[i][0]==bm2e[k][1]) and (bm1e[i][1]==bm2e[k][0] or  bm1e[i][1]==bm2e[k][1]):
                f1+=1
    ## checks that edge is only present in rest network
    if f1==0 and f==0:
        ecur1.append(bm1e[i])
print("the edges which are only present or unique edges of rest is : ",ecur1)

##........................ unique edges in foot mi ........................
## track that a edge is present in bm4e and bm2e but not in bm1e i.e it is present in two network but not in third network
f=0  
## track that a edge is present in bm4e and bm1e but not in bm2e i.e it is present in two network but not in third network
f1=0
ecuf4=[]
i,j,k=0,0,0
for i in range (0,bm4e.shape[0],1):
    f=0
    f1=0
    ## loop checks  that a edge is present in  bm4e and bm2e but not in bm1e
    for j in range(0,bm1e.shape[0],1):
        if (bm4e[i][0]==bm1e[j][0] or  bm4e[i][0]==bm1e[j][1]) and (bm4e[i][1]==bm1e[j][0] or  bm4e[i][1]==bm1e[j][1]) :
            f+=1
    ## loop checks that a edge is present in bm4e and bm1e but not in bm2e
    for k in range(0,bm2e.shape[0],1):
        if  (bm4e[i][0]==bm2e[k][0] or  bm4e[i][0]==bm2e[k][1]) and (bm4e[i][1]==bm2e[k][0] or  bm4e[i][1]==bm2e[k][1]):
                f1+=1
    ## checks that edge is only present in foot mi network
    if f1==0 and f==0:
        ecuf4.append(bm4e[i])
print("the edges which are only present or unique edges of foot mi is : ",ecuf4)
