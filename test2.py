import numpy as np
import pgmpy
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch
import pandas as pd
import scipy.io
from pgmpy.estimators import ExpertKnowledge


## Load the .mat file
## Harsh 
mat = scipy.io.loadmat("C:/Users/harsh/Desktop/BAYESIAN NETWORK/DATASET 1/BCICIV_calib_ds1a.mat",struct_as_record=False, squeeze_me=True)
## SP
#mat = scipy.io.loadmat("C:\\Users\\sapta\\Documents\\GitHub\\HarshBCI\\assets\\BCIC_IV_ds1\\Hz100\\BCICIV_calib_ds1a", struct_as_record = False, squeeze_me = True)

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

## Define a function to create tensor for the two class of motor imagery        
def tform(ndf,t):
    i=0
    j=0
    k=0
    while i < ndf.shape[0]:
            t[k][j]=ndf.iloc[i:(i+1),0:59]
            if j>=399:
                j= 0
                if k<99:
                  k=k+1
            else:
                j=j+1
            i=i+1
    return t

###.........................for left hand mi...........................

## extract the eeg data where motor imagery is left(2)
ndf2=df[df["MI"]==2]

## remove mi and time column for constuction of tensor
ndf2 = ndf2.drop(["MI","Time"], axis=1)

##.....................for foot mi.................

## extract the eeg data where motor imagery is foot(4)
ndf4=df[df["MI"]==4]

## remove mi and time column for constuction of tensor
ndf4 = ndf4.drop(["MI","Time"], axis=1)

###.........................for foot mi...........................

## extract the eeg data where motor imagery is rest(1)
ndf1=df[df["MI"]==1]

## Create the required tensor to store data of motor imagery
t2=np.empty ((100, 400 ,59))
t4=np.empty ((100, 400 ,59))

## Create required tensor for left(2) motor imagery
t2=tform(ndf2, t2)

## Create required tensor for foot (4) motor imagery
t4=tform(ndf4, t4)

##Function that extracts time interval
def time_interval(t, t1,t2):
    return t[: , [t1,t2] , :]

##Function that create the bayesian network each time when called
def rec_bn(te,t1,t2):
    data={}
    for i in range(te.shape[2]):
        data[f'{clab[i]}_t{t1}']=te[: ,0 ,i]
        data[f'{clab[i]}_t{t2}']=te[: ,1 ,i]
    d=pd.DataFrame(data,index=None)
    fl=[]        # stores forbidden edges
    for j in range(te.shape[2]):
        for k in range(te.shape[2]):
            fl.append((f'{clab[k]}_t{t2}',f'{clab[j]}_t{t1}'))
            fl.append((f'{clab[j]}_t{t1}', f'{clab[k]}_t{t1}' ))
            fl.append((f'{clab[j]}_t{t2}' ,f'{clab[k]}_t{t2}'))
    # expert knowledge for forbidden edges        
    ek = ExpertKnowledge(forbidden_edges=fl)
    ## apply  hillclimb search on mi data
    hc = HillClimbSearch(data=d,use_cache=True)
    ## apply estimation function to find the best network structure of  mi
    bn = hc.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True ,expert_knowledge=ek)
    return bn

##Function that stores the every time stored network formed and develops the dynamic  bayesian network
def dbn(tensor):
    dbn = []
    for t in range(0, 399):  # (0–399) gives t and t+1
        ste = time_interval(tensor, t, t+1)
        bn = rec_bn(ste,t,t+1)
        dbn.append(bn)
    return dbn

##Extract the edges of the model
def all_edges(dbn_list):
    edges = set()
    for model in dbn_list:
        edges.update(model.edges())
    return edges

# # Function for drawing and saving the network structure
def draw(m , na):
    G = nx.DiGraph(m)
    pos = nx.spring_layout(G,k=1.5)  
    # plt.figure(figsize=(10000, 10000)) 
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.savefig(f'network of 1a {na} .png')
    plt.savefig("networkof 1a {na} .svg")
    plt.show()

## Function that find unique edges in m1 with respect to m2
def unique_edges(m1,m2):
    f=0  
    ue=[]
    for i in range (0,len(m1),1):
        f=0
        ## loop checks  that a edge is present in m1  but not in m2
        for j in range(0,len(m2),1):
            if (list(m1)[i][0]==list(m2)[j][0] or  list(m1)[i][0]==list(m2)[j][1]) and (list(m1)[i][1]==list(m2)[j][0] or  list(m1)[i][1]==list(m2)[j][1]) :
                f+=1
        if  f==0:
            ue.append(m1[i])
    return ue
<<<<<<< HEAD
## Function to create a csv for which the edges can be stored
def make_csv(ed,na):
    # Convert set of edges to a DataFrame
    le_df = pd.DataFrame(list(ed), columns=["From", "To"])
    # Save to CSV
    le_df.to_csv(f'{na}_edges.csv', index=False)
=======
>>>>>>> f2cd2f1f9e6426cde6496dadaf61025749b2e1ff

## Find the edges of dynamic bayesian network for left mi
le=all_edges(dbn(t2))
## Find the edges of dynamic bayesian network for foot mi
ft=all_edges(dbn(t4))
## Find the unique edges of dynamic bayesian network for left mi w.r.t foot mi model
uel=unique_edges(list(le),list(ft))
## Find the unique edges of dynamic bayesian network for foot mi w.r.t left mi model
uef=unique_edges(list(ft),list(le))
<<<<<<< HEAD

## Function that check unique path 
def unique_path(e1,e2,lp):
    cp=set()
    i=0 
    j=0 
    while i<len(e1):
         j=0
         f=0
         while j<len(e2):
            if i+lp<len(e1) and j+lp<len(e2):
               if e1[i:i+lp]==e2[j:j+lp]:
                  cp.add(tuple(e2[j:j+lp]))
                  f=f+1
            j=j+1
         if f>0:
             i=i+lp
         else:
              i=i+1
    return ( set(e1).difference(cp))
=======
>>>>>>> f2cd2f1f9e6426cde6496dadaf61025749b2e1ff
