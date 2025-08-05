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


## remove mi and time column for constuction of tensor
ndf1 = ndf1.drop(["MI","Time"], axis=1)

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

## Function to create a csv for which the edges can be stored
def make_csv(ed,na):
    # Convert set of edges to a DataFrame
    le_df = pd.DataFrame(list(ed), columns=["From", "To"])
    # Save to CSV
    le_df.to_csv(f'{na}_edges.csv', index=False)

## Find the edges of dynamic bayesian network for left mi
le=all_edges(dbn(t2))
## Find the edges of dynamic bayesian network for foot mi
ft=all_edges(dbn(t4))

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
    return cp

## Function that find unique edges in m1 with respect to m2
def unique_edges(m1,m2):
    f=0  
    ue=[]
    for i in range (0,len(m1),1):
        f=0
        ## loop checks  that a edge is present in m1  but not in m2
        for j in range(0,len(m2),1):
            if ( m1[i][0]==m2[j][0] or  m1[i][0]==m2[j][1]) and ( m1[i][1]==m2[j][0] or  m1[i][1]==m2[j][1]) :
                f+=1
            if f>0:
                break
        if  f==0:
            ue.append(m1[i])
    return ue

## Create dataframe based on left edges
dl=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\github\HarshBCI\left_edges.csv")
## Remove the index column
dl = dl.drop(["Unnamed: 0"], axis=1)

## Create dataframe based on foot edges
df=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\github\HarshBCI\foot edges of 1a.csv")

## Find the unique edges of dynamic bayesian network for left mi w.r.t foot mi model
uel=unique_edges(dl.values,df.values)
## Find the unique edges of dynamic bayesian network for foot mi w.r.t left mi model
uef=unique_edges(df.values,dl.values)

## Create dataframe based on unique foot edges
dfuf=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\github\HarshBCI\unique foot MI edges of 1a's .csv")

## Create dataframe based on unique left  edges
dful=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\github\HarshBCI\unique left MI edges of 1a's .csv")

##convert the unique foot edges dataframe to list
ufe=dfuf.values.tolist()
##convert the unique foot edges dataframe to list
ule=dful.values.tolist()

##method to divide unique edges into bin and find  edges without temporal dependency (b0,b1,b2,b3,ue)
def edge_bin(u):
     ue=set()
     b0=set()
     b1=set()
     b2=set()
     b3=set()
     for i in range(len(u)):
         ue.add((u[i][0].split("_")[0],u[i][1].split("_")[0]))
         if 0<=int(u[i][0].split("_")[1][1:])<100:
             b0.add((u[i][0].split("_")[0],u[i][1].split("_")[0]))
         elif 99<int(u[i][0].split("_")[1][1:])<200:
             b1.add((u[i][0].split("_")[0],u[i][1].split("_")[0]))
         elif 199<int(u[i][0].split("_")[1][1:])<300:
             b2.add((u[i][0].split("_")[0],u[i][1].split("_")[0]))
         elif 299<int(u[i][0].split("_")[1][1:])<400:
             b3.add((u[i][0].split("_")[0],u[i][1].split("_")[0]))
     return b0,b1,b2,b3,ue

##bin - wise unique edges , edges without temporal dependency  and bin - wise temporal edges for left
lb0,lb1,lb2,lb3,ltfe=edge_bin(ule)
##bin - wise unique edges , edges without temporal dependency and and bin - wise temporal edges for foot
fb0,fb1,fb2,fb3,ftfe=edge_bin(ufe)
##completely temporal free unique edges of left wrt foot
cule=ltfe-ftfe
##completely temporal free unique edges of foot wrt left
cufe=ftfe-ltfe

def shd(tg, lg):
    g1 = set(tg)
    g2 = set(lg)
    
    # Directed edge mismatches (insertion + deletion)
    insertions = g2 - g1
    deletions = g1 - g2
    
    # Check for edge reversals
    flips = set()
    for u, v in deletions.copy():  # only check those missing in G2
        if (v, u) in insertions:
            flips.add((u, v))
            insertions.remove((v, u))
            deletions.remove((u, v))

    shd = len(insertions) + len(deletions) + len(flips)
    return shd

 
##...............................FOR TEST DATA...........................................
m = scipy.io.loadmat("C:/Users/harsh/Desktop/BAYESIAN NETWORK/DATASET 1/BCICIV_eval_ds1a.mat",struct_as_record=False, squeeze_me=True)
cnte = m["cnt"].astype(float)
cnte = 0.1*cnte
d= pd.DataFrame(cnte, columns=m["nfo"].clab)
cl=[]
for i in range (0,int(cnte.shape[0]),100):
    be=set()
    if i + 100 < cnte.shape[0]:
         da = pd.DataFrame(cnte[i:i+100], columns=m["nfo"].clab)
         # da["original_index"] = np.arange(i, i+400)
    print(da)
    h = HillClimbSearch(da,use_cache=True)
    ## apply estimation function to find the best network structure of  mi
    b = h.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
    be=set(b.edges())
    cl.append((f"t{i}_to_t{i+100}","left",len(be&lb0),len(be&lb1),len(be&lb2),len(be&lb3),len(be&cule),shd(lb0,be),shd(lb1,be),shd(lb2,be),shd(lb3,be)))
    cl.append((f"t{i}_to_t{i+100}","foot",len(be&fb0),len(be&fb1),len(be&fb2),len(be&fb3),len(be&cufe),shd(fb0,be),shd(fb1,be),shd(fb2,be),shd(fb3,be)))   
sbmec=sorted(cl, key=lambda x: sum(x[2:6]),reverse=True)
sbmshd=sorted(cl, key=lambda x: sum(x[7:11]),reverse=False)







  
    
ad=[]    
## for testing rest
for i in range (0,500,100):
    be=set()
    if i+100<ndf1.shape[0]:
      h = HillClimbSearch(ndf1.iloc[i:i+100],use_cache=True)
    ## apply estimation function to find the best network structure of  mi
    b = h.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
    be=set(b.edges())
    cl.append((f"t{i}_to_t{i+100}","left",len(be&lb0),len(be&lb1),len(be&lb2),len(be&lb3),len(be&cule),shd(lb0,be),shd(lb1,be),shd(lb2,be),shd(lb3,be)))
    ad.append((f"t{i}_to_t{i+100}","left",len(be&lb0)+len(be&lb1)+len(be&lb2)+len(be&lb3),shd(lb0,be)+shd(lb1,be)+shd(lb2,be)+shd(lb3,be),"rest"))
    cl.append((f"t{i}_to_t{i+100}","foot",len(be&fb0),len(be&fb1),len(be&fb2),len(be&fb3),len(be&cufe),shd(fb0,be),shd(fb1,be),shd(fb2,be),shd(fb3,be)))
    ad.append((f"t{i}_to_t{i+100}","foot",len(be&fb0)+len(be&fb1)+len(be&fb2)+len(be&fb3),shd(fb0,be)+shd(fb1,be)+shd(fb2,be)+shd(fb3,be),"rest"))


import networkx as nx
from dodiscover.metrics import structure_hamming_dist
## for testing left
for i in range (200,400,100):
    be=set()
    if i+100<ndf2.shape[0]:
      h = HillClimbSearch(ndf2.iloc[i:i+100],use_cache=True)
    ## apply estimation function to find the best network structure of  mi
    b = h.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
    be=set(b.edges())
    cl.append((f"t{i}_to_t{i+100}","left",len((be-lb0)-(be-fb0)),len((be-lb1)-(be-fb1)),len((be-lb2)-(be-fb2)),len((be-lb3)-(be-fb3)),len(be-cule), structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be))))
    ad.append((f"t{i}_to_t{i+100}","left",len((be-lb0)-(be-fb0))+len((be-lb1)-(be-fb1))+len((be-lb2)-(be-fb2))+len((be-lb3)-(be-fb3)),structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be)),"left"))
    cl.append((f"t{i}_to_t{i+100}","foot",len((be-fb0)-(be-lb0)),len((be-fb1)-(be-lb1)),len((be-fb2)-(be-lb2)),len((be-fb3)-(be-lb3)),len(be-cufe),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be))))
    ad.append((f"t{i}_to_t{i+100}","foot",len((be-fb0)-(be-lb0))+len((be-fb1)-(be-lb1))+len((be-fb2)-(be-lb2))+len((be-fb3)-(be-lb3)),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be)),"left"))
    # cl.append((f"t{i}_to_t{i+100}","left",len(be-lb0),len(be-lb1),len(be-lb2),len(be-lb3),len(be-cule), structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","left",len(be-lb0)+len(be-lb1)+len(be-lb2)+len(be-lb3),structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be)),"left"))
    # cl.append((f"t{i}_to_t{i+100}","foot",len(be-fb0),len(be-fb1),len(be-fb2),len(be-fb3),len(be-cufe),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","foot",len(be-fb0)+len(be-fb1)+len(be-fb2)+len(be-fb3),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be)),"left"))
    # # cl.append((f"t{i}_to_t{i+100}","left",len(be&lb0),len(be&lb1),len(be&lb2),len(be&lb3),len(be&cule),shd(lb0,be),shd(lb1,be),shd(lb2,be),shd(lb3,be)))
    # ad.append((f"t{i}_to_t{i+100}","left",len(be&lb0)+len(be&lb1)+len(be&lb2)+len(be&lb3),shd(lb0,be)+shd(lb1,be)+shd(lb2,be)+shd(lb3,be),"left"))
    # cl.append((f"t{i}_to_t{i+100}","foot",len(be&fb0),len(be&fb1),len(be&fb2),len(be&fb3),len(be&cufe),shd(fb0,be),shd(fb1,be),shd(fb2,be),shd(fb3,be)))
    # ad.append((f"t{i}_to_t{i+100}","foot",len(be&fb0)+len(be&fb1)+len(be&fb2)+len(be&fb3),shd(fb0,be)+shd(fb1,be)+shd(fb2,be)+shd(fb3,be),"left"))
adl = pd.DataFrame(ad, columns=["Time interval ID", "Expected MI","Total Edges Match","Total SHD","True MI"])   
## for testing foot
for i in range (200,400,100):
    be=set()
    if i+100<ndf4.shape[0]:
      h = HillClimbSearch(ndf4.iloc[i:i+100],use_cache=True)
    ## apply estimation function to find the best network structure of  mi
    b = h.estimate(scoring_method='bic-g',max_indegree=3,show_progress=True)
    be=set(b.edges())
    cl.append((f"t{i}_to_t{i+100}","left",len(be-lb0),len(be-lb1),len(be-lb2),len(be-lb3),len(be-cule), structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be))))
    ad.append((f"t{i}_to_t{i+100}","left",len(be-lb0)+len(be-lb1)+len(be-lb2)+len(be-lb3),structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be)),"foot"))
    cl.append((f"t{i}_to_t{i+100}","foot",len(be-fb0),len(be-fb1),len(be-fb2),len(be-fb3),len(be-cufe),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be))))
    ad.append((f"t{i}_to_t{i+100}","foot",len(be-fb0)+len(be-fb1)+len(be-fb2)+len(be-fb3),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be)),"foot"))
    # cl.append((f"t{i}_to_t{i+100}","left",len((be-lb0)-(be-fb0)),len((be-lb1)-(be-fb1)),len((be-lb2)-(be-fb2)),len((be-lb3)-(be-fb3)),len(be-cule), structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","left",len((be-lb0)-(be-fb0))+len((be-lb1)-(be-fb1))+len((be-lb2)-(be-fb2))+len((be-lb3)-(be-fb3)),structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be)),"foot"))
    # cl.append((f"t{i}_to_t{i+100}","foot",len((be-fb0)-(be-lb0)),len((be-fb1)-(be-lb1)),len((be-fb2)-(be-lb2)),len((be-fb3)-(be-lb3)),len(be-cufe),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","foot",len((be-fb0)-(be-lb0))+len((be-fb1)-(be-lb1))+len((be-fb2)-(be-lb2))+len((be-fb3)-(be-lb3)),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be)),"foot"))
    # cl.append((f"t{i}_to_t{i+100}","left",len(be&lb0),len(be&lb1),len(be&lb2),len(be&lb3),len(be&cule), structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","left",len(be&lb0)+len(be&lb1)+len(be&lb2)+len(be&lb3),structure_hamming_dist(nx.DiGraph(lb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(lb3),nx.DiGraph(be)),"foot"))
    # cl.append((f"t{i}_to_t{i+100}","foot",len(be&fb0),len(be&fb1),len(be&fb2),len(be&fb3),len(be&cufe),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be)),structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be))))
    # ad.append((f"t{i}_to_t{i+100}","foot",len(be&fb0)+len(be&fb1)+len(be&fb2)+len(be&fb3),structure_hamming_dist(nx.DiGraph(fb0),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb1),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb2),nx.DiGraph(be))+structure_hamming_dist(nx.DiGraph(fb3),nx.DiGraph(be)),"foot"))

ha=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\detailed Analysis for classifier for left and foot MI.csv")
h=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\Analysis for classifier for left and foot MI.csv")
##machine learning
haa=ha
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Prepare the data
haa = ha.copy()  # Assuming 'ha' is your labeled DataFrame

# Extract features and labels
X = haa.iloc[:, 2:6].copy()
X['unique_edge_match'] = haa.iloc[:, 6]
X['SHD_bin0'] = haa.iloc[:, 7]
X['SHD_bin1'] = haa.iloc[:, 8]
X['SHD_bin2'] = haa.iloc[:, 9]
X['SHD_bin3'] = haa.iloc[:, 10]

y = haa['Expected MI']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # e.g., foot=1, left=0

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 3. Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict on test set
y_pred = clf.predict(X_test)

# 5. Evaluate
print("RandomForest Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. If you want to predict on full data again:
haa['Predicted MI'] = le.inverse_transform(clf.predict(X))

#for test
har=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\BAYESIAN NETWORK\detailed Analysis for classifier for foot of extra 5 samples.csv")
# Step 1: Extract features from 'har' DataFrame
X_har = har.iloc[:, 2:6].copy()  # SHD Left/Foot + Total Edge Match
X_har['unique_edge_match'] = har.iloc[:, 6]
X_har['SHD_bin0'] = har.iloc[:, 7]
X_har['SHD_bin1'] = har.iloc[:, 8]
X_har['SHD_bin2'] = har.iloc[:, 9]
X_har['SHD_bin3'] = har.iloc[:, 10]

# Step 2: Predict using trained model
y_pred_har = clf.predict(X_har)

# Step 3: Decode predicted labels
pred_labels_har = le.inverse_transform(y_pred_har)

# Step 4: Add predictions to the har DataFrame
har['Predicted MI'] = pred_labels_har

# Step 5 (Optional): Evaluate accuracy if true labels exist
if 'Expected MI' in har.columns:
    y_true = le.transform(har['Expected MI'])
    accuracy = (y_true == y_pred_har).mean()
    print(f"Test Accuracy on har: {accuracy:.2%}")
