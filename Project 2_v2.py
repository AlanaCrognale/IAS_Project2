#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

#initialize parameters
N = 10
M = 50


# In[4]:


#import training data
import pandas as pd

wave1 = np.loadtxt('ECE5242Proj2-train/wave01.txt')
wave2 = np.loadtxt('ECE5242Proj2-train/wave02.txt')
wave3 = np.loadtxt('ECE5242Proj2-train/wave03.txt')
wave4 = np.loadtxt('ECE5242Proj2-train/wave05.txt')
wave5 = np.loadtxt('ECE5242Proj2-train/wave07.txt')
wave = np.concatenate((wave1,wave2,wave3,wave4,wave5))

inf1 = np.loadtxt('ECE5242Proj2-train/inf11.txt')
inf2 = np.loadtxt('ECE5242Proj2-train/inf13.txt')
inf3 = np.loadtxt('ECE5242Proj2-train/inf16.txt')
inf4 = np.loadtxt('ECE5242Proj2-train/inf18.txt')
inf5 = np.loadtxt('ECE5242Proj2-train/inf112.txt')
inf = np.concatenate((inf1,inf2,inf3,inf4,inf5))

eight1 = np.loadtxt('ECE5242Proj2-train/eight01.txt')
eight2 = np.loadtxt('ECE5242Proj2-train/eight02.txt')
eight3 = np.loadtxt('ECE5242Proj2-train/eight04.txt')
eight4 = np.loadtxt('ECE5242Proj2-train/eight07.txt')
eight5 = np.loadtxt('ECE5242Proj2-train/eight08.txt')
eight = np.concatenate((eight1,eight2,eight3,eight4,eight5))

circle1 = np.loadtxt('ECE5242Proj2-train/circle12.txt')
circle2 = np.loadtxt('ECE5242Proj2-train/circle13.txt')
circle3 = np.loadtxt('ECE5242Proj2-train/circle14.txt')
circle4 = np.loadtxt('ECE5242Proj2-train/circle17.txt')
circle5 = np.loadtxt('ECE5242Proj2-train/circle18.txt')
circle = np.concatenate((circle1,circle2,circle3,circle4,circle5))

beat31 = np.loadtxt('ECE5242Proj2-train/beat3_01.txt')
beat32 = np.loadtxt('ECE5242Proj2-train/beat3_02.txt')
beat33 = np.loadtxt('ECE5242Proj2-train/beat3_03.txt')
beat34 = np.loadtxt('ECE5242Proj2-train/beat3_06.txt')
beat35 = np.loadtxt('ECE5242Proj2-train/beat3_08.txt')
beat3 = np.concatenate((beat31,beat32,beat33,beat34,beat35))

beat41 = np.loadtxt('ECE5242Proj2-train/beat4_01.txt')
beat42 = np.loadtxt('ECE5242Proj2-train/beat4_03.txt')
beat43 = np.loadtxt('ECE5242Proj2-train/beat4_05.txt')
beat44 = np.loadtxt('ECE5242Proj2-train/beat4_08.txt')
beat45 = np.loadtxt('ECE5242Proj2-train/beat4_09.txt')
beat4 = np.concatenate((beat41,beat42,beat43,beat44,beat45))

training = np.concatenate((wave,inf,eight,circle,beat3,beat4))
df = pd.DataFrame(training[:,1:]) #ignore time column


# In[56]:


#k-means - used for further algorithm reference https://mubaris.com/posts/kmeans-clustering/
import math
import copy
k = 50 #50 clusters

#initialize random centroids
C_0 = np.random.randint(math.ceil(min(df[0])), math.floor(max(df[0])), size=k)
C_1 = np.random.randint(math.ceil(min(df[1])), math.floor(max(df[1])), size=k)
C_2 = np.random.randint(math.ceil(min(df[2])), math.floor(max(df[2])), size=k)
C_3 = np.random.randint(math.ceil(min(df[3])), math.floor(max(df[3])), size=k)
C_4 = np.random.randint(math.ceil(min(df[4])), math.floor(max(df[4])), size=k)
C_5 = np.random.randint(math.ceil(min(df[5])), math.floor(max(df[5])), size=k)

C = np.array(list(zip(C_0,C_1,C_2,C_3,C_4,C_5)), dtype = np.float32)
C_temp = np.zeros(C.shape)
C_labels = np.zeros(len(df))

#euclidean distance between random centroids and origin centroid(zeros)
error = np.linalg.norm(C - C_temp, axis=None)

#loop until converges i.e. error equals 0 or max iterations reached
max_iter = 100
i = 0
while error !=0 and i < max_iter:
    
    #assign each datapoint to closest cluster
    for j in range(len(df)):
        datapoint = np.array(df.loc[j])
        distances = np.linalg.norm(datapoint-C,axis=1)
        cluster = np.argmin(distances)
        C_labels[j] = cluster

    #update temp centroid values
    C_temp = copy.deepcopy(C)
    
    #average of each cluster to find new centroids, update
    for x in range(k):
        points = []
        for z in range(len(df)):
            if C_labels[z] == x:
                points.append(np.array(df.loc[z]))
        if len(points) == 0:
            C[x] = np.array(df.loc[x]) #If empty cluster, replace with random data point
        else:
            C[x] = np.mean(points,axis=0)
        
    #check if cluster assignments changed
    error = np.linalg.norm(C-C_temp,axis=None)  
    i = i + 1
    #print(error)
    #print(i)


# In[79]:


A_wave = np.random.rand(N,N)#number of states x number of states (10x10)
A_wave = A_wave/A_wave.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1


B_wave = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_wave = B_wave/B_wave.sum(axis=1,keepdims=1)

id_wave = np.random.rand(1,N) #1 x number of states (1x10)
id_wave = id_wave/id_wave.sum(axis=1,keepdims=1)

#train wave hmm model - used for further reference 'A Revealing Introduction to Hidden Markov Models' by Mark Stamp, October 17, 2018
T = wave.shape[0]
wave_labels = C_labels[0:T] #wave observation sequence
wave_labels = wave_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_wave[i,wave_labels[0]] == 0:
        B_wave[i,wave_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_wave[0,i]*B_wave[i,wave_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
    
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_wave[j,i])
        a[i,t] = a[i,t]*B_wave[i,wave_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_wave[i,j]*B_wave[j,wave_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_wave[i,j]*B_wave[j,wave_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_wave[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_wave[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (wave_labels[t] == j):
                n = n + g[i,t]
        B_wave[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

print(it)
print(max_it)
print(lp)
print(old_lp)
while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_wave[i,wave_labels[0]] == 0:
            B_wave[i,wave_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_wave[0,i]*B_wave[i,wave_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_wave[j,i])
            a[i,t] = a[i,t]*B_wave[i,wave_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_wave[i,j]*B_wave[j,wave_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_wave[i,j]*B_wave[j,wave_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_wave[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_wave[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (wave_labels[t] == j):
                    n = n + g[i,t]
            B_wave[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1
    
    


# In[80]:


A_inf = np.random.rand(N,N)#number of states x number of states (10x10)
A_inf = A_inf/A_inf.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1

B_inf = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_inf = B_inf/B_inf.sum(axis=1,keepdims=1)

id_inf = np.random.rand(1,N) #1 x number of states (1x10)
id_inf = id_inf/id_inf.sum(axis=1,keepdims=1)

#train inf hmm model
index = wave.shape[0] #index from previous model labels (wave)
T = inf.shape[0]
inf_labels = C_labels[index+1:index+1+T] #inf observation sequence 
inf_labels = inf_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_inf[i,inf_labels[0]] == 0:
        B_inf[i,inf_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_inf[0,i]*B_inf[i,inf_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_inf[j,i])
        a[i,t] = a[i,t]*B_inf[i,inf_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_inf[i,j]*B_inf[j,inf_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_inf[i,j]*B_inf[j,inf_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_inf[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_inf[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (inf_labels[t] == j):
                n = n + g[i,t]
        B_inf[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_inf[i,inf_labels[0]] == 0:
            B_inf[i,inf_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_inf[0,i]*B_inf[i,inf_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_inf[j,i])
            a[i,t] = a[i,t]*B_inf[i,inf_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_inf[i,j]*B_inf[j,inf_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_inf[i,j]*B_inf[j,inf_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_inf[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_inf[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (inf_labels[t] == j):
                    n = n + g[i,t]
            B_inf[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1
    


# In[81]:


A_eight = np.random.rand(N,N)#number of states x number of states (10x10)
A_eight = A_eight/A_eight.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1

B_eight = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_eight = B_eight/B_eight.sum(axis=1,keepdims=1)

id_eight = np.random.rand(1,N) #1 x number of states (1x10)
id_eight = id_eight/id_eight.sum(axis=1,keepdims=1)

#train eight hmm model
index = wave.shape[0]+inf.shape[0] #index from previous model labels (up to inf)
T = eight.shape[0]
eight_labels = C_labels[index+1:index+1+T] #eight observation sequence 
eight_labels = eight_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_eight[i,eight_labels[0]] == 0:
        B_eight[i,eight_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_eight[0,i]*B_eight[i,eight_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_eight[j,i])
        a[i,t] = a[i,t]*B_eight[i,eight_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_eight[i,j]*B_eight[j,eight_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_eight[i,j]*B_eight[j,eight_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_eight[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_eight[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (eight_labels[t] == j):
                n = n + g[i,t]
        B_eight[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_eight[i,eight_labels[0]] == 0:
            B_eight[i,eight_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_eight[0,i]*B_eight[i,eight_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_eight[j,i])
            a[i,t] = a[i,t]*B_eight[i,eight_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_eight[i,j]*B_eight[j,eight_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_eight[i,j]*B_eight[j,eight_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_eight[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_eight[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (eight_labels[t] == j):
                    n = n + g[i,t]
            B_eight[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1
    


# In[82]:


A_circle = np.random.rand(N,N)#number of states x number of states (10x10)
A_circle = A_circle/A_circle.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1

B_circle = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_circle = B_circle/B_circle.sum(axis=1,keepdims=1)

id_circle = np.random.rand(1,N) #1 x number of states (1x10)
id_circle = id_circle/id_circle.sum(axis=1,keepdims=1)


#train circle hmm model
index = wave.shape[0]+inf.shape[0]+eight.shape[0] #index from previous model labels (up to eight)
T = circle.shape[0]
circle_labels = C_labels[index+1:index+1+T] #circle observation sequence 
circle_labels = circle_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_circle[i,circle_labels[0]] == 0:
        B_circle[i,circle_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_circle[0,i]*B_circle[i,circle_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_circle[j,i])
        a[i,t] = a[i,t]*B_circle[i,circle_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_circle[i,j]*B_circle[j,circle_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_circle[i,j]*B_circle[j,circle_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_circle[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_circle[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (circle_labels[t] == j):
                n = n + g[i,t]
        B_circle[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_circle[i,circle_labels[0]] == 0:
            B_circle[i,circle_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_circle[0,i]*B_circle[i,circle_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_circle[j,i])
            a[i,t] = a[i,t]*B_circle[i,circle_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_circle[i,j]*B_circle[j,circle_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_circle[i,j]*B_circle[j,circle_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_circle[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_circle[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (circle_labels[t] == j):
                    n = n + g[i,t]
            B_circle[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1
    


# In[83]:


A_beat3 = np.random.rand(N,N)#number of states x number of states (10x10)
A_beat3 = A_beat3/A_beat3.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1

B_beat3 = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_beat3 = B_beat3/B_beat3.sum(axis=1,keepdims=1)

id_beat3 = np.random.rand(1,N) #1 x number of states (1x10)
id_beat3 = id_beat3/id_beat3.sum(axis=1,keepdims=1)


#train beat3 hmm model
index = wave.shape[0]+inf.shape[0]+eight.shape[0]+circle.shape[0] #index from previous model labels (up to circle)
T = beat3.shape[0]
beat3_labels = C_labels[index+1:index+1+T] #beat3 observation sequence 
beat3_labels = beat3_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_beat3[i,beat3_labels[0]] == 0:
        B_beat3[i,beat3_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_beat3[0,i]*B_beat3[i,beat3_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_beat3[j,i])
        a[i,t] = a[i,t]*B_beat3[i,beat3_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_beat3[i,j]*B_beat3[j,beat3_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_beat3[i,j]*B_beat3[j,beat3_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_beat3[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_beat3[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (beat3_labels[t] == j):
                n = n + g[i,t]
        B_beat3[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_beat3[i,beat3_labels[0]] == 0:
            B_beat3[i,beat3_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_beat3[0,i]*B_beat3[i,beat3_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_beat3[j,i])
            a[i,t] = a[i,t]*B_beat3[i,beat3_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_beat3[i,j]*B_beat3[j,beat3_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_beat3[i,j]*B_beat3[j,beat3_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_beat3[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_beat3[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (beat3_labels[t] == j):
                    n = n + g[i,t]
            B_beat3[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1


# In[84]:


A_beat4 = np.random.rand(N,N)#number of states x number of states (10x10)
A_beat4 = A_beat4/A_beat4.sum(axis=1,keepdims=1) #normalize s.t. each row adds to 1

B_beat4 = np.random.rand(N,M) #number of states x number of observation symbols (10x50)
B_beat4 = B_beat4/B_beat4.sum(axis=1,keepdims=1)

id_beat4 = np.random.rand(1,N) #1 x number of states (1x10)
id_beat4 = id_beat4/id_beat4.sum(axis=1,keepdims=1)


#train beat4 hmm model
index = wave.shape[0]+inf.shape[0]+eight.shape[0]+circle.shape[0]+beat3.shape[0] #index from previous model labels (up to beat3)
T = beat4.shape[0]-1
beat4_labels = C_labels[index+1:] #beat4 observation sequence 
beat4_labels = beat4_labels.astype(int)

c = np.zeros((1,T)) 
a = np.zeros((N,T))
b = np.zeros((N,T))
g = np.zeros((N,T))
dg = np.zeros((T,N,N))
max_it = 50
it = 0
old_lp = -10000000000

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_beat4[i,beat4_labels[0]] == 0:
        B_beat4[i,beat4_labels[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_beat4[0,i]*B_beat4[i,beat4_labels[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_beat4[j,i])
        a[i,t] = a[i,t]*B_beat4[i,beat4_labels[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]

#compute beta
for i in range(0,N):
    b[i,T-1] = c[0,T-1]
    
for t in range(T-2,-1,-1):
    for i in range(0,N):
        b[i,t] = 0
        for j in range(0,N):
            b[i,t] = b[i,t] + (A_beat4[i,j]*B_beat4[j,beat4_labels[t+1]]*b[j,t+1])
        #scale beta
        b[i,t] = b[i,t]*c[0,t]

#compute gamma
for t in range(0,T-1):
    for i in range(0,N):
        g[i,t] = 0
        for j in range(0,N):
            dg[t,i,j] = a[i,t]*A_beat4[i,j]*B_beat4[j,beat4_labels[t+1]]*b[j,t+1]
            g[i,t] = g[i,t] + dg[t,i,j]
            
for i in range(0,N):
    g[i,T-1] = a[i,T-1]

#re-estimate pi,A,B parameters
for i in range(0,N):
    id_beat4[0,i] = g[0,i]

for i in range(0,N):
    d = 0
    for t in range(0,T-1):
        d = d + g[i,t]
    for j in range(0,N):
        n = 0
        for t in range(0,T-1):
            n = n + dg[t,i,j]
        A_beat4[i,j] = n/d
        
for i in range(0,N):
    d = 0
    for t in range(0,T):
        d = d + g[i,t]
    for j in range(0,M):
        n = 0
        for t in range(0,T):
            if (beat4_labels[t] == j):
                n = n + g[i,t]
        B_beat4[i,j] = n/d

#compute log prob + iterate if necessary
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

it = it + 1

while (it < max_it and lp > old_lp):
    print("it: ")
    print(it)
    print("lp: ")
    print(lp)
    old_lp = lp
    
    #below copy/pasted from above from first iteration
    #compute alpha_0
    c[0,0] = 0
    for i in range(0,N):
        if B_beat4[i,beat4_labels[0]] == 0:
            B_beat4[i,beat4_labels[0]] = 10**-10 #to account for division by 0 issues
        a[i,0] = id_beat4[0,i]*B_beat4[i,beat4_labels[0]]
        c[0,0] = c[0,0] + a[i,0]

    #scale alpha_0
    c[0,0] = 1/c[0,0]
    for i in range(0,N):
        a[i,0] = a[i,0]*c[0,0]

    #compute alpha_t
    for t in range(1,T):
        c[0,t] = 0
        for i in range(0,N):
            a[i,t] = 0
            for j in range(0,N):
                a[i,t] = a[i,t] + (a[j,t-1]*A_beat4[j,i])
            a[i,t] = a[i,t]*B_beat4[i,beat4_labels[t]]
            c[0,t] = c[0,t] + a[i,t]
        #scale alpha_t
        c[0,t] = 1/c[0,t]
        for i in range(0,N):
            a[i,t] = a[i,t]*c[0,t]

    #compute beta
    for i in range(0,N):
        b[i,T-1] = c[0,T-1]

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            b[i,t] = 0
            for j in range(0,N):
                b[i,t] = b[i,t] + (A_beat4[i,j]*B_beat4[j,beat4_labels[t+1]]*b[j,t+1])
            #scale beta
            b[i,t] = b[i,t]*c[0,t]

    #compute gamma
    for t in range(0,T-1):
        for i in range(0,N):
            g[i,t] = 0
            for j in range(0,N):
                dg[t,i,j] = a[i,t]*A_beat4[i,j]*B_beat4[j,beat4_labels[t+1]]*b[j,t+1]
                g[i,t] = g[i,t] + dg[t,i,j]

    for i in range(0,N):
        g[i,T-1] = a[i,T-1]

    #re-estimate pi,A,B parameters
    for i in range(0,N):
        id_beat4[0,i] = g[0,i]

    for i in range(0,N):
        d = 0
        for t in range(0,T-1):
            d = d + g[i,t]
        for j in range(0,N):
            n = 0
            for t in range(0,T-1):
                n = n + dg[t,i,j]
            A_beat4[i,j] = n/d

    for i in range(0,N):
        d = 0
        for t in range(0,T):
            d = d + g[i,t]
        for j in range(0,M):
            n = 0
            for t in range(0,T):
                if (beat4_labels[t] == j):
                    n = n + g[i,t]
            B_beat4[i,j] = n/d

    #compute log prob + iterate if necessary
    lp = 0
    for i in range(0,T):
        lp = lp + math.log(c[0,i],10)
    lp = -1*lp

    it = it + 1


# In[85]:


#save off model parameters in case kernel restarts so you can quickly load back in without having to retrain all
np.save('A_wave',A_wave)
np.save('B_wave',B_wave)
np.save('id_wave',id_wave)

np.save('A_inf',A_inf)
np.save('B_inf',B_inf)
np.save('id_inf',id_inf)

np.save('A_eight',A_eight)
np.save('B_eight',B_eight)
np.save('id_eight',id_eight)

np.save('A_circle',A_circle)
np.save('B_circle',B_circle)
np.save('id_circle',id_circle)

np.save('A_beat3',A_beat3)
np.save('B_beat3',B_beat3)
np.save('id_beat3',id_beat3)

np.save('A_beat4',A_beat4)
np.save('B_beat4',B_beat4)
np.save('id_beat4',id_beat4)


# In[113]:


#test

test = np.loadtxt('ECE5242Proj2-test/test8.txt') #change directory to whichever .txt file you are testing
df_test = pd.DataFrame(test[:,1:]) #ignore time column

#assign each datapoint to closest cluster - equivalent to sklearn.kmeans.predict
test_obs = np.zeros(len(df_test))

for i in range(len(df_test)):
    datapoint = np.array(df_test.loc[i])
    distances = np.linalg.norm(datapoint-C,axis=1)
    cluster = np.argmin(distances)
    test_obs[i] = cluster
    
T = test_obs.shape[0]
test_obs = test_obs.astype(int)


# In[114]:


#compute log likelihoods for each hmm model 

#wave
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_wave[i,test_obs[0]] == 0:
        B_wave[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_wave[0,i]*B_wave[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_wave[j,i])
        if B_wave[i,test_obs[t]] == 0:
            B_wave[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_wave[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("wave log probability: ")
print(lp)



#inf
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_inf[i,test_obs[0]] == 0:
        B_inf[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_inf[0,i]*B_inf[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_inf[j,i])
        if B_inf[i,test_obs[t]] == 0:
            B_inf[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_inf[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("inf log probability: ")
print(lp)



#eight
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_eight[i,test_obs[0]] == 0:
        B_eight[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_eight[0,i]*B_eight[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_eight[j,i])
        if B_eight[i,test_obs[t]] == 0:
            B_eight[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_eight[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("eight log probability: ")
print(lp)


#circle
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_circle[i,test_obs[0]] == 0:
        B_circle[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_circle[0,i]*B_circle[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_circle[j,i])
        if B_circle[i,test_obs[t]] == 0:
            B_circle[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_circle[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("circle log probability: ")
print(lp)



#beat3
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_beat3[i,test_obs[0]] == 0:
        B_beat3[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_beat3[0,i]*B_beat3[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_beat3[j,i])
        if B_beat3[i,test_obs[t]] == 0:
            B_beat3[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_beat3[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]
    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("beat3 log probability: ")
print(lp)



#beat4
a = np.zeros((N,T))
c = np.zeros((1,T)) 

#compute alpha_0
c[0,0] = 0
for i in range(0,N):
    if B_beat4[i,test_obs[0]] == 0:
        B_beat4[i,test_obs[0]] = 10**-10 #to account for division by 0 issues
    a[i,0] = id_beat4[0,i]*B_beat4[i,test_obs[0]]
    c[0,0] = c[0,0] + a[i,0]
    
#scale alpha_0
c[0,0] = 1/c[0,0]
for i in range(0,N):
    a[i,0] = a[i,0]*c[0,0]

#compute alpha_t
for t in range(1,T):
    c[0,t] = 0
    for i in range(0,N):
        a[i,t] = 0
        for j in range(0,N):
            a[i,t] = a[i,t] + (a[j,t-1]*A_beat4[j,i])
        if B_beat4[i,test_obs[t]] == 0:
            B_beat4[i,test_obs[t]] = 10**-10 #to account for division by 0 issues
        a[i,t] = a[i,t]*B_beat4[i,test_obs[t]]
        c[0,t] = c[0,t] + a[i,t]

    #scale alpha_t
    c[0,t] = 1/c[0,t]
    for i in range(0,N):
        a[i,t] = a[i,t]*c[0,t]
lp = 0
for i in range(0,T):
    lp = lp + math.log(c[0,i],10)
lp = -1*lp

print("beat4 log probability: ")
print(lp)


# In[ ]:




