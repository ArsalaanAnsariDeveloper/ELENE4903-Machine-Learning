
# coding: utf-8

# In[242]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import eigs


# 1. Markov Chain

# In[243]:



# Load team names
teamNames = np.loadtxt("TeamNames.txt" , dtype='str')
print(teamNames[0])


# In[244]:


# Initialize M matrix with zeros
M = np.zeros((763,763))
print(M)


# In[245]:


# Initialize random Walk matrix 

gameData = np.genfromtxt('CFB2017_scores.csv', delimiter=",")


# In[246]:


# Update M based on the data from file

for row in gameData:
    teamAin = int(row[0])
    aPts = int(row[1])
    teamBin = int(row[2])
    bPts = int(row[3])
    
    i = teamAin - 1
    j = teamBin - 1
    
    weight = aPts/(aPts + bPts)
    win = int(aPts > bPts)
    
    M[i,i] += win + weight
    M[j,j] += (1 - win) + (1 - weight)
    M[j,i] += win + weight
    M[i,j] +=  (1 - win) + (1 - weight)
    

    
M =  M / (np.sum(M, axis = 1).reshape(-1,1))


# In[247]:


print(np.sum(M[2]))


# In[248]:


# eigenval calcs

w_inf= eigs(M.T,1)[1].flatten()
w_inf = w_inf/(np.sum(w_inf))


# In[249]:


# update w vector
w = np.repeat(1/763, 763)
tRank = set([10, 100, 1000, 10000])
tCol = []
w_inf_t = np.zeros(10000)

for t in range(1,10001):
    w = np.dot(w,M)
    w_inf_t[t-1] = np.sum(abs(w - w_inf)) 
    if(t in tRank):
        
        if(t in tRank):
            wordsNum = np.argsort(w)[::-1][:25]
            print(len(wordsNum))
            tCol.append(teamNames[wordsNum])
        


# In[250]:


print(w_inf_t)


# In[251]:


# Create ranking data frame 
tArray = tCol[0]
print(tCol[1])

for t in range(1, len(tCol)):
    tArray = np.vstack((tArray, tCol[t]))
tArray = tArray.T



# In[252]:


# print rankings data frame
rankingsDf = pd.DataFrame(tArray, columns=["t = 10", "t = 100", " t = 1000", "t = 10000"])
rankingsDf.index = np.arange(1, 26)
rankingsDf.to_html('rankingsTable.html')
print(rankingsDf)


# In[253]:


import matplotlib as mpl
mpl.style.use('seaborn')
plt.figure()
plt.plot(range(1,10001), w_inf_t)
plt.xlabel("Iterations t")
plt.ylabel("|w_t - w_inf|")
plt.title("|w_t - w_inf| vs t")
plt.savefig('differential.png')


# Problem 2 NMF

# In[351]:


#Initialize X

X = np.zeros((3012, 8447))


# In[352]:



raw_nyt_data = np.loadtxt('nyt_data.txt', dtype='str', delimiter="\n")



# In[353]:


for i in range(0, len(raw_nyt_data)):
    temp = raw_nyt_data[i].split(',')
    for w in temp:
        temp2  = w.split(':')
        index = int(temp2[0])
        count = int(temp2[1])
        X[index-1, i] = count


# In[354]:


print(X)


# In[355]:


#Loading words 
vocab = np.loadtxt("nyt_vocab.dat" , dtype='str')



# In[356]:


print(vocab)


# In[357]:


# Initializing W and H
W = np.random.uniform(1,2,(3012,25))
H = np.random.uniform(1,2,(25,8447))



# In[358]:


# training model

objective = np.zeros(100)
matWH = np.dot(W, H)
A = np.divide(X, (matWH) + 1e-16)


# In[359]:


for i in range(0,100):
    H = np.multiply(H, np.dot(W.T, A))/(np.sum(W, axis = 0)).reshape(25,1)
    matWH = np.dot(W, H)
    A = np.divide(X, (matWH) + 1e-16)
    W = (W * np.dot(A, H.T))/(np.sum(H, axis = 1)).reshape(1,25)
    matWH = np.dot(W, H)
    A = np.divide(X, (matWH) + 1e-16)
    objVal = np.sum((np.log(1/(matWH + 1e-16)) * X ) + matWH)
    objective[i] = objVal


# In[360]:


# Print plot of objective function
mpl.style.use('seaborn')
plt.plot(range(1,101), objective)
plt.xlabel("Iterations t")
plt.ylabel("Objective Value")
plt.title("Objective Value Over Iterations t")
plt.savefig("figure2a.png")


# In[361]:


print(W)


# In[362]:


# normalize and get words
Wnorm = W /(np.sum(W, axis=0).reshape(1,-1))
print(np.sum(Wnorm[:,0]))


# In[363]:


windex = np.zeros((10,25),dtype=int)
for i in range(0,25):
    windex[:,i] = np.argsort(Wnorm[:,i][-10:][::-1])

data = pd.DataFrame(index=range(1,11), columns=['Topic %d'%i for i in range(1,26)])
for i in range(25):
    for x in range(0,10):
        data.iloc[:,i][x] = {vocab[windex[:,i]][x]: format(Wnorm[windex[:,i],i][x], '.4f')}
            
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
display(results)


results.to_html('figure2b.html',index=False)

