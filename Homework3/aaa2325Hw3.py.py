
# coding: utf-8

# In[86]:


import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from __future__ import division
from numba import autojit
plt.rcParams["figure.figsize"] = (10,8)


# In[87]:


xtr = np.genfromtxt('hw3-data/gaussian_process/X_train.csv', delimiter = ',')
ytr = np.genfromtxt('hw3-data/gaussian_process/y_train.csv')
xt = np.genfromtxt('hw3-data/gaussian_process/x_test.csv', delimiter  =',')
yt = np.genfromtxt('hw3-data/gaussian_process/y_test.csv')


# In[88]:


class Gaussian_process:
    def __init__(self, x_train, y_train, x_test, y_test):       
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test        


# In[89]:


import scipy as sp
from scipy.spatial.distance import cdist
def kernelize(x_1, x_2, b):
    p_sq_dists = cdist(x_1, x_2, 'sqeuclidean')
    K = np.exp((-1/b)*p_sq_dists)   
    return K   


# In[90]:


def gaussian_predict(K_test, K_train, y,  v):
    varianceMatrix = np.dot(v, np.identity(K_train.shape[0]))
    K_train_dot_y = np.dot(np.linalg.inv(K_train + varianceMatrix), y)
    mean = np.dot(K_test, K_train_dot_y)
    return mean    


# In[91]:


import pandas as pd
b_matrix = np.array([5,7,9,11,13,15])
v_matrix = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7 , 0.8, 0.9, 1.0])
rsmeMatrix = pd.DataFrame(index = b_matrix, columns = v_matrix) 
print(rsmeMatrix)


# In[92]:



def rsmeCalc(y_actual, y_pred):
    return np.sqrt(((y_pred - y_actual) ** 2).mean())
    


# In[93]:


gp  = Gaussian_process(xtr,ytr,xt,yt)

for b in b_matrix:
    for v in v_matrix:
        k_train = kernelize(gp.x_train, gp.x_train, b )
        k_test = kernelize(gp.x_test, gp.x_train, b)
        pred = gaussian_predict(k_test, k_train, gp.y_train, v)
        rsmeVal = rsmeCalc(gp.y_test, pred)
        rsmeMatrix.loc[b,v] = rsmeVal
print(rsmeMatrix)
rsmeMatrix.to_html('table.html')


# In[94]:


x_train_4 = gp.x_train[:,3].reshape(-1,1)
gp_dim_4 = Gaussian_process(x_train_4, gp.y_train, x_train_4, gp.y_train)
K_dim_4_train = kernelize(gp_dim_4.x_train, gp_dim_4.x_train, 5)
K_dim_4_test = kernelize(gp_dim_4.x_test, gp_dim_4.x_train, 5)
pred_dim_4  = gaussian_predict(K_dim_4_test, K_dim_4_train, gp_dim_4.y_train, 2)
print(pred_dim_4.shape)
print(gp_dim_4.y_test.shape)


# In[95]:


import seaborn as sb

plt.figure()
x_axis = x_train_4.flatten()
x_sort = np.argsort(x_axis)
sb.regplot(x = x_axis, y = gp_dim_4.y_test, fit_reg = False, color = "g")
plt.plot(x_axis[x_sort], pred_dim_4[x_sort], 'Black')
plt.xlabel("x[4] training data")
plt.ylabel("y training data ")
plt.title("Visualizing kernel performace through one dimension")
plt.savefig("kernelPlot.png")


plt.show()


# In[96]:


# Boosting Part
xtrB = np.genfromtxt('hw3-data/boosting/X_train.csv', delimiter = ',')
ytrB = np.genfromtxt('hw3-data/boosting/y_train.csv')
xtB = np.genfromtxt('hw3-data/boosting/x_test.csv', delimiter  =',')
ytB = np.genfromtxt('hw3-data/boosting/y_test.csv')


# In[97]:


# adding a column of ones
xtrB = np.column_stack((np.ones(xtrB.shape[0]),xtrB))
xtB = np.column_stack((np.ones(xtB.shape[0]),xtB))


# In[98]:


class BoostData:
    
    def __init__(self, x_train,y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


# In[99]:


def linearFit(x_train, y_train):
    return np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), x_train.T), y_train)
    


# In[100]:


def linearPred(x_train, w):
    return np.sign(np.dot(x_train, w))


# In[101]:


def bootstrapSample(n, w):
    return np.random.choice(n,n, p = w)


# In[102]:


def adaBoost(x_train, y_train, x_test, y_test, n, t_iter):
    w = np.ones(n)/n
    e_array = np.zeros(t_iter)
    a_array = np.zeros(t_iter)
    hist = np.zeros(n)
    ls_w_arr = []
    train_sum = np.zeros(n)
    test_sum = np.zeros(x_test.shape[0])
    for t in range (t_iter):
        b_index = bootstrapSample(n,w)
        
        for b in b_index:
            hist[b] += 1

        
        Bx_train = x_train[b_index]
        By_train = y_train[b_index]
        
        ls_w = linearFit(Bx_train, By_train)
        ls_pred = linearPred(x_train, ls_w)
              
        error = np.sum((y_train != ls_pred) * w)
        
        if error > 0.5:
            ls_w = -ls_w
            ls_pred = linearPred(x_train, ls_w)
            error = np.sum((y_train != ls_pred) * w)
        
        
        e_array[t] = error
        
        alpha = 0.5 *  np.log((1-error)/error)
        
        a_array[t] = alpha
        
   
        ls_w_arr.append(ls_w)
        
        w = w * np.exp(-alpha * y_train * ls_pred)
        w = w/(np.sum(w))
        
    
    return (a_array, e_array, ls_w_arr, hist)


# In[103]:


bd = BoostData(xtrB, ytrB, xtB, ytB)
alphas, epsilons, ls_w_arr, hist = adaBoost(bd.x_train, bd.y_train, bd.x_test, bd.y_test, bd.x_train.shape[0], 1500)


# In[104]:


def train_test_errors(x_train, x_test, y_train, y_test, alphas, ls_w_arr):
    train_err_arr = np.zeros(1500)
    test_err_arr = np.zeros(1500)
    print(ls_w_arr[0])
    for t in range(1,1501):
        train_sum = np.zeros(x_train.shape[0])
        test_sum = np.zeros(x_test.shape[0])
        for i in range(t):
            train_sum += alphas[i] * linearPred(x_train,ls_w_arr[i])
            test_sum += alphas[i] * linearPred(x_test, ls_w_arr[i])
        train_err_arr[t-1] = np.sum(y_train != np.sign(train_sum))/y_train.shape[0]
        test_err_arr[t-1] = np.sum(y_test != np.sign(test_sum))/y_test.shape[0]
    return train_err_arr, test_err_arr


# In[105]:


autojit(linearPred)
autojit(train_test_errors)


# In[106]:


train_err_arr, test_err_arr = train_test_errors(bd.x_train, bd.x_test,bd.y_train, bd.y_test, alphas, ls_w_arr)


# In[107]:


plt.figure()
sb.axes_style("darkgrid")
plt.plot(range(1,1501), train_err_arr, 'g', label = "training error")
plt.plot(range(1,1501), test_err_arr, 'r', label  = "testing error")
plt.xlabel("Iteratons t")
plt.ylabel("Error")
plt.title("Training and Testing errors vs Iteration t")
plt.legend()
plt.savefig("trainTest.png")
plt.show()


# In[108]:


upperbound = np.zeros(1500)
for t in range(1,1501):
    sum_ub = 0
    for i in range(t):
        sum_ub += (0.5 - epsilons[i])**2
    upperbound[t-1] = np.exp(sum_ub * -2)


# In[109]:


plt.figure()
sb.axes_style("darkgrid")
plt.plot(range(1,1501), upperbound, 'g')
plt.xlabel("Iterations t")
plt.ylabel("Upperbound Error")
plt.title("Upperbound Error vs Iterations t")

plt.show()


# In[110]:


# plot for epsilons
plt.figure()
sb.axes_style("darkgrid")
plt.plot(range(1,1501), epsilons, 'g')
plt.xlabel("Iterations t")
plt.ylabel("Epsilon")
plt.title("Epsilon vs Iteration t")
plt.savefig("epsilonPlot.png")
plt.show()


# In[111]:


# plot for alphas
plt.figure()
sb.axes_style("darkgrid")
plt.plot(range(1,1501), alphas, 'g')
plt.xlabel("Iterations t")
plt.ylabel("Alpha")
plt.title("Alpha vs Iteration t")
plt.savefig("alphaPlot.png")
plt.show()


# In[112]:


print(0.5 *  np.log((1-epsilons[0])/epsilons[0]))


# In[63]:


plt.figure()
print(bd.x_train.shape[0])
plt.bar(range(0,bd.x_train.shape[0]), hist)
plt.xlabel("Dimensions Selected")
plt.ylabel("Frequency")
plt.title("Frequency of Dimensions Selected Due to Bootstrapping")
plt.savefig("histogram.png")
plt.show()

