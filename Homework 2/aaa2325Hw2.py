
# coding: utf-8

# In[3]:


import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[118]:


# Read in data for y_train and calculate the prior

y_train = np.genfromtxt('y_train.csv')
yPi = np.mean(y_train)
yN1 = 0
indicesY1 = []
indicesY0 = []

for i in range(0, len(y_train)):
    if y_train[i] == 1:
        yN1 += 1
        indicesY1.append(i)
    else:
        indicesY0.append(i)
        
yN0 = len(y_train) - yN1


# In[119]:


# Separating the Bernoulli and pareto data
x_train = np.genfromtxt('x_train.csv', delimiter=',')
bernX = x_train[:, 0:54]
parX = x_train[:, 54:57]
bernX_1 = bernX[indicesY1]
bernX_0 = bernX[indicesY0]
parX_1 = parX[indicesY1]
parX_0 = parX[indicesY0]

print(np.shape(bernX_0))


# In[120]:


# Calculating the bernoulli paramters
bernTheta1 = np.zeros(54)
bernTheta0 = np.zeros(54)

for c in range(0, len(bernX_1.T)):
    bernTheta1[c] = np.mean(bernX_1.T[c])

for c in range(0, len(bernX_0.T)):
    bernTheta0[c] = np.mean(bernX_0.T[c])

print(bernTheta1)
print(bernTheta0)



# In[124]:


print(bernTheta0[51])


# In[7]:


# Calculate the pareto parameters 
parTheta1 = np.zeros(3)
parTheta0 = np.zeros(3)

for c in range(0, len(parX_1.T)):
    parTheta1[c] = yN1/(np.sum(np.log(parX_1.T[c])))

for c in range(0, len(parX_0.T)):
    parTheta0[c] = yN0/(np.sum(np.log(parX_0.T[c])))
    
print(parTheta1)
print(parTheta0)


# In[106]:


# Load data from testing set
y_test = np.genfromtxt('y_test.csv')
x_test = np.genfromtxt('X_test.csv', delimiter=',')



# In[9]:


# Takes the data point, the thetas for the bernoulli and calculates the class conditional probability
def classConditional(data, bernTheta, parTheta):
    bernData = data[0:54]
    parData = data[54:57]
    bernProd = 1
    parProd = 1
    for i in range(0, len(bernData)):
        bernProd *= np.power(bernTheta[i], bernData[i]) * np.power((1-bernTheta[i]), (1-bernData[i]))
    for i in range(0, len(parData)):
        parProd *= parTheta[i]* np.power(parData[i], -(parTheta[i] + 1))
    
    return bernProd * parProd
        


# In[10]:


def predictY(testData,bernTheta0, bernTheta1, parTheta0, parTheta1, yPi):
    predictions = []
    zero = 0
    one = 0
    for i in range (0, len(testData)):
        zero = (1- yPi) * classConditional(testData[i], bernTheta0, parTheta0)
        one = (yPi) * classConditional(testData[i], bernTheta1, parTheta1)
        if(one > zero):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[11]:


predictions = predictY(x_test, bernTheta0, bernTheta1, parTheta0, parTheta1, yPi)
print(predictions)


# In[96]:


def confusion_matrix(predictions, actual):
    correctOnes = 0
    correctZeros = 0
    incorrectOnes = 0
    incorrectZeros = 0
    for i in range (0, len(predictions)):
        if(predictions[i] == actual[i]):
            if(predictions[i] == 1):
                correctOnes += 1
            else:
                correctZeros += 1
        else:
            if(predictions[i] == 1):
                incorrectOnes += 1
            else:
                incorrectZeros += 1
        
    return (correctZeros, incorrectOnes, incorrectZeros, correctOnes, (correctZeros + correctOnes)/93 * 100)  


# In[13]:


print(confusion_matrix(predictions, y_test))


# In[14]:


fig = plt.figure(figsize=(15, 4))


markerline, stemlines, baseline =plt.stem(np.arange(1,55), bernTheta0)
markerline2, stemlines2, baseline2 = plt.stem(np.arange(1,55), bernTheta1)

plt.setp(stemlines2, color = 'BLACK', linewidth=1, linestyle='-')
plt.setp(stemlines, color = 'BLACK', linewidth=1, linestyle='-')

plt.setp (markerline, color = 'r')

plt.setp (markerline2, color = 'g')
plt.xlabel("Parameter Number")
plt.ylabel("Bernoulli Parameter Value")

red_patch = mpatches.Patch(color='red', label=' Class 0')
green_patch = mpatches.Patch(color='green', label=' Class 1')
plt.legend(handles=[red_patch, green_patch])


plt.show()


# In[15]:


# implementing KNN prediction algorithm and how ties are decided (randomly)

import random

def voteChoose(selections):
    numberOnes = selections.count(1)
    numberZeros = selections.count(0)
    
    if(numberOnes == numberZeros):
        return float(random.randint(0,1))
    elif(numberOnes > numberZeros):
        return 1.0
    else:
        return 0.0

        


def KNNpredict(k,x_train, y_train, x_testVal):
    distanceArray = []
    selections = []
    
    for i in range (0, len(x_train)):
        absdist = np.sum(np.absolute(x_testVal - x_train[i, :]))
        distanceArray.append((absdist, i))
    
    distanceArray.sort()
    
    for d in range(0,k):
        i = distanceArray[d][1]
        selections.append(y_train[i])
        
    return voteChoose(selections)

         
        
        
    


# In[16]:


knnpredictions = np.zeros((20, len(x_test)) )

for k in range(1, 21):
    for i in range(0, len(x_test)):
        knnpredictions[k-1][i] = KNNpredict(k, x_train, y_train, x_test[i, :])
        
print(knnpredictions)


# In[17]:


# Calculating the prediction accuracies using the confusion matrix format used before
knnpredacc = []
for i in range (0, 20):
    knnpredacc.append(confusion_matrix(knnpredictions[i], y_test)[4])

print(knnpredacc)


    


# In[18]:


from matplotlib.ticker import MaxNLocator

newfig = plt.figure(figsize=(14, 4)).gca()

kxaxis = np.arange(1,21)
print(kxaxis)
plt.plot(kxaxis, knnpredacc)
plt.xlabel("K values")
plt.ylabel("Prediction Accuracy (%)")
newfig.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()


# In[107]:


# Change all 0's to -1's and extend with ones

y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
print(x_train)
print(y_test)


# In[8]:


import scipy as sp

def sigmoid1(yi, xi, w):
    denominator = 1.0 + np.exp(-(yi* np.dot(xi, w)))
    return 1/denominator
    

def sigmoid2(yi, xi, w):
    
    result = sp.special.expit(yi * np.dot(xi, w))
    return result


# In[9]:


def gradient(x_train, y_train, w):
    result = 0.0
    for i in range(0, len(x_train)):
        result += (1 - sigmoid2(y_train[i], x_train[i], w)) *y_train[i] * x_train[i]
    return result

def gradVec(x_train, y_train, w):
    
    v1 = 1 - sp.special.expit(np.multiply(y_train, np.dot(x_train, w)))
    v2 = np.multiply(y_train, v1)
    v3 = x_train * v2[:, np.newaxis] 
    result = np.sum(v3, axis = 0)
    
    return result
    


# In[10]:


winit = np.zeros(58)
wtest = gradVec(x_train, y_train, winit)


# In[11]:


print(wtest)
print(gradient(x_train, y_train, winit))


# In[12]:


def etaCalc(t):
    return (1.0/((10**5) * np.sqrt(t + 1)))
    
def gradientDescent(x_train, y_train, t_iter):
    wArray = np.zeros((t_iter, len(x_train[1, :])))
    
    w = np.zeros(len(x_train[1, :]))

    for t in range(1, t_iter + 1):
        
        w = w + etaCalc(t) * gradVec(x_train, y_train, w)
        wArray[t-1] = w
    return wArray
        


# In[13]:


wArray = gradientDescent(x_train, y_train, 10000)

print(np.shape(wArray))
print(wArray[-1])


# In[14]:


winit = wArray[-1]
wtest = gradVec(x_train, y_train, winit)
wtest2 = gradient(x_train, y_train, winit)
print(wtest)
print(wtest2)


# In[15]:


def objectiveFast(x_train, y_train, w):
    
    v1 = np.dot(x_train, w) * y_train[:, np.newaxis]
    v2 = np.sum(np.log(sp.special.expit(v1)))
    
    
    return v2
    



def objective(x_train, y_train, w):
    result = 0.0
    
    for i in range (0, len(x_train)):
        v = y_train[i]* np.dot(x_train[i], w)
        if (v > 709):
            v = 709
        result += (v - np.log(1 + np.exp(v)))
    
    return result

        


# In[16]:


objectiveVals = np.zeros(10000)

for i in range(0, 10000):
    objectiveVals[i] = objective(x_train, y_train, wArray[i])
    


# In[111]:


tArray = np.arange(1, 10001)
fig = plt.figure(figsize=(15,6))

plt.plot(tArray, objectiveVals)
plt.xlabel("Iterations t")
plt.ylabel("Objective training function")

plt.show()





# In[18]:


#calculating eta for newton method 


    


# In[45]:


def hessian(x_train, y_train, w):
    
    v1 = sp.special.expit(np.multiply(y_train, np.dot(x_train, w)))
    v2 = v1*(1 - v1)
    
    result = 0
    
    for i in range(0, len(v2)):
        result += v2[i] * np.outer(x_train[i], x_train[i])
    
    
    return -result


# In[46]:


winit = np.zeros(58)
wtest = hessian(x_train, y_train, winit)
print(wtest)


# In[56]:


def gradientDescentNewt(x_train, y_train, t_iter):
    wArray = np.zeros((t_iter, len(x_train[1, :])))
    
    w = np.zeros(len(x_train[1, :]))

    for t in range(1, t_iter + 1):
        
        w = w - (1/(np.sqrt(t+1))) * np.dot(np.linalg.inv(hessian(x_train, y_train, w)), gradVec(x_train, y_train, w))
        wArray[t-1] = w
    return wArray


# In[64]:


newtonWArray = gradientDescentNewt(x_train, y_train, 100)


# In[65]:


print(newtonWArray[-1])


# In[66]:


hesObjectiveVals = np.zeros(100)

for i in range(0, 100):
    hesObjectiveVals[i] = objective(x_train, y_train, newtonWArray[i])


# In[112]:


tArray = np.arange(1, 101)
fig = plt.figure(figsize=(15,6))

plt.plot(tArray, hesObjectiveVals)

plt.xlabel("Iterations t")
plt.ylabel("Objective training function")



plt.show()


# In[68]:


print(hesObjectiveVals[-1])


# In[71]:


def logisticPred(xi, w):
    if(sp.special.expit(np.dot(xi, w)) >= 0.5):
        return 1
    else:
        return -1
    


# In[108]:


x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))


# In[109]:


logPredictions = np.zeros(len(y_test))

for i in range(0, len(y_test)):
    logPredictions[i] = logisticPred(x_test[i], newtonWArray[-1])
    
print(logPredictions)


# In[110]:


print(confusion_matrix(logPredictions, y_test)[4])
print(y_test)
                            


# In[115]:


oldLogPredictions = np.zeros(len(y_test))

for i in range(0, len(y_test)):
    oldLogPredictions[i] = logisticPred(x_test[i], wArray[-1])
    
print(oldLogPredictions)


# In[116]:


print(confusion_matrix(oldLogPredictions, y_test)[4])

