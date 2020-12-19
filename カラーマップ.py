#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


C_S = np.array([[1], [0], [0], [0]])
I = np.array([[1, 0], [0, 1]])
X = np.array([[0,1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])

H = 1 / math.sqrt(2) * np.array([[1,1],
                                 [1,-1]])
H_2 = np.kron(H,H)


def func(x1, x2, i):
    x12 = np.pi * x1 * x2
    
    #X軸変換
    U1 = np.array([[np.cos(x1/2), -1j*np.sin(x1/2)],
                   [-1j*np.sin(x1/2), np.cos(x1/2)]])
    U2 = np.array([[np.cos(x2/2), -1j*np.sin(x2/2)],
                   [-1j*np.sin(x2/2), np.cos(x2/2)]])
    
    #Y軸変換
    #U1 = np.array([[np.cos(x1/2), -np.sin(x1/2)],
    #               [np.sin(x1/2), np.cos(x1/2)]])
    #U2 = np.array([[np.cos(x2/2), -np.sin(x2/2)],
    #               [np.sin(x2/2), np.cos(x2/2)]])
    
    #Z軸変換
    #U1 = np.array([[1, 0],
    #               [0, np.cos(x1)+1j*np.sin(x1)]])
    #U2 = np.array([[1, 0],
    #               [0, np.cos(x2)+1j*np.sin(x2)]])
    
    
    U_2 = np.kron(U1, U2)
    
    #Z軸
    #U12 = np.kron(np.array([[1,0], [0,1]]), np.array([[1,0],[0,np.cos(x12) + 1j*np.sin(x12)]]))
    #X軸
    #U12 = np.kron(np.array([[1,0], [0,1]]), np.array([[np.cos(x12/2), -1j*np.sin(x12/2)], [-1j*np.sin(x12/2), np.cos(x12/2)]]))
    #Y軸
    U12 = np.kron(np.array([[1,0], [0,1]]), np.array([[np.cos(x12/2), -np.sin(x12/2)], [np.sin(x12/2), np.cos(x12/2)]]))
    
    
    F_1 = np.matmul(CNOT, np.matmul(U12, np.matmul(CNOT, np.matmul(U_2, H_2))))
    F_2 = np.matmul(F_1, F_1)
    C = np.matmul(F_2, C_S)
    C_H = np.conjugate(C.T)
    C_F = np.matmul(C, C_H)
    
    
    A = np.trace(np.matmul(C_F, np.kron(i[0], i[1])))/4
    
    return A


# In[6]:


import itertools
lis = [I, X, Y, Z]
pair = []
pair = list([[I, I], [I, X], [I, Y], [I, Z],
          [X, I], [X, X], [X, Y], [X, Z],
          [Y, I], [Y, X], [Y, Y], [Y, Z],
          [Z, I], [Z, X], [Z, Y], [Z, Z]])


# In[7]:


pair[0][0]


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')



np.random.seed(2018)
fig = plt.figure()
for idx, p in enumerate(pair):
    
    a = np.linspace(-1, 1, 50)
    b = np.linspace(-1, 1, 50)
    A, B = np.meshgrid(a, b)
    
    C = []
    for i in range(50):
        for j in range(50):
            c = func(A[i][j], B[i][j], p)
            C.append(c)
    

    C = np.array(C).reshape(50, 50)
    
    
    
    
    
    ax1 = fig.add_subplot(4, 4, idx+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    ax1.contourf(A, B, C, 20, cmap="jet")
    


plt.show()


# In[219]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')


C_all = 0
np.random.seed(2018)

for idx, p in enumerate(pair):
    
    a = np.linspace(-1, 1, 50)
    b = np.linspace(-1, 1, 50)
    A, B = np.meshgrid(a, b)
    C = []
    for i in range(50):
        for j in range(50):
            c = func(A[i][j], B[i][j], p)
            C.append(c)
    
    C = np.array(C).reshape(50, 50)
    
    C_all = C_all + C
    
    
    
C_all = C_all / 16
    

    
plt.contour(A, B, C_all, 20, cmap="jet")
    


plt.show()


# In[ ]:




