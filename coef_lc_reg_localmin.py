"""
Using multiple linear regression with constraint on coef and intercept,
the combination with minimal residuals is found. That combination is treated
as the starting point for a local minimizer of chi2. 
"""
import time
start_time = time.time()
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from scipy import optimize
#read normalized, merged data
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx')
#introduce the data matrix and the result vector
#subject to lb <= x <= ub
A=norm[['s2_norm_mu','s3_norm_mu','s4_norm_mu','s5_norm_mu','s6_norm_mu']].to_numpy()
b=norm['s1_norm_mu'].to_numpy()
#lb and ub needs modification when more rows are added to norm
lb=np.array([0.0,0.0,0.0,0.0,0.0],np.float64)
ub=np.array([1.0,1.0,1.0,1.0,1.0],np.float64)
res = lsq_linear(A, b, bounds=(lb, ub))
print(res)
known_num=norm.shape[1]-2
def chisq(c):
    """

    Parameters
    ----------
    c : list
        c records the coefficients used in the model. c should have a length 
        equal to known_num

    Returns
    -------
    chisq : float
        model is the predicted behavior of c2*s2+c3*s3+...+cn*sn. s2,s3,...sn are 
        the known samples presented in norm. chisq records the chi-square 
        between model and norm['s1_orm_mu'], which is the unknown sample.

    """
    model=np.zeros(norm.shape[0])
    for i in range(known_num):
        name="s"+str(i+2)+"_norm_mu"
        model=model+c[i]*norm[name].to_numpy()
    actual=norm['s1_norm_mu'].to_numpy()
    chisq=np.sum(((actual.astype(np.float64)-model)**2)/(model))
    return chisq

# use scipy.optimize.minimize to optimize the list c, starting from the best result 
# gained at linear regression
c0=res.x
# bounds needs change when more known structures are added to norm.
bounds=[(0,1),(0,1),(0,1),(0,1),(0,1)]
#constraint that the sum of elements in c is 1
def con(c):
    """
    

    Parameters
    ----------
    c : list
        c records the coefficients used in the model. c should have a length 
        equal to known_num

    Returns
    -------
    c_sum-1: float64
             the difference the sum of all elements in c and 1

    """
    c_sum=0
    for i in range (len(c)):
        c_sum=c_sum+c[i]
    return (c_sum-1)
cons = {'type':'eq', 'fun': con}
res=optimize.minimize(chisq,c0,bounds=bounds,constraints=cons)
print(res)

#record the best result
c=res.x
model=np.zeros(norm.shape[0])
for i in range(known_num):
        name="s"+str(i+2)+"_norm_mu"
        model=model+c[i]*norm[name].to_numpy()
#plot the figure to compare and print the chisq using scipy.stats to verify
plt.plot(norm["Energy"],norm["s1_norm_mu"],label="actual")
plt.plot(norm["Energy"],model,label="predicted")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
