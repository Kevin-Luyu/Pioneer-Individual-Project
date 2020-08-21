"""
The program defines a function that calculates chi-square given a possible 
combination. Then scipy.optimize.shgo is used to find the combination that yields
a global min of chi square under range (0.001,1) and constraint the sum of c
is 1. Then a comparing figure is plotted.  
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
#read normalized, merged data
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
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

#use scipy.optimize.shgo to optimize the list c, no initial conditioned needed
#never use (0,1) because that leads to a fail in global minimalization
    
# bounds needs change when more known structures are added to norm.
bounds=[(0.001,1),(0.001,1),(0.001,1),(0.001,1)]
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
res=optimize.shgo(chisq,bounds,constraints=cons)
print(res)

#record the best result of c and model
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

