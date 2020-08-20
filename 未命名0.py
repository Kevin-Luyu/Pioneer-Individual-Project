# -*- coding: utf-8 -*-
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
def chisq(observe, predicted):
    #takes two array of equal length as parameters
    #returns the chi-square value
    result=np.float64(0.0)
    for i in range(len(observe)):
        add=(observe[i]-predicted[i])**2
        add=add/predicted[i]
        result=result+add
    return result
a=np.array([1,2,3,10,100],dtype=np.float64)
b=np.array([1.1,2,3,9,101],dtype=np.float64)
print(chisq(b, a))
print(sp.stats.chisquare(b,a))
# norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
# center2=0.0030864197530864196*1000
# center3=0.125*1000
# center4=0.8305898491083676*1000
# center5=0.12157064471879286*1000
# predicted=center2*norm['s2_norm_mu']+center3*norm['s3_norm_mu']+center4*norm['s4_norm_mu']+center5*norm['s5_norm_mu']
# actual=1000*norm['s1_norm_mu']
# print(sp.stats.chisquare(actual,f_exp=predicted,ddof=318))