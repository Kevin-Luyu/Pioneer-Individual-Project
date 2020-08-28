# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from lmfit import models
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx')
x=norm.Energy
y=norm['s1_norm_mu']
plt.plot(x,y)

center=2486
amplitude=1.0
def atan(x,amplitude,center,length):
    model=amplitude*((1/np.pi)*np.arctan((x-center)/length)+1/2)
    return model
plt.plot(x,y-atan(x,amplitude,center,1))
plt.show()
norm['s1_norm_mu']=y-atan(x,amplitude,center,1)
norm.to_excel(r'C:\Users\lenovo\OneDrive\桌面\atannorm.xlsx', index = False, header=True)