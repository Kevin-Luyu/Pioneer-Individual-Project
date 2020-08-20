# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
#introduce the data matrix and the result vector
#subject to lb <= x <= ub
A=norm[['s2_norm_mu','s3_norm_mu','s4_norm_mu','s5_norm_mu']].to_numpy()
print(A)
b=norm['s1_norm_mu'].to_numpy()
print(b)
lb=np.array([0.0,0.0,0.0,0.0],np.float64)
ub=np.array([1.0,1.0,1.0,1.0],np.float64)
res = lsq_linear(A, b, bounds=(lb, ub))