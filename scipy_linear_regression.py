# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
#introduce the data matrix and the result vector
#subject to lb <= x <= ub
A=norm[['s2_norm_mu','s3_norm_mu','s4_norm_mu','s5_norm_mu']].to_numpy()
b=norm[['s1_norm_mu']].to_numpy()
lb=0.0
ub=1.0
res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)