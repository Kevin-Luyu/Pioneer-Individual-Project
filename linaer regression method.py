# -*- coding: utf-8 -*-
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
s=norm[['s2_norm_mu','s3_norm_mu','s4_norm_mu','s5_norm_mu']]
s1=norm[['s1_norm_mu']]
reg = LinearRegression().fit(s, s1)
print(reg.coef_)
print(reg.intercept_)
