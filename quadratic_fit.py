# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx')
actual=norm['s1_norm_mu'].to_numpy()
quad=np.polyfit(norm['Energy'][100:206],actual[100:206],2)
eq=np.poly1d(quad)
print(eq)
norm_mu=actual/eq(norm.Energy)
plt.plot(norm.Energy,actual)
plt.plot(norm.Energy,eq(norm.Energy))
plt.plot(norm.Energy,norm_mu)