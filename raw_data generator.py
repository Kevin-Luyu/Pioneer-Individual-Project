# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:22:59 2020

@author: lenovo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
s1 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sasph008.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
s2 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbs034.xlsx')
s3 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbso042.xlsx')
s4 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbt029.xlsx')
s5 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sfeso4051.xlsx')
s6 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbso2.xlsx')
plt.plot(s1['Energy'],7*s1['PIPS']/s1['I0'],label="Asphaltene")
plt.plot(s2['Energy'],s2['PIPS']/s2['I0']+0.05,label="DBS")
plt.plot(s3['Energy'],2*s3['PIPS']/s3['I0']+0.15,label="DBSO")
plt.plot(s4['Energy'],s4['PIPS']/s4['I0']+0.10,label="DBT")
plt.plot(s5['Energy'],0.5*s5['PIPS']/s5['I0']+0.40,label="FeSO4")
plt.plot(s6['Energy'],0.1*s6['PIPS']/s6['I0']+0.35,label="DBSO2")
plt.legend(loc="upper right")
plt.xlabel('Energy(eV)')
plt.ylabel('Rescaled Absorption Coefficient $\mu$')
plt.show()