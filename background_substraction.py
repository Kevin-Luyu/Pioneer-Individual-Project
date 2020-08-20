# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import excel files and construct numpy array
s1 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sasph008.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
s2 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbs034.xlsx')
s3 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbso042.xlsx')
s4 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbt029.xlsx')
s5 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sfeso4051.xlsx')
s1_energy=s1.Energy.to_numpy()
s1_I0=s1.I0.to_numpy()
s1_PIPS=s1.PIPS.to_numpy()
s1_mu=s1_PIPS/s1_I0
#construct the slope array defined as d(mu)/d(energy)
#to match the dimension, an additional element was added(value=the second last value)
s1_mu_diff=np.diff(s1_mu)
s1_mu_diff=np.append(s1_mu_diff,s1_mu_diff[-1])
s1_energy_diff=np.diff(s1_energy)
s1_energy_diff=np.append(s1_energy_diff,s1_energy_diff[-1])
s1_mu_slope=s1_mu_diff/s1_energy_diff
s1['mu']=s1_mu
s1['slope of mu']=s1_mu_slope
#test for where the peak arises
#algorithm: if the slope is 10 times larger than the average of all of its past
#then the start of the peak is considered to be 3 values prior to the result
#the pre-tail is taken for 10 data
s1_mu_meanslope=np.zeros(1*len(s1_mu_slope))
for i in range(len(s1_mu_slope)):
    s1_mu_meanslope[i]=np.mean(s1_mu_slope[0:i+1],dtype=np.float64)
for i in range(len(s1_mu_slope)):
    if s1_mu_slope[i]>10*s1_mu_meanslope[i]:
        result=30
        break
end=result
start=end-20
#pretail: 2427-2467
s1_energy_pre=s1_energy[start:end].reshape((-1,1))
s1_mu_pre=s1_mu[start:end]
s1_model = LinearRegression().fit(s1_energy_pre, s1_mu_pre)
s1_mu_back=s1_model.intercept_+s1_model.coef_*s1_energy
s1_mu_subs=s1_mu-s1_mu_back
s1['substracted mu']=s1_mu_subs
s1_musubs_min=np.amin(-1*s1_mu_subs)
i_max=np.where(-s1_musubs_min==s1_mu_subs)[0].item()
s1_energy_min=s1_energy[i_max]
s1_r2=s1_model.score(s1_energy_pre, s1_mu_pre)
print("the r sqaured for regression of s1 is "+str(s1_r2))
print("energy for pre-tails are between" +str(s1_energy[start])+" eV and "+str(s1_energy[end])+" eV")
print("the energy of peak in s1 is "+str(s1_energy_min)+"eV")
print()
# plt.plot(s1_energy,s1_mu,label="raw")
# plt.plot(s1_energy,s1_mu_subs,label="substracted")
# plt.plot(s1_energy_pre,s1_mu_pre)
# plt.plot(s1_energy_pre,s1_mu_back[start:end])
# plt.legend(loc='best')
# plt.show()
# plt.plot(s1_energy_pre,s1_mu_pre)
# plt.plot(s1_energy_pre,s1_mu_back[start:end])
# plt.show()
# print(s1)

#s2
s2_energy=s2.Energy.to_numpy()
s2_I0=s2.I0.to_numpy()
s2_PIPS=s2.Ipips.to_numpy()
s2_mu=s2_PIPS/s2_I0
s2_mu_diff=np.diff(s2_mu)
s2_mu_diff=np.append(s2_mu_diff,s2_mu_diff[-1])
s2_energy_diff=np.diff(s2_energy)
s2_energy_diff=np.append(s2_energy_diff,s2_energy_diff[-1])
s2_mu_slope=s2_mu_diff/s2_energy_diff
s2['mu']=s2_mu
s2['slope of mu']=s2_mu_slope
s2_mu_meanslope=np.zeros(1*len(s2_mu_slope))
for i in range(len(s2_mu_slope)):
    s2_mu_meanslope[i]=np.mean(s2_mu_slope[0:i+1],dtype=np.float64)
for i in range(len(s2_mu_slope)):
    if abs(s2_mu_slope[i])>10*abs(s2_mu_meanslope[i]):
        result=24
        break
end=result
start=end-20
#pretail:2426-2452
s2_energy_pre=s2_energy[start:end].reshape((-1,1))
s2_mu_pre=s2_mu[start:end]
s2_model = LinearRegression().fit(s2_energy_pre, s2_mu_pre)
s2_mu_back=s2_model.intercept_+s2_model.coef_*s2_energy
s2_mu_subs=s2_mu-s2_mu_back
s2['substracted mu']=s2_mu_subs
s2_musubs_min=np.amin(-1*s2_mu_subs)
i_max=np.where(-s2_musubs_min==s2_mu_subs)[0].item()
s2_energy_min=s2_energy[i_max]
s2_r2=s2_model.score(s2_energy_pre, s2_mu_pre)
print("the r sqaured for regression of s2 is "+str(s2_r2))
print("energy for pre-tails are between" +str(s2_energy[start])+" eV and "+str(s2_energy[end])+" eV")
print("the energy of peak in s2 is "+str(s2_energy_min)+"eV")
print()
# plt.plot(s2_energy,s2_mu,label="raw")
# plt.plot(s2_energy,s2_mu_subs,label="substracted")
# plt.legend(loc='best')
# plt.show()
# plt.plot(s2_energy_pre,s2_mu_pre)
# plt.plot(s2_energy_pre,s2_mu_back[start:end])
# plt.show()
# print(s2)
#s3
s3_energy=s3.Energy.to_numpy()
s3_I0=s3.I0.to_numpy()
s3_PIPS=s3.Ipips.to_numpy()
s3_mu=s3_PIPS/s3_I0
s3_mu_diff=np.diff(s3_mu)
s3_mu_diff=np.append(s3_mu_diff,s3_mu_diff[-1])
s3_energy_diff=np.diff(s3_energy)
s3_energy_diff=np.append(s3_energy_diff,s3_energy_diff[-1])
s3_mu_slope=s3_mu_diff/s3_energy_diff
s3['mu']=s3_mu
s3['slope of mu']=s3_mu_slope
s3_mu_meanslope=np.zeros(1*len(s3_mu_slope))
for i in range(len(s3_mu_slope)):
    s3_mu_meanslope[i]=np.mean(s3_mu_slope[0:i+1],dtype=np.float64)
for i in range(len(s3_mu_slope)):
    if abs(s3_mu_slope[i])>10*abs(s3_mu_meanslope[i]):
        result=68
        break
end=result
start=end-20
s3_energy_pre=s3_energy[start:end].reshape((-1,1))
s3_mu_pre=s3_mu[start:end]
s3_model = LinearRegression().fit(s3_energy_pre, s3_mu_pre)
s3_mu_back=s3_model.intercept_+s3_model.coef_*s3_energy
s3_mu_subs=s3_mu-s3_mu_back
s3['substracted mu']=s3_mu_subs
s3_musubs_min=np.amin(-1*s3_mu_subs)
i_max=np.where(-s3_musubs_min==s3_mu_subs)[0].item()
s3_energy_min=s3_energy[i_max]
s3_r2=s3_model.score(s3_energy_pre, s3_mu_pre)
print("the r sqaured for regression of s3 is "+str(s3_r2))
print("energy for pre-tails are between" +str(s3_energy[start])+" eV and "+str(s3_energy[end])+" eV")
print("the energy of peak in s3 is "+str(s3_energy_min)+"eV")
print()
# plt.plot(s3_energy,s3_mu,label="raw")
# plt.plot(s3_energy,s3_mu_subs,label="substracted")
# plt.legend(loc='best')
# plt.show()
# plt.plot(s3_energy_pre,s3_mu_pre)
# plt.plot(s3_energy_pre,s3_mu_back[start:end])
# plt.legend(loc='best')
# plt.show()
# print(s3)

#s4
s4_energy=s4.Energy.to_numpy()
s4_I0=s4.I0.to_numpy()
s4_PIPS=s4.Ipips.to_numpy()
s4_mu=s4_PIPS/s4_I0
s4_mu_diff=np.diff(s4_mu)
s4_mu_diff=np.append(s4_mu_diff,s4_mu_diff[-1])
s4_energy_diff=np.diff(s4_energy)
s4_energy_diff=np.append(s4_energy_diff,s4_energy_diff[-1])
s4_mu_slope=s4_mu_diff/s4_energy_diff
s4['mu']=s4_mu
s4['slope of mu']=s4_mu_slope
s4_mu_meanslope=np.zeros(1*len(s4_mu_slope))
for i in range(len(s4_mu_slope)):
    s4_mu_meanslope[i]=np.mean(s4_mu_slope[0:i+1],dtype=np.float64)
for i in range(len(s4_mu_slope)):
    if abs(s4_mu_slope[i])>0.001 and abs(s4_mu_slope[i])>10*abs(s4_mu_meanslope[i]):
        result=80
        break
end=result
start=end-20
#pretail:2423-2452
s4_energy_pre=s4_energy[start:end].reshape((-1,1))
s4_mu_pre=s4_mu[start:end]
s4_model = LinearRegression().fit(s4_energy_pre, s4_mu_pre)
s4_mu_back=s4_model.intercept_+s4_model.coef_*s4_energy
s4_mu_subs=s4_mu-s4_mu_back
s4['substracted mu']=s4_mu_subs
s4_musubs_min=np.amin(-1*s4_mu_subs)
i_max=np.where(-s4_musubs_min==s4_mu_subs)[0].item()
s4_energy_min=s4_energy[i_max]
s4_r2=s4_model.score(s4_energy_pre, s4_mu_pre)
print("the r sqaured for regression of s4 is "+str(s4_r2))
print("energy for pre-tails are between" +str(s3_energy[start])+" eV and "+str(s3_energy[end])+" eV")
print("the energy of peak in s4 is "+str(s4_energy_min)+"eV")
print()
# plt.plot(s4_energy,s4_mu,label="raw")
# plt.plot(s4_energy,s4_mu_subs,label="substracted")
# plt.show()
# plt.plot(s4_energy_pre,s4_mu_pre)
# plt.plot(s4_energy_pre,s4_mu_back[start:end])
# plt.legend(loc='best')
# plt.show()
# print(s4)
#s5
s5_energy=s5.Energy.to_numpy()
s5_I0=s5.I0.to_numpy()
s5_PIPS=s5.Ipips.to_numpy()
s5_mu=s5_PIPS/s5_I0
s5_mu_diff=np.diff(s5_mu)
s5_mu_diff=np.append(s5_mu_diff,s5_mu_diff[-1])
s5_energy_diff=np.diff(s5_energy)
s5_energy_diff=np.append(s5_energy_diff,s5_energy_diff[-1])
s5_mu_slope=s5_mu_diff/s5_energy_diff
s5['mu']=s5_mu
s5['slope of mu']=s5_mu_slope
s5_mu_meanslope=np.zeros(1*len(s5_mu_slope))
for i in range(len(s5_mu_slope)):
    s5_mu_meanslope[i]=np.mean(s5_mu_slope[0:i+1],dtype=np.float64)
for i in range(len(s5_mu_slope)):
    if abs(s5_mu_slope[i])>0.001 and abs(s5_mu_slope[i])>10*abs(s5_mu_meanslope[i]):
        result=99
        break
end=result
start=end-20
#pretail:2460-2464
s5_energy_pre=s5_energy[start:end].reshape((-1,1))
s5_mu_pre=s5_mu[start:end]
s5_model = LinearRegression().fit(s5_energy_pre, s5_mu_pre)
s5_mu_back=s5_model.intercept_+s5_model.coef_*s5_energy
s5_mu_subs=s5_mu-s5_mu_back
s5['substracted mu']=s5_mu_subs
s5_musubs_min=np.amin(-1*s5_mu_subs)
i_max=np.where(-s5_musubs_min==s5_mu_subs)[0].item()
s5_energy_min=s5_energy[i_max]
s5_r2=s5_model.score(s5_energy_pre, s5_mu_pre)
print("the r sqaured for regression of s5 is "+str(s5_r2))
print("energy for pre-tails are between" +str(s3_energy[start])+" eV and "+str(s3_energy[end])+" eV")
print("the energy of peak in s5 is "+str(s5_energy_min)+"eV")
print()
# plt.plot(s5_energy,s5_mu,label="raw")
# plt.plot(s5_energy,s5_mu_subs,label="substracted")
# plt.show()
# plt.plot(s5_energy_pre,s5_mu_pre)
# plt.plot(s5_energy_pre,s5_mu_back[start:end])
# plt.legend(loc='best')
# plt.show()
# print(s5)
"""
space=0
plt.plot(s1_energy,s1_mu_subs+0*space,label="sasph008")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
plt.plot(s2_energy,s2_mu_subs+1*space,label="sdbs034")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
plt.plot(s3_energy,s3_mu_subs+2*space,label="sdbso042")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
plt.plot(s4_energy,s4_mu_subs+3*space,label="sdbt029")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
plt.plot(s5_energy,s5_mu_subs+4*space,label="sfeso4051")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
"""
#plot a superposed figure
space=0.065
plt.plot(s1_energy,s1_mu_subs+0*space,label="sasph008")
plt.plot(s2_energy,s2_mu_subs+1*space,label="sdbs034")
plt.plot(s3_energy,s3_mu_subs+2*space,label="sdbso042")
plt.plot(s4_energy,s4_mu_subs+3*space,label="sdbt029")
plt.plot(s5_energy,s5_mu_subs+4*space,label="sfeso4051")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()
#export data to excel files
subs1={'Energy':s1_energy,'subs mu':s1_mu_subs}
sub_s1 = pd.DataFrame(subs1, columns = ['Energy', 'subs mu'])
sub_s1.to_excel(r'C:\Users\lenovo\OneDrive\桌面\subs_s1.xlsx', index = False, header=True)

subs2={'Energy':s2_energy,'subs mu':s2_mu_subs}
sub_s2 = pd.DataFrame(subs2, columns = ['Energy', 'subs mu'])
sub_s2.to_excel(r'C:\Users\lenovo\OneDrive\桌面\subs_s2.xlsx', index = False, header=True)

subs3={'Energy':s3_energy,'subs mu':s3_mu_subs}
sub_s3 = pd.DataFrame(subs3, columns = ['Energy', 'subs mu'])
sub_s3.to_excel(r'C:\Users\lenovo\OneDrive\桌面\subs_s3.xlsx', index = False, header=True)

subs4={'Energy':s4_energy,'subs mu':s4_mu_subs}
sub_s4 = pd.DataFrame(subs4, columns = ['Energy', 'subs mu'])
sub_s4.to_excel(r'C:\Users\lenovo\OneDrive\桌面\subs_s4.xlsx', index = False, header=True)

subs5={'Energy':s5_energy,'subs mu':s5_mu_subs}
sub_s5 = pd.DataFrame(subs5, columns = ['Energy', 'subs mu'])
sub_s5.to_excel(r'C:\Users\lenovo\OneDrive\桌面\subs_s5.xlsx', index = False, header=True)