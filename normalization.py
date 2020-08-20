# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import excel files and construct numpy array
s1 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\subs_s1.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
s2 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\subs_s2.xlsx')
s3 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\subs_s3.xlsx')
s4 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\subs_s4.xlsx')
s5 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\subs_s5.xlsx')
midlength=50
#1
#eye-spot the quiet energy:2575eV
s1_energy=s1.Energy.to_numpy()
s1_subs_mu=s1['subs mu'].to_numpy()
s1_mid=281
s1_fac=np.mean(s1_subs_mu[s1_mid-int(midlength/2):s1_mid+int(midlength/2)],dtype=np.float64)
s1_norm_mu=s1_subs_mu/s1_fac
# plt.plot(s1_energy,s1_norm_mu)
# plt.show()

#2
s2_energy=s2.Energy.to_numpy()
s2_subs_mu=s2['subs mu'].to_numpy()
s2_mid=443
#range:2550-2600
s2_fac=np.mean(s2_subs_mu[s2_mid-midlength:s2_mid+midlength],dtype=np.float64)
s2_norm_mu=s2_subs_mu/s2_fac
# plt.plot(s2_energy,s2_norm_mu)
# plt.show()

#3
s3_energy=s3.Energy.to_numpy()
s3_subs_mu=s3['subs mu'].to_numpy()
s3_mid=443
s3_fac=np.mean(s3_subs_mu[s3_mid-midlength:s3_mid+midlength],dtype=np.float64)
s3_norm_mu=s3_subs_mu/s3_fac
# plt.plot(s3_energy,s3_norm_mu)
# plt.show()

#4
s4_energy=s4.Energy.to_numpy()
s4_subs_mu=s4['subs mu'].to_numpy()
s4_mid=443
s4_fac=np.mean(s4_subs_mu[s4_mid-midlength:s4_mid+midlength],dtype=np.float64)
s4_norm_mu=s4_subs_mu/s4_fac
# plt.plot(s4_energy,s4_norm_mu)
# plt.show()

#5
s5_energy=s5.Energy.to_numpy()
s5_subs_mu=s5['subs mu'].to_numpy()
s5_mid=443
s5_fac=np.mean(s5_subs_mu[s5_mid-midlength:s5_mid+midlength],dtype=np.float64)
s5_norm_mu=s5_subs_mu/s5_fac
# plt.plot(s5_energy,s5_norm_mu)
# plt.show()

# #plot a superposed figure
# space=0
# plt.plot(s1_energy,s1_norm_mu+0*space,label="sasph008")
# plt.plot(s2_energy,s2_norm_mu+1*space,label="sdbs034")
# plt.plot(s3_energy,s3_norm_mu+2*space,label="sdbso042")
# plt.plot(s4_energy,s4_norm_mu+3*space,label="sdbt029")
# plt.plot(s5_energy,s5_norm_mu+4*space,label="sfeso4051")
# plt.legend(loc="upper right")
# plt.set_xlabel="energy"
# plt.set_ylabel="absorption"
# plt.show()

#construct dataframe of known and unknown samples respectively
norms1={"Energy":s1_energy, "s1_norm_mu":s1_norm_mu}
norm_s1=pd.DataFrame(norms1, columns=['Energy','s1_norm_mu'])
normknown={'Energy':s2_energy[0:614],'s2_norm_mu':s2_norm_mu[0:614],'s3_norm_mu':s3_norm_mu[0:614],
           's4_norm_mu':s4_norm_mu[0:614],'s5_norm_mu':s5_norm_mu[0:614]}
norm_known=pd.DataFrame(normknown,columns=['Energy','s2_norm_mu',
                                           's3_norm_mu','s4_norm_mu','s5_norm_mu'])

#average values and construct a merged dataframe
i=0
j=0
while((i<len(norm_s1)-1) and (j<len(norm_known)-1)):
    while(norm_s1['Energy'][i]>norm_known['Energy'][j]):
        if(not(norm_known['Energy'][j+1]>norm_s1['Energy'][i])):
            j=j+1
        else:
            norm_s2_value=(norm_s1['Energy'][i]-norm_known['Energy'][j])*(norm_known['s2_norm_mu'][j+1]-norm_known['s2_norm_mu'][j])
            norm_s2_value=norm_s2_value/(norm_known['Energy'][j+1]-norm_known['Energy'][j])
            norm_s2_value=norm_known['s2_norm_mu'][j]+norm_s2_value
            
            norm_s3_value=(norm_s1['Energy'][i]-norm_known['Energy'][j])*(norm_known['s3_norm_mu'][j+1]-norm_known['s3_norm_mu'][j])
            norm_s3_value=norm_s3_value/(norm_known['Energy'][j+1]-norm_known['Energy'][j])
            norm_s3_value=norm_known['s3_norm_mu'][j]+norm_s3_value
            
            norm_s4_value=(norm_s1['Energy'][i]-norm_known['Energy'][j])*(norm_known['s4_norm_mu'][j+1]-norm_known['s4_norm_mu'][j])
            norm_s4_value=norm_s4_value/(norm_known['Energy'][j+1]-norm_known['Energy'][j])
            norm_s4_value=norm_known['s4_norm_mu'][j]+norm_s4_value 
            
            norm_s5_value=(norm_s1['Energy'][i]-norm_known['Energy'][j])*(norm_known['s5_norm_mu'][j+1]-norm_known['s5_norm_mu'][j])
            norm_s5_value=norm_s5_value/(norm_known['Energy'][j+1]-norm_known['Energy'][j])
            norm_s5_value=norm_known['s5_norm_mu'][j]+norm_s5_value 
            
            data=pd.DataFrame({'Energy':norm_s1['Energy'][i],
                               's2_norm_mu':norm_s2_value,
                               's3_norm_mu':norm_s3_value,
                               's4_norm_mu':norm_s4_value,
                               's5_norm_mu':norm_s5_value},index=[j+0.5])
            norm_known = norm_known.append(data, ignore_index=False)
            norm_known = norm_known.sort_index().reset_index(drop=True)
            j=j+1
    i=i+1        
    
#plot the superposed normalized merged figures
norm=pd.merge(norm_s1,norm_known,how="inner",on="Energy")
plt.plot(norm['Energy'],norm['s1_norm_mu'],label="sasph008")
plt.plot(norm['Energy'],norm['s2_norm_mu'],label="sdbs034")
plt.plot(norm['Energy'],norm['s3_norm_mu'],label="sdbso042")
plt.plot(norm['Energy'],norm['s4_norm_mu'],label="sdbt029")
plt.plot(norm['Energy'],norm['s5_norm_mu'],label="sfeso4051")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()

#export the excel file
norm.to_excel(r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx', index = False, header=True)