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
def back_subs(df, end, num):
    """
    Parameters
    ----------
    df : Pandas DataFrame
         df is the dataframe that is directly read from excel file. After the 
         execution of this function, two more columns will be added to the 
         DataFrame. The first is "mu", recording the linear absorption coef.
         The second is "substracted_mu", recording the linear absorption coef
         after doing background substraction
    end : int
        it records the index of the last element in pre-tail energy in df
    num : TYPE
        it records how much elements are considered as pre-tail energy

    Returns
    -------
    dict : Python dictionary
           "mu_subs": it records the linear absorption coef 
           after doing background substraction.
           "r2": it records the coef of determination for the linear regression 
           line used in the background substraction
           "range": it records the energy range of pre-tail
           "Ep": it records the energy associated with the highest 
           linear absorption coef (peak) in df after background substraction.
           "msg": the message that can be printed indicating information 
           about r2, range, and Ep

    """
    df_energy=df.Energy.to_numpy()
    df_I0=df.I0.to_numpy()
    df_PIPS=df.PIPS.to_numpy()
    df_mu=df_PIPS/df_I0
    df['mu']=df_mu
    #determine the end and start energy
    start=end-num
    df_energy_pre=df_energy[start:end].reshape((-1,1))
    df_mu_pre=df_mu[start:end]
    df_model = LinearRegression().fit(df_energy_pre, df_mu_pre)
    df_mu_back=df_model.intercept_+df_model.coef_*df_energy
    df_mu_subs=df_mu-df_mu_back
    df['substracted_mu']=df_mu_subs
    df_musubs_min=np.amin(-1*df_mu_subs)
    i_max=np.where(-df_musubs_min==df_mu_subs)[0].item()
    df_energy_min=df_energy[i_max]
    df_r2=df_model.score(df_energy_pre, df_mu_pre)
    msg=("the r sqaured for regression of is "+
            str(df_r2)+"\n energy for pre-tails are between" +str(df_energy[start])+
            " eV and "+str(df_energy[end])+" eV"+"\n the energy of peak in s2 is "
            +str(df_energy_min)+"eV")
    return {"mu_subs":df_mu_subs,"r2":df_r2,"range":[df_energy[start],df_energy[end]],
            "Ep":df_energy_min,"msg":msg}

s1_info=back_subs(s1,30,20)
s2_info=back_subs(s2,24,20)
s3_info=back_subs(s3,68,20)
s4_info=back_subs(s4,80,20)
s5_info=back_subs(s5,99,20)

#plot a superposed figure
space=0.065
plt.plot(s1['Energy'],back_subs(s1,30,20)["mu_subs"]+0*space,label="sasph008")
plt.plot(s2['Energy'],back_subs(s2,24,20)["mu_subs"]+1*space,label="sdbs034")
plt.plot(s3['Energy'],back_subs(s3,68,20)["mu_subs"]+2*space,label="sdbso042")
plt.plot(s4['Energy'],back_subs(s4,80,20)["mu_subs"]+3*space,label="sdbt029")
plt.plot(s5['Energy'],back_subs(s5,99,20)["mu_subs"]+4*space,label="sfeso4051")
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()

# s1_energy=s1.Energy.to_numpy()
# s1_subs_mu=s1['subs mu'].to_numpy()
# s1_mid=281
# s1_fac=np.mean(s1_subs_mu[s1_mid-int(midlength/2):s1_mid+int(midlength/2)],dtype=np.float64)
# s1_norm_mu=s1_subs_mu/s1_fac
def normalization(df,center,midrange):
    df_fac=np.mean(df['mu_subs'])
    
