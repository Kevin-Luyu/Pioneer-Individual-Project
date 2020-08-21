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
    num : int
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

def normalization(df,center,midrange):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        df is the dataframe that is directly read from excel file.
    center : int
        it records the center of ranges of data taken for average when normalization
    midrange : int
        it records half the number of data taken for average when normalization

    Returns
    -------
    df_norm_mu : Pandas DataFrame
        it is a DataFrame of width 1 that records the data after background substraction
        and normalization

    """
    df_fac=np.mean(df['substracted_mu'][center-midrange:center+midrange])
    df_norm_mu=df['substracted_mu']/df_fac
    return df_norm_mu

s1["mu_norm"]=normalization(s1,281,25)
s2["mu_norm"]=normalization(s2,443,50)
s3["mu_norm"]=normalization(s3, 443, 50)
s4["mu_norm"]=normalization(s4, 443, 50)
s5["mu_norm"]=normalization(s5, 443, 50)

def norm_merge(df1,df2):
    """
    

    Parameters
    ----------
    df1 : Pandas DataFrame
        The "base" dataframe for merging. It is suggested that you take the dataframe
        with fewer elements as df1 for higher accuracy. df1 should have only
        one target to merge. (You cannot do a nested method for norm_merge)
    df2 : Pandas DataFrame
        df2 is being merged into df1.

    Returns
    -------
    merged : Pandas DataFrame
        based on taking averages, more rows will be added to df2 so that there is 
        a corresponding value in df2 for every energy in df1. Those rows with energy not 
        present in df1 will be deleted. merged is the DataFrame after merging 
        df1 and df2. df2 should have only one target to merge. 
        (You cannot do a nested method for norm_merge)

    """
    i=0
    j=0
    while((i<df1.shape[0]-1)and(j<df2.shape[0]-1)):
        while(df1.Energy[i]>df2.Energy[j]):
            if(not(df2.Energy[j+1])>df1.Energy[i]):
                j=j+1
            else:
                df2_value=(df1.Energy[i]-df2.Energy[j])*(df2.mu_norm[j+1]-df2.mu_norm[j])
                df2_value=df2_value/(df2.Energy[j+1]-df2.Energy[j])
                df2_value=df2_value+df2.mu_norm[j]
                data=pd.DataFrame({'Energy':df1.Energy[i],
                                   'mu_norm':df2_value},index=[j+0.5])
                df2=df2.append(data,ignore_index=False)
                df2=df2.sort_index().reset_index(drop=True)
                j=j+1
        i=i+1
    merged=pd.merge(df1,df2,how="inner",on="Energy")
    return merged
s1=s1[6:]
s1=s1.sort_index().reset_index(drop=True)
norm=pd.DataFrame({'Energy':s1['Energy'],"s1_norm_mu":norm_merge(s1,s2[0:614])['mu_norm_x'],
                   "s2_norm_mu":norm_merge(s1,s2[0:614])['mu_norm_y'],
                   "s3_norm_mu":norm_merge(s1,s3[0:614])['mu_norm_y'],
                   "s4_norm_mu":norm_merge(s1,s4[0:614])['mu_norm_y'],
                   "s5_norm_mu":norm_merge(s1,s5[0:614])['mu_norm_y']})

#plot the superposed normalized merged figures
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
# norm.to_excel(r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx', index = False, header=True)
