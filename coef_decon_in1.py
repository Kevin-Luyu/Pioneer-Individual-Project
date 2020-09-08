import time
start_time = time.time()
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lmfit.models import StepModel, LorentzianModel
import win32com.client
import winsound
from win10toast import ToastNotifier
from pushover import init, Client
#import excel files and construct numpy array

res_coef=pd.DataFrame(columns=['Post-edge Energy Range','sdbs034(0)',
                                           'sdbt029(0)','dbso042(+2)','sdbso2(+2)',
                                           'sfeso4051(+6)','chi2'])
s1 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sasph008.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
s2 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbs034.xlsx')
s3 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbso042.xlsx')
s4 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbt029.xlsx')
s5 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sfeso4051.xlsx')
s6 = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\sdbso2.xlsx')
#define the start and legth of post-edge
for energy_start in [2510]:
    for energy_end in [2530]:
        if energy_end > energy_start:
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
            s6_info=back_subs(s6,40,20)
            #plot a superposed figure
            space=0.065
            plt.plot(s1['Energy'],back_subs(s1,30,20)["mu_subs"]+0*space,label="sasph008")
            plt.plot(s2['Energy'],back_subs(s2,24,20)["mu_subs"]+1*space,label="sdbs034")
            plt.plot(s3['Energy'],back_subs(s3,68,20)["mu_subs"]+2*space,label="sdbso042")
            plt.plot(s4['Energy'],back_subs(s4,80,20)["mu_subs"]+3*space,label="sdbt029")
            plt.plot(s5['Energy'],back_subs(s5,99,20)["mu_subs"]+4*space,label="sfeso4051")
            plt.plot(s6['Energy'],back_subs(s6,40,20)["mu_subs"]+5*space,label="sdbso2")
            plt.legend(loc="upper right")
            plt.set_xlabel="energy"
            plt.set_ylabel="absorption"
            plt.show()
            
            def normalization(df,start,end):
                """
            
                Parameters
                ----------
                df : Pandas DataFrame
                    df is the dataframe that is directly read from excel file.
                start : int
                    it records the start of data taken for average when normalization
                end : int
                    it records the end of data taken for average when normalization
            
                Returns
                -------
                [df_norm_mu] : list
                    it is a list with only 1 element df_norm_mu
                    it is a DataFrame of width 1 that records the data after background substraction
                    and normalization
            
                """
                df_fac=np.mean(df['substracted_mu'][start:end])
                df_norm_mu=df['substracted_mu']/df_fac
                return [df_norm_mu]
            def normalization1(df,start,end):
                """
                
            
                Parameters
                ----------
                df : Pandas DataFrame
                    df is the dataframe that is directly read from excel file.
                start : int
                    it records the start of data taken for regression when normalization
                end : int
                    it records the end of data taken for regression when normalization
            
                Returns
                -------
                [norm_mu,eq(df.Energy)] : list
                    it is a list with 2 elements
                    norm_mu is a DataFrame of width 1 that records the data after background substraction
                    and normalization
                    
                    eq(df.Energy) is a column pandas DataFrame of length (df.Energy).shape[0]
                    it records the associated regression linear equation
            
                """
                quad=np.polyfit(df.Energy[start:end],df['substracted_mu'][start:end],1)
                eq=np.poly1d(quad)
                norm_mu=df['substracted_mu']/eq(df.Energy)
                return [norm_mu,eq(df.Energy)]
            
            def normalization2(df,start,end):
                """
                
            
                Parameters
                ----------
                df : Pandas DataFrame
                    df is the dataframe that is directly read from excel file.
                start : int
                    it records the start of data taken for regression when normalization
                end : int
                    it records the end of data taken for regression when normalization
            
                Returns
                -------
                [norm_mu,eq(df.Energy)] : list
                    it is a list with 2 elements
                    norm_mu is a DataFrame of width 1 that records the data after background substraction
                    and normalization
                    
                    eq(df.Energy) is a column pandas DataFrame of length (df.Energy).shape[0]
                    it records the associated regression quadratic equation
            
                """
                quad=np.polyfit(df.Energy[start:end],df['substracted_mu'][start:end],2)
                eq=np.poly1d(quad)
                norm_mu=df['substracted_mu']/eq(df.Energy)
                return [norm_mu,eq(df.Energy)]
            
            def close_index(df,value):
                """
                
            
                Parameters
                ----------
                df : pandas DataFrame
                    a 1-width column of DataFrame within which you want to find the 
                    closest number to value
                value : float
                    the number that you want to approach
            
                Returns
                -------
                index : int
                    such that df[index] is the closest number in df to the value input.
            
                """
                index=abs(df - value).idxmin(axis=1,skipna=True)
                return index 
            
            s6_start=close_index(s6['Energy'],energy_start)
            s6_end=close_index(s6['Energy'],energy_end)
            
            s1["mu_norm"]=normalization(s1,close_index(s1['Energy'],energy_start),close_index(s1['Energy'],energy_end))[0]
            s2["mu_norm"]=normalization(s2,close_index(s2['Energy'],energy_start),close_index(s2['Energy'],energy_end))[0]
            s3["mu_norm"]=normalization(s3, close_index(s3['Energy'],energy_start),close_index(s3['Energy'],energy_end))[0]
            s4["mu_norm"]=normalization(s4, close_index(s4['Energy'],energy_start),close_index(s4['Energy'],energy_end))[0]
            s5["mu_norm"]=normalization(s5, close_index(s5['Energy'],energy_start),close_index(s5['Energy'],energy_end))[0]
            s6["mu_norm"]=normalization(s6,s6_start,s6_end)[0]
            
            #define the start and end of spectra calculating chi2
            start_energy=2461
            end_energy=2525
            
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
                    first slice df1 and df2 so that they have the same starting and ending
                    energy start_energy and end_energy specified before
                    based on taking averages, more rows will be added to df2 so that there is 
                    a corresponding value in df2 for every energy in df1. Those rows with energy not 
                    present in df1 will be deleted. merged is the DataFrame after merging 
                    df1 and df2. df2 should have only one target to merge. 
                    (You cannot do a nested method for norm_merge)
            
                """
                df1=df1[close_index(df1['Energy'],start_energy):close_index(df1['Energy'],end_energy)]
                df1=df1.sort_index().reset_index(drop=True)
                df2=df2[close_index(df2['Energy'],start_energy):close_index(df2['Energy'],end_energy)]
                df2=df2.sort_index().reset_index(drop=True)
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
            norm=pd.DataFrame({'Energy':norm_merge(s1,s2)['Energy'],"s1_norm_mu":norm_merge(s1,s2)['mu_norm_x'],
                               "s2_norm_mu":norm_merge(s1,s2)['mu_norm_y'],
                               "s3_norm_mu":norm_merge(s1,s3)['mu_norm_y'],
                               "s4_norm_mu":norm_merge(s1,s4)['mu_norm_y'],
                               "s5_norm_mu":norm_merge(s1,s5)['mu_norm_y'],
                               "s6_norm_mu":norm_merge(s1,s6)['mu_norm_y']})
            #plot the superposed normalized merged figures
            plt.plot(norm['Energy'],norm['s1_norm_mu'],label="sasph008")
            plt.plot(norm['Energy'],norm['s2_norm_mu'],label="sdbs034")
            plt.plot(norm['Energy'],norm['s3_norm_mu'],label="sdbso042")
            plt.plot(norm['Energy'],norm['s4_norm_mu'],label="sdbt029")
            plt.plot(norm['Energy'],norm['s5_norm_mu'],label="sfeso4051")
            plt.plot(norm['Energy'],norm['s6_norm_mu'],label="sdbso2")
            plt.legend(loc="upper right")
            plt.set_xlabel="energy"
            plt.set_ylabel="absorption"
            plt.show()
            
            def inflection(energy,df):
                """
                
            
                Parameters
                ----------
                energy : pandas DataFrame
                    energy is a column-wise Pandas DataFrame that records the 
                    energy of data
                df : pandas DataFrame
                    df is a column-wise Pandas DataFrame that records the linear 
                    absorption coef for the known sample.
            
                Returns
                -------
                float
                    returns the energy where the point of inflection of df happens.
            
                """
                energy_diff=np.diff(energy)
                df_diff=np.diff(df)
                slope=(df_diff/energy_diff)
                slope_min=np.amin(-1*slope)
                index=np.where(-slope_min-slope==0)[0]
                return energy[index].item()
            #make parameters for the deconvolution of known
            arctan_mod=StepModel(form='atan',prefix='arctan_')
            paras=arctan_mod.make_params()
            #construct the model with 5 arctangents and 5 Lorentzians
            atan2=StepModel(form='atan',prefix='atan2_')
            atan3=StepModel(form='atan',prefix='atan3_')
            atan4=StepModel(form='atan',prefix='atan4_')
            atan5=StepModel(form='atan',prefix='atan5_')
            atan6=StepModel(form='atan',prefix='atan6_')
            lor2=LorentzianModel(prefix='l2_')
            lor3=LorentzianModel(prefix='l3_')
            lor4=LorentzianModel(prefix='l4_')
            lor5=LorentzianModel(prefix='l5_')
            lor6=LorentzianModel(prefix='l6_')
            model=atan2+atan3+atan4+atan5+atan6+lor2+lor3+lor4+lor5+lor6
            model.set_param_hint('l2_amplitude', min=0.0)
            model.set_param_hint('l3_amplitude', min=0.0)
            model.set_param_hint('l4_amplitude', min=0.0)
            model.set_param_hint('l5_amplitude', min=0.0)
            model.set_param_hint('l6_amplitude', min=0.0)
            paras_un=model.make_params()
            #put constraints on the amplitude
            paras_un['l2_amplitude'].set(min=0.0)
            paras_un['l3_amplitude'].set(min=0.0)
            paras_un['l4_amplitude'].set(min=0.0)
            paras_un['l5_amplitude'].set(min=0.0)
            paras_un['l6_amplitude'].set(min=0.0)
            def make_lor(df,num,center,length):
                """
                This method do a single-peak Lorentzian deconvolution for a given dataframe
            
                Parameters
                ----------
                df : pandas dataframe
                    df is a column-wise dataframe that records the data you want to 
                    deconvolve.
                num : int
                    a positional keyword, indicating which index of center array the 
                    Lorentzian in this method is building its center on.
                center : array
                    the array recording the centers of all peaks.
                length : float 
                    the maximum allowed variance in the optimized position of peaks from their
                    initila values indicated in the center array.
            
                Returns
                -------
                dict
                    model: the single-peak Lorentzian corresponded.
                    paras: the parameters optimized through this function. This is useful
                    in updating the parameters object paras
            
                """
                pref='l'+str(num)+'_'
                model=LorentzianModel(prefix=pref)
                paras.update(model.guess(df,x=norm.Energy,center=center[num]))
                name=pref+'center'
                paras[name].set(value=center[num],min=center[num]-length,max=center[num]+length)
                paras[pref+'amplitude'].set(min=0.0)
                paras[pref+'sigma'].set(min=0.0)
                return {'model':model,'paras':paras}
            def decon_known(df,center):
                """
                This method uses multiple make_lor to do a multiple-peak deconvolutoin using 
                multiple Lorentzians and one arc tangent step function
            
                Notice: parameters are NOT automatically deleted if they are not used in the 
                deconvolution of data with fewer peaks than the last time
                
                Parameters
                ----------
                df : pandas dataframe
                    df is a column-wise dataframe that records the data you want to 
                    deconvolve.
                center : array
                    the array recording the centers of all peaks.
            
                Returns
                -------
                out : lmfit.Model
                    the composit multiple-peak model for the deconvolution.
            
                """
                arctan_mod=StepModel(form='atan',prefix='arctan_')
                paras.update(arctan_mod.make_params())
                paras['arctan_center'].set(value=inflection(norm.Energy,df),vary=False,min=0.0)
                paras['arctan_amplitude'].set(value=1.0,vary=False)
                paras['arctan_sigma'].set(value=1.0,min=0.0)
                mod=arctan_mod
                for i in range(len(center)):
                    this=make_lor(df,i,center,0.7)['model']
                    mod=mod+this
                    paras.update(make_lor(df,i,center,0.7)['paras'])
                out=mod.fit(df,params=paras,x=norm.Energy)
                return {'out':out}
            
            # print(decon_known(norm['s2_norm_mu'],[2472.5,2475.4,2478.9,2483.2]).fit_report())
            def show_decon_known(df,center):
                """
                
                This method displays the result of deconvolution by
                1. pring the report of fitting
                2. draw a figure displaying the best-fit compared with raw data and the 
                splitting of peaks
                
               
                Parameters
                ----------
                df : pandas dataframe
                    df is a column-wise dataframe that records the data you want to 
                    deconvolve.
                center : array
                    the array recording the centers of all peaks.
            
                Returns
                -------
                None.
            
                """
                out=decon_known(df,center)['out']
                print(out.fit_report())
                
                fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
                axes[0].plot(norm.Energy, df, 'b')
                axes[0].plot(norm.Energy, out.best_fit, 'r-', label='best fit')
                axes[0].legend(loc='best')
                
                comps = out.eval_components(x=norm.Energy)
                axes[1].plot(norm.Energy, df, 'b')
                for i in range (len(center)):
                    name='l'+str(i)+'_'
                    lb='Lorentzian component '+str(i)
                    axes[1].plot(norm.Energy,comps[name],label=lb)
                axes[1].plot(norm.Energy, comps['arctan_'], label='arctangent component')
                axes[1].legend(loc='best')
                return {'ratio':((out.params['arctan_sigma'].value)),
                        'center':out.params['l0_center'],'amp':out.params['l0_amplitude'].value,}
            
            # deconvolute for s2
            paras['arctan_center'].set(value=inflection(norm.Energy,norm['s2_norm_mu']),vary=False)
            paras['arctan_amplitude'].set(value=1.0,vary=False)
            paras['arctan_sigma'].set(value=1.0,min=0.0)
            s2_info=show_decon_known(norm['s2_norm_mu'],[2472.5,2475.4,2478.9,2483.2])
            s2_ratio,s2_amp=s2_info['ratio'],s2_info['amp']
            s2_cen=s2_info['center']
            paras_un.add('s2_cen',value=s2_cen,vary=False)
            paras_un.add('s2_ratio',value=s2_ratio,vary=False)
            paras_un.add('s2_amp',value=s2_amp,vary=False)
            
            
            #deconvolve for s3
            paras['arctan_center'].set(value=inflection(norm.Energy,norm['s3_norm_mu']),vary=False)
            paras['arctan_sigma'].set(value=1.0,min=0.0)
            s3_info = show_decon_known(norm['s3_norm_mu'],[2475.26,2478.20,2489.90,2484.52])
            s3_ratio,s3_amp=s3_info['ratio'],s3_info['amp']
            paras_un.add('s3_ratio',value=s3_ratio,vary=False)
            paras_un.add('s3_amp',value=s3_amp,vary=False)
            s3_cen=s3_info['center']
            paras_un.add('s3_cen',value=s3_cen,vary=False)
            
            #deconvolve for s4
            paras['arctan_center'].set(value=inflection(norm.Energy,norm['s4_norm_mu']),vary=False)
            paras['arctan_sigma'].set(value=1.0,min=0.0)
            s4_info=show_decon_known(norm['s4_norm_mu'],[2473.05,2476.22,2481.59])
            s4_ratio,s4_amp=s4_info['ratio'],s4_info['amp']
            paras_un.add('s4_ratio',value=s4_ratio,vary=False)
            paras_un.add('s4_amp',value=s4_amp,vary=False)
            s4_cen=s4_info['center']
            paras_un.add('s4_cen',value=s4_cen,vary=False)
            
            #deconvolve for s5
            paras['arctan_center'].set(value=inflection(norm.Energy,norm['s5_norm_mu']),vary=False)
            paras['arctan_sigma'].set(value=1.0,min=0.0)
            s5_info=show_decon_known(norm['s5_norm_mu'],[2481.76,2497.41])
            s5_ratio,s5_amp=s5_info['ratio'],s5_info['amp']
            paras_un.add('s5_ratio',value=s5_ratio,vary=False)
            paras_un.add('s5_amp',value=s5_amp,vary=False)
            s5_cen=s5_info['center']
            paras_un.add('s5_cen',value=s5_cen,vary=False)
            
            #deconvolve for s6
            paras['arctan_center'].set(value=inflection(norm.Energy,norm['s6_norm_mu']),vary=False)
            paras['arctan_sigma'].set(value=1.0,min=0.0)
            s6_info=show_decon_known(norm['s6_norm_mu'],[2478.79,2484.32,2494.86])
            s6_ratio,s6_amp=s6_info['ratio'],s6_info['amp']
            paras_un.add('s6_ratio',value=s6_ratio,vary=False)
            paras_un.add('s6_amp',value=s6_amp,vary=False)
            s6_cen=s6_info['center']
            paras_un.add('s6_cen',value=s6_cen,vary=False)
            
            
            
            #allowed variance of position of peak for each known structure
            length=0.2
            #put constraints on the amplitude
            paras_un['l2_amplitude'].set(min=0.0)
            paras_un['l3_amplitude'].set(min=0.0)
            paras_un['l4_amplitude'].set(min=0.0)
            paras_un['l5_amplitude'].set(min=0.0)
            paras_un['l6_amplitude'].set(min=0.0)
            #set values of center and sigma for arctangent steps
            paras_un['atan2_center'].set(value=inflection(norm.Energy,norm['s2_norm_mu']),vary=False)
            paras_un['atan3_center'].set(value=inflection(norm.Energy,norm['s3_norm_mu']),vary=False)
            paras_un['atan4_center'].set(value=inflection(norm.Energy,norm['s4_norm_mu']),vary=False)
            paras_un['atan5_center'].set(value=inflection(norm.Energy,norm['s5_norm_mu']),vary=False)
            paras_un['atan6_center'].set(value=inflection(norm.Energy,norm['s6_norm_mu']),vary=False)
            #determine the ratio of sigma and amplitude and add that into paras
            paras_un['atan2_sigma'].set(expr='s2_ratio')
            paras_un['atan3_sigma'].set(expr='s3_ratio')
            paras_un['atan4_sigma'].set(expr='s4_ratio')
            paras_un['atan5_sigma'].set(expr='s5_ratio')
            paras_un['atan6_sigma'].set(expr='s6_ratio')
            
            #set centers and guess other attributes of Lorentzians
            paras_un.update(lor2.guess(norm['s2_norm_mu'],x=norm['Energy'],center=s2_cen))
            paras_un['l2_center'].set(value=s2_cen,min=s2_cen-length,max=s2_cen+length)
            
            paras_un.update(lor3.guess(norm['s3_norm_mu'],x=norm['Energy'],center=s3_cen))
            paras_un['l3_center'].set(value=s3_cen,min=s3_cen-length,max=s3_cen+length)
            
            paras_un.update(lor4.guess(norm['s4_norm_mu'],x=norm['Energy'],center=s4_cen))
            paras_un['l4_center'].set(value=s4_cen,min=s4_cen-length,max=s4_cen+length)
            
            paras_un.update(lor5.guess(norm['s5_norm_mu'],x=norm['Energy'],center=s5_cen))
            paras_un['l5_center'].set(value=s5_cen,min=s5_cen-length,max=s5_cen+length)
            
            paras_un.update(lor6.guess(norm['s6_norm_mu'],x=norm['Energy'],center=s6_cen))
            paras_un['l6_center'].set(value=s6_cen,min=s6_cen-length,max=s6_cen+length)
            
            #put the constraint on arc tangent steps 
            paras_un.add('tot_area',expr='l2_amplitude+l3_amplitude+l4_amplitude+l5_amplitude+l6_amplitude')
            # print(paras)
            paras_un['atan2_amplitude'].set(expr='l2_amplitude/tot_area')
            paras_un['atan3_amplitude'].set(expr='l3_amplitude/tot_area')
            paras_un['atan4_amplitude'].set(expr='l4_amplitude/tot_area')
            paras_un['atan5_amplitude'].set(expr='l5_amplitude/tot_area')
            paras_un['atan6_amplitude'].set(expr='l6_amplitude/tot_area')
            paras_un['l2_amplitude'].set(min=0.0)
            paras_un['l3_amplitude'].set(min=0.0)
            paras_un['l4_amplitude'].set(min=0.0)
            paras_un['l5_amplitude'].set(min=0.0)
            paras_un['l6_amplitude'].set(expr='s6_amp*(1-(l2_amplitude/s2_amp)-(l3_amplitude/s3_amp)-(l4_amplitude/s4_amp)-(l5_amplitude/s5_amp))',min=0.0)
            out=model.fit(norm['s1_norm_mu'],paras_un,x=norm.Energy)
            # print(out.fit_report())
            #calculate the coefficients
            c2=out.params['l2_amplitude'].value/out.params['s2_amp']
            c3=out.params['l3_amplitude'].value/out.params['s3_amp']
            c4=out.params['l4_amplitude'].value/out.params['s4_amp']
            c5=out.params['l5_amplitude'].value/out.params['s5_amp']
            c6=out.params['l6_amplitude'].value/out.params['s6_amp']
            tot_coef=c2+c3+c4+c5+c6
            c2=c2/tot_coef
            c3=c3/tot_coef
            c4=c4/tot_coef
            c5=c5/tot_coef
            c6=c6/tot_coef
            print(tot_coef)
            print(out.chisqr)
            fix_result={'c2':c2,'c3':c3,'c4':c4,'c5':c5,'c6':c6}
            print(fix_result)
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
            axes[0].plot(norm.Energy, norm['s1_norm_mu'], 'b')
            axes[0].plot(norm.Energy, out.best_fit, 'r-', label='best fit')
            axes[0].legend(loc='best')
            
            comps = out.eval_components(x=norm.Energy)
            axes[1].plot(norm.Energy, norm['s1_norm_mu'], 'b')
            axes[1].plot(norm.Energy, comps['l2_'], label='Lorentzian sdbs034')
            axes[1].plot(norm.Energy, comps['l3_'], label='Lorentzian dbso042')
            axes[1].plot(norm.Energy, comps['l4_'], label='Lorentzian sdbt029')
            axes[1].plot(norm.Energy, comps['l5_'], label='Lorentzian sfeso4051')
            axes[1].plot(norm.Energy, comps['l6_'], label='Lorentzian Sdbso2')
            axes[1].plot(norm.Energy, comps['atan2_'], label='arctangent sdbs034')
            axes[1].plot(norm.Energy, comps['atan3_'], label='arctangent dbso042')
            axes[1].plot(norm.Energy, comps['atan4_'], label='arctangent sdbt029')
            axes[1].plot(norm.Energy, comps['atan5_'], label='arctangent sfeso4051')
            axes[1].plot(norm.Energy, comps['atan6_'], label='arctangent Sdbso2')
            axes[1].legend(loc='best')
            
            
            plt.show()
            
            
            res_coef=res_coef.append({'Post-edge Energy Range':(str(energy_start)+'-'+str(energy_end)),
                                      'sdbs034(0)': c2,'sdbt029(0)': c4,'dbso042(+2)': c3,
                                      'sdbso2(+2)': c6,'sfeso4051(+6)': c5,
                                      'chi2': out.chisqr},ignore_index=True)
            
            print("--- %s seconds ---" % (time.time() - start_time))
# res_coef.to_excel(r'C:\Users\lenovo\OneDrive\桌面\res_coef.xlsx', index = False, header=True)
speak=win32com.client.Dispatch('SAPI.SPVOICE')
#play sound on this computer
# winsound.Beep(2015,3000)
# speak.Speak('Program Terminated Sucessfully. Please check the result!')

#send a desktop message to this computer
toaster = ToastNotifier()
toaster.show_toast("Program Terminated Successfully","You spend in total"+"%s seconds"% (time.time() - start_time))

#send a message to iphone pushover
msga="Time used: "+str(time.time() - start_time)+"s. "+"Python Program has calculated the coefficients using deconvolution successfully."
init("avf2ozva1ptdir7caubhvgff1sm6o2")
Client("u6x9d8dfz493h4ixf775ow3edky6jy").send_message(msga,title="Program Terminated Successfully")
