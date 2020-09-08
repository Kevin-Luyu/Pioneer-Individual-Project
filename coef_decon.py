# -*- coding: utf-8 -*-
import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from lmfit.models import StepModel, LorentzianModel
import win32com.client
import winsound
from win10toast import ToastNotifier
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx')
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

arctan_mod=StepModel(form='atan',prefix='arctan_')
paras=arctan_mod.make_params()
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
    paras[pref+'amplitude'].set(min=0)
    paras[pref+'sigma'].set(min=0)
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
    paras=arctan_mod.make_params()
    paras['arctan_center'].set(value=inflection(norm.Energy,df),vary=False,min=0)
    paras['arctan_amplitude'].set(value=1.0,vary=False)
    paras['arctan_sigma'].set(value=1.0,min=0)
    mod=arctan_mod
    for i in range(len(center)):
        this=make_lor(df,i,center,2.0)['model']
        mod=mod+this
        paras.update(make_lor(df,i,center,2.0)['paras'])
    out=mod.fit(df,params=paras,x=norm.Energy)
    return out

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
    out=decon_known(df,center)
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

# deconvolute for s2
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s2_norm_mu']),vary=False)
paras['arctan_amplitude'].set(value=1.0,vary=False)
paras['arctan_sigma'].set(value=1.0,min=0)
show_decon_known(norm['s2_norm_mu'],[2472.5,2475.4,2478.9,2483.2])

#deconvolve for s3
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s3_norm_mu']),vary=False)
paras['arctan_sigma'].set(value=1.0,min=0.01)
show_decon_known(norm['s3_norm_mu'],[2475.26,2478.20,2489.90,2484.52])
# s=stats.chisquare(norm['s3_norm_mu'],f_exp=decon_known(norm['s3_norm_mu'],[2475.26,2478.20,2489.90,2484.52]))

# print(s)
"""
#deconvolve for s4
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s4_norm_mu']),vary=False)
paras['arctan_sigma'].set(value=1.0,min=0)
show_decon_known(norm['s4_norm_mu'],[2473.05,2476.22,2481.59])

#deconvolve for s5
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s5_norm_mu']),vary=False)
paras['arctan_sigma'].set(value=1.0,min=0)
show_decon_known(norm['s5_norm_mu'],[2481.76,2497.41])

#deconvolve for s6
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s6_norm_mu']),vary=False)
paras['arctan_sigma'].set(value=1.0,min=0)
show_decon_known(norm['s6_norm_mu'],[2478.79,2484.32,2494.86])
"""
plt.show()
toaster = ToastNotifier()
msga="Time used: "+str(time.time() - start_time)+"s. "+"Python Program has calculated the coefficients using deconvolution successfully."
toaster.show_toast("Program Terminated Successfully",msga)