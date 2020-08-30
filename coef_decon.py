# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from lmfit.models import StepModel, LorentzianModel
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

def atan_model(x,amplitude,center,length):
    """

    Parameters
    ----------
    x : numpy array
        the energy array.
    amplitude : float
        the amplitude of the model.
    center : float
        the center of the peak.
    length : float
        characteristic widths of the model (half-length of half-peak).

    Returns
    -------
    model : numpy array
        the peak array.

    """
    model=1*((1/np.pi)*np.arctan((x-center)/length)+1/2)
    return model
def atan_known(energy,df):
    """
    

    Parameters
    ----------
    energy : numpy array
        the energy array.
    df : numpy array
        the array records linear absorption coef of a known sample.

    Returns
    -------
    numpy array
        the arctangent step function associate with a known function.

    """
    center=inflection(energy,df)
    return atan_model(energy,1.0,center,1.0)

"""norm['s2_norm_mu']=norm['s2_norm_mu']-atan_known(norm.Energy,norm['s2_norm_mu'])
norm['s3_norm_mu']=norm['s3_norm_mu']-atan_known(norm.Energy,norm['s3_norm_mu'])
norm['s4_norm_mu']=norm['s4_norm_mu']-atan_known(norm.Energy,norm['s4_norm_mu'])
norm['s5_norm_mu']=norm['s5_norm_mu']-atan_known(norm.Energy,norm['s5_norm_mu'])
norm['s6_norm_mu']=norm['s6_norm_mu']-atan_known(norm.Energy,norm['s6_norm_mu'])
"""

#deconvolute for s2
arctan_mod=StepModel(form='atan',prefix='arctan_')
paras=arctan_mod.make_params()
paras['arctan_center'].set(value=inflection(norm.Energy,norm['s2_norm_mu']),vary=False)
paras['arctan_amplitude'].set(value=1.0,vary=False)
paras['arctan_sigma'].set(value=1.0)

lor1=LorentzianModel(prefix='l1_')
paras.update(lor1.guess(norm['s2_norm_mu'],x=norm['Energy'],center=2472.5))
#allowed variance of peak position
length=1.0
paras['l1_center'].set(value=2472.5,min=2472.5-length,max=2472.5+length)

lor2=LorentzianModel(prefix='l2_')
paras.update(lor2.guess(norm['s2_norm_mu'],x=norm['Energy'],center=2475.4))
paras['l2_center'].set(value=2475.4,min=2475.4-length,max=2475.4+length)

lor3=LorentzianModel(prefix='l3_')
paras.update(lor3.guess(norm['s2_norm_mu'],x=norm['Energy'],center=2478.9))

lor4=LorentzianModel(prefix='l4_')
paras.update(lor4.guess(norm['s2_norm_mu'],x=norm['Energy'],center=2483.2))

mod=arctan_mod+lor1+lor2+lor3+lor4
init = mod.eval(paras, x=norm.Energy)
out=mod.fit(norm['s2_norm_mu'],paras,x=norm.Energy)
print(out.fit_report())

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
axes[0].plot(norm.Energy, norm['s2_norm_mu'], 'b')
axes[0].plot(norm.Energy, init, 'k--', label='initial fit')
axes[0].plot(norm.Energy, out.best_fit, 'r-', label='best fit')
axes[0].legend(loc='best')

comps = out.eval_components(x=norm.Energy)
axes[1].plot(norm.Energy, norm['s2_norm_mu'], 'b')
axes[1].plot(norm.Energy, comps['l1_'], label='Lorentzian component 1')
axes[1].plot(norm.Energy, comps['l2_'], label='Lorentzian component 2')
axes[1].plot(norm.Energy, comps['l3_'], label='Lorentzian component 3')
axes[1].plot(norm.Energy, comps['l4_'], label='Lorentzian component 4')
axes[1].plot(norm.Energy, comps['arctan_'], label='arctangent component')
axes[1].legend(loc='best')

plt.show()
