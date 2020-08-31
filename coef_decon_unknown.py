# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from lmfit.models import StepModel, LorentzianModel
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\norm.xlsx')
#allowed variance of position of peak for each known structure
length=0.3
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
paras=model.make_params()
#set values of center and sigma for arctangent steps
paras['atan2_center'].set(value=2471.8,vary=False)
paras['atan3_center'].set(value=2474.647,vary=False)
paras['atan4_center'].set(value=2472.553,vary=False)
paras['atan5_center'].set(value=2480.952,vary=False)
paras['atan6_center'].set(value=2477.498,vary=False)
paras['atan2_sigma'].set(expr='0.25959158*atan2_amplitude')
paras['atan3_sigma'].set(expr='0.02740968*atan3_amplitude')
paras['atan4_sigma'].set(expr='0.5073644*atan4_amplitude')
paras['atan5_sigma'].set(expr='(3.55*10**(-12)) *atan5_amplitude')
paras['atan6_sigma'].set(expr='0.0432*atan6_amplitude')

#set centers and guess other attributes of Lorentzians
paras.update(lor2.guess(norm['s2_norm_mu'],x=norm['Energy'],center=2472.46340))
paras['l2_center'].set(value=2472.46340,min=2472.46340-length,max=2472.46340+length)

paras.update(lor3.guess(norm['s3_norm_mu'],x=norm['Energy'],center=2475.22029))
paras['l3_center'].set(value=2475.22029,min=2475.22029-length,max=2475.22029+length)

paras.update(lor4.guess(norm['s4_norm_mu'],x=norm['Energy'],center=2473.02235))
paras['l4_center'].set(value=2473.02235,min=2473.02235-length,max=2473.02235+length)

paras.update(lor5.guess(norm['s5_norm_mu'],x=norm['Energy'],center=2481.72013))
paras['l5_center'].set(value=2481.72013,min=2481.72013-length,max=2481.72013+length)

paras.update(lor6.guess(norm['s6_norm_mu'],x=norm['Energy'],center=2478.62882))
paras['l6_center'].set(value=2478.62882,min=2478.62882-length,max=2478.62882+length)

#put the constraint on arc tangent steps 
tot_area=paras['l2_amplitude'].value*paras['l2_sigma'].value
tot_area+=paras['l3_amplitude'].value*paras['l3_sigma'].value
tot_area+=paras['l4_amplitude'].value*paras['l4_sigma'].value
tot_area+=paras['l5_amplitude'].value*paras['l5_sigma'].value
tot_area+=paras['l6_amplitude'].value*paras['l6_sigma'].value
paras.add('tot_area',value=tot_area)
paras['atan2_amplitude'].set(expr='l2_amplitude*l2_sigma/tot_area')
paras['atan3_amplitude'].set(expr='l3_amplitude*l3_sigma/tot_area')
paras['atan4_amplitude'].set(expr='l4_amplitude*l4_sigma/tot_area')
paras['atan5_amplitude'].set(expr='l5_amplitude*l5_sigma/tot_area')
paras['atan6_amplitude'].set(expr='l6_amplitude*l6_sigma/tot_area')

out=model.fit(norm['s1_norm_mu'],paras,x=norm.Energy)
print(out.fit_report())

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