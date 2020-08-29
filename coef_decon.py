# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from lmfit import models
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
norm['s2_norm_mu']=norm['s2_norm_mu']-atan_known(norm.Energy,norm['s2_norm_mu'])
norm['s3_norm_mu']=norm['s3_norm_mu']-atan_known(norm.Energy,norm['s3_norm_mu'])
norm['s4_norm_mu']=norm['s4_norm_mu']-atan_known(norm.Energy,norm['s4_norm_mu'])
norm['s5_norm_mu']=norm['s5_norm_mu']-atan_known(norm.Energy,norm['s5_norm_mu'])
norm['s6_norm_mu']=norm['s6_norm_mu']-atan_known(norm.Energy,norm['s6_norm_mu'])


# norm.to_excel(r'C:\Users\lenovo\OneDrive\桌面\atannorm.xlsx', index = False, header=True)