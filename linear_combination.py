import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
#create an empty dataframe to store chisq of all possible combinations
column_names = ["c2",'c3','c4','c5',"chisq"]
df = pd.DataFrame(columns = column_names)
# find the minimum element larger than 1, return the idex
def find_min_position(array):
    plus_array = [elem for elem in array]
    min_elem = min(plus_array)
    return array.index(min_elem)
# define the step(1/tot), center of range, and lenth of range
tot=7
center2=np.float64(0.5)
center3=np.float64(0.5)
center4=np.float64(0.5)
center5=np.float64(0.5)
hfrange=np.float64(0.5)
#iterate over (0,1), then adjust the center and range accordingly to approximate
for t in range (5):
    for c5 in np.linspace(max([0,center5-hfrange]),center5+hfrange,tot):
        for c4 in np.linspace(max([0,center4-hfrange]),center4+hfrange,tot):
            for c3 in np.linspace(max([0,center3-hfrange]),center3+hfrange,tot):  
                for c2 in np.linspace(max([0,center2-hfrange]),center2+hfrange,tot):
                    #times 1000 because chi2 test is invalid when data are too small
                    actual=norm['s1_norm_mu']
                    predicted=c2*norm['s2_norm_mu']+c3*norm['s3_norm_mu']+c4*norm['s4_norm_mu']+c5*norm['s5_norm_mu']
                    predicted=predicted
                    #remember to take the absolute value to prevent those small negative values
                    xsq=sp.stats.chisquare(np.absolute(actual),f_exp=np.absolute(predicted))[0]
                    df = df.append({'c2' : c2, 'c3':c3,'c4':c4,'c5':c5,'chisq' : xsq},  
                                ignore_index = True)         
    indexmin=find_min_position(df['chisq'].tolist())
    print ("c2= "+str(df.c2[indexmin])+"c3= "+str(df.c3[indexmin])+"c4= "+str(df.c4[indexmin])+"c5= "+str(df.c5[indexmin])+"chisq="+str(df.chisq[indexmin]))
    center2=df.c2[indexmin]
    center3=df.c3[indexmin]
    center4=df.c4[indexmin]
    center5=df.c5[indexmin]
    hfrange=hfrange/3
#forcing the sum to be 1
total=center2+center3+center4+center5
center2=center2/total
center3=center3/total
center4=center4/total
center5=center5/total
actual=norm['s1_norm_mu']
predicted=center2*norm['s2_norm_mu']+center3*norm['s3_norm_mu']+center4*norm['s4_norm_mu']+center5*norm['s5_norm_mu']
predicted=predicted
print(sp.stats.chisquare(np.absolute(actual),f_exp=np.absolute(predicted),ddof=318))
plt.plot(norm['Energy'],actual,label="actual")
plt.plot(norm['Energy'],predicted,label='predicted')
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()