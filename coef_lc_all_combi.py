"""
The program uses the most preliminary method to find the optimal c. After taking 
all possible combinations of c, the corresponding values are stored in a list.
Then the optimal value is found, and another round of program starts, with range
halved and center change to the previous optimal value.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#read normalized, merged data
norm = pd.read_excel (r'C:\Users\lenovo\OneDrive\桌面\Physics\Pioneer Academics\Independent Project\Data\norm.xlsx')
#create an empty dataframe to store chisq of all possible combinations
column_names = ["c2",'c3','c4','c5',"chisq"]
all_cases = pd.DataFrame(columns = column_names)
# define the initial center of range and lenth of range
tot=5
center2=np.float64(0.5)
center3=np.float64(0.5)
center4=np.float64(0.5)
center5=np.float64(0.5)
hfrange=np.float64(0.5)
#iterate over (0,1), then adjust the center and range accordingly in the next round
#but constrain the range of coef smaller than [0,1]
for t in range (5):
    for c5 in np.linspace(max([0,center5-hfrange]),min([1,center5+hfrange]),tot):
        for c4 in np.linspace(max([0,center4-hfrange]),min([1,center4+hfrange]),tot):
            for c3 in np.linspace(max([0,center3-hfrange]),min([1,center3+hfrange]),tot):  
                for c2 in np.linspace(max([0,center2-hfrange]),min([1,center2+hfrange]),tot):
                    #forcing the sum to be 1
                    total=center2+center3+center4+center5
                    center2=center2/total
                    center3=center3/total
                    center4=center4/total
                    center5=center5/total
                    #define the predicted as the linear combination
                    actual=norm['s1_norm_mu']
                    predicted=(c2*norm['s2_norm_mu']+c3*norm['s3_norm_mu']+
                               c4*norm['s4_norm_mu']+c5*norm['s5_norm_mu'])
                    #take the absolute value to prevent those small negative values
                    #otherwise, the chi2 is significantly affected by noise
                    xsq=stats.chisquare(np.absolute(actual),f_exp=np.absolute(predicted))[0]
                    all_cases = all_cases.append({'c2' : c2, 'c3':c3,'c4':c4,'c5':c5,'chisq' : xsq},  
                                ignore_index = True)         
    indexmin=all_cases['chisq'].idxmin(axis=1,skipna=True)
    #update new center in the next round
    hfrange=hfrange/3
    center2=all_cases.c2[indexmin]
    center3=all_cases.c3[indexmin]
    center4=all_cases.c4[indexmin]
    center5=all_cases.c5[indexmin]
    print ("c2= "+str(center2)+" c3= "+str(center3)+
           " c4= "+str(center4)+" c5= "+str(center5)+
           " chisq= "+str(all_cases.chisq[indexmin]))
    print()
#plot to compare the predicted and actual figure
predicted=center2*norm['s2_norm_mu']+center3*norm['s3_norm_mu']+center4*norm['s4_norm_mu']+center5*norm['s5_norm_mu']
plt.plot(norm['Energy'],actual,label="actual")
plt.plot(norm['Energy'],predicted,label='predicted')
plt.legend(loc="upper right")
plt.set_xlabel="energy"
plt.set_ylabel="absorption"
plt.show()