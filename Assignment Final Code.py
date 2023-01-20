# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 01:22:46 2023

@author: nithi
"""
# Numpy is used for mathamatical operation
import numpy as np
# Pandas is used to read the data
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
# Matplotlib.pyplot is used to show the data in graph
import matplotlib.pyplot as plt
import seaborn as sns
# Wbgapi is best while using world bank data
import wbgapi as wb
import itertools as iter
# Read the data of economic indicators 
data = pd.read_csv(r"D:\Applied Data Science 1\Assignment_3\World_Data_Indicators.csv", low_memory=False)
#Information about the data
data.info()
# Data's initial rows
data.head()
# Transposing the dataset
data.transpose().head()
# Indicators to be chosen for analysis
#NE.IMP.GNFS.ZS: Countries import total
#NY.GDP.MKTP.CD: Countries GDP in USD
econm = ['NE.IMP.GNFS.ZS','NY.GDP.MKTP.CD']
cntry = ["JPN","AUS",'BMU','LUX','IND','BRA','ARG','ESP','GBR','CHL']
#EG.CFT.ACCS.RU.ZS: Clean fuels and technologies access in rural places
#EG.CFT.ACCS.UR.ZS: Clean fuels and technologies access in urban places
clim=['EG.CFT.ACCS.RU.ZS','EG.CFT.ACCS.UR.ZS']
data_econm  = wb.data.DataFrame(econm, cntry, mrv=7)
data_clim  = wb.data.DataFrame(clim, cntry, mrv=7)
# Indicators which are related to economy
data_econm.columns = [f.replace('YR','') for f in data_econm.columns]  
# Stack().unstack is used to reshape the data of ecconomy    
data_econm = data_econm.stack().unstack(level=1)                             
data_econm.index.names = ['Country_Nme', 'Year']                           
data_econm.columns                                                     
data_econm.fillna(0)
data_econm.head()
# Indicators which are related to climate
data_clim.columns = [f.replace('YR','') for f in data_clim.columns]      
data_clim = data_clim.stack().unstack(level=1)                             
data_clim.index.names = ['Country_Nme', 'Year']                           
data_clim.columns                                                     
data_clim.fillna(0)
data_clim.head()
#Cleaning the final dataset
ec1 = data_econm.reset_index()
ec2 = ec1.fillna(0)
cl1 = data_clim.reset_index()
cl2 = cl1.fillna(0)
#Preapring the final dataset
final = pd.merge(ec2, cl2)
final.head()
#Scaling the final data
final1 = final.iloc[:,2:]
final.iloc[:,2:] = (final1-final1.min())/ (final1.max() - final1.min())
final.head()
#Scatter plot - Association between Total imports and Total GDP of a country(Brazil)
b = final[(final['Country_Nme']=='BRA')]
txt = b.values
x, y = txt[:, 2], txt[:, 3]
plt.scatter(x, y,color="green")
plt.title('GDP vs Total Imports (Brazil)')
plt.ylabel('Total Imports')
plt.xlabel('Total GDP of a country')
plt.show()
#K-means Clustering form
# drop is used to remove specific column
final2 = final.drop('Country_Nme', axis = 1)
kmns = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(final2)
#Clean fuels access and technologies access clustering in rural areas
sns.scatterplot(data=final, x="Country_Nme", y="EG.CFT.ACCS.RU.ZS", hue=kmns.labels_)
plt.legend(loc='lower right')
plt.title("Rural Areas")
plt.show()
#curve_fit function implementation for association between GDP and total imports - Country with low rural access to fuel and technology
l = final[(final['Country_Nme']=='IND')]
txt3 = l.values
x, y = txt3[:, 2], txt3[:, 3]
# Defining function for low rural country so that it can process when its called
def Low_Rural_functn(x, a, b, c):
    return a*x**2+b*x+c
param, covar = curve_fit(Low_Rural_functn, x, y)
param, _ = curve_fit(Low_Rural_functn, x, y)
print("Function Covariance ->", covar)
print("Function Parameters ->", param)
a, b, c = param[0], param[1], param[2]
y_fit = a*x**2+b*x+c
import warnings
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fit, label="y=a*x**2+b*x+c",color="green")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Original Y",color="red")
    plt.ylabel('Total Imports')
    plt.title('GDP vs Total Imports India (Low Rural)')
    plt.xlabel('Total GDP of a country')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show() 
#curve_fit function implementation for association between GDP and total imports - Country with medium rural access to fuel and technology
m = final[(final['Country_Nme']=='BRA')]
txt2 = m.values
x, y = txt2[:, 2], txt2[:, 3]
# Defining function for Medium rural country so that it can process when its called
def Medium_Rural_functn(x, a, b, c):
    return a*x**2+b*x+c
param, covar = curve_fit(Medium_Rural_functn, x, y)
param, _ = curve_fit(Medium_Rural_functn, x, y)
print("Function Covariance ->", covar)
print("Function Parameters ->", param)
a, b, c = param[0], param[1], param[2]
y_fit =a*x**2+b*x+c
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fit, label="y=a*x**2+b*x+c",color="green")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Original Y",color="red")
    plt.ylabel('Total Imports')
    plt.title('GDP vs Total Imports Brazil(Medium Rural)')
    plt.xlabel('Total GDP of a country')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show() 
#curve_fit function implementation for association between GDP and total imports - Country with higher rural access to fuel and technology
k = final[(final['Country_Nme']=='LUX')]
txt1 = k.values
x, y = txt1[:, 2], txt1[:, 3]
# Defining function for low rural country so that it can process when its called
def High_Rural_functn(x, a, b, c):
    return a*x**2+b*x+c
# Curve_fit() is used to for optimization problem that finds a line that best fits a collection of observations
param, covar = curve_fit(High_Rural_functn, x, y)
param, _ = curve_fit(High_Rural_functn, x, y)
print("Function Covariance ->", covar)
print("Function Parameters ->", param)
a, b, c = param[0], param[1], param[2]
y_fit = a*x**2+b*x+c
with warnings.catch_warnings(record=True):
    plt.plot(x, y_fit, label="y=a*x**2+b*x+c",color="green")
    plt.grid(True)
    plt.plot(x, y, 'bo', label="Original Y",color="red")
    plt.ylabel('Total Imports')
    plt.title('GDP vs Total Imports Luxembourg (High Rural)')
    plt.xlabel('Total GDP of a country')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show() 
#Error range function definition
def err_ranges(x, func, param, sigma):
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))   
    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)  
    return lower, upper