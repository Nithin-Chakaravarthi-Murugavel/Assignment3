# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 04:03:06 2023

@author: nithi
"""
import wbgapi as wb
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as iter
import numpy as np
#Load the economic indicators globally
data=pd.read_csv(r"D:\Applied Data Science 1\Assignment_3\World_Data_Indicators.csv", low_memory=False)
#Information about the data
data.info()
#Data's initial rows
data.head()
#Transposing the dataset
data.transpose().head()
#Indicators to be chosen for analysis
econm = ['NE.IMP.GNFS.ZS','NY.GDP.MKTP.CD']
cntry = ["JPN","AUS",'BMU','LUX','IND','BRA','ARG','ESP','GBR','CHL']
clim=['EG.CFT.ACCS.RU.ZS','EG.CFT.ACCS.UR.ZS']
data_econm  = wb.data.DataFrame(econm, cntry, mrv=7)
data_clim  = wb.data.DataFrame(clim, cntry, mrv=7)
#NE.IMP.GNFS.ZS: Countries import total
#NY.GDP.MKTP.CD: Countries GDP in USD
#EG.CFT.ACCS.RU.ZS: Clean fuels and technologies access in rural places
#EG.CFT.ACCS.UR.ZS: Clean fuels and technologies access in urban places
# Indicators which are related to economy
data_econm.columns = [f.replace('YR','') for f in data_econm.columns]      
data_econm=data_econm.stack().unstack(level=1)                             
data_econm.index.names = ['Country_Nme', 'Year']                           
data_econm.columns                                                     
data_econm.fillna(0)
data_econm.head()
# Indicators which are related to climate
data_clim.columns = [f.replace('YR','') for f in data_clim.columns]      
data_clim=data_clim.stack().unstack(level=1)                             
data_clim.index.names = ['Country_Nme', 'Year']                           
data_clim.columns                                                     
data_clim.fillna(0)
data_clim.head()
#Cleaning the final dataset
ec1=data_econm.reset_index()
ec2=ec1.fillna(0)
cl1=data_clim.reset_index()
cl2=cl1.fillna(0)
#Preapring the final dataset
final = pd.merge(ec2, cl2)
final.head()
#K-means Clustering form
final2 = final.drop('Country_Nme', axis = 1)
kmns = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(final2)
#Clean fuels access and technologies access clustering in rural areas
sns.scatterplot(data=final, x="Country_Nme", y="EG.CFT.ACCS.RU.ZS", hue=kmns.labels_)
plt.legend(loc='lower right')
plt.title("Rural Areas")
plt.show()
#Scatter plot - Association between Total imports and Total GDP of a country(Brazil)
b=final[(final['Country_Nme']=='BRA')]
txt = b.values
x, y = txt[:, 2], txt[:, 3]
plt.scatter(x, y,color="green")
plt.title('GDP vs Total Imports (Brazil)')
plt.ylabel('Total Imports')
plt.xlabel('Total GDP of a country')
plt.show()