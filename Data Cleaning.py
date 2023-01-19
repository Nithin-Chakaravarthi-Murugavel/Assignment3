# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 04:03:06 2023

@author: nithi
"""
import wbgapi as wb
import pandas as pd
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