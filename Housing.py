#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:10:43 2017

@author: phil
"""

import os
import re
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import itertools

from matplotlib import pyplot as plt
import seaborn as sb
import missingno as mn

sb.set(font_scale=0.75)

##  Working Dir and Data Load
os.chdir('github/datasets')
housing = pd.read_csv('housing/housing.csv')
housing.shape

##  Initial Look at the Data
housing.info()
housing['ocean_proximity'].value_counts()
housing.hist(bins=50)
max(housing['median_house_value'])

##  Missing Values
mn.matrix(housing)
[print(s,' ',sum(housing[s].isnull())/len(housing)) for s in set(housing)]
sum(housing.total_bedrooms.isnull())/len(housing)
housing['total_bedrooms'].hist(bins=50)

##  Trimming top value as "high prices" due to forced 500k cutoff
hiPrice = housing[housing['median_house_value'] >= 500001]
housing = housing[housing['median_house_value'] < 500001]
housing.shape
housing[housing['housing_median_age'] == max(housing['housing_median_age'])].size/housing.size
housing.hist(bins=50)

##  Properties colored by median house value
housing.plot(kind='scatter',
             x='longitude',
             y='latitude',
             alpha=0.4,
             label='population',
             c='median_house_value',
             cmap=plt.get_cmap('jet'),
             colorbar=True)

housing['ocean_proximity'].value_counts()
housing[housing['ocean_proximity'] == "ISLAND"]

##  Properties colored by house median age
housing.plot(kind='scatter',
             x='longitude',
             y='latitude',
             alpha=0.4,
             label='population',
             c='housing_median_age',
             cmap=plt.get_cmap('jet'),
             colorbar=True)

##  Geo chart colored by ocean proximity
fig, ax = plt.subplots()
for cat in housing['ocean_proximity'].unique():
    ax.plot(housing['longitude'][housing['ocean_proximity']==cat],
            housing['latitude'][housing['ocean_proximity']==cat],
            marker='o', linestyle='', alpha=0.4, label=cat)
ax.legend()
plt.show()

##  Highly Priced Properties
fig, ax = plt.subplots()
for cat in hiPrice['ocean_proximity'].unique():
    ax.plot(hiPrice['longitude'][hiPrice['ocean_proximity']==cat],
            hiPrice['latitude'][hiPrice['ocean_proximity']==cat],
            marker='o', linestyle='', alpha=0.4, label=cat)
ax.legend()
plt.show()

##  Density plots of median house value separated by ocean proximity
fig, ax = plt.subplots()
for cat in housing['ocean_proximity'].unique():
    sb.distplot(housing["median_house_value"][housing["ocean_proximity"] == cat], hist=False, label=cat)
plt.legend()
plt.show()

##  pairwize t-test for house value difference between ocean proximity values
for tPair in itertools.combinations(housing['ocean_proximity'].unique(),2):
    a = np.random.choice(housing['median_house_value'][housing['ocean_proximity'] == tPair[0]], 60)
    b = np.random.choice(housing['median_house_value'][housing['ocean_proximity'] == tPair[1]], 60)
    print(tPair, " : ", np.round(ttest_ind(a,b).pvalue,2))

housing['opBool'] = [int(i != 'INLAND') for i in housing.ocean_proximity]


sb.pairplot(housing.dropna(),
            vars=['housing_median_age','total_rooms','population','median_income','median_house_value'],
            markers='+',
            kind='reg',
            hue='opBool',
            plot_kws = {'scatter_kws': {'alpha':0.1}})

##  Looking for any difference in patterns in the highly priced properties
sb.pairplot(hiPrice.dropna(), vars=['housing_median_age','total_rooms','population','median_income'])

##  Significantly higher distribution in median income.
##  Considering no real price information from the high priced properties,
##      leaving them out will give a better estimate of their actual price,
##      since median income will definitely be a factor.
##  With this in mind, a multiple regression will likely perform better than
##      a regression tree.

housing['incomeOverAge'] = housing.median_income / housing.housing_median_age
housing['roomsPerHouse'] = housing.total_rooms / housing.households
housing['ageOverPopulation'] = housing.housing_median_age / housing.population
housing['fracBeds'] = housing.total_bedrooms / housing.total_rooms

##  Log Transforms
housing['logHouseholds'] = np.log(housing.households)
housing['logIncome'] = np.log(housing.median_income)
housing['logPop'] = np.log(housing.population)
housing['logRooms'] = np.log(housing.total_rooms)
housing['logBeds'] = np.log(housing.total_bedrooms)
housing['logFB'] = np.log(housing.fracBeds)
housing['logRpH'] = np.log(housing.roomsPerHouse)

sb.heatmap(housing.drop('opBool',axis=1).dropna().corr(),
           vmin = -1,
           vmax = 1,
           cmap = 'RdBu',
           annot=True)

sb.pairplot(housing.dropna(),
            vars=sum([re.findall('log.+',s) for s in set(housing)],[]) + ['median_house_value'],
            hue='opBool')

sb.pairplot(housing.dropna(),
            vars=['incomeOverAge','roomsPerHouse','population','median_income','median_house_value'],
            hue='opBool')

##  Seaborn Pairs doesn't seem to have an alpha parameter, and I want to see more detail in the spread of the points.
housing.plot(kind='scatter',
             x='incomeOverAge',
             y='median_house_value',
             alpha=0.4,
             c='opBool',
             cmap='RdBu')

pd.pivot_table(housing,
               values=['housing_median_age','total_rooms','total_bedrooms','households','median_income','median_house_value'],
               columns='opBool',
               aggfunc=np.mean)

sb.heatmap(housing.drop(['longitude','latitude','ocean_proximity','opBool'],axis=1).dropna().corr(),
           cmap='RdBu',
           vmin=-1,
           vmax=1,
           annot=True)

housing = housing.drop(['incomeOverAge','roomsPerHouse','ageOverPopulation'], axis=1)

##  It's Go Time!
##  Select variables, Impute missing values, Stratified Random Train/Test selection, and model spot checks

housing = housing[['housing_median_age','total_rooms','median_income','opBool','fracBeds','median_house_value']]
sb.pairplot(housing.dropna(),
            vars=['housing_median_age','total_rooms','median_income','fracBeds','median_house_value'],
            hue='opBool',
            kind='reg')
sb.heatmap(housing.dropna().drop('opBool',axis=1).corr(),
           cmap = 'RdBu',
           vmin=-1,
           vmax=1,
           annot=True)

##  I need to impute some missing values for fracBeds (total_bedrooms / total_rooms)
##  A median would probably suffice for so few missing, but I want to get used to some of these regression commands


##  Stratified Test/Training set split

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for iTrain, iTest in split.split(housing, housing.opBool):
    stratTrain = housing.loc[iTrain]
    stratTest = housing.loc[iTest]

stratTrain.shape[0] + stratTest.shape[0] - housing.shape[0]

####

##  Principal Component Analysis

tPCA = PCA().fit(stratTrain.drop(['median_house_value','ocean_proximity','opBool'],axis=1).dropna())
plt.plot(np.cumsum(tPCA.explained_variance_ratio_))
plt.xlabel('Number of Components')

tPCA = PCA(n_components=2)
tProjected = tPCA.fit_transform(stratTrain.drop(['median_house_value','ocean_proximity','opBool'],axis=1).dropna())
tProjected.shape

np.linalg.eig(stratTrain.drop(['median_house_value','ocean_proximity','opBool'],axis=1).dropna().corr())

plt.scatter(tProjected[:,0], tProjected[:,1],
            c=stratTrain.dropna().median_house_value,
            cmap=plt.get_cmap('jet'),
            alpha=0.4)
plt.scatter(tProjected[:,0],
            stratTrain.dropna().median_house_value,
            c=stratTrain.dropna().opBool,
            alpha=0.4,
            cmap='RdBu')
plt.scatter(tProjected[:,1],
            stratTrain.dropna().median_house_value,
            c=stratTrain.dropna().opBool,
            alpha=0.4,
            cmap='RdBu')


def regImpute(tDF, sVarX, sVarY):
    tRegressor = LinearRegression(normalize=True)
    varX = tDF.dropna()[sVarX]
    varY = tDF.dropna()[sVarY]
    tRegressor.fit(varX,varY)
    tDF[sVarY][tDF[sVarY].isnull()] = tRegressor.predict(tDF[sVarX][tDF[sVarY].isnull()])
    return(tDF)








