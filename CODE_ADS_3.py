#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:13:36 2023

@author: mukeshavudaiappan
"""

# Importing required libraries
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter

# Inserting the indicator IDs for World Development Indicators(WDI) dataset
indicator1 = ["EN.ATM.CO2E.PC", "EG.USE.ELEC.KH.PC"]
indicator2 = ["EN.ATM.METH.KT.CE", "EG.ELC.ACCS.ZS"]

# Selecting country codes representing the countries of interest
country_code = ['AUS', 'GBR', 'CAN', 'IND', 'MYS']

# Read func returns data for most recent 30 yrs for each indicator & country


def read(indicator, country_code):
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df


# Reads a CSV file with CO2 emissions data and returns a pandas DataFrame
file = "co2 emission.csv"

# Function to read indicator1 and country_code
data = read(indicator1, country_code)

# Preprocessing data by removing 'YR' suffix from column & giving new index
data.columns = [i.replace('YR', '') for i in data.columns]
data = data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
data.columns

# creating another dataframe
data1 = read(indicator2, country_code)

# removing YR and giving index names to data1
data1.columns = [i.replace('YR', '') for i in data1.columns]
data1 = data1.stack().unstack(level=1)
data1.index.names = ['Country', 'Year']
data1.columns

# creating indices for dt1 and dt2
dt1 = data.reset_index()
dt2 = data1.reset_index()
dt = pd.merge(dt1, dt2)
dt

# dropping the column
dt.drop(['EG.USE.ELEC.KH.PC'], axis=1, inplace=True)
dt.drop(['EG.ELC.ACCS.ZS'], axis=1, inplace=True)
dt
dt["Year"] = pd.to_numeric(dt["Year"])

# function to normalise the data


def norm_df(df):
    y = df.iloc[:, 2:]
    df.iloc[:, 2:] = (y-y.min()) / (y.max() - y.min())
    return df


dt_norm = norm_df(dt)
df_fit = dt_norm.drop('Country', axis=1)
k = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df_fit)
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC",
                palette='magma', hue=k.labels_, alpha=0.7)  # increase transparency
plt.xticks(rotation=50)  # rotate x-axis labels
plt.xlabel('Country')
plt.ylabel('CO2 emissions (normalized)')
plt.title('Distribution of CO2 emissions across countries')
plt.grid(True)
plt.legend()
plt.tight_layout()  # adjust spacing between plot elements
plt.savefig("plot.png")
plt.show()

# function to find the error


def err_ranges(x, func, param, sigma):

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []

# list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


dt1 = dt[(dt['Country'] == 'IND')]
dt1

# curve fitting for India
val = dt1.values
x, y = val[:, 1], val[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x, y)
dt1["pop_log"] = fct(x, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x, dt1["pop_log"], label="Fit", color="red")
plt.plot(x, y, label="Data", color="blue")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in INDIA")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.savefig("Ind.png")
plt.show()

# extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)

# finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt2 = dt[(dt['Country'] == 'CAN')]
dt2

# curve fitting for Canada
val2 = dt2.values
x2, y2 = val2[:, 1], val2[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x2, y2)
dt2["pop_log"] = fct(x2, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x2, dt2["pop_log"], label="Fit", color="red")
plt.plot(x2, y2, label="Data", color="blue")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Canada")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.savefig("Can.png")
plt.show()

# extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x2, fct, prmet, sigma)

# finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt3 = dt[(dt['Country'] == 'GBR')]
dt3

# curve fitting for UK
val3 = dt3.values
x3, y3 = val3[:, 1], val3[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x3, y3)
dt3["pop_log"] = fct(x3, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x3, dt3["pop_log"], label="Fit", color="red")
plt.plot(x3, y3, label="Data", color="blue")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Great Britain - UK")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.savefig("UK.png")
plt.show()

# extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x3, fct, prmet, sigma)

# finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
