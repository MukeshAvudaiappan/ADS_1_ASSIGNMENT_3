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
df_1 = read(indicator1, country_code)

# Preprocessing data by removing 'YR' suffix from column & giving new index
df_1.columns = [i.replace('YR', '') for i in df_1.columns]
df_1 = df_1.stack().unstack(level=1)
df_1.index.names = ['Country', 'Year']
df_1.columns

# Funtion to read indicator2 and country_code
df_2 = read(indicator2, country_code)

# Removing YR and giving index names to df_2
df_2.columns = [i.replace('YR', '') for i in df_2.columns]
df_2 = df_2.stack().unstack(level=1)
df_2.index.names = ['Country', 'Year']
df_2.columns

# Creating indices for dt1 and dt2
dt1 = df_1.reset_index()
dt2 = df_2.reset_index()
dt = pd.merge(dt1, dt2)
dt

# Dropping the column
dt.drop(['EG.USE.ELEC.KH.PC'], axis=1, inplace=True)
dt.drop(['EG.ELC.ACCS.ZS'], axis=1, inplace=True)
dt
dt["Year"] = pd.to_numeric(dt["Year"])

# Function to return normalise DataFrame


def norm_df(df):
    y = df.iloc[:, 2:]
    df.iloc[:, 2:] = (y-y.min()) / (y.max() - y.min())
    return df


dt_norm = norm_df(dt)
df_fit = dt_norm.drop('Country', axis=1)
k = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df_fit)
sns.set_style("whitegrid")
sns.violinplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC",
               hue=k.labels_, split=True, inner="stick",
               bw=.2, cut=1, linewidth=1, alpha=.7)
plt.xticks(rotation=50)
plt.xlabel('Country')
plt.ylabel('CO2 emissions (normalized)')
plt.title('Distribution of CO2 emissions across countries')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='best', fancybox=True, shadow=True, fontsize=10)
ax = plt.gca()
ax.set_facecolor('#f5f5f5')
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()

# Function to find the error


def err_ranges(x, func, param, sigma):

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    uplow = []

# List to hold upper and lower limits for parameters
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


dt1 = dt[(dt['Country'] == 'AUS')]
dt1

# Curve fitting for Australia
val = dt1.values
x, y = val[:, 1], val[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x, y)
dt1["pop_log"] = fct(x, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
sns.set_theme(style="whitegrid")
sns.lineplot(data=dt1, x=x, y="pop_log", label="Fit",
             palette="magma", linewidth=2, marker=".", markersize=10)
sns.lineplot(x=x, y=y, label="Data", palette="magma", linewidth=1.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 Emission Rate in Australia")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='best', fancybox=True, shadow=True, fontsize=10)
ax = plt.gca()
ax.set_facecolor('#f5f5f5')
plt.tight_layout()
plt.savefig("AUS.png", dpi=300)
plt.show()


# Extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)

# Finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt2 = dt[(dt['Country'] == 'IND')]
dt2

# Curve fitting for India
val2 = dt2.values
x2, y2 = val2[:, 1], val2[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x2, y2)
dt2["pop_log"] = fct(x2, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
sns.set_theme(style="whitegrid")
sns.lineplot(data=dt2, x=x2, y="pop_log", label="Fit",
             palette="viridis", linewidth=2, marker=".", markersize=10)
sns.lineplot(x=x2, y=y2, label="Data", palette="viridis", linewidth=1.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in India")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='best', fancybox=True, shadow=True, fontsize=10)
ax = plt.gca()
ax.set_facecolor('#f5f5f5')
plt.tight_layout()
plt.savefig("IND.png", dpi=300)
plt.show()

# Extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x2, fct, prmet, sigma)

# Finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
dt3 = dt[(dt['Country'] == 'GBR')]
dt3

# Curve fitting for UK
val3 = dt3.values
x3, y3 = val3[:, 1], val3[:, 2]


def fct(x, a, b, c):
    return a*x**2+b*x+c


prmet, cov = opt.curve_fit(fct, x3, y3)
dt3["pop_log"] = fct(x3, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
sns.set_theme(style="whitegrid")
sns.lineplot(x=x3, y=dt3["pop_log"], label="Fit",
             palette="viridis", linewidth=2, marker=".", markersize=10)
sns.lineplot(x=x3, y=y3, label="Data", palette="viridis", linewidth=1.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Great Britain - UK")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='best', fancybox=True, shadow=True, fontsize=10)
ax = plt.gca()
ax.set_facecolor('#f5f5f5')
plt.tight_layout()
plt.savefig("UK.png", dpi=300)
plt.show()

# Extracting the sigma
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x3, fct, prmet, sigma)

# Finding the emission rate in the coming 10 years
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fct, prmet, sigma)
print("2030 between", low, "and", up)
