# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:12:59 2023

@author: Yousha
"""

## DROP NOT RATED RESTAURANTS

import pandas as pd
from dython import nominal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 100)

df = pd.read_csv('zomato.csv',encoding=('ISO-8859-1'))
c = pd.read_csv('Country-Code.csv')
df = pd.merge(df,c,on='Country Code')

df.head()
df.dtypes
df.columns

df.plot.scatter(x='Aggregate rating',y='Votes')

df.isnull().sum()
df.drop('Switch to order menu', axis=1,inplace = True)
df = df.dropna()

########
unrated = df[df['Rating text'] == 'Not rated'].copy()
df.drop(unrated.index, inplace=True)
#######

df['Aggregate rating'].value_counts()
df['Restaurant Name'].value_counts().head(10)
df.Locality.value_counts()

df['Has Table booking'].value_counts()
df['Has Online delivery'].head()
df['Is delivering now'].head()
df.City.value_counts()
df.Currency.value_counts()
df['Country Code'].value_counts()

## CUISINES

df['cuisine_list'] = df.Cuisines.apply(lambda x: x.lower().replace(' ','')\
                                       .strip())

    
# def cuisine_kw(**kwargs):           ## NOICE ##
#     for key, value in kwargs.items():
#         df[str(key)] = df.cuisine_list.apply(lambda x: value in x)

# cuisine_kw(north_indian='northindian',chinese='chinese',\
#            south_indian='south_indian',seafood='seafood',\
#            japanese='japanese',american='american',fast_food='fast_food',\
#            cafe='cafe',afghani='afghani',desserts='desserts')

df['north_indian'].value_counts()
cuisine = df.cuisine_list.value_counts().head(10)

for i in cuisine.index:
    df[i.replace(',','_')] = df.cuisine_list.apply(lambda x: i in x)

## What i was doing wrong was looking at the SERIES, not the ELEMENTS in the series.
## here, x means objects in cuisine_list.

def corr_cat(x):
    import warnings
    warnings.filterwarnings("ignore")
    from dython import nominal
    nominal.associations(x,figsize=(40,20),mark_columns=True,\
                         display_columns='Aggregate rating') # Categorical Correlation matrix
    plt.show()

corr_cat(df)

df.columns
df.drop([ 'bakery', 'bakery_desserts'],axis=1,inplace=True)

# localities --done--
# countries --done--
# locality verbose --done--
# currency --done--
# price range --done--
# longitude --done--
# latitude --
# address --done
# city -done-
# restaurant name -done-

def dum_col(x):
    return x.strip().lower().replace(' ','_')

def dummy(lst,column):
    for i in lst.index:
        df[dum_col(i)] = df[column].apply(lambda x: i in x)

## Restaurants
restaurants = df['Restaurant Name'].value_counts().head(10)
dummy(restaurants,'Restaurant Name')

#corr_cat(df)

#df.columns
df.drop(["mcdonald's", 'green_chick_chop', 'pizza_hut', 'keventers','giani', 'barista'],\
        axis=1,inplace=True)
## Cities
cities = df['City'].value_counts().head(10)
dummy(cities,'City')

#corr_cat(df)

df.drop(['ghaziabad', 'amritsar'],axis=1,inplace=True)

## Locality
local = df['Locality'].value_counts().head(10)
dummy(local,'Locality')
#corr_cat(df)

df.drop(['rajouri_garden', 'malviya_nagar','defence_colony', 
         'satyaniketan', 'pitampura', 
         'sector_18', 'karol_bagh'],axis=1,inplace=True)

## Country
# country = df['Country'].value_counts()
# dummy(country,'Country') ## Good correlations

#corr_cat(df)
# df.drop(['australia','singapore','canada','sri_lanka'],axis=1,inplace=True)

## Locality Verbose
# loc_verb = df['Locality Verbose'].value_counts().head(10)
# dummy(loc_verb,'Locality Verbose') ## Bad corrs

## Currency
# currency = df['Currency'].value_counts()
# dummy(currency,'Currency') ## Good

# corr_cat(df)
# df.drop(['qatari_rial(qr)', 'sri_lankan_rupee(lkr)'],axis=1,inplace=True)

## Price Range
# price = df['Price range'].value_counts()
# price = price.apply(lambda x: str(x))
# df['Price range'] = df['Price range'].apply(lambda x: str(x))
# dummy(price,'Price range') ## Good and worth it

## Address
# address = df['Address'].value_counts().head(10)
# dummy(address,'Address') ## Bad

## Longitude
# longit = df['Longitude'].value_counts().head(10)
# longit = longit.apply(lambda x: str(x))
# df['Longitude'] = df['Longitude'].apply(lambda x: str(x))
# dummy(longit,'Longitude') ## Bad

## Latitude
# lat = df['Latitude'].value_counts().head(10)
# lat = lat.apply(lambda x: str(x))
# df['Latitude'] = df['Latitude'].apply(lambda x: str(x))
# dummy(lat,'Latitude') ## Bad

corr_cat(df)


## EXPORTING ##

df['aggregate_rating'] = df['Aggregate rating']
df.drop('Aggregate rating',axis=1,inplace=True)
df.columns
df.dtypes

df.to_csv('Zomato final 3.csv',index=None)
























