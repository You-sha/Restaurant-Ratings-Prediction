# <p align="center"> Restaurant Rating Prediction </p>

<p align="center">Predicting the aggregate <b>rating</b> of Zomato restaurants using <b>Machine Learning</b>.</p>

<p align="center"> Tools used: Python (<b>Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, Dython</b>)</p>

#

**Sections:**
* [Data Analysis and Cleaning](https://github.com/You-sha/Restaurant-Ratings-Prediction#-data-analysis-and-cleaning-)
* [Feature Engineering and Preprocessing](https://github.com/You-sha/Restaurant-Ratings-Prediction#-feature-engineering-and-preprocessing-)
* [Model building and Tuning](https://github.com/You-sha/Zomato-Ratings-Prediction#model-building-and-tuning)
* [Results](https://github.com/You-sha/Zomato-Ratings-Prediction#-results-)

#

First, we import the libraries that we will be using:

```python
import numpy as np
import pandas as pd
import seaborn as sns
from dython import nominal
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.core.display import HTML    # To centralize the plots
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
```

Importing the data:


```python
df = pd.read_csv('zomato.csv', encoding='ISO-8859-1') # Specifying the encoding is important or it will raise UTF error
```

# <p align="center"> Data Analysis and Cleaning </p>

Let's get to know our data:


```python
df.shape
```




    (9551, 21)



So we have **9551 rows** and **21 columns**.

Let's see the columns:


```python
df.columns
```




    Index(['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address',
           'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines',
           'Average Cost for two', 'Currency', 'Has Table booking',
           'Has Online delivery', 'Is delivering now', 'Switch to order menu',
           'Price range', 'Aggregate rating', 'Rating color', 'Rating text',
           'Votes'],
          dtype='object')



Let's take a look at the first 5 rows of the dataset, to get an idea of the data:


```python
pd.set_option('display.max_columns',21)
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Restaurant Name</th>
      <th>Country Code</th>
      <th>City</th>
      <th>Address</th>
      <th>Locality</th>
      <th>Locality Verbose</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Cuisines</th>
      <th>Average Cost for two</th>
      <th>Currency</th>
      <th>Has Table booking</th>
      <th>Has Online delivery</th>
      <th>Is delivering now</th>
      <th>Switch to order menu</th>
      <th>Price range</th>
      <th>Aggregate rating</th>
      <th>Rating color</th>
      <th>Rating text</th>
      <th>Votes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6317637</td>
      <td>Le Petit Souffle</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Third Floor, Century City Mall, Kalayaan Avenu...</td>
      <td>Century City Mall, Poblacion, Makati City</td>
      <td>Century City Mall, Poblacion, Makati City, Mak...</td>
      <td>121.027535</td>
      <td>14.565443</td>
      <td>French, Japanese, Desserts</td>
      <td>1100</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>4.8</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6304287</td>
      <td>Izakaya Kikufuji</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Little Tokyo, 2277 Chino Roces Avenue, Legaspi...</td>
      <td>Little Tokyo, Legaspi Village, Makati City</td>
      <td>Little Tokyo, Legaspi Village, Makati City, Ma...</td>
      <td>121.014101</td>
      <td>14.553708</td>
      <td>Japanese</td>
      <td>1200</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>4.5</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>591</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6300002</td>
      <td>Heat - Edsa Shangri-La</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Edsa Shangri-La, 1 Garden Way, Ortigas, Mandal...</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City, Ma...</td>
      <td>121.056831</td>
      <td>14.581404</td>
      <td>Seafood, Asian, Filipino, Indian</td>
      <td>4000</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.4</td>
      <td>Green</td>
      <td>Very Good</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6318506</td>
      <td>Ooma</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Fashion Hall, SM Megamall, O...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.056475</td>
      <td>14.585318</td>
      <td>Japanese, Sushi</td>
      <td>1500</td>
      <td>Botswana Pula(P)</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.9</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6314302</td>
      <td>Sambo Kojin</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Atrium, SM Megamall, Ortigas...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.057508</td>
      <td>14.584450</td>
      <td>Japanese, Korean</td>
      <td>1500</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.8</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>229</td>
    </tr>
  </tbody>
</table>
</div>



Let's take a closer look at the columns:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9551 entries, 0 to 9550
    Data columns (total 21 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Restaurant ID         9551 non-null   int64  
     1   Restaurant Name       9551 non-null   object 
     2   Country Code          9551 non-null   int64  
     3   City                  9551 non-null   object 
     4   Address               9551 non-null   object 
     5   Locality              9551 non-null   object 
     6   Locality Verbose      9551 non-null   object 
     7   Longitude             9551 non-null   float64
     8   Latitude              9551 non-null   float64
     9   Cuisines              9542 non-null   object 
     10  Average Cost for two  9551 non-null   int64  
     11  Currency              9551 non-null   object 
     12  Has Table booking     9551 non-null   object 
     13  Has Online delivery   9551 non-null   object 
     14  Is delivering now     9551 non-null   object 
     15  Switch to order menu  9551 non-null   object 
     16  Price range           9551 non-null   int64  
     17  Aggregate rating      9551 non-null   float64
     18  Rating color          9551 non-null   object 
     19  Rating text           9551 non-null   object 
     20  Votes                 9551 non-null   int64  
    dtypes: float64(3), int64(5), object(13)
    memory usage: 1.5+ MB
    

**Observation:** It seems ```Cuisines``` has some null values. We'll take a look at that.

#


```python
df.describe() # Looking at just the numerical columns
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Country Code</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Average Cost for two</th>
      <th>Price range</th>
      <th>Aggregate rating</th>
      <th>Votes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.551000e+03</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.051128e+06</td>
      <td>18.365616</td>
      <td>64.126574</td>
      <td>25.854381</td>
      <td>1199.210763</td>
      <td>1.804837</td>
      <td>2.666370</td>
      <td>156.909748</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.791521e+06</td>
      <td>56.750546</td>
      <td>41.467058</td>
      <td>11.007935</td>
      <td>16121.183073</td>
      <td>0.905609</td>
      <td>1.516378</td>
      <td>430.169145</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.300000e+01</td>
      <td>1.000000</td>
      <td>-157.948486</td>
      <td>-41.330428</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.019625e+05</td>
      <td>1.000000</td>
      <td>77.081343</td>
      <td>28.478713</td>
      <td>250.000000</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.004089e+06</td>
      <td>1.000000</td>
      <td>77.191964</td>
      <td>28.570469</td>
      <td>400.000000</td>
      <td>2.000000</td>
      <td>3.200000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.835229e+07</td>
      <td>1.000000</td>
      <td>77.282006</td>
      <td>28.642758</td>
      <td>700.000000</td>
      <td>2.000000</td>
      <td>3.700000</td>
      <td>131.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.850065e+07</td>
      <td>216.000000</td>
      <td>174.832089</td>
      <td>55.976980</td>
      <td>800000.000000</td>
      <td>4.000000</td>
      <td>4.900000</td>
      <td>10934.000000</td>
    </tr>
  </tbody>
</table>
</div>



Looks like no restaurant has full 5 star rating. Interesting.

Now let's take a look at the **null** values of our columns:


```python
sns.heatmap(df.isnull().sum().values.reshape(-1,1), \
            annot=True, cmap=plt.cm.Blues, yticklabels=df.columns)
plt.xlabel('Null Values')
plt.show()
```


    
![png](readme_files/readme_17_0.png)
    


**Observation:** ```Cuisines``` has 9 null values.

Since we can't determince what cuisines a restaurant has from the other features in the dataset, we will just drop these null values.


```python
df.dropna(inplace=True)
```

There. Let's take a look at the null counts again, just to check:


```python
sns.heatmap(df.isnull().sum().values.reshape(-1,1), \
            annot=True, cmap=plt.cm.Blues, yticklabels=df.columns)
plt.xlabel('Null Values')
plt.show()
```


    
![png](readme_files/readme_21_0.png)
    


Perfect.

#

There is something interesting about the ```Switch to order menu``` column:


```python
df['Switch to order menu']
```




    0       No
    1       No
    2       No
    3       No
    4       No
            ..
    9546    No
    9547    No
    9548    No
    9549    No
    9550    No
    Name: Switch to order menu, Length: 9542, dtype: object




```python
df['Switch to order menu'].value_counts()
```




    No    9542
    Name: Switch to order menu, dtype: int64



**Observation:** ```Switch to order menu``` has no other value than **'No'**.

Since that is not much use for us, we are going to drop it.


```python
df.drop('Switch to order menu', axis=1, inplace = True)
```

#

Since once of the categorical columns turned out to be useless for us, it makes sense to also take a look at the rest of them:


```python
df.columns
```




    Index(['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address',
           'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines',
           'Average Cost for two', 'Currency', 'Has Table booking',
           'Has Online delivery', 'Is delivering now', 'Price range',
           'Aggregate rating', 'Rating color', 'Rating text', 'Votes'],
          dtype='object')




```python
df['Restaurant Name'].value_counts()
```




    Cafe Coffee Day             83
    Domino's Pizza              79
    Subway                      63
    Green Chick Chop            51
    McDonald's                  48
                                ..
    The Town House Cafe          1
    The G.T. Road                1
    The Darzi Bar & Kitchen      1
    Smoke On Water               1
    Walter's Coffee Roastery     1
    Name: Restaurant Name, Length: 7437, dtype: int64




```python
df.Locality.value_counts().value_counts() # Remember, we can specify a column both as df['column'] and df.column
```




    1      550
    2      172
    3      103
    4       51
    5       42
          ... 
    44       1
    45       1
    50       1
    51       1
    122      1
    Name: Locality, Length: 82, dtype: int64




```python
df['Has Table booking'].value_counts()
```




    No     8384
    Yes    1158
    Name: Has Table booking, dtype: int64




```python
df['Has Online delivery'].value_counts()
```




    No     7091
    Yes    2451
    Name: Has Online delivery, dtype: int64




```python
df['Is delivering now'].value_counts()
```




    No     9508
    Yes      34
    Name: Is delivering now, dtype: int64




```python
df.City.value_counts()
```




    New Delhi         5473
    Gurgaon           1118
    Noida             1080
    Faridabad          251
    Ghaziabad           25
                      ... 
    Lincoln              1
    Lakeview             1
    Lakes Entrance       1
    Inverloch            1
    Panchkula            1
    Name: City, Length: 140, dtype: int64



**Observation:** So, all of these columns do have more than one value. That means they could actually be useful.

#

Now we are going to use the **Dython** library to make a correlation plot of all the features. What I like about this library is that it lets you easily plot the correlation between both categorical and continuous features, something that is not easy to do with Pandas.


```python
nominal.associations(df,figsize=(20,10),mark_columns=True,title="Correlation Matrix") # correlation matrix
plt.show()
```


    
![png](readme_files/readme_38_0.png)
    


# <p align="center"> Feature Engineering and Preprocessing </p>

If we look at the ```Aggregate rating (con)``` row, we can see how correlated it is with the rest of the features.

The first highly correlated feature is the ```Restaurant name (nom)``` column, with 95%. Let's take a look at this column and see what we can do.


```python
print( f"Total number of restaurants:    {df['Restaurant Name'].value_counts().shape[0]}")
print(f"Restaurants with 1 value count: {(df['Restaurant Name'].value_counts() == 1).sum()}")
```

    Total number of restaurants:    7437
    Restaurants with 1 value count: 6703
    

That's a **lot** of restaurants. and a lot of them also value count of just 1. 

We won't be able to include all of these in a model. So let's just pick the top 10.


```python
df['Restaurant Name'].value_counts().head(10)
```




    Cafe Coffee Day     83
    Domino's Pizza      79
    Subway              63
    Green Chick Chop    51
    McDonald's          48
    Keventers           34
    Pizza Hut           30
    Giani               29
    Baskin Robbins      28
    Barbeque Nation     26
    Name: Restaurant Name, dtype: int64



Now we are going to define a function to get **dummies** just for these 10 restaurants. Dummies are columns with values **0** and **1**; 0 meaning false and 1 meaning true.

So, for example, if we make a dummy column for "Cafe Coffee Day", the rows in the dummy column will have 1 as the value if the restaurant's name is 'Cafe Coffee Day', and 0 if not.


```python
def dummy(rest_name,column):
    df[column] = df['Restaurant Name'].apply(lambda x: 1 if str(x).strip()==rest_name\
                                             else 0)
```


```python
dummy('Cafe Coffee Day','cafe_coffee_day')
```

Here is a visual example to see how the columns look:


```python
df.loc[df['cafe_coffee_day']==1].head(3)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Restaurant Name</th>
      <th>Country Code</th>
      <th>City</th>
      <th>Address</th>
      <th>Locality</th>
      <th>Locality Verbose</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Cuisines</th>
      <th>Average Cost for two</th>
      <th>Currency</th>
      <th>Has Table booking</th>
      <th>Has Online delivery</th>
      <th>Is delivering now</th>
      <th>Price range</th>
      <th>Aggregate rating</th>
      <th>Rating color</th>
      <th>Rating text</th>
      <th>Votes</th>
      <th>cafe_coffee_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>932</th>
      <td>9650</td>
      <td>Cafe Coffee Day</td>
      <td>1</td>
      <td>Faridabad</td>
      <td>SCF 42, Shopping Centre, Main Huda Market, Sec...</td>
      <td>Sector 15</td>
      <td>Sector 15, Faridabad</td>
      <td>77.323611</td>
      <td>28.395267</td>
      <td>Cafe</td>
      <td>450</td>
      <td>Indian Rupees(Rs.)</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>3.3</td>
      <td>Orange</td>
      <td>Average</td>
      <td>67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>8590</td>
      <td>Cafe Coffee Day</td>
      <td>1</td>
      <td>Ghaziabad</td>
      <td>1st Floor, Shipra Mall, Gulmohar Road, Indirap...</td>
      <td>Shipra Mall, Indirapuram</td>
      <td>Shipra Mall, Indirapuram, Ghaziabad</td>
      <td>77.370208</td>
      <td>28.634047</td>
      <td>Cafe</td>
      <td>450</td>
      <td>Indian Rupees(Rs.)</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>3.2</td>
      <td>Orange</td>
      <td>Average</td>
      <td>63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1283</th>
      <td>631</td>
      <td>Cafe Coffee Day</td>
      <td>1</td>
      <td>Gurgaon</td>
      <td>Upper Ground Floor, DLF Mega Mall, DLF Phase 1...</td>
      <td>DLF Mega Mall, DLF Phase 1</td>
      <td>DLF Mega Mall, DLF Phase 1, Gurgaon</td>
      <td>77.093595</td>
      <td>28.475489</td>
      <td>Cafe</td>
      <td>450</td>
      <td>Indian Rupees(Rs.)</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>2.6</td>
      <td>Orange</td>
      <td>Average</td>
      <td>27</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Wherever the ```Restaurant Name``` column's value is "Cafe Coffee Day", the value of the ```cafe_coffee_day``` column is 1, and the value for the rest of the new columns is zero.

We will apply this function for all of the 10 most frequent restaurants:


```python
def dum_col(x):
    return x.strip().lower().replace(' ','_')

def dummy(lst,column):
    for i in lst.index:
        df[dum_col(i)] = df[column].apply(lambda x: i in x)
```


```python
restaurants = df['Restaurant Name'].value_counts().head(10)
dummy(restaurants,'Restaurant Name')
```


```python
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Restaurant Name</th>
      <th>Country Code</th>
      <th>City</th>
      <th>Address</th>
      <th>Locality</th>
      <th>Locality Verbose</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Cuisines</th>
      <th>...</th>
      <th>cafe_coffee_day</th>
      <th>domino's_pizza</th>
      <th>subway</th>
      <th>green_chick_chop</th>
      <th>mcdonald's</th>
      <th>keventers</th>
      <th>pizza_hut</th>
      <th>giani</th>
      <th>baskin_robbins</th>
      <th>barbeque_nation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6317637</td>
      <td>Le Petit Souffle</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Third Floor, Century City Mall, Kalayaan Avenu...</td>
      <td>Century City Mall, Poblacion, Makati City</td>
      <td>Century City Mall, Poblacion, Makati City, Mak...</td>
      <td>121.027535</td>
      <td>14.565443</td>
      <td>French, Japanese, Desserts</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6304287</td>
      <td>Izakaya Kikufuji</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Little Tokyo, 2277 Chino Roces Avenue, Legaspi...</td>
      <td>Little Tokyo, Legaspi Village, Makati City</td>
      <td>Little Tokyo, Legaspi Village, Makati City, Ma...</td>
      <td>121.014101</td>
      <td>14.553708</td>
      <td>Japanese</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6300002</td>
      <td>Heat - Edsa Shangri-La</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Edsa Shangri-La, 1 Garden Way, Ortigas, Mandal...</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City, Ma...</td>
      <td>121.056831</td>
      <td>14.581404</td>
      <td>Seafood, Asian, Filipino, Indian</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6318506</td>
      <td>Ooma</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Fashion Hall, SM Megamall, O...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.056475</td>
      <td>14.585318</td>
      <td>Japanese, Sushi</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6314302</td>
      <td>Sambo Kojin</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Atrium, SM Megamall, Ortigas...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.057508</td>
      <td>14.584450</td>
      <td>Japanese, Korean</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>

Now we have **True** or **False** values for each of the top 10 restaurants. In python, True and False can also be written as **1** and **0**.

Let's take a look at how many restaurants are named 'Cafe Coffee Day', using our new column:


```python
print(f"Number of Cafe Coffee Day's: {df.loc[df['cafe_coffee_day']==1].size}")
```

    Number of Cafe Coffee Day's: 2730
    


```python
df.shape
```




    (9542, 30)



**Observation:** So out of our **9542** different restaurants, **2730** are Cafe Coffee Day's.

#

Now let's take a look at the correlation between the ```Aggregate rating``` and the new columns that we have created.


```python
features = ['Price range','Votes','Country Code','Restaurant ID','Longitude',
            'Has Table booking','Has Online delivery','cafe_coffee_day',
            "domino's_pizza",'subway','green_chick_chop',"mcdonald's",'keventers',
            'pizza_hut','giani','baskin_robbins','barbeque_nation',
            'Aggregate rating']# --> Only added to see correlation, must be removed later
```


```python
nominal.associations(df[features],figsize=(20,10),mark_columns=True,\
                     title="Correlation Matrix (features)")
plt.show()
```


    
![png](readme_files/readme_60_0.png)
    


**Observation:** Except for ```barbeque_nation```, the rest of the created features seem to have extremely low correlations.

Since the best practice is to keep the model simplistic and use only the best features, we are going to drop all the features except for this one.


```python
features = ['Price range','Votes','Country Code','Restaurant ID','Longitude',
            'Has Table booking','Has Online delivery','barbeque_nation']
```

This is going to be our final list of features for training and testing our model.

**Important Note:** **We are not going to include the features ```Rating color``` and ```Rating text``` in this list. Their inclusion will not result in an actually useful model.**

# <p align="center"> Model Building and Tuning </p>

Now, using these features, we are going to build models to predict our target variable.

## Building

We know that predicting the ```Aggregate rating``` feature is a **regression** problem. Since its correlation with other features is not high enough, a linear model like **Linear Regression** will **not** be optimal.

Instead, we are going to use a **Random Forest Regressor** model for this problem.

First, we are going to split the data into **independent variables** **(Features)** and a **dependent variable** **(Target)**.

So, our features (the columns we will use to predict):


```python
X = pd.get_dummies(df[features])
X
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price range</th>
      <th>Votes</th>
      <th>Country Code</th>
      <th>Restaurant ID</th>
      <th>Longitude</th>
      <th>barbeque_nation</th>
      <th>Has Table booking_No</th>
      <th>Has Table booking_Yes</th>
      <th>Has Online delivery_No</th>
      <th>Has Online delivery_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>314</td>
      <td>162</td>
      <td>6317637</td>
      <td>121.027535</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>591</td>
      <td>162</td>
      <td>6304287</td>
      <td>121.014101</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>270</td>
      <td>162</td>
      <td>6300002</td>
      <td>121.056831</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>365</td>
      <td>162</td>
      <td>6318506</td>
      <td>121.056475</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>229</td>
      <td>162</td>
      <td>6314302</td>
      <td>121.057508</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9546</th>
      <td>3</td>
      <td>788</td>
      <td>208</td>
      <td>5915730</td>
      <td>28.977392</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9547</th>
      <td>3</td>
      <td>1034</td>
      <td>208</td>
      <td>5908749</td>
      <td>29.041297</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9548</th>
      <td>4</td>
      <td>661</td>
      <td>208</td>
      <td>5915807</td>
      <td>29.034640</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9549</th>
      <td>4</td>
      <td>901</td>
      <td>208</td>
      <td>5916112</td>
      <td>29.036019</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9550</th>
      <td>2</td>
      <td>591</td>
      <td>208</td>
      <td>5927402</td>
      <td>29.026016</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9542 rows × 10 columns</p>
</div>



Our target (the column we want to predict):


```python
y = df['Aggregate rating']
```

Now, we want to split them into **train** and **test** sets. 

We will use the train set to train the model, and the test set to test the performance of the model.


```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
```

Next, we will import the model that we want to use, i.e, RandomForestRegressor:


```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 2)
```

Now we **fit** the **train** sets into the model, and use it to predict the test set:


```python
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
```

And check its **performance** using the **test** and the **prediction** sets:


```python
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
mae = metrics.mean_absolute_error(y_test, y_pred)
medae = metrics.median_absolute_error(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Median Absolute Error (MEDAE): {medae}")
print(f'Test variance: {np.var(y_test)}')
```

    Mean Squared Error (MSE): 0.08397072865374543
    Root Mean Squared Error (RMSE): 0.28977703265397936
    Mean Absolute Error (MAE): 0.18649607124148773
    Median Absolute Error (MEDAE): 0.11999999999999966
    Test variance: 2.2502005690560023
    

Let's plot the **residuals**:


```python
residuals = y_test - y_pred
# plot the residuals
plt.scatter(np.linspace(0,5,1909), residuals,c=residuals,cmap='magma', edgecolors='black', linewidths=.1)
plt.colorbar(label="Quality", orientation="vertical")
# plot a horizontal line at y = 0
plt.hlines(y = 0,
xmin = 0, xmax=5,
linestyle='--',colors='black')
# set xlim
plt.xlim((0, 5))
plt.xlabel('Aggregate Rating'); plt.ylabel('Residuals')
plt.show()
```


    
![png](readme_files/readme_80_0.png)
    


A **residual** is the difference between the **observed** value of the target and the **predicted** value. The closer the residual is to **0**, the **better** job our model is doing.


```python
print(f"Error range: {residuals.max()-residuals.min()}")
```

    Error range: 2.7820000000000076
    

So our prediction's **error** range is around **2.782**.

## Tuning

Now we are going to run **RandomizedSearchCV** to tune the model by improving the **hyperparameters**.


```python
# from sklearn.model_selection import RandomizedSearchCV

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rf2 = RandomForestRegressor(random_state=2)

# rf_rscv = RandomizedSearchCV(estimator=rf2, param_distributions=random_grid,\
#                              n_iter = 100, cv = 3, verbose=2, random_state=2, n_jobs = -1)
# rf_rscv.fit(X_train,y_train)
# print(rf_rscv.best_params_)

# Output:
#      n_estimators= 1200,
#      min_samples_split= 10,
#      min_samples_leaf= 1,
#      max_depth = 30,
#      bootstrap= True,
#      random_state=2
```

A **hyperparameter** is a machine learning parameter whose value is chosen **before** a learning algorithm is trained. It has an impact on the model's performance.

Now we are going to use these hyperparameters to make a new Random Forests model, fit the data into it and then score it:


```python
rf_random = RandomForestRegressor(
      n_estimators= 1200,
      min_samples_split= 10,
      min_samples_leaf= 1,
      max_depth = 30,
      max_features='sqrt',
      bootstrap= True,
      random_state=2) # Best RandomizedSearch parameters

rf_random.fit(X_train,y_train)
random_pred = rf_random.predict(X_test)
```


```python
random_mse = metrics.mean_squared_error(y_test, random_pred)
random_rmse = metrics.mean_squared_error(y_test, random_pred, squared=False)
random_mae = metrics.mean_absolute_error(y_test, random_pred)
random_medae = metrics.median_absolute_error(y_test, random_pred)

print(f"Mean Squared Error (MSE): {random_mse}")
print(f"Root Mean Squared Error (RMSE): {random_rmse}")
print(f"Mean Absolute Error (MAE): {random_mae}")
print(f"Median Absolute Error (MEDAE): {random_medae}")
print(f'Test variance: {np.var(y_test)}')
```

    Mean Squared Error (MSE): 0.07950896506171087
    Root Mean Squared Error (RMSE): 0.2819733410478921
    Mean Absolute Error (MAE): 0.18367410616812943
    Median Absolute Error (MEDAE): 0.1146495007615349
    Test variance: 2.2502005690560023
    


```python
print('Improvements:')
print(f"Mean Squared Error (MSE):       {mse} => {random_mse}")
print(f"Root Mean Squared Error (RMSE): {rmse} => {random_rmse}")
print(f"Mean Absolute Error (MAE):      {mae} => {random_mae}")
print(f"Median Absolute Error (MEDAE):  {mae} => {random_medae}")
print(f'Test variance: {np.var(y_test)}')
```

    Improvements:
    Mean Squared Error (MSE):       0.08397072865374543 => 0.07950896506171087
    Root Mean Squared Error (RMSE): 0.28977703265397936 => 0.2819733410478921
    Mean Absolute Error (MAE):      0.18649607124148773 => 0.18367410616812943
    Median Absolute Error (MEDAE):  0.18649607124148773 => 0.1146495007615349
    Test variance: 2.2502005690560023
    

There is decrease in the model's errors.

We can also run **GridSearchCV** on the parameters around these to maybe tune the model further. But we are done with model tuning for this project.

# <p align="center"> Results </p>

Let's plot the residuals for this final model:


```python
f_residuals = y_test - random_pred
# plot the residuals
plt.scatter(np.linspace(0,5,1909), f_residuals, c = f_residuals, cmap='magma', edgecolors='black', linewidths=.1)
plt.colorbar(label = "Quality", orientation = "vertical")
# plot a horizontal line at y = 0
plt.hlines(y = 0, xmin = 0, xmax = 5, linestyle = '--', colors = 'black')
# set xlim
plt.xlim((0, 5))
plt.xlabel('Aggregate Rating'); plt.ylabel('Residuals')
plt.show()
```


    
![png](readme_files/readme_94_0.png)
    



```python
print(f"Errors range of first model: {residuals.max() - residuals.min()}")
print(f"Errors range of second model: {f_residuals.max() - f_residuals.min()}")
print(f"Error difference of models: {(residuals.max() - residuals.min()) - (f_residuals.max() - f_residuals.min())}")
```

    Errors range of first model: 2.7820000000000076
    Errors range of second model: 2.554883167428941
    Error difference of models: 0.2271168325710664
    

When compared to the previous model (with **default** hyperparameters), our final model has a **22.7** reduction in range of error.

 # <p align="center"> FIN </p>
