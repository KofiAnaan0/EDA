# Explanatory Data Analysis (EDA) for PowerCo Dataset: Determining Customer Churn

This project focuses on performing an Explanatory Data Analysis (EDA) on PowerCo's dataset to predict whether a customer will churn or not. The datasets used include `client_data.csv` and `price_data.csv`.

## 1. Importing Necessary Packages

The first step was importing the necessary packages for the analysis. This was done using Python in a Google Colab notebook. Key libraries include:
- `pandas` for data manipulation
- `numpy` for numerical computations
- `matplotlib` and `seaborn` for data visualization

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Loading data with Pandas

Next was loading the dataset from google drive to google colab notebook such that the EDA can be done.

```python
#Uploading the file using google drive
from google.colab import drive

drive.mount('/content/drive')

client_data = pd.read_csv('/content/drive/My Drive/BCGX/client_data.csv')
price_data = pd.read_csv('/content/drive/My Drive/BCGX/price_data.csv')
```

### 3. Exploring the Data

Viewing the top 5 & the least 5 data from both client_data & price_data

```python
client_data.head(5)
price_data.head(5)
```
![client_data](assets/)

## a. Data Types and Structure

I also examined the data types and structure of both datasets using info().

```python
client_df.info()
price_df.info()
```

![client_data](assets/)

From this, I gained insights into the number of null values, data types, and memory usage.

## b. Statistical Summary

A statistical summary was generated using the describe() function:

```python
client_df.describe()
price_df.describe()
```
![client_data](assets/)

Observations:
- Client Data: The key takeaway was the presence of highly skewed data, as indicated by the percentile values.
- Price Data: Overall, the price data appeared to be well-structured and free from major anomalies.

### 4. Data Visualization

To better understand the data distribution and key insights, visualizations were created. A function was written to streamline this process.

Functions for plotting the graph

```python
def stacked_bar_chart(dataframe, title_, size_=(18, 4), rot_=0, legend_="upper right"):
  """
  Plot stacked bars with annotations
  """
  ax = dataframe.plot(
      kind="bar",
      stacked=True,
      title=title_,
      figsize=size_,
      rot=rot_
  )

  #Annotate bars
  annotate_stack_bars(ax, textsize=14)

  #Rename legends
  plt.legend(["Retention", "churn"], loc=legend_)

  #Labels
  plt.ylabel("Company base (%)")

  #Plot the graph
  plt.show()

def annotate_stack_bars(ax, pad=0.99, colour="white", textsize=13):
  """
  Annotate the stack bars
  """
  for p in ax.patches:

    #Calculate Annotation
    value = str(round(p.get_height(), 1))

    #If value is 0 don't annotate
    if value == "0.0":
      continue

    ax.annotate(
        value,
        ((p.get_x()+p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
        color=colour,
        size=textsize
    )
```

## Customer Churn

Visualising churn rate for each company

```python
churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100

# Ploting a stacked bar graph
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")
```
![stacked_bar](assets/)





