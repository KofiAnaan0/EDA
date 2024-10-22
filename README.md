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

About 10% of the total customers have churned.

## Sales channel

```python
#creating channel dataframe
channel = client_data[["id", "churn", "code of the sales channel"]]
channel = channel.groupby(["code of the sales channel", "churn"])['id'].count()
channel = channel.unstack(level=1).fillna(0)
channel_churn = (channel.div(channel.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)
channel_churn.head(10)

#Ploting the graph
stacked_bar_chart(channel_churn, "Churn by Sales Channel", rot_=30)
```

![stacked_bar](assets/)

The churning customers are distributed over 5 different values for channel_sales. As well as this, the value of MISSING has a churn rate of 7.6%. MISSING indicates a missing value. This feature could be an important feature when it comes to building our model.

## Consumption

The distribution of consumption in the last year and month. Since the consumption data is univariate, I used histograms to visualize their distribution.

```python
# creating consumption dataframe
consumption = client_data[["id", "churn", "electricity consumption of the past 12 months", "gas consumption of the past 12 months", "electricity consumption of the last month", "has_gas", "current paid consumption"]]
```

```python
#function for plotting the histogram
def plot_distribution(dataframe, column, ax, bins_=50, width=0.85):
    """
    Plot variable distribution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    # Plot the histogram
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')
```

```python
fig, ax = plt.subplots(nrows=4, figsize=(16, 20))

plot_distribution(consumption, "electricity consumption of the past 12 months", ax[0])
plot_distribution(consumption[consumption['has_gas'] =='t'], "gas consumption of the past 12 months", ax[1])
plot_distribution(consumption, "electricity consumption of the last month", ax[2])
plot_distribution(consumption, "current paid consumption", ax[3])
```

![stacked_bar](assets/)

Clearly, the consumption data is highly positively skewed, presenting a very long right-tail towards the higher values of the distribution. The values on the higher and lower end of the distribution are likely to be outliers. So I used a boxplot to visualise the outliers in more detail. A boxplot is a standardized way of displaying the distribution based on a five number summary:

- Minimum
- First quartile (Q1)
- Median
- Third quartile (Q3)
- Maximum
  
It can reveal outliers and what their values are. It can also determine if our data is symmetrical, how tightly the data is grouped and if/how our data is skewed.

```python
fig, axs = plt.subplots(nrows=4, figsize=(18,25))

# Plot histogram
sns.boxplot(consumption["cons_12m"], ax=axs[0])
sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
sns.boxplot(consumption["cons_last_month"], ax=axs[2])
sns.boxplot(consumption["imp_cons"], ax=axs[3])

# Remove scientific notation
for ax in axs:
    ax.ticklabel_format(style='plain', axis='x')
    # Set x-axis limit
    axs[0].set_xlim(-200000, 2000000)
    axs[1].set_xlim(-200000, 2000000)
    axs[2].set_xlim(-20000, 100000)
    plt.show()
```

![stacked_bar](assets/)









