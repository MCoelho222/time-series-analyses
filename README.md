# Time Series Representativeness (RHIS)

This repository presents 4 methods that can be simultaneously applied in time series for variability patterns detection. Essentially, these methods check if the time series are compliant with the basic assumptions for statistical representativeness, i. e., the application of statistical methods for frequency analysis depends on the compliance with the hypotheses of randomness, homogeneity, independence, and stationarity (RHIS). If at least one of these hypotheses is rejected, it means that there is high chance for the presence of a variability pattern, such as trend, shift, and/or seasonality. In this situation, statistical methods likely will not produce representative results. In order to get representativeness, a period of data should be selected, which actually represents the current conditions and can possibly be used for planning future periods. 

Generally, there are 3 patterns that can cause rejection of RHIS hypotheses:

* **Trends**
* **Seasonality**
* **Shifts**

In the context of water resources management, for example, the continuous and disordered urban expansion processes iof cities makes the waters from rain to reach the river channel each time faster due to waterproofing of the soil. So, the streamflow data becomes greater year after year characterizing a trend in the time series. When this is statistically confirmed, an strategy for data selection or treatment should be implemented. Finally, the selected data must then be compliant with the RHIS hypotheses.

# How this package can help you?

When you want to know if a time series is representative, the first thing to do is to apply the RHIS tests in the complete time series. If one or more hypotheses are rejected, one strategy is to select only a period of the data where the actual characteristics are represented. However, we have to include the maximum quantity of information by selecting the maximum data we can. This selection period must be RHIS-compliant, so we would have to test every time we include one more data in the selection. Here is where this program comes in. 

It provides a method that applies the RHIS hypotheses in a time series with an increasing number of elements. The initial time series will have only the first 5 or 10 elements and then this number will be increasing one by one, and the tests applied each time this number increases. When the results of the tests are plotted in sequence, if there is a representative period, it will be possible to see the exact time when data starts to be complete compliant and you should select the data considering this point as a divisor. 

# Methods - Hypothesis (RHIS)

* **Runs Test** - randomness
* **Mann-Whitney** - homogeneity
* **Wald-Wolfovitz** - independence
* **Mann-Kendall** - stationarity (trend)

## Randomness

Briefly, the runs method checks if there are too many values above or below the median or if there is another pattern in the positioning of the values above and below the median. Randomness is rejected if a pattern is detected for a given significance level.

## Homogeneity

The homogeneity test checks if the halves of the time series are statistically equal. Homogeneity is rejected if one of the halves is greater or smaller than the other, for a given significance level.

## Independence

A time series has dependency when a value influences the next. For example, if it rains a lot one day and it stops, the daily streamflow measurements of the next days will have a pattern, being each day higher while the underground water keeps flowing into the river, and the opposite when it stops flowing. Independece is rejected if this kind of pattern occurs, for a given significance level.

## Stationarity

When a time series consistently increases or decreases with time, it is considered non-stationary if the hypothesis is rejected for a given significance level.


# Scientific foundations

These tests were used in my doctorate thesis to check the representativeness of water quality time series in the Alto Iguaçu Watershed in the south of Brazil.

If you are interested, please check the article below.

[Uncertainty analysis in the detection of trends, cycles, and shifts in water resources time series](https://link.springer.com/article/10.1007/s11269-019-02210-1)

# Run to see an example

```
python -m venv .venv
```
```
.venv/scripts/activate
```
```
pip install -e .[dev]
```
```
rhis-ts
```

# Example

The **RHIS** tests were applied to increasing slices from a time series.

For example:

```py
ts = [22, 10, 30, 4, 25, 12, 7]

slices = [[22, 10, 30, 4, 25], [22, 10, 30, 4, 25, 12], [22, 10, 30, 4, 25, 12, 7]]
```

The result is similar to this:

```py
evol_rhis = {
        'randomness': [0.234, 0.321, 0.001],
        'homogeneity': [0.532, 0.731, 0.091],
        'independence': [0.444, 0.624, 0.656],
        'stationarity': [0.121, 0.000, 0.001],
    }
```

In this way, the first p-values are the result from the application of the **RHIS** tests in the first 5 values, and the last p-values are the result from the entire time series.
    
In the example below, the time series has an upward shift after the 80th value. Clearly something has changed in the process that generates these data. In this situation, the estimation of statistical parameters with the whole data would not be reasonable, since two different populations might be present. The boxplot evolution shows that the median, the 25th, 75th and higher percentiles are increasing from the 80th value, changing the whole population distribution over time.

The last graph, which shows the evolution of the **RHIS** results, confirms that indeed, the statictical characteristics of the population has changed after the 80th value. The homogeneity, independence and stationarity hypotheses were rejected right after this point. The p-values go to 0. Despite randomness was not rejected, it had a significant decrease. Anyway, the rejection of just one of these 4 hypotheses is enough to conclude that the time series is no longer representative of only one population after this point.

With that in mind, one should decide which part of the time series to use to calculate statistical parameters or make any inference analysis. It will depend on each ones objectives.

## Time series

![TimeSeries](src/rhis_timeseries/example_plots/original_ts.png)

## Boxplots evolution

![BoxplotEvolution](src/rhis_timeseries/example_plots/boxplot_evolution.png)

## RHIS evolution

![RHISEvolution](src/rhis_timeseries/example_plots/representativeness_evolution.png)
