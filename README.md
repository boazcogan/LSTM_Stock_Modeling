## Replicating *Enhancing Time Series Momentum Strategies Using Deep Neural Networks* - Project for CS496: Advanced Topics in Deep Learning

### Sections
1. Introduction
2. CLC Dataset
3.

### Introduction
Time series momentum, or trend momentum, is measured by a portfolio comprised of long assets that have had recent positive returns and short assets that have had recent negative returns. Academic evidence suggests that strategies that optimize time series momentum improve a portfolio's risk-adjusted returns.

These time series momentum strategies have two main attributes, trend estimation and position sizing; and recent studies have implemented various supervised learning models to predict these components. These models, however, do not account for volatility and other risk characteristics, and require manual specifications.

In *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*, the authors propose Deep Momentum Networks (DMNs) as an approach to momentum strategy that improves on the shortcomings of prior supervised learning models. DMNs are intended to generate positions directly in a way that simultaneously learns both trend estimation and position sizing in the same function.

The authors investigated four common deep neural network architectures - linear regression, multilayer perceptron (MLP), WaveNet, and Long Short-Term Memory (LSTM) - and compared the generated positions to those calculated from existing momentum strategies. In our experimentation, we implemented three of the four network types, omitting WaveNet due to time constraints, and compared our predicted returns with those of the DMNs implemented by the authors.

### CLC Dataset
The data used in the original paper and our experimentation can be accessed through the [Pinnacle Data Corp. CLC Database](https://pinnacledata2.com/clc.html). The database contains ratio-adjusted continuous futures contracts that span a variety of asset classes - including commodities, fixed income, and currency futures. The authors extracted 88 contracts that contained prices from 1990 to 2015. The data is given in CSV format, one for each asset, and formatted as follows:

| Date | Open | High | Low | Close | Open Interest | Volume |
| --- | --- | --- | --- | --- | --- | --- |
| 01/19/2010 | 83.07 | 83.07 | 83.07 | 83.07 | 14 | 145215 |

For our experimentation, we read the contents of the CSVs for a given asset class into numpy arrays in the function `data.GetDataset.get_dataset_by_category()`. For simplicity, we truncated the data, so the features given to the networks only include the data for **Open, High, Low,** and **Close.**

Not all neural networks are capable of infering information from sequence data. To accommodate for systems with no short term memory, such as a linear model, 5 datapoints were aggregated together. The LSTM does need the aggregated datapoints since it contains a hidden and cell state which allow it to infer information from a sequence.

### Next Section

### Further Reading
This section includes the original paper and additional resources related to the experiment.
#### Papers
[Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/pdf/1904.04912.pdf), 2019.
#### Posts
[Chapter 8. Recurrent Neural Networks, Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/index.html)

[Time Series Momentum (aka Trend-Following): A Good Time for a Refresh](https://alphaarchitect.com/2018/02/08/time-series-momentum-aka-trend-following-the-historical-evidence/#:~:text=Time%2Dseries%20momentum%2C%20also%20called,have%20had%20recent%20negative%20returns.)
