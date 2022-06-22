# DLinear

DLinear conducts direct multi-step forecasting by decomposing the time series into a trend and a remainder series and employs two one-layer linear networks to model these two series for the forecasting task. Then, two one-layer linear networks are applied to the two series.

<p align="center">
<img src=".\utils\pic\DLinear.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall architecture of DLinear model.
</p>


Although DLinear is simple, it has some compelling characteristics:

- *An $O(1)$ maximum signal traversing path length*: The shorter the path, the better the dependencies are captured [18], making DLinear capable of capturing both short-range and long-range temporal relations.
- *High-efficiency*: As each branch has only one linear layer, it costs much lower memory and fewer parameters and has a faster inference speed than existing Transformers.
- *Interpretability*: After training, we can visualize weights from the seasonality and trend branches to have some insights on the predicted values [8].
- *Easy-to-use*: DLinear can be obtained easily without tuning model hyper-parameters.

More information can be found in: https://arxiv.org/abs/2205.13504


## Use cases

1. Electricity demand dataset
2. Air-Quality dataset (CO & NO2)


## Get Started
1. Install Python >= 3.7
2. Install requirements.txt
3. Run DLinear - UC:AirQuality.ipynb or DLinear - UC:Electricity.ipynb

## Requirements
- tqdm==4.62.3
- matplotlib==3.5.1
- numpy==1.21.2
- torch==1.7.0
- pandas==1.3.5
- scikit-learn==1.0.2


## Versions
- Version 1.0
  - Basic implementation
  - Use cases: Electricity demand & Air-Quality forecasting
  - Evaluation per value of forecasting horizon
  - Model Evaluation
    - Performance evaluation metrics: MAE, RMSE, MAPE, R2
    - Examine AutoCorrelation 
  - Examples 
