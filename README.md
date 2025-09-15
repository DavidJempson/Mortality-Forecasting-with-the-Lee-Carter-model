# Mortality Forecasting with the Lee Carter Model
In this project, I used publicly available data from the Human Mortality Database to forecast UK mortality rates. It uses historical data to fit the model using Singular Vaue Decomposition, forecasts the mortality trend using an ARIMA time series model, and then can create generation life tables. As an example application, I have included a function to calculate the net cost of an insurance premium with arbitrary inputs.

## Methodology
1.  **Data Preparation:** Sourced and cleaned death and exposure data from the Human Mortality Database for UK males and females (1950-2022).
2.  **Model Fitting:** Estimated the core Lee-Carter parameters (a<sub>x</sub>, b<sub>x</sub>, k<sub>t</sub>) using SVD on the log-mortality matrix.
3.  **Time Series Forecasting:** Modelled the resulting k_t time series as a random walk with drift (ARIMA(0,1,0)) to forecast the mortality trend to 2050.
4.  **Life Table Construction:** Recombined the parameters to build a projected period life table for 2050 and a generation life table for the 1980 birth cohort.
5.  **Insurance Premium Calculation** Calculated the Expected Present Value (EPV) of the benefit and premiums to find the net price of an insurance policy for anyone of a given age and sex, wanting a premium with a given term and sum assured.

## Results
The model identifies a statistically significant downward trend in mortality for males and females (p < 0.001), with mortality improving faster on average for females than males, though with higher variance. However, there is significant evidence that the model needs refinement, particularly for females. For example, the Ljung-Box test shows there is significant evidence that the residuals are autocorrelated (p = 0.01 (males), p = 0.00 (females). Additionally, the kurtosis is high for both males and females, and the Jarque-Bera test gives sufficient evidence (p = 0.00) for females that the model's residuals are not normally distributed.

Therefore, in the future I would use a more complex model to forecast future mortality, such as a full ARIMA model or a changepoint detection model such as facebook prophet, which would be more appropriate because it should be able to handle changes in UK mortality trends, like the slowdown since 2011. Additionally, I would use another mortality forecasting model such as an Age-Period-Cohort (APC) model or a machine learning model, to compare.

Interestingly, the insurance premium calculation often gave similar prices to what was quoted on comparison sites, and was consistently much better than using a period life table for the calculations.

## How to Run
Simply download the python file main.py and the data files to the same directory, and run. I was using python 3.13.

## Data Sources
Mortality data was obtained from the Human Mortality Database (https://www.mortality.org). I downloaded the yearly deaths and exposures files, and only used the data from 1950 to 2022.
