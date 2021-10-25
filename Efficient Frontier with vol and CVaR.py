import numpy
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
from EFF import efficientFrontier as eff

# parameter
lb = 0.0  # 0.0 lower bound
ub = 1.0  # 1.0  upper bound
useWeekly = False
alpha = 0.05
riskMeasure = 'CVaR'  # vol, VaR, CVaR


## objective function
## The optimize function should be put in the first parameter
def MV(w, cov_mat):
    return np.dot(w, np.dot(cov_mat, w.T))


def VaR(w, VaR_stock):
    return np.dot(VaR_stock.T, w)


def cVaR(w, VaR_stock):
    return np.dot(VaR_stock, w)


def stock_var(ret, alpha):
    return abs(np.percentile(ret, alpha * 100))  # ret.loc[ret <= np.percentile(ret, alpha)].mean()


# read_data - existing portfolio
data = pd.read_excel('biggestETFData.xlsx', engine='openpyxl', index_col=0, sheet_name="US-only")
data_int = pd.read_excel('biggestETFData.xlsx', engine='openpyxl', index_col=0, sheet_name="international")
data_CA = pd.read_excel('biggestETFData.xlsx', engine='openpyxl', index_col=0, sheet_name="crossAsset")
muRange = np.arange(0.0055, 0.010, 0.0002)


def transform_data(data, riskMeasure):
    data.index = pd.to_datetime(data.index)
    start = data.index[0]
    end = data.index[-1]
    monthend = pd.date_range(start, end, freq='BM')

    if useWeekly:
        data_m = data[data.index.dayofweek == 4]
    else:
        data_m = data[data.index.isin(monthend)]

    data_m_ret = (data_m / data_m.shift() - 1).dropna(how='all')

    n = len(data_m_ret.columns)
    muRange = np.arange(0.0055, 0.010, 0.0002)
    targetRange = np.zeros(len(muRange))
    R = data_m_ret.mean()
    omega = data_m_ret.cov()

    wgt = {}

    for i in range(len(muRange)):
        mu = muRange[i]
        wgt[mu] = []
        x_0 = np.ones(n) / n  # equal weight
        bndsa = ((lb, ub),)
        for j in range(1, n):
            bndsa = bndsa + ((lb, ub),)

        if riskMeasure == "vol":
            consTR = (
                {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}, {'type': 'eq', 'fun': lambda x: mu - np.dot(x, R)})
            w = minimize(MV, x_0, method='SLSQP', constraints=consTR, bounds=bndsa, args=(omega))
            targetRange[i] = np.dot(w.x, np.dot(omega, w.x.T)) ** 0.5
            wgt[mu].extend(np.squeeze(w.x))
        elif riskMeasure == "CVaR":
            stockvar = numpy.asarray([stock_var(data_m_ret[column].to_numpy(), alpha) for column in data_m_ret.columns])
            consTR = (
                {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}, {'type': 'eq', 'fun': lambda x: mu - np.dot(x, R)})
            w = minimize(cVaR, x_0, method='SLSQP', constraints=consTR, bounds=bndsa, args=(stockvar))
            targetRange[i] = cVaR(w.x, stockvar)
            wgt[mu].extend(np.squeeze(w.x))
        # elif riskMeasure == "VaR":
    print(wgt)

    return targetRange


us_target = transform_data(data, riskMeasure)
int_target = transform_data(data_int, riskMeasure)
CA_target = transform_data(data_CA, riskMeasure)

plt.plot(us_target, muRange, '-', int_target, muRange, '-', CA_target, muRange, '-')
plt.xlabel(f"{riskMeasure}")
plt.ylabel("Expected Return")
plt.show()
