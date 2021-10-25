import pandas as pd
import numpy as np
from numpy import linalg
from scipy import stats
import statsmodels.api as sm

equity = pd.read_excel('HW2/HW2.xlsx', sheet_name='equity', index_col=0)
factor = pd.read_excel('HW2/HW2.xlsx', sheet_name='factor', index_col=0)

equity_tmp = equity.pct_change().dropna(how='all')
equity_ret = equity_tmp.to_numpy()
factor_tmp = factor.pct_change().dropna(how='all')
factor_ret = factor_tmp.to_numpy()

reqExp = 0.8
reqCorr = 0.4
reqFcorr = 0.7

equity_ret = equity_ret - np.mean(equity_ret, axis=0)[np.newaxis, :]
equity_omega = np.cov(equity_ret.T)
w, v = linalg.eig(equity_omega)

lambda_sum = 0
min_PC = 0
while (lambda_sum / np.sum(w)) < reqExp:
    lambda_sum += w[min_PC]
    min_PC += 1

# print(f'The minimum number of component required is {min_PC}')

F_list = {}

for i in range(min_PC):
    PC = np.dot(equity_ret, v[:, i])  # or v.T[i]
    for j in range(factor_ret.shape[1]):
        factor_key = factor_tmp.columns[j]
        Corr = stats.pearsonr(PC, factor_ret.T[j])[0]
        if reqCorr < abs(Corr):
            F_list[factor_tmp.columns[j]] = abs(Corr)
        else:
            continue

tmp_lst = []
for i in F_list.keys():
    for j in F_list.keys():
        FCorr = abs(stats.pearsonr(factor_tmp[i].to_numpy(), factor_tmp[j].to_numpy())[0])
        if i == j:
            continue
        elif FCorr > reqFcorr:
            tmp_lst.extend([i if F_list[i] > F_list[j] else j])

for k in set(tmp_lst):
    F_list.pop(k)

F_list = dict(sorted(F_list.items(), key=lambda item: item[1], reverse=True))
factor_list = list(F_list.keys())

std_factor = ((factor_tmp - factor_tmp.mean()) / factor_tmp.std())[factor_list]
std_equity_ret = (equity_tmp - equity_tmp.mean()) / equity_tmp.std()
beta = []
t_value = []
Rsq = []

for sec in std_equity_ret.columns:
    Y = std_equity_ret[sec].to_numpy()
    X = sm.add_constant(std_factor.to_numpy())
    model = sm.OLS(Y, X)
    results = model.fit()
    beta.append(list(results.params))
    t_value.append(list(results.tvalues))
    Rsq.append(results.rsquared)

Column = ['Const']+factor_list
pd.DataFrame(beta, columns=Column, index=std_equity_ret.columns).to_csv('HW2/beta.csv')
pd.DataFrame(t_value, columns=Column, index=std_equity_ret.columns).to_csv('HW2/tvalue.csv')
pd.DataFrame(Rsq, columns=['R-square'], index=std_equity_ret.columns).to_csv('HW2/Rsq.csv')
