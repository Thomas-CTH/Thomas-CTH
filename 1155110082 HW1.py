import pandas as pd
import numpy as np

# read HSI.xlsx data and clean the dataset by removing nan rows
tmp_price = pd.read_excel("HSI.xlsx", index_col="Date")
price = tmp_price.dropna(how="all")
price.index = pd.to_datetime(price.index, format="%Y-%m-%d")

# get the week end and month end date for filtering weekly and monthly data
start_date = price.index[0]
end_date = price.index[-1]
week_end = pd.date_range(start_date, end_date, freq='W-FRI')
month_end = pd.date_range(start_date, end_date, freq='BM')

# slice the data frames into required monthly and weekly pricing data
price_daily = price
price_weekly = price[price.index.isin(week_end)]
price_monthly = price[price.index.isin(month_end)]

return_daily = price_daily.pct_change()
return_weekly = price_weekly.pct_change()
return_monthly = price_monthly.pct_change()

return_weekly.cov().to_csv("covHSI.csv")

import matplotlib.pyplot as plt

stock = return_weekly["700 HK"].dropna()
plt.hist(stock, bins=100, density=True)
plt.xlabel('Return', fontsize=8)
plt.ylabel('Density', fontsize=8)
plt.title('700 HK', fontsize=10)
plt.show()
plt.savefig('700 HK.png')


# Question 6: Standard deviation of stock return without using package
def standard_dev(data):
    variance = (data - (data.sum() / len(data))) ** 2
    return (variance.sum()/(len(variance)-1)) ** (1 / 2)


company_list = price_daily.columns
dict_temp = {}

for company in company_list:
    temp = {}
    for year in range(start_date.year, end_date.year + 1):
        year_return = return_daily[company][return_daily.index.year == year].dropna().to_numpy(dtype=float)
        temp[year] = standard_dev(year_return)
        for month in range(1, 13):
            month_return = return_daily[company][
                (return_daily.index.year == year) & (return_daily.index.month == month)].dropna().to_numpy(dtype=float)
            temp[year*100+month] = standard_dev(month_return)
    dict_temp[company] = temp

return_sd = pd.DataFrame.from_dict(dict_temp)
return_sd.replace(0, np.nan, inplace = True)
return_sd = return_sd.dropna(axis = 0, how = 'all')
return_sd.to_csv("HSI_vol.csv")

# standard deviation of stock using resample
year_sd = return_daily.resample('A').std()
month_sd = return_daily.resample('M').std()
vol_df = pd.concat([year_sd, month_sd])
vol_df.to_csv("HSI_vol_resample.csv")
