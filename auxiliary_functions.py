'''
Author: Tiago Da Costa
Date: 23/01/2023
'''
import torch
import pandas as pd
import numpy as np
from datetime import datetime as dt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import matplotlib.pyplot as plt

'''
Loads Data for use in the TFT file
'''
def load_data(
    data_directory: str,
    list_of_companies: list,
    data_start_date: str,
    data_end_date: str,
    val_start_date: str,
    gen_future_data: bool = False,
    simple_list: bool = False
):
    if simple_list: ticker_list = list_of_companies
    else:
        ticker_list = []
        for i in list_of_companies:
            temp = pd.read_csv(data_directory + i)
            ticker_list.append(temp['Symbol'])

    for u in range(len(ticker_list)):
        for i in range(len(ticker_list[0])):
            try:
                tickr = ticker_list[u][i]
                # If its the first entry
                if ticker_list[0][0] == tickr: first = True
                # Replace the '.' in the Symbols with an '_' to load the files.
                if('.' or '-' in tickr):
                    tickr = tickr.replace('.', '_')
                # Read the file
                temp = pd.read_csv(data_directory + tickr + '.csv')
                # Add the Symbol as a column 'Symbol'.
                temp['Symbol'] = ticker_list[u][i]

                # Generate future_data DataFrame to predict on.
                if gen_future_data:
                    if first:
                        future_data = temp[temp['Date'] >= data_end_date]
                    else: future_data = future_data.append(temp[temp['Date'] >= data_end_date])

                # Cutoff dates
                temp = temp[temp['Date'] >= data_start_date]
                temp = temp[temp['Date'] < data_end_date]
                if first: 
                    data = temp
                    first = False
                else: data = data.append(temp) 
            except:
                print('No file: ' + ticker_list[i])


    count = 0

    data = data.sort_values('Date').reset_index(drop=True)
    for i in data['Date'].unique():
        data.loc[data['Date'] == i, 'time_idx'] = count
        count += 1
    data['time_idx'] = data.time_idx.astype(int)

    if gen_future_data:
        future_data = future_data.sort_values('Date').reset_index(drop=True)
        for i in future_data['Date'].unique():
            future_data.loc[future_data['Date'] == i, 'time_idx'] = count
            count += 1
        future_data['time_idx'] = future_data.time_idx.astype(int)


    # Convert dates to datetime obj
    s_d = dt.strptime(val_start_date, '%Y-%m-%d')
    e_d = dt.strptime(data_end_date, '%Y-%m-%d')
    # Get delta between validation start date and data_end_date
    dlt = int(np.busday_count(s_d.date(), e_d.date()))

    # Convert dates to datetime obj
    s_d2 = dt.strptime(data_start_date, '%Y-%m-%d')
    e_d2 = dt.strptime(data_end_date, '%Y-%m-%d')
    # Calculate average work days between data_start_date and data_end_date (encoder length)
    avg_work_days = int((np.busday_count(s_d2.date(), e_d2.date())/12)/(e_d2.year - s_d2.year))

    if gen_future_data:
        past_data = data[lambda x: x.time_idx > x.time_idx.max() - (avg_work_days + 26)]
        future_data = pd.concat([past_data, future_data]).reset_index(drop=True)
        return data, future_data, dlt, avg_work_days
    else:
        return data, dlt, avg_work_days

'''
Returns the best predicting quantile from each n number of samples.
'''
def get_best_quantiles(
    predictions,
    actuals,
    samples: int = 10,
    quantiles: int = 7,
    raw: bool = True
    )->list[torch.Tensor]:
    best_quantiles = []

    # Output from predict_from_list
    if not raw:
        for i in range(samples):
            prev_mae = 100 # 100% error as baseline
            best = 0
            predi = predictions[i]['prediction'][0]
            for j in range(quantiles):
                mae = (actuals[i] - predi[0:, j]).abs().mean()
                if(mae < prev_mae):
                    best = j
                    prev_mae = mae
            best_quantiles.append(predi[0:, best])

    # Output from regular predict
    else:
        for i in range(samples):
            prev_mae = 100 # 100% error as baseline
            best = 0
            for j in range(quantiles):
                predi = predictions[0][i]
                mae = (actuals[i] - predi[0:, j]).abs().mean()
                if(mae < prev_mae):
                    best = j
                    prev_mae = mae
            best_quantiles.append(predi[0:, best])
    return best_quantiles

'''
Returns predictions made in same order as list that is passed in
'''
def predict_from_list(
    _list: list,
    best_tft: TemporalFusionTransformer,
    tsds: TimeSeriesDataSet
)-> tuple[list, list]:
    pred = []
    x = []
    for i in _list:
        try:
            _p, _x = best_tft.predict(
                tsds.filter(lambda x: (x.Symbol == i)), 
                mode='raw', return_x=True)
            pred.append(_p)
            x.append(_x)
        except ValueError:
            print(f'Value not found: {i}')
    
    return pred, x

'''
Returns actuals made in same order as list that is passed in
'''
def actuals_from_list(
    _list: list,
    tsds: TimeSeriesDataSet
)->list:
    actuals = []
    for i in _list:
        actuals.append(torch.cat([y[0] for _, y in iter(tsds.filter(lambda x: (x.Symbol == i)))]))
    return actuals

'''
Returns largest n number of companies by each business sector from Yahoo Finance data.
'''
def get_largest_n_companies_by_sector(temp: pd.DataFrame,
n: int = 5
)->list:
    sectors = temp.Sector.unique()
    _list = []
    for i in sectors:
        fl = temp.loc[temp.Sector == i]
        if (fl.shape[0] < n):
            for j in range(fl.shape[0]):
                _list.append(fl.Symbol.loc[fl['Market Cap'] == fl['Market Cap'].nlargest(n).values[j]].values[0])
        else:
            for j in range(n):
                _list.append(fl.Symbol.loc[fl['Market Cap'] == fl['Market Cap'].nlargest(n).values[j]].values[0])

    return _list

'''
Truncates dates to yyyy-mm-dd format
'''
def date_fixer(tickr: str,
     save: bool = False
):
    filen = f'./Data/StockData/{tickr}.csv'
    fix = pd.read_csv(filen)
    fixr = []
    for i in range(len(fix.Date)):
        fixr.append(fix.Date.values[i][0:10])

    fix['Date'] = fixr
    if save: fix.to_csv(filen)
    return fix

'''
Function to produce plots from predict_from_list
'''
def plot_list_predictions(
    predictions: list,
    actuals: list,
    x_ax: int,
    y_ax: int,
    best_quantiles: list = [],
    titles: list = [],
    ylabel: str = ""
):
    x_ax = 5
    y_ax = 11
    figure, axis = plt.subplots(y_ax, x_ax)
    figure.set_figheight(y_ax*10)
    figure.set_figwidth(x_ax*10)

    c = 0
    for idy in range(y_ax):
        for idx in range(x_ax):
            pred = predictions[c]['prediction'][0]
            axis[idy, idx].plot(actuals[c], label='Actual Values', color='b')
            axis[idy, idx].plot(np.average(pred, axis=1), label='Average Quantile', color='purple')
            if best_quantiles != []: axis[idy, idx].plot(best_quantiles[c], label='Best Tensor', color='red')
            if titles != []: axis[idy, idx].set_title(titles[c])
            axis[idy, idx].set(ylabel=ylabel)
            axis[idy, idx].legend()
            c += 1
    figure.show()
    return

def plot_joint_predictions(
    tftpredictions: list,
    stpredictions: list,
    lstmpredictions: list,
    actuals: list,
    x_ax: int,
    y_ax: int,
    titles: list = [],
    ylabel: str = "",
    skip: list = [],
    accuracies: pd.DataFrame = None
):
    _1x = False
    _1y = False
    figure, axis = plt.subplots(y_ax, x_ax)
    figure.set_figheight(y_ax*10)
    figure.set_figwidth(x_ax*10)

    best_quantiles = get_best_quantiles(tftpredictions, actuals, samples=len(tftpredictions), raw=False)
    alpha1 = 0.85
    dashed = (0, (5, 3))
    c = 0
    if x_ax == 1:
        _1x = True
    if y_ax == 1:
        _1y = True
    for idy in range(y_ax):
        for idx in range(x_ax):
            if skip != [] and c in skip:
                c += 1
            if c < len(tftpredictions):
                if _1x and _1y:
                    pred = tftpredictions[c]['prediction'][0]
                    axis.plot(actuals[c][22:], label='Actual Values', color='b')
                    if best_quantiles != []: axis.plot(best_quantiles[c][23:], label='TFT - Best Quantile', color='red', ls=dashed, alpha=alpha1)
                    axis.plot(np.average(pred, axis=1)[22:], label='TFT - Quantiles Average', color='red', ls=(0,(1,5)), alpha=0.75)
                    axis.plot(stpredictions[titles[c]], label='1D-CT', color='green', ls=dashed, alpha=alpha1)
                    axis.plot(lstmpredictions[titles[c]], label='LSTM', color='orange', ls=dashed, alpha=alpha1)
                    if titles != []: axis.set_title(titles[c])
                    axis.set(ylabel=ylabel, xlabel='Business Days since 31-01-2022')
                    axis.legend()
                    c += 1
                elif _1y:
                    pred = tftpredictions[c]['prediction'][0]
                    axis[idx].plot(actuals[c][22:], label='Actual Values', color='b')
                    if best_quantiles != []: axis[idx].plot(best_quantiles[c][23:], label='TFT - Best Quantile', color='red', ls=dashed, alpha=alpha1)
                    axis[idx].plot(np.average(pred, axis=1)[22:], label='TFT - Quantiles Average', color='red', ls=(0,(1,5)), alpha=0.75)
                    axis[idx].plot(stpredictions[titles[c]], label='1D-CT', color='green', ls=dashed, alpha=alpha1)
                    axis[idx].plot(lstmpredictions[titles[c]], label='LSTM', color='orange', ls=dashed, alpha=alpha1)
                    if titles != []: axis[idx].set_title(titles[c])
                    axis[idx].set(ylabel=ylabel, xlabel='Business Days since 31-01-2022')
                    axis[idx].legend()
                    c += 1
                elif _1x:
                    pred = tftpredictions[c]['prediction'][0]
                    axis[idy].plot(actuals[c][22:], label='Actual Values', color='b')
                    if best_quantiles != []: axis[idy].plot(best_quantiles[c][23:], label='TFT - Best Quantile', color='red', ls=dashed, alpha=alpha1)
                    axis[idy].plot(np.average(pred, axis=1)[22:], label='TFT - Quantiles Average', color='red', ls=(0,(1,5)), alpha=0.75)
                    axis[idy].plot(stpredictions[titles[c]], label='1D-CT', color='green', ls=dashed, alpha=alpha1)
                    axis[idy].plot(lstmpredictions[titles[c]], label='LSTM', color='orange', ls=dashed, alpha=alpha1)
                    if titles != []: axis[idy].set_title(titles[c])
                    axis[idy].set(ylabel=ylabel, xlabel='Business Days since 31-01-2022')
                    axis[idy].legend()
                    c+=1
                else:
                    pred = tftpredictions[c]['prediction'][0]
                    axis[idy, idx].plot(actuals[c][22:], label='Actual Values', color='b')
                    if best_quantiles != []: axis[idy, idx].plot(best_quantiles[c][23:], label='TFT - Best Quantile', color='red', ls=dashed, alpha=alpha1)
                    axis[idy, idx].plot(np.average(pred, axis=1)[22:], label='TFT - Quantiles Average', color='red', ls=(0,(1,5)), alpha=0.75)
                    axis[idy, idx].plot(stpredictions[titles[c]], label='1D-CT', color='green', ls=dashed, alpha=alpha1)
                    axis[idy, idx].plot(lstmpredictions[titles[c]], label='LSTM', color='orange', ls=dashed, alpha=alpha1)
                    if titles != []: axis[idy, idx].set_title(titles[c])
                    axis[idy, idx].set(ylabel=ylabel, xlabel='Business Days since 31-01-2022')
                    axis[idy, idx].legend()
                    c += 1

    figure.show()
    return

def tft_pearson(prediction, actuals):
    return (torch.sum(   (prediction - prediction.mean()) * (actuals - actuals.mean())   ) / \
(torch.sqrt(    torch.sum(  (prediction - prediction.mean()) **2    ) * torch.sum(    (actuals - actuals.mean())**2   )))).item()
def tft_mae(prediction, actuals):
    return torch.sqrt(((prediction - actuals)**2).mean()).item()
def tft_rmse(prediction, actuals):
    return (actuals - prediction).abs().mean().item()

def load_accuracies(
    file_path: str,
    _list: list
    )->tuple((np.array, np.array, np.array)):
    accuracies = pd.read_csv(file_path)
    accuracies = accuracies.set_index(['Model'])

    _tft = accuracies.loc['TFT']
    for i in range(len(_tft)): _tft[i] =_tft[i].replace('[', '').replace(']', '').split(sep=',')
    _st = accuracies.loc['Transformer']
    for i in range(len(_st)): _st[i] =_st[i].replace('[', '').replace(']', '').split(sep=',')
    _lstm = accuracies.loc['LSTM']
    for i in range(len(_lstm)): _lstm[i] =_lstm[i].replace('[', '').replace(']', '').split(sep=',')

    for i in range(len(_list)):
        for j in range(3):
            _tft[i][j] = float(_tft[i][j])
            _st[i][j] = float(_st[i][j])
            _lstm[i][j] = float(_lstm[i][j])

    _tft2, _st2, _lstm2 = [], [], []
    for i in range(len(_list)):
        _tft2.append(_tft[i])
        _st2.append(_st[i])
        _lstm2.append(_lstm[i])
        
    return np.array(_tft2), np.array(_st2), np.array(_lstm2)