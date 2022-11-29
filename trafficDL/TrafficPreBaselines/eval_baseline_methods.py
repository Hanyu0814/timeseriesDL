import argparse
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.var_model import VAR

from metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from utils import StandardScaler


def historical_average_predict(df, period, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test


def static_predict(df, n_forward, test_ratio=0.2):
    """
    Assumes $x^{t+1} = x^{t}$
    :param df:
    :param n_forward:
    :param test_ratio:
    :return:
    """
    test_num = int(round(df.shape[0] * test_ratio))
    y_test = df[-test_num:]
    y_predict = df.shift(n_forward).iloc[-test_num:]
    return y_predict, y_test


def var_predict(df, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test


def eval_static(traffic_reading_df):
    print('Static')
    horizons = [1, 3, 6, 12]
    print('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = static_predict(traffic_reading_df, n_forward=horizon, test_ratio=0.2)
        rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        line = 'Static\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        print(line)


def eval_historical_average(traffic_reading_df, period):
    y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    print('Historical Average')
    print('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        print(line)


def eval_var(traffic_reading_df, n_lags=3):
    n_forwards = [1, 3, 6, 12]
    y_predicts, y_test = var_predict(traffic_reading_df, n_forwards=n_forwards, n_lags=n_lags,
                                     test_ratio=0.2)
    print('VAR (lag=%d)' % n_lags)
    print('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        print(line)


def main(args):
    df = np.load(args.traffic_reading_filename, allow_pickle=True)
    num_tsteps, num_nodes, num_directions = df.shape[0], df.shape[1]*df.shape[2], df.shape[3]
    for di in range(num_directions):
        data = df[:, :, :, di].reshape((num_tsteps, num_nodes))
        pdata = pd.DataFrame(data)
        print('Evaluating baselines for direction {}'.format(di))
        eval_static(pdata)
        eval_historical_average(pdata, period=48)
        eval_var(pdata, n_lags=48)
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', default="data/i10_data.npy", type=str,
                        help='Path to the traffic Dataframe.')
    args = parser.parse_args()
    main(args)
