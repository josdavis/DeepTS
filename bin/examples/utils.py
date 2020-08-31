import pandas as pd
import numpy as np


def make_supervised_single_grpup(df, lookback, timestep,
                    nonTsFeatureCols, timeCol,
                    targetCol, tsCols=None,
                    variable_prefix="var",
                    target_prefix="tar"):
    """
    :param df: The pandas DataFrame that contains the TS data that needs to be
    converted to a supervised learning problem.
    :param lookback: The time steps to look back for each row
    :param timestep: The time steps ahead that needs to be forecast
    :param nonTsFeatureCols: The feature columns that are not TS and have to
    be retained.
    :param timeCol: The column that contains the time information.
    :param targetCol: The columns that is to be forecast or predicted.
    :param tsCols: The features which are a TS but not including the target
    column.
    :param variable_prefix: The prefix to be assigned to the columns for past
    time steps
    :param target_prefix: The prefix of the target column
    :return: Pandas data frame which contains TS data represented as a
    supervised learning problem.
    """
    if tsCols:
        targetCol = targetCol + tsCols
    df = df.set_index(timeCol)
    tsDf = df[[targetCol]].copy()
    nonTsDf = df[nonTsFeatureCols]

    n_vars = tsDf.shape[1]
    # input sequence (t-n, ... t-1)
    cols, names = list(), list()
    for i in range(lookback, 0, -1):
        cols.append(tsDf.shift(i))
        names += [('%s%d(t-%d)' % (variable_prefix,j + 1, i))
                  for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, timestep):
        cols.append(tsDf.shift(-i))
        if i == 0:
            names += [('%s%d(t)' % (target_prefix,j + 1))
                      for j in range(n_vars)]
        else:
            names += [('%s%d(t+%d)' % (target_prefix,j + 1, i))
                      for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg[nonTsDf.columns] = nonTsDf
    # Return only the sequences which have the full sequence
    # without Nans introduced due to shifting
    if timestep > 1:
        return agg[lookback:-timestep + 1]
    return agg[lookback:]


def make_supervised(df, lookback, timestep,
                    nonTsFeatureCols, timeCol,
                    targetCol, tsCols=None,
                    variable_prefix="var",
                    target_prefix="tar",
                    groupbyColList=None):
    """

    :param df: The pandas DataFrame that contains the TS data that needs to be
    converted to a supervised learning problem.
    :param lookback: The time steps to look back for each row
    :param timestep: The time steps ahead that needs to be forecast
    :param nonTsFeatureCols: The feature columns that are not TS and have to
    be retained.
    :param timeCol: The column that contains the time information.
    :param targetCol: The columns that is to be forecast or predicted.
    :param tsCols: The features which are a TS but not including the target
    column.
    :param variable_prefix: The prefix to be assigned to the columns for past
    time steps
    :param target_prefix: The prefix of the target column
    :param groupbyColList: List of columns to group by. If specified the make
    supervised will group by this key and treat each batch separately and
    then concatenate the results
    :return: Pandas data frame which contains TS data represented as a
    supervised learning problem.
    """
    if groupbyColList:
        dfList = list()
        for key, grp in df.groupby(groupbyColList):
            tdf = make_supervised_single_grpup(grp, lookback, timestep,
                                               nonTsFeatureCols, timeCol,
                                               targetCol, tsCols,
                                               variable_prefix,
                                               target_prefix)
            dfList.append(tdf)
        return pd.concat(dfList)
    else:
        return make_supervised_single_grpup(df, lookback, timestep,
                                            nonTsFeatureCols, timeCol,
                                            targetCol, tsCols,
                                            variable_prefix,
                                            target_prefix)


def transform_series_for_lstm(series_array, dim_timestep=1):
    """
    reshapes the series to be compatible with the KERAS LSTM cell
    which expects the inout to be of shape (BATCH_SIZE, TIME_WINDOW,
    DIMENSION_OF_EACH_TIMESTEP
    :param series_array: The numpy array which contains the sequences
    :param dim_timestep: The number of dimensions for each time step through the
    LSTM cell.
    :return:
    """
    series_array = series_array.reshape(series_array.shape + (dim_timestep,))
    return series_array


def normalize_series(series_array):
    """
    Normalizes the series by removing the mean and dividing by
    standard deviation
    :param series_array: The numpy array which has to be normalized
    :return: Tuple of normalized numpy array, series mean and
    series standard deviation
    """
    series_array = np.log1p(np.nan_to_num(series_array))
    series_mean = series_array.mean()
    series_std = series_array.std()
    series_array -= series_mean
    series_array /= series_std
    return series_array, series_mean, series_std