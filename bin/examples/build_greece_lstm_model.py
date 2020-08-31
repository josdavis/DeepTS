import pandas as pd
import numpy as np
from KerasFactory import KerasModelFactory
import yaml
from utils import make_supervised, transform_series_for_lstm, normalize_series

"""
Usage: 
python ./bin/examples/build_greece_lstm_model.py
"""
input_file = "/Users/josephdavis/Documents/ds_lead/ds_toolbox/code/DeepTs/data/hackathon_thessaly_dataset.csv" # TODO change hard coded file path

def get_data_frame():
    base_file = input_file
    df = pd.read_csv(base_file,sep=",")
    df = df[["TARGET_total_pest_count", "trap_crop", "date", "total_precip_mm_sum_1_weeks_ago_0_years_ago", "week"]]
    print(df.head())
    return df

def build_model():
    strm = open("/Users/josephdavis/Documents/ds_lead/ds_toolbox/code/DeepTs/model_config/lstm_model.yaml")
    lstm_config = yaml.load(strm)
    kmf = KerasModelFactory()
    model = kmf.create_keras_model(nn_architecture_dict=lstm_config)
    model.summary()
    return model

def extract_io(grped):
    supervisedDf = make_supervised(df=grped,
                                   lookback=14,
                                   timestep=1,
                                   nonTsFeatureCols=["trap_crop","week",
                                                     "total_precip_mm_sum_1_weeks_ago_0_years_ago"],
                                   timeCol="date",
                                   targetCol="TARGET_total_pest_count",
                                   tsCols=None,
                                   variable_prefix="var",
                                   target_prefix="tar",
                                   groupbyColList=["trap_crop"]
                                   )

    train = supervisedDf.copy()
    train_x = transform_series_for_lstm(train.drop(["tar1(t)",
                                                    "total_precip_mm_sum_1_weeks_ago_0_years_ago",
                                                    "trap_crop",
                                                    "week"],
                                                   axis=1).values)
    train_y = transform_series_for_lstm(train["tar1(t)"].values)

    dtev = train[["total_precip_mm_sum_1_weeks_ago_0_years_ago"]].values

    dtev, dtem, dtes = normalize_series(dtev)
    trap_crop_X = pd.get_dummies(train[["trap_crop"]]).values
    week_X = pd.get_dummies(train["week"].map(lambda x:
                                                            str(x))).values
    train_y = train_y.reshape((train_y.shape[0],1))
    return train_x,train_y,dtev,dtem, dtes, trap_crop_X, week_X


if __name__=="__main__":
    model = build_model()
    grped = get_data_frame()
    train_X, train_y, days_to_event_vals, days_to_event_vals_mean, \
        days_to_event_vals_std, trap_crop_X, week_X = \
            extract_io(grped=grped)
    history = model.fit([train_X, days_to_event_vals,trap_crop_X, week_X],
                        train_y,
                        batch_size=25,
                        epochs=40,
                        validation_split=0.1)