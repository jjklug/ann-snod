import os
import random
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class DLoader(Dataset):
    """
    Custom data loader class, mostly copied from pytorch docs
    """
    def __init__(self, data, labels):
        self.data = data  # Your features (e.g., list of tensors or file paths)
        self.labels = labels  # Your corresponding labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def load_data(dpath: os.PathLike):
    return pd.read_csv(dpath)


def load_model_data(dpath: os.PathLike, batch_size: int = 1):
    # load train or test datat (should behave the same way)
    data = load_data(dpath)
    truth_data = data["snod_delta"]
    # now we can safely drop the truth column from the training data
    data.drop(columns=["snod_delta"], inplace=True)

    # convert dataframe to a tensor
    # note: I'm sure we could do this all with tensors, but I'm just gonna use a list so I know wtf is going on
    # start with a simple batch size of 1, can get fancier later
    inputs = []
    for index, row in data.iterrows():
        # Convert to tensor
        tensor_data = torch.tensor(row.values, dtype=torch.float32)
        inputs.append(tensor_data)

    # and repeat for the label data
    labels = []
    for v in truth_data:
        tens = torch.tensor([v], dtype=torch.float32)
        labels.append(tens)

    return inputs, labels


def min_max_normalize(v: float, min_v: float, max_v: float):
    return 2 * ((v - min_v) / (max_v - min_v)) - 1


def denormalize_min_max(v: float, min_v: float, max_v: float):
    # assumes normalization was: 2 * ((v - min_v) / (max_v - min_v)) - 1
    return (max_v - min_v) * ( ( v + 1 ) / 2 )  + min_v


def normalize_dataframe(df_in: pd.DataFrame, exclude_columns: list[str] = None):

    # okay we'll need to store the normalization params to recover values later.
    # at the very least, we'll need to recover the predicted value. I suppose the rest may not matter
    normalization_parameters = pd.DataFrame({'key': ['max_val', 'min_val']})

    # let's use a min-max normalization
    new_df = pd.DataFrame()
    for column_name in df_in.columns:
        if exclude_columns is not None and column_name in exclude_columns:
            # don't normalize this column
            new_col = df_in[column_name]
        else:
            new_col = []
            max_val = df_in[column_name].max()
            min_val = df_in[column_name].min()
            for val in df_in[column_name]:
                norm_val = min_max_normalize(val, min_val, max_val)
                new_col.append(norm_val)
            normalization_parameters[column_name] = [max_val, min_val]
        # now add the new col to the new df
        new_df[column_name] = new_col

    return new_df, normalization_parameters


def partition_data(in_df: pd.DataFrame, train_pct: float):
    # let's just start with a train, test split (no val - for now). Not getting that fancy
    if train_pct >= 1 or train_pct <= 0:
        raise ValueError("Invalid training pct specified. Must be in range (0, 1)")

    train_df = pd.DataFrame(columns=in_df.columns)
    test_df = pd.DataFrame(columns=in_df.columns)
    for index, row in in_df.iterrows():

        sample = random.random()
        if sample < train_pct:
            # add this row to the train df
            train_df = pd.concat([train_df, pd.DataFrame([row])], ignore_index=True)
        else:
            # add this row to the test df
            test_df = pd.concat([test_df, pd.DataFrame([row])], ignore_index=True)

    # lastly, shuffle the rows
    print(f"Shuffling rows of TRAIN and TEST datasets")
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    print(f"Size of TEST dataset: {len(test_df)}")
    print(f"Size of TRAIN dataset: {len(train_df)}")
    return train_df, test_df


def build_dataset(df: pd.DataFrame):
    """
    Take an input dataframe in timeseries format and build up a set of model input rows
    where each row has the predictors from the previous day, the days weather, and the true snow delta
    """

    dformat = "%Y-%m-%d"
    # the structure below defines the structure of the resulting dataframe
    row_struct = {
        "previous_snod": [],
        "previous_snod_delta": [],
        "previous_3d_precip": [],
        "day_mean_temp": [],
        "day_max_temp": [],
        "day_min_temp": [],
        "day_total_solar": [],
        "day_pressure": [],
        "day_rel_humidity": [],
        "day_precip": [],
        "dav_avg_wind_speed": [],
        "day_max_wind_speed": [],
        "snod_delta": [],

        # everything below here are specific features of the site - which means they are duplicated
        "pai_mean": [],
        "pai_sd": [],
        "rumple": [],
        "height_max": [],
        "height_mean": [],
        "height_95pct": [],
        "canopy_relief_ratio": [],
        "entropy": [],
        "pct_conifer": [],
        "ndvi_total_mean": [],
        "ndvi_con_mean": [],
        "vertical_gap_fraction": [],
        "Elevation": [],
    }
    out_df = pd.DataFrame(row_struct)
    for index, row in df.iterrows():
        # skip the first index...
        if index == 0:
            continue
        previous_day = datetime.datetime.strptime(df.iloc[index  - 1]['date'], dformat)
        day = datetime.datetime.strptime(row['date'], dformat)

        if (day - previous_day) != datetime.timedelta(days=1):
            # skip this, don't have previous day. probable gap
            continue

        # otherwise build up a row
        new_row = {
            "previous_snod": df.iloc[index - 1]["SNOD"],
            "previous_snod_delta": df.iloc[index - 1]["delta"],
            "previous_3d_precip": df.iloc[index - 1]["ant3d_precip"],
            "day_mean_temp": row["mean_temp"],
            "day_max_temp": row["max_temp"],
            "day_min_temp": row["min_temp"],
            "day_total_solar": row["total_solar"],
            "day_pressure": row["pressure"],
            "day_rel_humidity": row["rel_humidity"],
            "day_precip": row["precip"],
            "dav_avg_wind_speed": row["avg_wind_speed"],
            "day_max_wind_speed": row["max_wind_speed"],
            "snod_delta": row["delta"],

            # everything below here are specific features of the site - which means they are duplicated
            "pai_mean": row["pai_mean"],
            "pai_sd": row["pai_sd"],
            "rumple": row["rumple"],
            "height_max": row["height_max"],
            "height_mean": row["height_mean"],
            "height_95pct": row["height_95pct"],
            "canopy_relief_ratio": row["canopy_relief_ratio"],
            "entropy": row["entropy"],
            "pct_conifer": row["pct_conifer"],
            "ndvi_total_mean": row["ndvi_total_mean"],
            "ndvi_con_mean": row["ndvi_con_mean"],
            "vertical_gap_fraction": row["vertical_gap_fraction"],
            "Elevation": row["Elevation"],
        }

        # now add the new row
        out_df.loc[len(out_df)] = new_row

    return out_df