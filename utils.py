import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def get_feature_tensor(path: str, train_fraction=1, evaluation_fraction=0) -> (
pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
pd.DataFrame):
    """
    Get train, evaluation (optional), and test tensor in the form of DataFrames.
    DataFrames are divided in three subgroups: Symbol, Lung, and Pleura features.
    :param path: root path to the data to be split into training, (evaluation, ) and test set
    :param train_fraction: fraction of the whole data found under <path> to be used for the training set.
        Must be a floating point between 0 and 1 (both inclusive).
    :param evaluation_fraction: fraction of the (1-train_fraction)*100 % of the data under <path> to be used for the
    evaluation set. Must be a floating point between 0 and 1 (both inclusive).
    :return: train, evaluation and test tensors for lung, pleura and symbol features each in the following order:
    Training feature Tensor for Lung features
    Training feature Tensor for Pleura features
    Training feature Tensor for Symbol features

    Evaluation feature Tensor for Lung features
    Evaluation feature Tensor for Pleura features
    Evaluation feature Tensor for Symbol features

    Testing feature Tensor for Lung features
    Testing feature Tensor for Pleura features
    Testing feature Tensor for Symbol features
    """
    features = pd.read_excel(path)
    features = features.fillna(-1)
    features["grouping"] = features["medicoID"]
    features.loc[features["grouping"].map(features["grouping"].value_counts()) == 1, "grouping"] = 0

    general_columns = features.loc[:,
                      ["Nachname", "Vorname", "Geburtsdatum", "medicoID", "grouping", "Untersuchungsdatum",
                       "technical_quality", "lateral_exposure"]].columns
    symbol_columns = features.filter(like="symbol_", axis=1).columns
    lung_columns = features.loc[:, "small_rounded_opacities_size_p":"large_opacities_lower_left"].columns
    pleura_columns = features.loc[:, "costophrenic_angle_obliteration_nad":"pleural_calcification_other_left"].columns

    symbol_train = features[general_columns.append(symbol_columns)]
    lung_train = features[general_columns.append(lung_columns)]
    pleura_train = features[general_columns.append(pleura_columns)]
    symbol_eval, lung_eval, pleura_eval = None, None, None
    symbol_test, lung_test, pleura_test = None, None, None
    if train_fraction < 1:
        # set medicoIDs of values being present only once, to 0 so that those patients appear in one group

        symbol_train, symbol_test = train_test_split(symbol_train, test_size=1 - train_fraction,
                                                     stratify=symbol_train["grouping"])
        lung_train, lung_test = train_test_split(lung_train, test_size=1 - train_fraction,
                                                 stratify=lung_train["grouping"])
        pleura_train, pleura_test = train_test_split(pleura_train, test_size=1 - train_fraction,
                                                     stratify=pleura_train["grouping"])

        if evaluation_fraction > 0:
            # split test in half for evaluation and test set:
            symbol_eval, symbol_test = train_test_split(symbol_test, test_size=1 - evaluation_fraction)
            lung_eval, lung_test = train_test_split(lung_test, test_size=1 - evaluation_fraction)
            pleura_eval, pleura_test = train_test_split(pleura_test, test_size=1 - evaluation_fraction)

    return lung_train, pleura_train, symbol_train, symbol_eval, lung_eval, pleura_eval, lung_test, pleura_test, symbol_test
