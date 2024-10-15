import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def get_feature_tensor(path: str, train_fraction=1) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    features = pd.read_excel(path)
    features = features.fillna(-1)
    features["grouping"] = features["medicoID"]
    features.loc[features["grouping"].map(features["grouping"].value_counts()) == 1, "grouping"] = 0

    general_columns = features.loc[:, ["Nachname", "Vorname", "Geburtsdatum", "medicoID", "grouping", "Untersuchungsdatum", "technical_quality", "lateral_exposure"]].columns
    symbol_columns = features.filter(like="symbol_", axis=1).columns
    lung_columns = features.loc[:, "small_rounded_opacities_size_p":"large_opacities_lower_left"].columns
    pleura_columns = features.loc[:, "costophrenic_angle_obliteration_nad":"pleural_calcification_other_left"].columns

    symbol_train = features[general_columns.append(symbol_columns)]
    lung_train = features[general_columns.append(lung_columns)]
    pleura_train = features[general_columns.append(pleura_columns)]
    symbol_test, lung_test, pleura_test = None, None, None
    if train_fraction < 1:
        # set medicoIDs of values being present only once, to 0 so that those patients appear in one group

        symbol_train, symbol_test = train_test_split(symbol_train, test_size=1-train_fraction, stratify=symbol_train["grouping"])
        lung_train, lung_test = train_test_split(lung_train, test_size=1-train_fraction, stratify=lung_train["grouping"])
        pleura_train, pleura_test = train_test_split(pleura_train, test_size=1-train_fraction, stratify=pleura_train["grouping"])
    return lung_train, pleura_train, symbol_train, lung_test, pleura_test, symbol_test