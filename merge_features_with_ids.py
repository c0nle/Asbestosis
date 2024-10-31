import pandas as pd

feature_path = "D:\\Projects\\Thorax\\ArbeitsRadio\\ArbeitsRadio\\merged_data.csv" # "/home/debora/Documents/Projects/Thorax/merged_data.csv"
id_path = "D:\\Projects\\Thorax\\st_Befundtext_RO_Thorax_AR.xlsx"  # "/home/debora/Documents/Projects/Thorax/st_Befundtext_RO_Thorax_AR.xlsx"

feature_data = pd.read_csv(feature_path)
feature_data['Geburtsdatum'] = pd.to_datetime(feature_data['Geburtsdatum'])
feature_data['Untersuchungsdatum'] = pd.to_datetime(feature_data['Untersuchungsdatum'])


id_data = pd.read_excel(id_path, skiprows=8)
id_data = id_data.drop(columns=["Unnamed: 0", "Unnamed: 6"])
id_data.rename(columns={"Name": "Nachname"}, inplace=True)

found_data = pd.merge(feature_data, id_data, how='inner', on=['Nachname', 'Vorname', 'Geburtsdatum', 'Untersuchungsdatum'])

found_data.to_excel("D:\\Projects\\Thorax\\found_merged_data.xlsx", index=False)