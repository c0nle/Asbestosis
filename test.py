import math
import os

import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_path = "D:\\Projects\\Thorax\\DeboraThorax\\" #  "/home/debora/Documents/Projects/Thorax/"  #
data = pd.read_excel(main_path + "found_merged_data.xlsx")  # merged_data_prepared.csv")
print(data["Geburtsdatum"].describe())
birth_col = pd.to_datetime(data["Geburtsdatum"], format="%d.%m.%Y")
exam_col = pd.to_datetime(data["Untersuchungsdatum"], format="%d.%m.%Y")
age = ((exam_col - birth_col) / pd.Timedelta(days=365.25)).astype(int)
print(age.min())
print(age.max())
print(age.mean())


symbol_columns = [col for col in data.columns if col.startswith('symbol')]
current_symbol_cols = [col for col in symbol_columns if col in data.columns]
symbols_df = data[symbol_columns]
#print(symbols_df.describe())

lung_columns = [
    "small_rounded_opacities_size_p",
    "small_rounded_opacities_size_q",
    "small_rounded_opacities_size_r",
    "small_rounded_opacities_profusion",
    "small_rounded_opacities_upper_right",
    "small_rounded_opacities_middle_right",
    "small_rounded_opacities_lower_right",
    "small_rounded_opacities_upper_left",
    "small_rounded_opacities_middle_left",
    "small_rounded_opacities_lower_left",
    "small_irregular_opacities_size_s",
    "small_irregular_opacities_size_t",
    "small_irregular_opacities_size_u",
    "small_irregular_opacities_profusion",
    "small_irregular_opacities_upper_right",
    "small_irregular_opacities_middle_right",
    "small_irregular_opacities_lower_right",
    "small_irregular_opacities_upper_left",
    "small_irregular_opacities_middle_left",
    "small_irregular_opacities_lower_left",
    "mixed_shapes_1",
    "mixed_shapes_2",
    "mixed_shapes_profusion",
    "mixed_shapes_upper_right",
    "mixed_shapes_middle_right",
    "mixed_shapes_lower_right",
    "mixed_shapes_upper_left",
    "mixed_shapes_middle_left",
    "mixed_shapes_lower_left",
    "large_opacities_nad",
    "large_opacities",
    "large_opacities_upper_right",
    "large_opacities_middle_right",
    "large_opacities_lower_right",
    "large_opacities_upper_left",
    "large_opacities_middle_left",
    "large_opacities_lower_left",
    "costophrenic_angle_obliteration_nad",
    "costophrenic_angle_obliteration_right",
    "costophrenic_angle_obliteration_left"
]
current_lung_cols = [col for col in lung_columns if col in data.columns]
lung_df = data[current_lung_cols]
#print(lung_df.describe())


pleura_columns = [
    "diffuse_pleural_thickening_nad",
    "diffuse_pleural_thickening_extent_right",
    "diffuse_pleural_thickening_width_right",
    "diffuse_pleural_thickening_small_right",
    "diffuse_pleural_thickening_face_on_right",
    "diffuse_pleural_thickening_extent_left",
    "diffuse_pleural_thickening_width_left",
    "diffuse_pleural_thickening_small_left",
    "diffuse_pleural_thickening_face_on_left",
    "diffuse_pleural_thickening_upper_right",
    "diffuse_pleural_thickening_middle_right",
    "diffuse_pleural_thickening_lower_right",
    "diffuse_pleural_thickening_upper_left",
    "diffuse_pleural_thickening_middle_left",
    "diffuse_pleural_thickening_lower_left",
    "localized_pleural_thickening_nad",
    "localized_pleural_thickening_extent_right",
    "localized_pleural_thickening_width_right",
    "localized_pleural_thickening_small_right",
    "localized_pleural_thickening_face_on_right",
    "localized_pleural_thickening_extent_left",
    "localized_pleural_thickening_width_left",
    "localized_pleural_thickening_small_left",
    "localized_pleural_thickening_face_on_left",
    "localized_pleural_thickening_diaphragm_right",
    "localized_pleural_thickening_diaphragm_left",
    "localized_pleural_thickening_chest_wall_right",
    "localized_pleural_thickening_chest_wall_left",
    "pleural_calcification_nad",
    "pleural_calcification_diaphragm_right",
    "pleural_calcification_diaphragm_left",
    "pleural_calcification_chest_wall_right",
    "pleural_calcification_chest_wall_left",
    "pleural_calcification_other_right",
    "pleural_calcification_other_left",
    "occupational_disease_nad",
    "occupational_disease_silicosis",
    "occupational_disease_silicotuberculosis",
    "occupational_disease_quartz_dust_lung_cancer",
    "occupational_disease_asbestosis",
    "occupational_disease_asbestos_pleura",
    "occupational_disease_asbestos_lung_cancer",
    "occupational_disease_asbestos_larynx_cancer",
    "occupational_disease_asbestos_mesothelioma",
    "occupational_disease_ionized_radiation"
]
current_pleura_cols = [col for col in pleura_columns if col in data.columns]
pleura_df = data[current_pleura_cols]
#print(pleura_df.describe())


def plot_distribution(df, title_value):
    numeric_cols = df.select_dtypes(include=['number']).columns
    color_map = {'0.0': 'b', '1.0': 'm', '2.0': 'y', 'NaN': 'g'}
    all_possible_values = ['0.0', '1.0', '2.0', 'NaN']

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(20, 20), sharey=True)
    fig.suptitle("Distribution of " + title_value + " values", fontsize=40)
    i = 0
    for row in ax:
        for col in row:
            if i >= len(numeric_cols):
                col.axis('off')
                continue
            col_name = numeric_cols[i]
            while col_name not in df.columns:
                i = i+1
                col_name = numeric_cols[i]
            column_values = df[col_name].fillna('NaN').astype(str).value_counts()
            counts = [column_values.get(v, 0) for v in all_possible_values]
            colors = [color_map[v] for v in all_possible_values]
            col.bar(all_possible_values, counts, color=colors)
            col.set_title(col_name.replace('_pleural', '').title())
            i = i+1
    plt.savefig(f'{main_path}data_analysis{os.sep}{title_value}.png')


# Funktion zur Analyse der Korrelation zwischen zwei Spalten
def analyze_correlation(df, col1, col2):
    correlation_matrix = df[[col1, col2]].corr()

    print(f"Korrelation zwischen {col1} und {col2}:")
    print(correlation_matrix)

def plot_correlation(df, title):
    data_encoded = df.copy()
    for col in df.select_dtypes(include='O').columns:
        data_encoded[col] = pd.factorize(df[col])[0]

    correlation_matrix = data_encoded.corr()
    threshold = 0.7
    high_corr_pairs = correlation_matrix[(correlation_matrix > threshold) | (correlation_matrix < -threshold)]
    plt.figure(figsize=(20, 16))
    sns.heatmap(high_corr_pairs, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Korrelationsmatrix {title}".format(title=title))
    plt.savefig(f'{main_path}data_analysis{os.sep}corr_{title}.png')

# Verteilung darstellen
distributions = pd.DataFrame({
    'Column': [col for col in current_lung_cols],
    'Entries': [data[col].unique() for col in current_lung_cols]
})
distributions.to_excel(f'{main_path}data_analysis{os.sep}Distributions_Lunge.xlsx', index=False)

distributions = pd.DataFrame({
    'Column': [col for col in current_pleura_cols],
    'Entries': [data[col].unique() for col in current_pleura_cols]
})
distributions.to_excel(f'{main_path}data_analysis{os.sep}Distributions_Pleura.xlsx', index=False)

distributions = pd.DataFrame({
    'Column': [col for col in current_symbol_cols],
    'Entries': [data[col].unique() for col in current_symbol_cols]
})
distributions.to_excel(f'{main_path}data_analysis{os.sep}Distributions_Symbols.xlsx', index=False)

#plot_correlation(pleura_df, "Pleura")
#plot_correlation(lung_df, "Lung")
#plot_correlation(symbols_df, "Symbols")

#plot_distribution(symbols_df)
plot_distribution(lung_df, "Lung")
plot_distribution(pleura_df, "Pleura")

# Korrelation zwischen zwei Spalten analysieren (z.B. Spalte1 und Spalte2)
#analyze_correlation(df, 'Spalte1', 'Spalte2')