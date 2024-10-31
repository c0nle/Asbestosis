import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("D:\\Projects\\Thorax\\found_merged_data.xlsx")

symbol_columns = [col for col in data.columns if col.startswith('symbol')]
symbols_df = data[symbol_columns]
print(symbols_df.describe())

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
lung_df = data[lung_columns]
print(lung_df.describe())


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
pleura_df = data[pleura_columns]
print(pleura_df.describe())


def plot_distribution(df):
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution von {col}')
        plt.xlabel(col)
        plt.ylabel('Häufigkeit')
        plt.savefig(f'D:\\Projects\\Thorax\\data_analysis\\{col}.png')


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
    plt.savefig(f'D:\\Projects\\Thorax\\data_analysis\\corr_{title}.png')

# Verteilung darstellen
'''distributions = pd.DataFrame({
    'Column': [col for col in pleura_columns],
    'Entries': [data[col].unique() for col in pleura_columns]
})
distributions.to_excel('D:\\Projects\\Thorax\\data_analysis\\Distributions_Pleura.xlsx', index=False)
'''
plot_correlation(pleura_df, "Pleura")
plot_correlation(lung_df, "Lung")
plot_correlation(symbols_df, "Symbols")

#plot_distribution(symbols_df)
#plot_distribution(lung_df)
#plot_distribution(pleura_df)

# Korrelation zwischen zwei Spalten analysieren (z.B. Spalte1 und Spalte2)
#analyze_correlation(df, 'Spalte1', 'Spalte2')