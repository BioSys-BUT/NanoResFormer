import glob
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# model_name = "model_20250825_*"
# model_name = "model_20250825_115512_best"
# model_name = "model_20250820_153720_best"
# model_name = "model_20250825_154418_best"

# model_name = "model_8x64_with_20_iter_params"
# results_dir = os.path.join(r"C:\Data\Jakubicek\NanoGeneNetV2\Results_Time",model_name)

model_name = "model_64x64_with_10_iter_params"
results_dir = os.path.join(r"C:\Data\Jakubicek\NanoGeneNetV2\Results_Time",model_name)

# model_name = "model_20250820_153720_best"
# results_dir = os.path.join(r"C:\Data\Jakubicek\NanoGeneNetV2\Results",model_name)

# results_dir = r"C:\Data\Jakubicek\NanoGeneNetV2\Results"

csv_files = glob.glob(os.path.join(results_dir, "**/*.csv"), recursive=True)


# Create an empty list to store DataFrames
dfs = []

# Iterate over all CSV files and read them into DataFrames
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Extract model name, OV, and BS from the file path
    match = re.search(r'results_OV(\d+)_BS(\d+)_D(.*)\.csv', os.path.basename(csv_file))
    model = model_name
    ov = int(match.group(1))
    bs = int(match.group(2))
    device = match.group(3)

    # Add metadata columns
    df['model'] = model
    df['OV'] = ov
    df['BS'] = bs
    df['Device'] = device.replace('.csv', '')
    dfs.append(df)

# Concatenate all DataFrames into one comprehensive DataFrame
full_df = pd.concat(dfs, ignore_index=True)

# Označ správné predikce (True Recognition, TR)
full_df['TR'] = full_df.apply(lambda row: int(row['Label']) in [int(x) for x in re.findall(r'\d+', str(row['Predictions']))], axis=1)


# Přepočítej Time na čas odpovídající 1 milion signálů
full_df['Time'] = full_df['Time'] * 1_000_000 /60/60
full_df = full_df[full_df['Time'] > 0.0]

full_df = full_df[full_df['OV'].isin([50, 75, 90])]

# # Výpočet úspěšnosti TR (True Recognition) pouze pro GPU a BS=64 v závislosti na OV
# tr_success_rate_gpu_bs64 = full_df[(full_df['Device'] == 'cuda') & (full_df['BS'] == 64)].groupby(['OV'])['TR'].mean().reset_index()
# print("Úspěšnost TR pro GPU a BS=64 podle OV:")
# print(tr_success_rate_gpu_bs64)

# plt.figure(figsize=(8, 5))
# sns.barplot(data=tr_success_rate_gpu_bs64, x='OV', y='TR', palette='viridis')
# plt.title('Úspěšnost TR podle OV (GPU, BS=64)')
# plt.xlabel('OV')
# plt.ylabel('Úspěšnost TR')
# plt.ylim(0, 1)

# # Vypočítej mean Time pro GPU a BS=64 podle OV
# mean_time_gpu_bs64 = full_df[(full_df['Device'] == 'cuda') & (full_df['BS'] == 64)].groupby(['OV'])['Time'].mean().reset_index()
# # sum_time_gpu_bs64 = full_df[(full_df['Device'] == 'cuda') & (full_df['BS'] == 64)].groupby(['OV'])['Time'].sum().reset_index()


# # num_signals = full_df[(full_df['Device'] == 'cuda') & (full_df['BS'] == 64) & (full_df['OV'] == 90)].shape[0]
# # mean_time_gpu_bs64 = sum_time_gpu_bs64.copy()
# # mean_time_gpu_bs64['Time'] = mean_time_gpu_bs64['Time'] * 1_000_000 / num_signals
# # mean_time_gpu_bs64['Time'] = mean_time_gpu_bs64['Time'] / 3600

# for i, ov in enumerate(tr_success_rate_gpu_bs64['OV']):
#     tr_value = tr_success_rate_gpu_bs64[tr_success_rate_gpu_bs64['OV'] == ov]['TR'].values[0]
#     mean_time_row = mean_time_gpu_bs64[mean_time_gpu_bs64['OV'] == ov]['Time']
#     if not mean_time_row.empty:
#         mean_time = mean_time_row.values[0]
#         plt.text(i, tr_value/2 + 0.05, f"{mean_time:.2f}\nTR: {tr_value:.2f}",
#                  ha='center', va='center', fontsize=10, color='black')
#     else:
#         plt.text(i, tr_value/2 + 0.05, f"N/A\nTR: {tr_value:.2f}", 
#                  ha='center', va='center', fontsize=10, color='red')


# plt.tight_layout()
# plt.show()



# Zobraz dva grafy: závislost OV na BS=1 a OV na BS=64
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Data pro BS=1, rozdělené podle Device
df_bs1 = full_df[full_df['BS'] == 1]
sns.boxplot(data=df_bs1, x='OV', y='Time', hue='Device', ax=axes[0])
axes[0].set_title('Non-paralelized (BatchSize=1)')
# Přidej průměrné hodnoty do boxplotu
mean_times_bs1 = df_bs1.groupby(['OV', 'Device'])['Time'].mean().reset_index()
for i, ov in enumerate(sorted(df_bs1['OV'].unique())):
    devices = sorted(df_bs1['Device'].unique())
    for j, device in enumerate(devices):
        mean_time = mean_times_bs1[(mean_times_bs1['OV'] == ov) & (mean_times_bs1['Device'] == device)]['Time']
        if not mean_time.empty:
            # x_pos = i - 1/5 + j * (2/5)
            x_pos = i - 1/10 + j * (4/10)
            q3_time = df_bs1[(df_bs1['OV'] == ov) & (df_bs1['Device'] == device)]['Time'].quantile(0.75) - 2
            axes[0].text(x_pos, q3_time, f"{mean_time.values[0]:.2f}",
                         ha='center', va='bottom', fontsize=9, color='black')
axes[0].set_xlabel('Overlap [%]')
axes[0].set_ylabel('Time [hrs]')
axes[0].set_ylim(0, 50)

# Data pro BS=64, rozdělené podle Device
df_bs64 = full_df[full_df['BS'] == 64]
sns.boxplot(data=df_bs64, x='OV', y='Time', hue='Device', ax=axes[1])
axes[1].set_title('Paralelized (BatchSize=64)')

# Přidej průměrné hodnoty do boxplotu
mean_times = df_bs64.groupby(['OV', 'Device'])['Time'].mean().reset_index()
for i, ov in enumerate(sorted(df_bs64['OV'].unique())):
    devices = sorted(df_bs64['Device'].unique())
    for j, device in enumerate(devices):
        mean_time = mean_times[(mean_times['OV'] == ov) & (mean_times['Device'] == device)]['Time']
        if not mean_time.empty:
            # x_pos = i - 1/5 + j * (2/5)
            x_pos = i - 1/10 + j * (4/10)
            q3_time = df_bs64[(df_bs64['OV'] == ov) & (df_bs64['Device'] == device)]['Time'].quantile(0.75) - 2
            axes[1].text(x_pos, q3_time, f"{mean_time.values[0]:.2f}",
                         ha='center', va='bottom', fontsize=9, color='black')

axes[1].set_xlabel('Overlap [%]')
axes[1].set_ylabel('Time [hrs]')
axes[1].set_ylim(0, 50)

axes[0].legend(loc='upper left', title='Device')
axes[1].legend(loc='upper left', title='Device')

plt.tight_layout()
plt.show()

