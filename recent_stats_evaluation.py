import os
import pandas as pd
import pybaseball
from pybaseball import statcast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings

# Suppress specific FutureWarnings from pybaseball
warnings.filterwarnings(
    "ignore",
    message=".*errors='ignore' is deprecated.*",
    category=FutureWarning,
    module='pybaseball.*'
)
warnings.filterwarnings(
    "ignore",
    message=".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning
)

pybaseball.cache.enable()

# -----------------------------------
# Step 1: Fetch Statcast Data
# -----------------------------------

def fetch_statcast_data(
    start_year=2013,
    end_year=2023,
    output_file='statcast_data.csv',
    max_retries=3
):
    if os.path.exists(output_file):
        print(f"[✓] Data file '{output_file}' already exists. Skipping download.")
        return pd.read_csv(output_file, low_memory=False)

    # Prepare CSV file with header
    header_written = False

    for year in range(start_year, end_year + 1):
        for month in range(3, 11):  # March to October
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-28"  # safe default

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Fetching data: {start_date} to {end_date} (Attempt {attempt})...")
                    df = statcast(start_dt=start_date, end_dt=end_date)

                    if df is not None and not df.empty:
                        mode = 'a' if header_written else 'w'
                        df.to_csv(output_file, mode=mode, header=not header_written, index=False)
                        header_written = True
                        print(f"[✓] Appended {len(df)} rows for {start_date} to {end_date}")
                    else:
                        print(f"[!] Empty data for {start_date} to {end_date}")
                    break
                except Exception as e:
                    print(f"[!] Failed to fetch {start_date} to {end_date}: {e}")
                    if attempt < max_retries:
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"[X] Giving up on {start_date} to {end_date} after {max_retries} attempts.")

    if not header_written:
        raise RuntimeError("No data was fetched. Exiting.")

    print(f"[✓] Data saved to {output_file}")
    return pd.read_csv(output_file, low_memory=False)

# -----------------------------------
# Step 2: Process Plate Appearances
# -----------------------------------

def get_pa_outcomes(df):
    df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df['is_on_base'] = df['events'].isin(['single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch']).astype(int)
    df['total_bases'] = df['events'].map({'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}).fillna(0)
    return df

def compute_rolling_stats(df, max_window=250):
    df = df.sort_values(by='game_date')
    results = []

    for window in range(1, max_window + 1):
        rolling_avg = df['is_hit'].rolling(window).mean()
        rolling_obp = df['is_on_base'].rolling(window).mean()
        rolling_slg = df['total_bases'].rolling(window).mean()

        temp = pd.DataFrame({
            'player_id': df['batter'],
            'game_date': df['game_date'],
            'rolling_window': window,
            'rolling_avg': rolling_avg,
            'rolling_obp': rolling_obp,
            'rolling_slg': rolling_slg,
            'next_is_hit': df['is_hit'].shift(-1),
            'next_is_on_base': df['is_on_base'].shift(-1),
            'next_total_bases': df['total_bases'].shift(-1),
        })

        results.append(temp)

    return pd.concat(results)

def process_all_players(data, output_file='rolling_stats.csv'):
    data = get_pa_outcomes(data)
    data = data[['game_date', 'batter', 'events', 'is_hit', 'is_on_base', 'total_bases']]
    data['game_date'] = pd.to_datetime(data['game_date'])

    grouped = data.groupby('batter')
    all_rolling = []

    for batter, group in tqdm(grouped, desc="Processing batters"):
        if len(group) < 260:
            continue
        stats = compute_rolling_stats(group)
        all_rolling.append(stats)

    final_df = pd.concat(all_rolling)
    final_df.dropna(inplace=True)
    final_df.to_csv(output_file, index=False)
    print(f"[✓] Saved rolling stats to {output_file}")
    return final_df

# -----------------------------------
# Step 3: Compute Correlations
# -----------------------------------

def compute_correlations(df, output_csv='correlation_table.csv'):
    grouped = df.groupby('rolling_window')

    results = []
    for window, group in grouped:
        corr_avg = group['rolling_avg'].corr(group['next_is_hit'])
        corr_obp = group['rolling_obp'].corr(group['next_is_on_base'])
        corr_slg = group['rolling_slg'].corr(group['next_total_bases'])

        results.append({
            'window': window,
            'avg_corr': corr_avg,
            'obp_corr': corr_obp,
            'slg_corr': corr_slg
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"[✓] Saved correlation table to {output_csv}")
    return result_df

# -----------------------------------
# Step 4: Plot Correlations
# -----------------------------------

def plot_correlation_graphs(df):
    def make_plot(df_slice, xlim, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(df_slice['window'], df_slice['avg_corr'], label='AVG', color='blue')
        plt.plot(df_slice['window'], df_slice['obp_corr'], label='OBP', color='green')
        plt.plot(df_slice['window'], df_slice['slg_corr'], label='SLG', color='red')
        plt.xlabel("Number of Prior Plate Appearances (N)")
        plt.ylabel("Correlation with Next Outcome")
        plt.title(title)
        plt.xlim(xlim)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[✓] Saved plot to {filename}")

    make_plot(df, xlim=(1, 250), title="Correlation vs Rolling Window (1–250)", filename="correlation_1_250.png")
    make_plot(df[df['window'] <= 10], xlim=(1, 10), title="Correlation vs Rolling Window (1–10)", filename="correlation_1_10.png")
    make_plot(df[df['window'] >= 11], xlim=(11, 250), title="Correlation vs Rolling Window (11–250)", filename="correlation_11_250.png")

# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def main():
    print("\n=== STEP 1: Fetching data ===")
    data = fetch_statcast_data()

    print("\n=== STEP 2: Processing rolling stats ===")
    rolling_df = process_all_players(data)

    print("\n=== STEP 3: Computing correlations ===")
    corr_df = compute_correlations(rolling_df)

    print("\n=== STEP 4: Plotting ===")
    plot_correlation_graphs(corr_df)

    print("\n✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
