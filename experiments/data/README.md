# Datasets

The raw datasets are **not** committed (they are large and redistributed under
their own UCI licenses). Run `./download_data.sh` from this directory to fetch
them into `data/real/`, which is where every experiment script expects them.

All five are from the UCI Machine Learning Repository.

| Dataset (script name) | File placed in `data/real/` | Sep | Header | UCI source |
|---|---|---|---|---|
| Census (`census`)     | `adult.data`                       | `,` | no  | Adult |
| Forest (`forest`)     | `covtype.data`                     | `,` | no  | Covertype |
| Power  (`power`)      | `household_power_consumption.txt`  | `;` | yes | Individual household electric power consumption |
| Wine   (red/white)    | `winequality-red.csv`, `winequality-white.csv` | `;` | yes | Wine Quality |
| Bike   (`bike`)       | `hour.csv`                         | `,` | yes | Bike Sharing |

## Column indices used by the experiments

Columns are referenced by 0-based index in the source files (no renaming).

**Cross-column pairs** (`crosscol_feedback_experiment.py`):
- bike `hour.csv`: (10 `temp`, 11 `atemp`), (14 `casual`, 16 `cnt`)
- wine-white: (5 `free SO2`, 6 `total SO2`)
- wine-red: (0 `fixed acidity`, 8 `pH`), (7 `density`, 10 `alcohol`)
- forest `covtype.data`: (6 `Hillshade_9am`, 8 `Hillshade_3pm`), (0 `Elevation`, 5 `Horizontal_Distance_To_Roadways`)
- census `adult.data`: (0 `age`, 12 `hours-per-week`)

**Single-column drift** (`run_real_local.sh` → held-out test sets):
- power col 2 (`Global_active_power`), forest col 0 (`Elevation`), census col 0 (`age`)

**Training pool** (`gen_train_pool.sh` → leakage-free, disjoint from the test sets):
- wine-red cols 0–10, wine-white cols 0–10, bike cols 10–16 (Wine + Bike only)

## License note

Each dataset is governed by its UCI dataset license / citation requirements;
cite the original dataset papers if you use them. UCI download URLs occasionally
change — if a download 404s, search the UCI repository for the dataset name.
