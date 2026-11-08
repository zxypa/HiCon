# Preparing datasets

Place chronological CSV files here (e.g., `ETTh1.csv`, `Weather.csv`).
Each row is a timestamp, and each numerical column is one feature. A text `date` column is allowed and automatically excluded.

`python -m hicon.train` trains on the first 70% of rows, validates on the next 10%, and evaluates on the last 20%.
**This generic split is not necessarily the official split for every TFB benchmark.** To reproduce published leaderboard numbers, adapt the dataset preprocessing, splits, and rolling evaluation to that exact benchmark protocol. Do not publish this project's numbers as if they were obtained using a different protocol.

Datasets are not redistributed here. Follow the respective datasets' original hosting terms and licenses.
