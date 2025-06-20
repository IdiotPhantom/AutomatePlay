import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setting import TRAINING_DATA_PATH

OLD_WIDTH = 720
OLD_HEIGHT = 1280
NEW_WIDTH = 960
NEW_HEIGHT = 540

input_file = os.path.join(TRAINING_DATA_PATH, "bannedtraining.csv")
output_file = os.path.join("training_converted.csv")

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) < 6:
            continue  # skip malformed rows

        path, task_id, x_norm_old, y_norm_old, done, confidence = row

        x_pixel = float(x_norm_old) * OLD_WIDTH
        y_pixel = float(y_norm_old) * OLD_HEIGHT

        x_norm_new = x_pixel / NEW_WIDTH
        y_norm_new = y_pixel / NEW_HEIGHT

        writer.writerow([
            path,
            task_id,
            f"{x_norm_new:.4f}",
            f"{y_norm_new:.4f}",
            done,
            confidence
        ])
