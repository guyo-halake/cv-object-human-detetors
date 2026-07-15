import pandas as pd
import os

class DataExporter:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        self.csv_data = []

    def log_frame(self, frame_num, timestamp, counts):
        row_data = {"frame": frame_num, "timestamp": timestamp}
        for c, count in counts["global"].items():
            if count > 0:
                row_data[f"{c}_total"] = count
        for c, count in counts["in"].items():
            if count > 0:
                row_data[f"{c}_in"] = count
        for c, count in counts["out"].items():
            if count > 0:
                row_data[f"{c}_out"] = count
        self.csv_data.append(row_data)

    def export_csv(self, filename="counts_log.csv"):
        path = os.path.join(self.output_dir, filename)
        if not self.csv_data:
            df = pd.DataFrame(columns=["frame", "timestamp"])
        else:
            df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=False)
        return path
