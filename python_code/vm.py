import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import pytz  # ใช้ timezone-aware timestamp

# ==== CONFIG ====
csv_path = "data.csv"  # path ไปยังไฟล์ CSV
bucket = "Systern-monirtor"
org = "TEE-TPE"
token = "Usc9_YwEAaHVe_TmynL8FkTOlpGOL11m6JErizQX8nDoXHv1MEkIQUqFUYvTyYS2G8r3X3edTrm_KNnQhwa5PQ=="
url = "http://localhost:8086"

# ==== เชื่อมต่อ InfluxDB ====
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# ==== อ่าน CSV ====
df = pd.read_csv(csv_path)

# แปลง Timestamp ให้เป็น datetime พร้อม timezone
df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize("Asia/Bangkok").dt.tz_convert("UTC")

# ==== เขียนลง InfluxDB ====
for _, row in df.iterrows():
    point = (
        Point("environment")  # ชื่อ measurement
        .tag("Component", row["Component"])
        .tag("Status", row["Status"])
        .field("dB", float(row["dB"]))
        .field("TopFreq1", float(row["TopFreq1"]))
        .field("TopFreq2", float(row["TopFreq2"]))
        .field("TopFreq3", float(row["TopFreq3"]))
        .time(row["Timestamp"], WritePrecision.NS)
    )
    write_api.write(bucket=bucket, org=org, record=point)

print("✅ Upload success!")
