import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "Usc9_YwEAaHVe_TmynL8FkTOlpGOL11m6JErizQX8nDoXHv1MEkIQUqFUYvTyYS2G8r3X3edTrm_KNnQhwa5PQ=="

org = "TEE-TPE"
url = "http://localhost:8086"

write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

bucket="Systern-monirtor"


write_api = write_client.write_api(write_options=SYNCHRONOUS)
   
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="TEE-TPE", record=point)
  time.sleep(1) # separate points by 1 second
