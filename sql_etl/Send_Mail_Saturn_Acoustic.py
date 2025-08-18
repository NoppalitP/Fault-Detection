import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import time
# SMTP Relay configuration
smtp_server = "mailrelay.hgst.com"
smtp_port = 25  # Standard SMTP port without TLS/SSL

# Email details
sender_email = "PDL-PRB-TEE-Automatic-Report@wdc.com"
receiver_emails = [
    # "chaiwat.chantana@wdc.com",
    #"nobpharit.phosrithat@wdc.com",
    # "kittisak.boo@wdc.com",
     #"ratchanon.nooraksa@wdc.com",
     "siwat.ninlachat@wdc.com"
]

# Email content
subject = "🔔 Saturn-Acoustic Tool Monitoring Alert"
body = """
<p>Dear Team,</p>

<p>This is an automated alert from the <strong>Saturn-Acoustic tool monitoring system</strong>.</p>

<p>Please check the system status. If any irregularities are found, kindly notify the System Administrator.</p>

<p>Thank you,<br>
Automated Notification System</p>
"""

# Sample data
d = {
    "Tester ID": ["IPC1"],
    "Sound Level (dB)": ["90"],
    "Timestamp": ["2025-06-07 17:00:00"]
}
df = pd.DataFrame(data=d)

# Style the HTML table with CSS
styled_table = df.to_html(index=False, classes="styled-table", border=0)

# Full HTML content
html = f"""\
<html>
  <head>
    <style>
      body {{
        font-family: Arial, sans-serif;
        font-size: 14px;
        color: #333333;
      }}
      .styled-table {{
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        min-width: 400px;
        border: 1px solid #dddddd;
      }}
      .styled-table thead tr {{
        background-color: #009879;
        color: #ffffff;
        text-align: left;
      }}
      .styled-table th,
      .styled-table td {{
        padding: 12px 15px;
        border: 1px solid #dddddd;
      }}
      .styled-table tbody tr:nth-child(even) {{
        background-color: #f3f3f3;
      }}
    </style>
  </head>
  <body>
    {body}
    <h3>📋 Acoustic Monitoring Data</h3>
    {styled_table}
  </body>
</html>
"""

# Create MIME message
message = MIMEMultipart("alternative")
message["From"] = sender_email
message["To"] = ", ".join(receiver_emails)
message["Subject"] = subject
message.attach(MIMEText(html, "html"))
i = 1
# Send email
while True:
  try:
      with smtplib.SMTP(smtp_server, smtp_port) as server:
          server.sendmail(sender_email, receiver_emails, message.as_string())
          time.sleep(2)
      print(f"{i} Email sent successfully.")
      i+=1
  except Exception as e:
      print(f"❌ An error occurred: {e}")
