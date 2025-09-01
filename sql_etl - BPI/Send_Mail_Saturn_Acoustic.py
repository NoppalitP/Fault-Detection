"""
Email Alert Utility — robust SMTP sender with HTML table payload
- Safe defaults (no infinite loops, retries with backoff, timeouts)
- Accepts row data as list or dict and renders a styled HTML table
- Adds plain‑text fallback and proper RFC 2822 headers
- Optional STARTTLS (disabled by default for port 25 relays)
"""
from __future__ import annotations

import smtplib
import socket
import time
import uuid
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

# ===============================
# Configuration
# ===============================
SMTP_SERVER = "mailrelay.hgst.com"
SMTP_PORT = 25  # usually 25 (no TLS) for internal relay
SMTP_TIMEOUT_SEC = 10
SMTP_USE_STARTTLS = False  # set True if your relay requires STARTTLS on this port

SENDER_EMAIL = "PDL-PRB-TEE-Automatic-Report@wdc.com"
RECEIVER_EMAILS: List[str] = [
    # "chaiwat.chantana@wdc.com",
    #"nobpharit.phosrithat@wdc.com",
    #"kittisak.boo@wdc.com",
    # "ratchanon.nooraksa@wdc.com",
    # "siwat.ninlachat@wdc.com",
]

SUBJECT = "🔔 Saturn‑Acoustic Tool Monitoring Alert"

# Retry policy
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SEC = 3.0  # exponential backoff

# Default body header
BODY_INTRO_HTML = (
    "<p>Dear Team,</p>"
    "<p>This is an automated alert from the <strong>Saturn‑Acoustic tool monitoring system</strong>.</p>"
    "<p>Please check the system status. If any irregularities are found, kindly notify the System Administrator.</p>"
    "<p>Thank you,<br>Automated Notification System</p>"
)

BODY_INTRO_TEXT = (
    "Dear Team,\n\n"
    "This is an automated alert from the Saturn‑Acoustic tool monitoring system.\n"
    "Please check the system status. If any irregularities are found, kindly notify the System Administrator.\n\n"
    "Thank you,\n"
    "Automated Notification System"
)

# ===============================
# Helpers
# ===============================
COLUMNS = ["Timestamp", "Component", "Status", "Sound Level (dB)", "Tester ID"]


def _coerce_row(row_data: Union[Sequence, Mapping]) -> List[str]:
    """Coerce row_data (list/tuple/dict) into a list of strings matching COLUMNS.

    Accepts:
        - list/tuple in the exact order of COLUMNS
        - dict with keys matching COLUMNS (case‑sensitive)
    """
    if isinstance(row_data, Mapping):
        try:
            return [str(row_data[col]) for col in COLUMNS]
        except KeyError as e:
            missing = [c for c in COLUMNS if c not in row_data]
            raise ValueError(f"Missing required keys in row_data: {missing}") from e
    else:
        seq = list(row_data)
        if len(seq) != len(COLUMNS):
            raise ValueError(
                f"row_data must have {len(COLUMNS)} elements in order {COLUMNS}; got {len(seq)}"
            )
        return [str(x) for x in seq]


def _render_html_table(row: List[str]) -> str:
    df = pd.DataFrame([row], columns=COLUMNS)
    # escape=True prevents HTML injection from values
    return df.to_html(index=False, classes="styled-table", border=0, escape=True)


def _wrap_html(body_html: str, table_html: str) -> str:
    return f"""
<html>
  <head>
    <meta charset=\"utf-8\">
    <style>
      body {{ font-family: Arial, sans-serif; font-size: 14px; color: #333; }}
      h3 {{ margin-top: 0; }}
      .styled-table {{
        border-collapse: collapse; margin: 12px 0; font-size: 14px; min-width: 420px; border: 1px solid #ddd;
      }}
      .styled-table thead tr {{ background-color: #009879; color: #fff; text-align: left; }}
      .styled-table th, .styled-table td {{ padding: 10px 12px; border: 1px solid #ddd; }}
      .styled-table tbody tr:nth-child(even) {{ background-color: #f6f8fa; }}
      .footer {{ margin-top: 12px; color: #555; font-size: 12px; }}
      .muted {{ color: #6b7280; }}
    </style>
  </head>
  <body>
    {body_html}
    <h3>📋 Acoustic Monitoring Data</h3>
    {table_html}
    <div class=\"footer muted\">This message was sent automatically at {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}.</div>
  </body>
</html>
"""


def _build_message(
    subject: str,
    sender: str,
    recipients: Sequence[str],
    html_body: str,
    text_body: Optional[str] = None,
) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    # Unique, well‑formed message id
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1])
    msg["Subject"] = subject

    if text_body:
        msg.attach(MIMEText(text_body, "plain", _charset="utf-8"))
    msg.attach(MIMEText(html_body, "html", _charset="utf-8"))
    return msg


# ===============================
# Public API
# ===============================

def send_alert(
    row_data: Union[Sequence, Mapping],
    *,
    body_html: str = BODY_INTRO_HTML,
    body_text: str = BODY_INTRO_TEXT,
    subject: str = SUBJECT,
    recipients: Optional[Sequence[str]] = None,
    sender: str = SENDER_EMAIL,
    smtp_server: str = SMTP_SERVER,
    smtp_port: int = SMTP_PORT,
    timeout_sec: int = SMTP_TIMEOUT_SEC,
    use_starttls: bool = SMTP_USE_STARTTLS,
    max_retries: int = MAX_RETRIES,
    initial_retry_delay_sec: float = INITIAL_RETRY_DELAY_SEC,
) -> bool:
    """Send an alert email with one row of monitoring data.

    Returns True on success, False on permanent failure after retries.
    Raises ValueError for invalid inputs.
    """
    if not recipients:
        recipients = RECEIVER_EMAILS
    if not recipients:
        raise ValueError("No recipients provided.")

    row = _coerce_row(row_data)
    table_html = _render_html_table(row)
    html = _wrap_html(body_html, table_html)
    msg = _build_message(subject, sender, recipients, html_body=html, text_body=body_text)

    delay = initial_retry_delay_sec
    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout_sec) as server:
                if use_starttls:
                    # STARTTLS if your relay supports it
                    server.starttls()
                server.sendmail(sender, list(recipients), msg.as_string())
            print(f"✅ Email sent successfully (attempt {attempt}).")
            return True
        except (smtplib.SMTPException, socket.timeout, OSError) as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2  # exponential backoff
    print("🚨 Failed to send email after retries.")
    return False


# ===============================
# Example usage
# ===============================

