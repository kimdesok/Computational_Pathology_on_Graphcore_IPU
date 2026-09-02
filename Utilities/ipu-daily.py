import csv
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
import pandas as pd

CSV_FILE = "ipu_metrics_log.csv"

# Columns expected in the CSV
FIELDNAMES = [
    "Date",
    "Time",
    "Card",
    "Global_ID",
    "PID",
    "Util",
    "Session_Util",
    "Daily_Avg_Util",
    "Daily_Avg_Session_Util",
    "Weekly_Avg_Util",
    "Weekly_Avg_Session_Util",
    "Monthly_Avg_Util",
    "Monthly_Avg_Session_Util",
]


def print_card(card):
    # Standard placeholder for your helper function
    pass

def parse_util_value(val):
    """Helper to convert '87.36%' string or numeric types to float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # Remove '%', clean whitespace, and convert
    clean_str = str(val).replace("%", "").strip()
    try:
        return float(clean_str)
    except ValueError:
        return 0.0


def get_actual_ipu_metrics():
    try:
        result = subprocess.run(
            ["gc-monitor", "-j"], stdout=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        metrics = []

        for card in data.get("cards", []):
            print_card(card)
            card_ip = card.get("IPU-M", "Unknown")

            for ipu_data in card.get("ipus", []):
                global_id = ipu_data.get("ID")
                util = ipu_data.get("IPU utilisation")
                util_session = ipu_data.get("IPU utilisation session")
                pid = ipu_data.get("PID")

                metrics.append(
                    {
                        "Card": card_ip,
                        "Global_ID": global_id,
                        "PID": pid,
                        "Util": parse_util_value(util),
                        "Session_Util": parse_util_value(util_session),
                    }
                )
        return metrics
    except Exception as e:
        print(f"Failed to parse telemetry block: {e}")
        return None


def calculate_rolling_averages(current_time, metrics):
    """Calculates daily, weekly, and monthly averages per Global_ID using past CSV history."""
    if not os.path.exists(CSV_FILE):
        # File doesn't exist yet, so current metric values are their own averages
        for m in metrics:
            m["Daily_Avg_Util"] = m["Util"]
            m["Daily_Avg_Session_Util"] = m["Session_Util"]
            m["Weekly_Avg_Util"] = m["Util"]
            m["Weekly_Avg_Session_Util"] = m["Session_Util"]
            m["Monthly_Avg_Util"] = m["Util"]
            m["Monthly_Avg_Session_Util"] = m["Session_Util"]
        return metrics

    # Read existing historical records
    df = pd.read_csv(CSV_FILE)
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    # Define time window boundaries
    one_day_ago = current_time - timedelta(days=1)
    one_week_ago = current_time - timedelta(days=7)
    one_month_ago = current_time - timedelta(days=30)

    for m in metrics:
        gid = m["Global_ID"]
        ipu_history = df[df["Global_ID"] == gid]

        # Daily Averages (Past 24 hours)
        d_mask = ipu_history["Timestamp"] >= one_day_ago
        d_util = list(ipu_history[d_mask]["Util"]) + [m["Util"]]
        d_sess = list(ipu_history[d_mask]["Session_Util"]) + [m["Session_Util"]]
        m["Daily_Avg_Util"] = round(sum(d_util) / len(d_util), 2)
        m["Daily_Avg_Session_Util"] = round(sum(d_sess) / len(d_sess), 2)

        # Weekly Averages (Past 7 days)
        w_mask = ipu_history["Timestamp"] >= one_week_ago
        w_util = list(ipu_history[w_mask]["Util"]) + [m["Util"]]
        w_sess = list(ipu_history[w_mask]["Session_Util"]) + [m["Session_Util"]]
        m["Weekly_Avg_Util"] = round(sum(w_util) / len(w_util), 2)
        m["Weekly_Avg_Session_Util"] = round(sum(w_sess) / len(w_sess), 2)

        # Monthly Averages (Past 30 days)
        m_mask = ipu_history["Timestamp"] >= one_month_ago
        m_util = list(ipu_history[m_mask]["Util"]) + [m["Util"]]
        m_sess = list(ipu_history[m_mask]["Session_Util"]) + [m["Session_Util"]]
        m["Monthly_Avg_Util"] = round(sum(m_util) / len(m_util), 2)
        m["Monthly_Avg_Session_Util"] = round(sum(m_sess) / len(m_sess), 2)

    return metrics


def log_metrics_to_csv():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    active_metrics = get_actual_ipu_metrics()
    if not active_metrics:
        print("No metrics collected.")
        return

    # Calculate rolling averages across history
    enriched_metrics = calculate_rolling_averages(now, active_metrics)

    # Check if header needs to be written
    file_exists = os.path.exists(CSV_FILE)

    # Append to CSV
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for m in enriched_metrics:
            writer.writerow(
                {
                    "Date": date_str,
                    "Time": time_str,
                    "Card": m["Card"],
                    "Global_ID": m["Global_ID"],
                    "PID": m["PID"],
                    "Util": m["Util"],
                    "Session_Util": m["Session_Util"],
                    "Daily_Avg_Util": m["Daily_Avg_Util"],
                    "Daily_Avg_Session_Util": m["Daily_Avg_Session_Util"],
                    "Weekly_Avg_Util": m["Weekly_Avg_Util"],
                    "Weekly_Avg_Session_Util": m["Weekly_Avg_Session_Util"],
                    "Monthly_Avg_Util": m["Monthly_Avg_Util"],
                    "Monthly_Avg_Session_Util": m["Monthly_Avg_Session_Util"],
                }
            )


# --- Polling Loop (Executes once per minute) ---
if __name__ == "__main__":
    while True:
        log_metrics_to_csv()
        time.sleep(300)
