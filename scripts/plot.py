import re
import pandas as pd
import plotly.express as px

# mapping Measurement Position -> phase label
POS_LABEL = {
    5: "input_transfer",
    7: "weight_transfer",
    8: "mla_execution",
    9: "output_transfer",
}

def build_worker_intervals(xls_dict):
    """Convert Excel sheets into {worker: [(start, end, label), ...]}"""
    row_data = {}

    for name, df in xls_dict.items():
        m = re.match(r"WORKER #(\d+)", str(name))
        if not m:
            continue  # skip non-worker sheets like scheduler
        worker_id = int(m.group(1))

        # normalize column names (strip, handle case)
        df = df.rename(columns=lambda c: c.strip())

        # detect start/end columns robustly (handles 'µs' vs 'us' vs 'μs', case/space variants)
        cols_lower = {c.lower().replace("μ", "µ"): c for c in df.columns}
        def find_col(prefix):
            for key, orig in cols_lower.items():
                if prefix in key and "[µs]" in key:
                    return orig
                if prefix in key and "[us]" in key:  # ASCII 'us'
                    return orig
            return None

        start_col = find_col("start")
        end_col = find_col("end")
        if start_col is None or end_col is None:
            raise ValueError(f"Couldn't find Start/End microsecond columns in sheet {name}. Columns: {list(df.columns)}")

        # get Measurement Position column (case-insensitive)
        pos_series = df.get("Measurement Position")
        if pos_series is None:
            # try case-insensitive match
            for c in df.columns:
                if c.strip().lower() == "measurement position":
                    pos_series = df[c]
                    break
        if pos_series is None:
            raise ValueError(f"Couldn't find 'Measurement Position' column in sheet {name}")

        intervals = []
        for pos, label in POS_LABEL.items():
            rows = df.loc[pos_series == pos]
            if rows.empty:
                continue
            start = rows[start_col].iloc[0]
            end   = rows[end_col].iloc[0]
            if pd.notna(start) and pd.notna(end) and end > start:
                intervals.append((int(start), int(end), label))
        if len(intervals) < len(POS_LABEL):
            missing = [lbl for p, lbl in POS_LABEL.items() if df.loc[pos_series == p].empty]
            if missing:
                print(f"Sheet {name}: missing phases -> {missing}")

        row_data[f"Worker {worker_id}"] = intervals

    return row_data

def plot_workers(row_data):
    """Flatten and plot worker intervals with Plotly"""
    rows = []
    for row_name, segs in row_data.items():
        for start, end, phase in segs:
            rows.append({"Row": row_name, "Start": start, "End": end, "Phase": phase})

    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        raise ValueError("No intervals extracted!")

    # normalize so earliest start is 0
    t0 = df_plot["Start"].min()
    df_plot["Start"] -= t0
    df_plot["End"]   -= t0

    fig = px.timeline(
        df_plot, x_start="Start", x_end="End", y="Row", color="Phase",
        color_discrete_map={
            "input_transfer": "blue",
            "weight_transfer": "green",
            "mla_execution": "orange",
            "output_transfer": "red",
        }
    )
    fig.update_yaxes(autorange="reversed")  # Worker 0 at top
    fig.update_layout(
        xaxis_title="Time (µs, normalized)",
        height=max(800, 18 * df_plot["Row"].nunique()),
        legend_title_text="Phase",
    )
    fig.show()

# --- main usage ---
# assuming you already did:
# xls = pd.read_excel("your_file.xlsx", sheet_name=None)

xls = pd.read_excel("data/ff2.xlsx", sheet_name=None)

row_data = build_worker_intervals(xls)
plot_workers(row_data)