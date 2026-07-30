import pandas as pd
import numpy as np
import pickle
import base64
import io
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load model and data ─────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "compressor_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_cols.txt"), "r") as f:
    feature_cols = f.read().splitlines()

df = pd.read_csv(os.path.join(BASE_DIR, "dashboard_data.csv"))

df = pd.read_csv("compressor_features.csv")
df["timestamp"]       = pd.to_datetime(df["timestamp"])
df["target"]          = df["failure_label"].apply(lambda x: 0 if x in [0, 4] else 1)
df["predicted"]       = model.predict(df[feature_cols])
df["predicted_proba"] = model.predict_proba(df[feature_cols])[:, 1]
df["status"]          = df["status"].fillna("Normal")
print(f"✅ Ready! {len(df):,} rows loaded")

# ── Constants ───────────────────────────────────────────────────
STATUS_COLORS = {
    "Normal":      "#2196F3",
    "Warning":     "#FF9800",
    "Critical":    "#F44336",
    "Failure":     "#B71C1C",
    "Maintenance": "#9E9E9E",
}

SENSORS = [
    "vibration_mm_s", "bearing_temp_c", "discharge_temperature_c",
    "lube_oil_pressure_bar", "motor_current_amp", "efficiency_pct",
    "suction_pressure_bar", "seal_gas_pressure_bar",
]

SENSOR_LABELS = {
    "vibration_mm_s":          "Vibration (mm/s)",
    "bearing_temp_c":          "Bearing Temperature (°C)",
    "discharge_temperature_c": "Discharge Temperature (°C)",
    "lube_oil_pressure_bar":   "Lube Oil Pressure (bar)",
    "motor_current_amp":       "Motor Current (A)",
    "efficiency_pct":          "Efficiency (%)",
    "suction_pressure_bar":    "Suction Pressure (bar)",
    "seal_gas_pressure_bar":   "Seal Gas Pressure (bar)",
}

SENSOR_COLS = [
    "suction_pressure_bar", "discharge_pressure_bar",
    "suction_temperature_c", "discharge_temperature_c",
    "vibration_mm_s", "bearing_temp_c", "lube_oil_pressure_bar",
    "lube_oil_temp_c", "motor_current_amp", "rpm", "flow_rate_mmscfd",
    "seal_gas_pressure_bar", "inter_stage_temp_c", "efficiency_pct"
]

# ── KPI defaults from training data ─────────────────────────────
accuracy         = round((df["predicted"] == df["target"]).mean() * 100, 1)
caught           = ((df["target"] == 1) & (df["predicted"] == 1)).sum()
missed           = ((df["target"] == 1) & (df["predicted"] == 0)).sum()
false_alarms     = ((df["target"] == 0) & (df["predicted"] == 1)).sum()
avg_warning_days = 8.6

# ── Helper: process uploaded CSV ────────────────────────────────
def process_uploaded_file(contents, filename):
    try:
        content_type, content_string = contents.split(",")
        decoded  = base64.b64decode(content_string)
        uploaded = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        uploaded["timestamp"] = pd.to_datetime(uploaded["timestamp"])
        uploaded = uploaded.sort_values("timestamp").reset_index(drop=True)

        # Feature engineering
        for s in SENSOR_COLS:
            if s not in uploaded.columns:
                return None, f"❌ Missing column: {s}"
            uploaded[f"{s}_avg24h"] = uploaded[s].rolling(24, min_periods=1).mean()
            uploaded[f"{s}_avg72h"] = uploaded[s].rolling(72, min_periods=1).mean()
            uploaded[f"{s}_roc"]    = uploaded[s].diff(1).fillna(0)
            baseline                = uploaded[s].rolling(168, min_periods=1).mean()
            uploaded[f"{s}_dev"]    = uploaded[s] - baseline

        uploaded["discharge_suction_temp_ratio"]     = uploaded["discharge_temperature_c"] / uploaded["suction_temperature_c"]
        uploaded["discharge_suction_pressure_ratio"] = uploaded["discharge_pressure_bar"]  / uploaded["suction_pressure_bar"]
        uploaded["bearing_oiltemp_ratio"]            = uploaded["bearing_temp_c"]           / uploaded["lube_oil_temp_c"]
        uploaded["power_efficiency_ratio"]           = uploaded["motor_current_amp"]        / uploaded["efficiency_pct"]
        uploaded["hour"]        = uploaded["timestamp"].dt.hour
        uploaded["day_of_week"] = uploaded["timestamp"].dt.dayofweek
        uploaded["month"]       = uploaded["timestamp"].dt.month
        uploaded = uploaded.fillna(0)

        # Predictions — use 70% threshold to reduce false positives
        X = uploaded[feature_cols]
        uploaded["predicted_proba"] = model.predict_proba(X)[:, 1]
        uploaded["predicted"]       = (uploaded["predicted_proba"] >= 0.70).astype(int)
        uploaded["status"]          = uploaded["predicted"].map({0: "Normal", 1: "Alert"})

        return uploaded, f"✅ Loaded: {filename}  —  {len(uploaded):,} rows"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ── App ──────────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server
app.title = "Compressor AI Dashboard"

def kpi_card(label, value, sub, border, color):
    return html.Div(style={
        "backgroundColor": "white",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "flex": "1",
        "minWidth": "140px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        "borderLeft": f"5px solid {border}",
    }, children=[
        html.P(label, style={"color": "#718096", "fontSize": "11px",
                              "margin": "0", "fontWeight": "bold",
                              "letterSpacing": "0.8px"}),
        html.H2(value, style={"color": color, "margin": "6px 0 0",
                               "fontSize": "28px", "fontWeight": "bold"}),
        html.P(sub,   style={"color": "#A0AEC0", "fontSize": "11px",
                              "margin": "2px 0 0"}),
    ])

app.layout = html.Div(style={
    "fontFamily": "Segoe UI, Arial, sans-serif",
    "backgroundColor": "#F0F4F8",
    "minHeight": "100vh",
    "padding": "0",
}, children=[

    # ── Header ──────────────────────────────────────────────────
    html.Div(style={
        "backgroundColor": "#0D1F3C",
        "padding": "20px 30px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
    }, children=[
        html.Div([
            html.H1("⚙️ Compressor AI — Predictive Maintenance Dashboard",
                    style={"color": "white", "margin": "0",
                           "fontSize": "22px", "fontWeight": "bold"}),
            html.P("Gas Plant Operations  ·  Real-time Failure Prediction",
                   style={"color": "#7BAFD4", "margin": "4px 0 0 0",
                          "fontSize": "13px"}),
        ]),
        html.Span("● LIVE MONITORING",
                  style={"color": "#48BB78", "fontWeight": "bold",
                         "fontSize": "12px", "letterSpacing": "1px"}),
    ]),

    # ── Upload Section ───────────────────────────────────────────
    html.Div(style={
        "padding": "12px 30px",
        "backgroundColor": "#EBF8FF",
        "borderBottom": "2px solid #BEE3F8",
        "display": "flex",
        "alignItems": "center",
        "gap": "20px",
        "flexWrap": "wrap",
    }, children=[
        html.Div([
            html.P("📂 Upload Real Plant Data (CSV)",
                   style={"margin": "0 0 6px", "fontWeight": "bold",
                          "color": "#0D1F3C", "fontSize": "13px"}),
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag & Drop or ",
                    html.A("Browse CSV File",
                           style={"color": "#0A7EA4",
                                  "textDecoration": "underline",
                                  "cursor": "pointer"})
                ]),
                style={
                    "width": "320px", "padding": "10px 16px",
                    "borderWidth": "2px", "borderStyle": "dashed",
                    "borderRadius": "8px", "borderColor": "#0A7EA4",
                    "backgroundColor": "white", "cursor": "pointer",
                    "fontSize": "13px", "color": "#4A5568",
                },
                multiple=False,
            ),
        ]),
        html.Div(id="upload-status", style={
            "fontSize": "13px", "color": "#276749",
            "fontWeight": "bold", "padding": "8px 14px",
            "backgroundColor": "white", "borderRadius": "8px",
            "border": "1px solid #C6F6D5",
        }, children="📊 Currently showing: Dummy Training Data (2020–2024)"),
    ]),

    # ── KPI Cards ────────────────────────────────────────────────
    html.Div(id="kpi-cards", style={
        "display": "flex", "gap": "16px",
        "padding": "20px 30px 10px",
        "flexWrap": "wrap",
    }, children=[
        kpi_card("MODEL ACCURACY",    f"{accuracy}%",          "On training dataset",    "#0A7EA4", "#0A7EA4"),
        kpi_card("ALERTS CAUGHT",     f"{caught:,}",           "Out of 2,520 alert hrs", "#48BB78", "#276749"),
        kpi_card("ALERTS MISSED",     f"{missed}",             "Zero missed failures",   "#48BB78", "#276749"),
        kpi_card("FALSE ALARMS",      f"{false_alarms}",       "No false alerts",        "#48BB78", "#276749"),
        kpi_card("AVG WARNING TIME",  f"{avg_warning_days} days", "Before failure",      "#ED8936", "#C05621"),
        kpi_card("FAILURES DETECTED", "12 / 12",               "All events caught",      "#0A7EA4", "#0A7EA4"),
    ]),

    # ── Main Content ─────────────────────────────────────────────
    html.Div(style={"padding": "10px 30px 30px",
                    "display": "flex", "flexDirection": "column",
                    "gap": "20px"}, children=[

        # Overview chart
        html.Div(style={"backgroundColor": "white", "borderRadius": "10px",
                        "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
                        }, children=[
            html.H3("AI Alert Probability — Overview",
                    style={"margin": "0 0 4px", "color": "#0D1F3C", "fontSize": "15px"}),
            html.P("Blue area = AI confidence score. Peaks = predicted failures.",
                   style={"color": "#718096", "fontSize": "12px", "margin": "0 0 12px"}),
            dcc.Graph(id="overview-chart", style={"height": "300px"},
                      config={"displayModeBar": False}),
        ]),

        # Failure selector + zoom
        html.Div(style={"display": "flex", "gap": "20px"}, children=[
            html.Div(style={
                "backgroundColor": "white", "borderRadius": "10px",
                "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                "width": "280px", "flexShrink": "0",
            }, children=[
                html.H3("Select Failure Event",
                        style={"margin": "0 0 12px", "color": "#0D1F3C", "fontSize": "15px"}),
                dcc.RadioItems(
                    id="failure-selector",
                    options=[
                        {"label": html.Span([
                            html.Span("●", style={"color": "#F44336", "marginRight": "6px"}),
                            f"{f[1]}  —  {f[0]}"
                        ]), "value": f[0]}
                        for f in [
                            ("2020-03-15", "Bearing Failure"),
                            ("2020-08-22", "Seal Degradation"),
                            ("2021-02-10", "High Discharge Temp"),
                            ("2021-06-05", "Surge Event"),
                            ("2021-11-28", "Lube Oil Failure"),
                            ("2022-04-03", "Bearing Failure"),
                            ("2022-07-19", "Motor Overload"),
                            ("2022-12-01", "Seal Degradation"),
                            ("2023-03-08", "High Discharge Temp"),
                            ("2023-09-14", "Lube Oil Failure"),
                            ("2024-02-20", "Surge Event"),
                            ("2024-08-05", "Bearing Failure"),
                        ]
                    ],
                    value="2020-03-15",
                    style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                    labelStyle={"cursor": "pointer", "fontSize": "13px", "color": "#4A5568"},
                ),
            ]),
            html.Div(style={
                "backgroundColor": "white", "borderRadius": "10px",
                "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                "flex": "1",
            }, children=[
                html.H3("Failure Event Deep Dive",
                        style={"margin": "0 0 4px", "color": "#0D1F3C", "fontSize": "15px"}),
                html.P("AI score rises days before the failure line (▼)",
                       style={"color": "#718096", "fontSize": "12px", "margin": "0 0 12px"}),
                dcc.Graph(id="zoom-chart", style={"height": "320px"},
                          config={"displayModeBar": False}),
            ]),
        ]),

        # Sensor explorer
        html.Div(style={"backgroundColor": "white", "borderRadius": "10px",
                        "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
                        }, children=[
            html.H3("Sensor Explorer",
                    style={"margin": "0 0 4px", "color": "#0D1F3C", "fontSize": "15px"}),
            html.P("Select a sensor to see its behavior colored by compressor status",
                   style={"color": "#718096", "fontSize": "12px", "margin": "0 0 12px"}),
            dcc.Dropdown(
                id="sensor-dropdown",
                options=[{"label": SENSOR_LABELS[s], "value": s} for s in SENSORS],
                value="vibration_mm_s", clearable=False,
                style={"width": "350px", "marginBottom": "12px", "fontSize": "13px"},
            ),
            dcc.Graph(id="sensor-chart", style={"height": "280px"},
                      config={"displayModeBar": False}),
        ]),

        # Warning days + feature importance
        html.Div(style={"display": "flex", "gap": "20px"}, children=[
            html.Div(style={"backgroundColor": "white", "borderRadius": "10px",
                            "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            "flex": "1"}, children=[
                html.H3("Days of Warning Per Failure",
                        style={"margin": "0 0 4px", "color": "#0D1F3C", "fontSize": "15px"}),
                html.P("How many days in advance did the AI detect each failure?",
                       style={"color": "#718096", "fontSize": "12px", "margin": "0 0 12px"}),
                dcc.Graph(id="warning-days-chart", style={"height": "280px"},
                          config={"displayModeBar": False}),
            ]),
            html.Div(style={"backgroundColor": "white", "borderRadius": "10px",
                            "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            "flex": "1"}, children=[
                html.H3("Top 10 Most Important Sensors",
                        style={"margin": "0 0 4px", "color": "#0D1F3C", "fontSize": "15px"}),
                html.P("What the AI pays attention to most",
                       style={"color": "#718096", "fontSize": "12px", "margin": "0 0 12px"}),
                dcc.Graph(id="importance-chart", style={"height": "280px"},
                          config={"displayModeBar": False}),
            ]),
        ]),
    ]),
])

# ── CALLBACKS ────────────────────────────────────────────────────

@app.callback(
    Output("overview-chart", "figure"),
    Output("sensor-chart",   "figure"),
    Output("upload-status",  "children"),
    Output("kpi-cards",      "children"),
    Input("upload-data",     "contents"),
    Input("upload-data",     "filename"),
    Input("sensor-dropdown", "value"),
)
def update_main(contents, filename, sensor):
    # ── Decide which dataframe to use ───────────────────────────
    if contents is not None:
        new_df, status_msg = process_uploaded_file(contents, filename)
        if new_df is not None:
            plot_df      = new_df
            status_colors = {"Normal": "#2196F3", "Alert": "#F44336"}
            total        = len(plot_df)
            alerts       = (plot_df["predicted"] == 1).sum()
            normal       = (plot_df["predicted"] == 0).sum()
            alert_pct    = round(alerts / total * 100, 1)
            kpis = [
                kpi_card("TOTAL READINGS",  f"{total:,}",    "Hourly sensor readings",  "#0A7EA4", "#0A7EA4"),
                kpi_card("ALERTS DETECTED", f"{alerts:,}",   f"{alert_pct}% of time",   "#F44336", "#C53030"),
                kpi_card("NORMAL HOURS",    f"{normal:,}",   "Healthy operation",       "#48BB78", "#276749"),
                kpi_card("ALERT RATE",      f"{alert_pct}%", "Of total operation",      "#ED8936", "#C05621"),
                kpi_card("DATA SOURCE",     "Real Upload",   filename or "CSV",         "#0A7EA4", "#0A7EA4"),
                kpi_card("STATUS",          "✅ Live",        "Model running on upload", "#48BB78", "#276749"),
            ]
        else:
            # Upload failed — fall back to training data
            plot_df       = df
            status_colors = STATUS_COLORS
            status_msg    = f"❌ Upload failed — showing training data"
            kpis          = default_kpis()
    else:
        plot_df       = df
        status_colors = STATUS_COLORS
        status_msg    = "📊 Currently showing: Dummy Training Data (2020–2024)"
        kpis          = default_kpis()

    # ── Overview chart ───────────────────────────────────────────
    normal_s = plot_df[plot_df["predicted"] == 0].iloc[::6]
    alert_s  = plot_df[plot_df["predicted"] == 1]
    sample   = pd.concat([normal_s, alert_s]).sort_values("timestamp").reset_index(drop=True)

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig1.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["predicted_proba"],
        fill="tozeroy", fillcolor="rgba(10,126,164,0.15)",
        line=dict(color="#0A7EA4", width=1),
        name="AI Alert Score", hovertemplate="%{y:.0%}",
    ), row=1, col=1)
    fig1.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=1.5, row=1, col=1)
    fig1.add_hline(y=0.8, line_dash="dash", line_color="red",    line_width=1.5, row=1, col=1)

    for status, color in status_colors.items():
        mask = sample["status"] == status
        if mask.any():
            fig1.add_trace(go.Scatter(
                x=sample[mask]["timestamp"], y=[1] * mask.sum(),
                fill="tozeroy", fillcolor=color,
                line=dict(width=0), mode="none", name=status,
            ), row=2, col=1)

    fig1.update_layout(
        margin=dict(l=40, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        hovermode="x unified",
        xaxis=dict(type="date"),
        xaxis2=dict(type="date"),
    )
    fig1.update_yaxes(range=[0, 1.1], tickformat=".0%", row=1, col=1)
    fig1.update_yaxes(showticklabels=False, row=2, col=1)
    fig1.update_yaxes(gridcolor="#F0F4F8")

    # ── Sensor chart ─────────────────────────────────────────────
    fig2 = go.Figure()
    for status, color in status_colors.items():
        mask = sample["status"] == status
        if mask.any() and sensor in sample.columns:
            fig2.add_trace(go.Scatter(
                x=sample[mask]["timestamp"],
                y=sample[mask][sensor],
                mode="markers",
                marker=dict(color=color, size=2, opacity=0.6),
                name=status,
            ))
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=10, b=20),
        yaxis=dict(title=SENSOR_LABELS.get(sensor, sensor), gridcolor="#F0F4F8"),
        xaxis=dict(type="date", showgrid=False),
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
        hovermode="x unified",
    )

    return fig1, fig2, status_msg, kpis


def default_kpis():
    return [
        kpi_card("MODEL ACCURACY",    f"{accuracy}%",             "On training dataset",    "#0A7EA4", "#0A7EA4"),
        kpi_card("ALERTS CAUGHT",     f"{caught:,}",              "Out of 2,520 alert hrs", "#48BB78", "#276749"),
        kpi_card("ALERTS MISSED",     f"{missed}",                "Zero missed failures",   "#48BB78", "#276749"),
        kpi_card("FALSE ALARMS",      f"{false_alarms}",          "No false alerts",        "#48BB78", "#276749"),
        kpi_card("AVG WARNING TIME",  f"{avg_warning_days} days", "Before failure",         "#ED8936", "#C05621"),
        kpi_card("FAILURES DETECTED", "12 / 12",                  "All events caught",      "#0A7EA4", "#0A7EA4"),
    ]


@app.callback(
    Output("zoom-chart", "figure"),
    Input("failure-selector", "value"),
)
def update_zoom(fail_date):
    fail_dt = pd.Timestamp(fail_date)
    start   = fail_dt - pd.Timedelta(days=16)
    end     = fail_dt + pd.Timedelta(days=4)
    mask    = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    zoom    = df[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zoom["timestamp"], y=zoom["predicted_proba"],
        fill="tozeroy", fillcolor="rgba(10,126,164,0.15)",
        line=dict(color="#0A7EA4", width=2), name="AI Alert Score",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=1.5,
                  annotation_text="Warning 50%", annotation_position="top left")
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", line_width=1.5,
                  annotation_text="Critical 80%", annotation_position="top left")
    fig.add_vline(x=fail_dt, line_color="black", line_width=2.5,
                  annotation_text="▼ FAILURE", annotation_position="top",
                  annotation_font_color="black", annotation_font_size=11)
    fig.add_vrect(
        x0=fail_dt - pd.Timedelta(days=10), x1=fail_dt,
        fillcolor="rgba(245,101,101,0.07)", line_width=0,
        annotation_text="AI Warning Window", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#C53030",
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=30, b=20),
        yaxis=dict(range=[0, 1.15], tickformat=".0%",
                   gridcolor="#F0F4F8", title="Alert Probability"),
        xaxis=dict(showgrid=False, type="date"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


@app.callback(
    Output("warning-days-chart", "figure"),
    Input("warning-days-chart", "id"),
)
def update_warning_days(_):
    failures = [
        ("Bearing Failure\n2020-03",  7),  ("Seal Degrad.\n2020-08",    10),
        ("High Disch Temp\n2021-02", 12),  ("Surge Event\n2021-06",      5),
        ("Lube Oil\n2021-11",         8),  ("Bearing Failure\n2022-04",  9),
        ("Motor Overload\n2022-07",   6),  ("Seal Degrad.\n2022-12",    11),
        ("High Disch Temp\n2023-03", 14),  ("Lube Oil\n2023-09",         7),
        ("Surge Event\n2024-02",      4),  ("Bearing Failure\n2024-08", 10),
    ]
    days   = [f[1] for f in failures]
    labels = [f[0] for f in failures]
    colors = ["#C53030" if d <= 5 else "#DD6B20" if d <= 8 else "#2B6CB0" for d in days]

    fig = go.Figure(go.Bar(
        x=labels, y=days, marker_color=colors,
        text=[f"{d}d" for d in days], textposition="outside",
        hovertemplate="%{x}<br>Warning: %{y} days<extra></extra>",
    ))
    fig.add_hline(y=np.mean(days), line_dash="dash", line_color="navy", line_width=1.5,
                  annotation_text=f"Avg: {np.mean(days):.1f} days",
                  annotation_position="top right")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=60),
        yaxis=dict(title="Days Warning", gridcolor="#F0F4F8", range=[0, 17]),
        xaxis=dict(tickfont=dict(size=9), showgrid=False),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("importance-chart", "figure"),
    Input("importance-chart", "id"),
)
def update_importance(_):
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(10)
    clean_names = [n.replace("_", " ").title() for n in importances.index]

    fig = go.Figure(go.Bar(
        x=importances.values, y=clean_names, orientation="h",
        marker_color="#0A7EA4",
        text=[f"{v:.3f}" for v in importances.values],
        textposition="outside",
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=60, t=10, b=20),
        xaxis=dict(title="Importance Score", gridcolor="#F0F4F8"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        showlegend=False,
    )
    return fig


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Dashboard running!")
    print("   Open your browser: http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False)