import os
import sys
import json
import re
import time
import yaml
import copy
import glob
import random
import locale
import subprocess
import ctypes
import platform
import shutil
import threading
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QCoreApplication
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QPlainTextEdit,
    QFrame,
    QDialog
)

from typing import Dict

def keep_awake():
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-dimsu"])
    elif os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    else:
        try:
            subprocess.Popen(
                ["systemd-inhibit", "--what=idle", "--why=Keep app awake", "--mode=block", "sleep", "infinity"]
            )
        except FileNotFoundError:
            pass

keep_awake()

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

def random_pastel():
    return (
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5,
    )

def load_config(simulation_type):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, simulation_type, "configuration", "config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)

def load_adaptation(simulation_type):
    try:
        cfg = load_config(simulation_type)
        return cfg.get("adaptation", "None")
    except Exception:
        return "None"

def _latest_round_csv(simulation_type: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subdir = "Docker" if str(simulation_type).strip().lower() == "docker" else "Local"
    perf_dir = os.path.join(base_dir, subdir, "performance")
    raw_files = glob.glob(os.path.join(perf_dir, "FLwithAP_performance_metrics_round*.csv"))
    if not raw_files:
        return None, None
    def _rnum(p):
        m = re.search(r"round(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    lastf = max(raw_files, key=_rnum)
    return _rnum(lastf), lastf

def _extract_ap_prev(df: pd.DataFrame) -> dict:
    ap_map = {p: False for p in [
        "client_selector","client_cluster","message_compressor",
        "model_co-versioning_registry","multi-task_model_trainer","heterogeneous_data_handler"
    ]}
    col = None
    for c in df.columns:
        if "AP List" in c or (isinstance(c, str) and c.strip().lower() == "ap list"):
            col = c
            break
    if col is None or df.empty:
        return ap_map
    val = str(df[col].dropna().iloc[-1]).strip().strip("{}[]() ")
    parts = [x.strip().upper() for x in val.split(",") if x.strip()]
    order = list(ap_map.keys())
    for i, name in enumerate(order):
        ap_map[name] = (parts[i] == "ON") if i < len(parts) else False
    return ap_map

def _aggregate_round(df: pd.DataFrame) -> dict:
    safe = lambda name: df[name].astype(float).mean() if name in df.columns else None
    return {
        "mean_f1": safe("Val F1") or safe("F1"),
        "mean_total_time": safe("Total Time of FL Round") or safe("Total Time (s)") or safe("TotalTime"),
        "mean_training_time": safe("Training Time") or safe("Training (s)") or safe("Training Time (s)"),
        "mean_comm_time": safe("Communication Time") or safe("Comm (s)") or safe("server_comm_time"),
    }

class AIAgentsLogViewer(QDialog):
    def __init__(self, parent, adaptation_policy: str, log_path: str):
        super().__init__(parent)
        self.setWindowTitle("AI-Agents Adaptation")
        self.setStyleSheet("background-color: black;")
        self.log_path = log_path

        self.title = QLabel(adaptation_policy)
        self.title.setTextFormat(Qt.RichText)
        self.title.setStyleSheet("color: white; font-weight: bold; font-size: 16px; margin-bottom: 6px;")

        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet(
            "QPlainTextEdit { background-color: black; color: white; font-family: Courier; font-size: 12px; border: 1px solid #333; }"
        )

        lay = QVBoxLayout()
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lay.addWidget(self.title)
        lay.addWidget(self.out)
        self.setLayout(lay)

        self.timer = QTimer(self)
        self.timer.setInterval(800)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    def refresh(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            txt = "No logs yet."
        self.out.setPlainText(txt)
        self.out.moveCursor(self.out.textCursor().End)

def agent_log_path():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "Docker", "logs", "ai_agent_decisions.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        open(p, "w", encoding="utf-8").close()
    return p

class DashboardWindow(QWidget):
    def __init__(self, simulation_type):
        super().__init__()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.simulation_type = simulation_type
        self.setWindowTitle("Live Dashboard")
        self.setStyleSheet("background-color: white;")
        self.resize(1200, 800)
        self.pattern_names = [
            "client_selector",
            "client_cluster",
            "message_compressor",
            "model_co-versioning_registry",
            "multi-task_model_trainer",
            "heterogeneous_data_handler",
        ]
        cfg = load_config(self.simulation_type)
        self.client_configs = {int(d["client_id"]): d for d in cfg.get("client_details", [])}
        model_name = ""
        dataset_name = ""
        if cfg.get("client_details"):
            number_of_rounds = cfg.get("rounds")
            number_of_clients = cfg.get("clients")
            first = cfg["client_details"][0]
            model_name = first.get("model", "")
            dataset_name = first.get("dataset", "")
        self.color_f1 = random_pastel()
        self.color_tot = random_pastel()
        self.client_colors = {}
        self.clients = []
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)
        lbl_mod = QLabel(
            f'Model: <b>{model_name}</b> • '
            f'Dataset: <b>{dataset_name}</b> • '
            f'Number of Clients: <b>{number_of_clients}</b> • '
            f'Number of Rounds: <b>{number_of_rounds}</b>'
        )
        big_font = QFont()
        big_font.setPointSize(16)
        big_font.setWeight(QFont.Normal)
        lbl_mod.setFont(big_font)
        lbl_mod.setStyleSheet("color: black;")
        lbl_mod.setTextFormat(Qt.RichText)
        top_row_layout = QHBoxLayout()
        main_layout.addLayout(top_row_layout)
        self.metrics_panel = QWidget()
        self.metrics_panel.setStyleSheet("background-color:#f9f9f9; border:1px solid #ddd; border-radius:6px;")
        self.metrics_panel.setFixedWidth(420)
        metrics_panel_layout = QVBoxLayout(self.metrics_panel)
        metrics_panel_layout.setContentsMargins(8, 8, 8, 8)
        metrics_panel_layout.setSpacing(8)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(["Round", "Client", "F1", "Total Time (s)", "Training (s)", "Comm (s)"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setStyleSheet(
            """
            QTableWidget { background-color: white; alternate-background-color: #f4f4f4; gridline-color: #c0c0c0; font-size: 12px; color: black; border: 1px solid #bbbbbb; }
            QHeaderView::section { background-color: #dcdcdc; color: black; font-weight: bold; font-size: 12px; border: 1px solid #aaaaaa; padding: 2px 4px; }
            QTableWidget::item { padding: 2px 4px; }
            """
        )
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.metrics_table.setColumnWidth(0, 45)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.metrics_table.setColumnWidth(1, 70)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        metrics_panel_layout.addWidget(self.metrics_table)
        top_row_layout.addWidget(self.metrics_panel)
        self.plots_panel = QWidget()
        self.plots_panel_layout = QVBoxLayout(self.plots_panel)
        self.plots_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_panel_layout.setSpacing(8)
        self.sim_info_label = QLabel(
            f"Model: <b>{model_name}</b> - "
            f"Dataset: <b>{dataset_name}</b> - "
            f"Number of Clients: <b>{number_of_clients}</b> - "
            f"Number of Rounds: <b>{number_of_rounds}</b>"
        )
        self.sim_info_label.setWordWrap(True)
        self.sim_info_label.setAlignment(Qt.AlignCenter)
        big_font = QFont()
        big_font.setPointSize(18)
        big_font.setWeight(QFont.Normal)
        self.sim_info_label.setFont(big_font)
        self.sim_info_label.setStyleSheet("color:black; background-color: transparent; border:none;")
        self.plots_panel_layout.addWidget(self.sim_info_label)
        plots_container = QWidget()
        plots_grid = QGridLayout(plots_container)
        plots_grid.setContentsMargins(0, 0, 0, 0)
        plots_grid.setHorizontalSpacing(16)
        plots_grid.setVerticalSpacing(16)
        def make_plot_canvas(fig_width_px, fig_height_px):
            fig, ax = plt.subplots()
            fig.set_size_inches(fig_width_px / 100.0, fig_height_px / 100.0)
            fig.patch.set_facecolor("white")
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(fig_width_px, fig_height_px)
            canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return fig, ax, canvas
        self.fig_f1, self.ax_f1, self.canvas_f1 = make_plot_canvas(360, 220)
        plots_grid.addWidget(self.canvas_f1, 0, 0)
        self.fig_tot, self.ax_tot, self.canvas_tot = make_plot_canvas(360, 220)
        plots_grid.addWidget(self.canvas_tot, 0, 1)
        self.fig_train, self.ax_train, self.canvas_train = make_plot_canvas(360, 220)
        plots_grid.addWidget(self.canvas_train, 1, 0)
        self.fig_comm, self.ax_comm, self.canvas_comm = make_plot_canvas(360, 220)
        plots_grid.addWidget(self.canvas_comm, 1, 1)
        self.plots_panel_layout.addWidget(plots_container)
        top_row_layout.addWidget(self.plots_panel)
        self.patterns_panel = QFrame()
        self.patterns_panel.setFrameShape(QFrame.NoFrame)
        self.patterns_panel.setFrameShadow(QFrame.Plain)
        self.patterns_panel.setStyleSheet("background-color:#f9f9f9; border:none; border-radius:6px;")
        self.patterns_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        patterns_layout = QVBoxLayout(self.patterns_panel)
        patterns_layout.setContentsMargins(8, 8, 8, 8)
        patterns_layout.setSpacing(6)
        patterns_title = QLabel("Architectural Patterns Activation per Round")
        patterns_title.setStyleSheet("font-weight:bold; font-size:13px; color:black; background-color: transparent; border:none;")
        patterns_layout.addWidget(patterns_title)
        self.pattern_grid = QWidget()
        self.pattern_grid_layout = QGridLayout(self.pattern_grid)
        self.pattern_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.pattern_grid_layout.setHorizontalSpacing(8)
        self.pattern_grid_layout.setVerticalSpacing(4)
        patterns_layout.addWidget(self.pattern_grid, alignment=Qt.AlignLeft)
        main_layout.addWidget(self.patterns_panel)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)

    def _pretty_pattern_name(self, raw_name: str) -> str:
        parts = raw_name.replace("_", " ").split()
        parts = [p.capitalize() for p in parts]
        return " ".join(parts)

    def parse_ap_list(self, ap_str: str) -> Dict[str, bool]:
        active_map: Dict[str, bool] = {}
        if not isinstance(ap_str, str):
            return active_map
        cleaned = ap_str.strip().strip("{}")
        if not cleaned:
            for pname in self.pattern_names:
                active_map[pname] = False
            return active_map
        parts = [p.strip().upper() for p in cleaned.split(",")]
        for i, pname in enumerate(self.pattern_names):
            status = parts[i] if i < len(parts) else "OFF"
            active_map[pname] = (status == "ON")
        return active_map

    def update_pattern_grid(self, pattern_matrix_data):
        while self.pattern_grid_layout.count():
            item = self.pattern_grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not pattern_matrix_data:
            return
        rounds_list = [row["round"] for row in pattern_matrix_data]
        rounds_list = sorted(list(dict.fromkeys(rounds_list)))
        round_lookup = {}
        for row in pattern_matrix_data:
            rnd = row["round"]
            round_lookup[rnd] = {}
            for pname in self.pattern_names:
                round_lookup[rnd][pname] = row.get(pname, False)
        for col_idx, rnd in enumerate(rounds_list, start=1):
            rnd_lbl = QLabel(str(rnd))
            rnd_lbl.setStyleSheet("font-size:12px; color:black;background-color: transparent; border: none;")
            rnd_lbl.setAlignment(Qt.AlignCenter)
            self.pattern_grid_layout.addWidget(rnd_lbl, 0, col_idx, alignment=Qt.AlignCenter)
        for row_idx, pname in enumerate(self.pattern_names, start=1):
            pretty = self._pretty_pattern_name(pname)
            pat_lbl = QLabel(pretty)
            pat_lbl.setStyleSheet("font-size:12px; color:black; background-color: transparent; border: none;")
            pat_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.pattern_grid_layout.addWidget(pat_lbl, row_idx, 0, alignment=Qt.AlignVCenter)
            for col_idx, rnd in enumerate(rounds_list, start=1):
                active = round_lookup.get(rnd, {}).get(pname, False)
                color = "#4CAF50" if active else "#D32F2F"
                square = QWidget()
                square.setFixedSize(18, 18)
                square.setStyleSheet(f"background-color:{color}; border:1px solid #333; border-radius:2px;")
                self.pattern_grid_layout.addWidget(square, row_idx, col_idx, alignment=Qt.AlignCenter)
        self.pattern_grid_layout.setColumnStretch(0, 0)
        for col_idx in range(1, len(rounds_list) + 1):
            self.pattern_grid_layout.setColumnStretch(col_idx, 0)

    def _extract_ap_map_for_round(self, df: pd.DataFrame) -> Dict[str, bool]:
        ap_colname = None
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower().startswith("ap list"):
                ap_colname = col
                break
        if ap_colname is None:
            return {p: False for p in self.pattern_names}
        series_nonnull = df[ap_colname].dropna()
        if series_nonnull.empty:
            return {p: False for p in self.pattern_names}
        ap_str = str(series_nonnull.iloc[-1])
        return self.parse_ap_list(ap_str)

    def _client_label(self, cid) -> str:
        return str(cid)

    def update_data(self):
        base_dir = os.path.dirname(__file__)
        subdir = "Docker" if self.simulation_type == "Docker" else "Local"
        perf_dir = os.path.join(base_dir, subdir, "performance")
        raw_files = glob.glob(os.path.join(perf_dir, "FLwithAP_performance_metrics_round*.csv"))
        files_info = []
        for f in raw_files:
            m = re.search(r"round(\d+)", f)
            if m:
                rnd = int(m.group(1))
                files_info.append((rnd, f))
        if not files_info:
            return
        files_info.sort(key=lambda x: x[0])
        if not self.clients:
            first_round, first_file = files_info[0]
            df0 = pd.read_csv(first_file)
            cids_seen = []
            for cid in df0["Client ID"].dropna().tolist():
                if cid not in cids_seen:
                    cids_seen.append(cid)
            self.clients = cids_seen
            for cid in self.clients:
                self.client_colors[cid] = random_pastel()
        rounds_list = []
        f1_list = []
        total_times_list = []
        table_rows = []
        pattern_matrix_data = []
        for rnd, fpath in files_info:
            df = pd.read_csv(fpath)
            df_nonan = df.dropna(subset=["Train Loss"])
            if df_nonan.empty:
                continue
            last = df_nonan.iloc[-1]
            f1_val = last.get("Val F1", float("nan"))
            total_time_val = last.get("Total Time of FL Round", float("nan"))
            rounds_list.append(rnd)
            f1_list.append(f1_val)
            total_times_list.append(total_time_val)
            for cid in self.clients:
                row_c = df[(df["Client ID"] == cid) & (df["FL Round"] == rnd)]
                if row_c.empty:
                    continue
                row0 = row_c.iloc[0]
                train_t = row0.get("Training Time", float("nan"))
                comm_t = row0.get("Communication Time", float("nan"))
                table_rows.append(
                    (
                        rnd,
                        self._client_label(cid),
                        f"{f1_val:.2f}",
                        f"{total_time_val:.0f}",
                        f"{train_t:.2f}",
                        f"{comm_t:.2f}",
                    )
                )
            active_map = self._extract_ap_map_for_round(df)
            pat_row = {"round": rnd}
            for pname in self.pattern_names:
                pat_row[pname] = active_map.get(pname, False)
            pattern_matrix_data.append(pat_row)
        self.metrics_table.setRowCount(len(table_rows))
        for r_idx, rowvals in enumerate(table_rows):
            for c_idx, cellval in enumerate(rowvals):
                item = QTableWidgetItem(str(cellval))
                item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(r_idx, c_idx, item)
        current_row = 0
        total_rows = len(table_rows)
        while current_row < total_rows:
            this_round = table_rows[current_row][0]
            span_len = 1
            i = current_row + 1
            while i < total_rows and table_rows[i][0] == this_round:
                span_len += 1
                i += 1
            if span_len > 1:
                self.metrics_table.setSpan(current_row, 2, span_len, 1)
                self.metrics_table.setSpan(current_row, 3, span_len, 1)
                for r in range(current_row + 1, current_row + span_len):
                    empty_f1 = QTableWidgetItem("")
                    empty_f1.setTextAlignment(Qt.AlignCenter)
                    self.metrics_table.setItem(r, 2, empty_f1)
                    empty_tt = QTableWidgetItem("")
                    empty_tt.setTextAlignment(Qt.AlignCenter)
                    self.metrics_table.setItem(r, 3, empty_tt)
            current_row += span_len
        self.update_pattern_grid(pattern_matrix_data)
        self.ax_f1.clear()
        sns.lineplot(x=rounds_list, y=f1_list, marker="o", ax=self.ax_f1, color=self.color_f1)
        self.ax_f1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_f1.set_title("Global Model Accuracy", fontweight="bold")
        self.ax_f1.set_xlabel("Federated Learning Round")
        self.ax_f1.set_ylabel("F1 Score")
        self.canvas_f1.draw()
        self.ax_tot.clear()
        sns.lineplot(x=rounds_list, y=total_times_list, marker="o", ax=self.ax_tot, color=self.color_tot)
        self.ax_tot.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_tot.set_title("Total Round Time", fontweight="bold")
        self.ax_tot.set_xlabel("Federated Learning Round")
        self.ax_tot.set_ylabel("Total Round Time (sec)")
        self.canvas_tot.draw()
        self.ax_train.clear()
        self.ax_comm.clear()
        for cid in self.clients:
            rds = []
            tv = []
            cv = []
            for rnd, fpath in files_info:
                df = pd.read_csv(fpath)
                row_c = df[(df["Client ID"] == cid) & (df["FL Round"] == rnd)]
                if row_c.empty:
                    continue
                rds.append(rnd)
                tv.append(row_c["Training Time"].values[0])
                cv.append(row_c["Communication Time"].values[0])
            color_for_client = self.client_colors.get(cid, random_pastel())
            sns.lineplot(x=rds, y=tv, marker="o", ax=self.ax_train, label=self._client_label(cid), color=color_for_client)
            sns.lineplot(x=rds, y=cv, marker="o", ax=self.ax_comm, label=self._client_label(cid), color=color_for_client)
        self.ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_comm.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax, title in [(self.ax_train, "Training Time"), (self.ax_comm, "Communication Time")]:
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Training Time (sec)" if title == "Training Time" else "Communication Time (sec)")
            ax.legend()
        self.canvas_train.draw()
        self.canvas_comm.draw()

class SimulationPage(QWidget):
    def __init__(self, config, num_supernodes=None):
        super().__init__()
        self.config = config
        self.setWindowTitle("Simulation Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        self.output_area = QPlainTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: black;
                font-family: Courier;
                font-size: 12px;
                color: white;
            }
            """
        )
        layout.addWidget(self.output_area)
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(20)
        buttons_row.setContentsMargins(0, 0, 0, 0)
        self.dashboard_button = QPushButton("📈 Real-Time Performance Analysis")
        self.dashboard_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dashboard_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007ACC; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #005F9E; }
            QPushButton:pressed { background-color: #004970; }
            """
        )
        self.dashboard_button.clicked.connect(self.open_dashboard)
        self.XAI_button = QPushButton("🤖 AI-Agents Reasoning")
        self.XAI_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.XAI_button.setCursor(Qt.PointingHandCursor)
        self.XAI_button.setStyleSheet(
            """
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #00b300; }
            QPushButton:pressed { background-color: #008000; }
            """
        )
        self.XAI_button.clicked.connect(self.open_agents_viewer)
        adaptation_mode = self.config.get("adaptation", "None")
        if str(adaptation_mode).strip().lower() == "none":
            self.XAI_button.setEnabled(False)
            self.XAI_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #888;
                    color: #eee;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                """
            )
        buttons_row.addWidget(self.dashboard_button, 1)
        buttons_row.addWidget(self.XAI_button, 1)
        layout.addLayout(buttons_row)
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #ff6666; }
            QPushButton:pressed { background-color: #cc0000; }
            """
        )
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.agent_viewer = None
        self.start_simulation(num_supernodes)

    def open_dashboard(self):
        self.db = DashboardWindow(self.config["simulation_type"])
        self.db.show()

    def open_agents_viewer(self):
        policy = str(self.config.get("adaptation", "AI-Agents")).strip()
        dlg = AIAgentsLogViewer(self, policy, agent_log_path())
        dlg.resize(900, 600)
        dlg.show()
        self.agent_viewer = dlg


    def start_simulation(self, num_supernodes):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sim_type = self.config["simulation_type"]
        rounds = self.config["rounds"]
        if sim_type == "Docker":
            work_dir = os.path.join(base_dir, "Docker")
            dc_in = os.path.join(work_dir, "docker-compose.yml")
            dc_out = os.path.join(work_dir, "docker-compose.dynamic.yml")
            self.output_area.appendPlainText("Launching Docker Compose...")
            with open(dc_in, "r") as f:
                compose = yaml.safe_load(f)
            server_svc = compose["services"].get("server")
            client_tpl = compose["services"].get("client")
            if not server_svc or not client_tpl:
                self.output_area.appendPlainText("Error: Missing server or client service in docker-compose.yml")
                return
            new_svcs = {"server": server_svc}
            for detail in self.config["client_details"]:
                cid = detail["client_id"]
                cpu = detail["cpu"]
                ram = detail["ram"]
                svc = copy.deepcopy(client_tpl)
                svc.pop("image", None)
                svc.pop("deploy", None)
                svc["container_name"] = f"Client{cid}"
                svc["cpus"] = cpu
                svc["mem_limit"] = f"{ram}g"
                env = svc.setdefault("environment", {})
                env["NUM_ROUNDS"] = str(rounds)
                env["NUM_CPUS"] = str(cpu)
                env["NUM_RAM"] = str(ram)
                env["CLIENT_ID"] = str(cid)
                new_svcs[f"client{cid}"] = svc
            compose["services"] = new_svcs
            with open(dc_out, "w") as f:
                yaml.safe_dump(compose, f)
            cmd = "/opt/homebrew/bin/docker"
            args = ["compose", "-f", dc_out, "up"]
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)
            env = QProcessEnvironment.systemEnvironment()
            env.insert("COMPOSE_BAKE", "true")
            self.process.setProcessEnvironment(env)
            self.process.start(cmd, args)
            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Error: Docker Compose failed to start")
                self.output_area.appendPlainText(self.process.errorString())
                return
        else:
            work_dir = os.path.join(base_dir, "Local")
            cmd = "flower-simulation"
            args = ["--server-app", "server:app", "--client-app", "client:app", "--num-supernodes", str(num_supernodes)]
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)
            self.process.start(cmd, args)
            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Local simulation failed to start")

    def handle_stdout(self):
        sender = self.sender()
        if sender is None:
            return
        data = sender.readAllStandardOutput()
        if not data:
            data = sender.readAllStandardError()
        try:
            encoding = locale.getpreferredencoding(False)
            stdout = bytes(data).decode(encoding)
        except UnicodeDecodeError:
            stdout = bytes(data).decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            cleaned = self.remove_ansi_sequences(line)
            stripped = cleaned.lstrip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("Network "):
                continue
            if stripped.startswith("Container "):
                continue
            if stripped.startswith("Attaching to"):
                continue
            if "flower-super" in stripped:
                continue
            lower = stripped.lower()
            if any(
                key in lower
                for key in [
                    "deprecated",
                    "to view usage",
                    "to view all available options",
                    "warning",
                    "entirely in future versions",
                    "files already downloaded and verified",
                    "client pulling",
                ]
            ):
                continue
            if re.match(r"^[^|]+\|\s*$", cleaned):
                continue
            if "client" in lower:
                html = f"<span style='color:#4caf50;'>{cleaned}</span>"
            elif "server" in lower:
                html = f"<span style='color:#2196f3;'>{cleaned}</span>"
            else:
                html = f"<span style='color:white;'>{cleaned}</span>"
            self.output_area.appendHtml(html)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def remove_ansi_sequences(self, text):
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def process_finished(self):
        self.output_area.appendPlainText("Simulation finished.")
        if hasattr(self, "title_label") and self.title_label:
            self.title_label.setText("Simulation Results")
        self.stop_button.setText("Close")
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.close_application)
        self.process = None

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")
        if hasattr(self, "title_label") and self.title_label:
            self.title_label.setText("Simulation Terminated")
        self.stop_button.setText("Close")
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.close_application)

    def close_application(self):
        try:
            if hasattr(self, "process") and self.process is not None:
                self.process.terminate()
                self.process.waitForFinished()
        except Exception:
            pass
        try:
            if hasattr(self, "agent_viewer") and self.agent_viewer is not None:
                self.agent_viewer.close()
        except Exception:
            pass
        QCoreApplication.quit()
