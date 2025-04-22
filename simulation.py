import os
import sys
import json
import re
import glob
import random
import pandas as pd
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QPushButton, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from simulation_results import SimulationResults


def random_pastel():
    return (random.random() * 0.5 + 0.5,
            random.random() * 0.5 + 0.5,
            random.random() * 0.5 + 0.5)

class DashboardWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Dashboard")
        self.setStyleSheet("background-color: white;")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Persistent pastel colors for session
        self.color_f1 = random_pastel()
        self.color_tot = random_pastel()
        self.client_colors = {}
        self.clients = []

        # Model section
        lbl_mod = QLabel("Model")
        lbl_mod.setStyleSheet("font-weight: bold; font-size: 16px; color: black;")
        layout.addWidget(lbl_mod)
        self.model_area = QTextEdit()
        self.model_area.setReadOnly(True)
        self.model_area.setStyleSheet("background-color: #f9f9f9; color: black;")
        layout.addWidget(self.model_area)

        # Model plots: F1 and Total Time
        h_model = QHBoxLayout()
        self.fig_f1, self.ax_f1 = plt.subplots()
        self.canvas_f1 = FigureCanvas(self.fig_f1)
        h_model.addWidget(self.canvas_f1)
        self.fig_tot, self.ax_tot = plt.subplots()
        self.canvas_tot = FigureCanvas(self.fig_tot)
        h_model.addWidget(self.canvas_tot)
        layout.addLayout(h_model)

        # Clients section
        lbl_cli = QLabel("Clients")
        lbl_cli.setStyleSheet("font-weight: bold; font-size: 16px; color: black;")
        layout.addWidget(lbl_cli)
        self.client_area = QTextEdit()
        self.client_area.setReadOnly(True)
        self.client_area.setStyleSheet("background-color: #f9f9f9; color: black;")
        layout.addWidget(self.client_area)

        # Client plots: Training and Comm
        h_client = QHBoxLayout()
        self.fig_train, self.ax_train = plt.subplots()
        self.canvas_train = FigureCanvas(self.fig_train)
        h_client.addWidget(self.canvas_train)
        self.fig_comm, self.ax_comm = plt.subplots()
        self.canvas_comm = FigureCanvas(self.fig_comm)
        h_client.addWidget(self.canvas_comm)
        layout.addLayout(h_client)

        # Timer to update every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)

    def update_data(self):
        perf_dir = os.path.join(os.path.dirname(__file__), 'Local', 'performance')
        files = sorted(glob.glob(os.path.join(perf_dir, 'FLwithAP_performance_metrics_round*.csv')))
        if not files:
            return

        # Determine fixed clients from first file
        first_df = pd.read_csv(files[0])
        if not self.clients:
            self.clients = first_df['Client ID'].tolist()
            for cid in self.clients:
                self.client_colors[cid] = random_pastel()

        # Collect model metrics across rounds
        rounds, f1s, totals = [], [], []
        html = ''
        for f in files:
            df = pd.read_csv(f)
            last = df.dropna(subset=['Train Loss']).iloc[-1]
            rnd = int(re.search(r'round(\d+)', f).group(1))
            rounds.append(rnd)
            f1s.append(last['Val F1'])
            totals.append(last['Total Time of FL Round'])
            html += f"<b>Round {rnd}</b>  F1: {last['Val F1']:.4f}<br>"
        self.model_area.setHtml(html)

        # Plot F1
        self.ax_f1.clear()
        self.ax_f1.plot(rounds, f1s, marker='o', color=self.color_f1)
        self.ax_f1.set_title('Validation F1 vs Round', fontweight='bold')
        self.ax_f1.set_xlabel('Round')
        self.ax_f1.set_ylabel('F1 Score')
        self.canvas_f1.draw()

        # Plot Total Round Time
        self.ax_tot.clear()
        self.ax_tot.plot(rounds, totals, marker='o', color=self.color_tot)
        self.ax_tot.set_title('Total Round Time vs Round', fontweight='bold')
        self.ax_tot.set_xlabel('Round')
        self.ax_tot.set_ylabel('Time (s)')
        self.canvas_tot.draw()

        # Latest client metrics for text
        last_df = pd.read_csv(files[-1])
        html_c = ''
        for cid in self.clients:
            row = last_df[last_df['Client ID'] == cid].iloc[0]
            html_c += f"<b>{cid}</b>  Train: {row['Training Time']:.2f}s  Comm: {row['Communication Time']:.2f}s<br>"
        self.client_area.setHtml(html_c)

        # Plot client trends across rounds
        self.ax_train.clear()
        self.ax_comm.clear()
        for cid in self.clients:
            rds, tvs, cvs = [], [], []
            for f in files:
                df = pd.read_csv(f)
                sel = df[df['Client ID'] == cid]
                if not sel.empty:
                    r = int(re.search(r'round(\d+)', f).group(1))
                    rds.append(r)
                    tvs.append(sel['Training Time'].values[0])
                    cvs.append(sel['Communication Time'].values[0])
            col = self.client_colors[cid]
            self.ax_train.plot(rds, tvs, marker='o', label=cid, color=col)
            self.ax_comm.plot(rds, cvs, marker='o', label=cid, color=col)

        for ax, title in [(self.ax_train, 'Training Time per Client'),
                          (self.ax_comm, 'Communication Time per Client')]:
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Round')
            ax.set_ylabel('Time (s)')
            ax.legend()
        self.canvas_train.draw()
        self.canvas_comm.draw()

class SimulationPage(QWidget):
    def __init__(self, num_supernodes=None):
        super().__init__()
        self.setWindowTitle('Simulation Output')
        self.setStyleSheet('background-color: white;')
        self.resize(800, 600)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Title
        tl = QHBoxLayout()
        tl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label = QLabel('Running the Simulation...')
        self.title_label.setStyleSheet('color: black; font-size: 24px; font-weight: bold;')
        tl.addWidget(self.title_label)
        tl.addStretch()
        layout.addLayout(tl)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet('background-color: black; color: white; font-family: Courier; font-size: 12px;')
        layout.addWidget(self.output_area)

        # Stop button
        self.stop_button = QPushButton('Stop Simulation')
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet('background-color: #ee534f; color: white; font-size: 14px; padding: 8px 16px; border-radius: 5px;')
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button, alignment=Qt.AlignCenter)

        # Dashboard button
        self.dashboard_button = QPushButton('📈 Live Dashboard')
        self.dashboard_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_button.setStyleSheet('background-color: #2196f3; color: white; font-size: 14px; padding: 8px 16px; border-radius: 5px;')
        self.dashboard_button.clicked.connect(self.open_dashboard)
        bl = QHBoxLayout()
        bl.addStretch()
        bl.addWidget(self.dashboard_button)
        bl.addStretch()
        layout.addLayout(bl)

        # Setup process
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.start_simulation(num_supernodes)

    def start_simulation(self, num_supernodes):
        base = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base, 'Local')
        docker_dir = os.path.join(base, 'Docker')
        cfg = os.path.join(base, 'configuration', 'config.json')
        sim_type, n_rounds = 'Local', 2
        if os.path.exists(cfg):
            try:
                js = json.load(open(cfg))
                sim_type = js.get('simulation_type','Local')
                n_rounds = js.get('rounds',2)
                if num_supernodes is None:
                    num_supernodes = js.get('clients',2)
            except:
                pass
        if sim_type.lower()=='docker':
            self.process.setWorkingDirectory(docker_dir)
            env = QProcessEnvironment.systemEnvironment()
            env.insert('NUM_ROUNDS', str(n_rounds))
            self.process.setProcessEnvironment(env)
            cmd, args = 'docker-compose', ['up','--scale',f'client={num_supernodes}']
        else:
            self.process.setWorkingDirectory(local_dir)
            cmd, args = 'flower-simulation',[ '--server-app','server:app','--client-app','client:app','--num-supernodes',str(num_supernodes) ]
        self.process.start(cmd,args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8')
        for ln in text.splitlines():
            cl = self.remove_ansi_sequences(ln)
            if any(x in cl for x in ['deprecated','future versions']):continue
            if 'WARNING' in cl: continue
            if 'Client' in cl: color='#4caf50'
            elif 'Server' in cl: color='#2196f3'
            else: color='white'
            self.output_area.insertHtml(f"<span style='color:{color};'>{cl}</span>")
            self.output_area.insertPlainText('\n')
        sb = self.output_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def remove_ansi_sequences(self,text):
        return re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])').sub('',text)

    def process_finished(self):
        self.output_area.insertPlainText('Simulation finished.\n')
        self.stop_button.setText('Close')
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.close)

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.insertPlainText('Simulation terminated by user.\n')
            self.stop_button.setText('Close')
            self.stop_button.clicked.disconnect()
            self.stop_button.clicked.connect(self.close)

    def open_dashboard(self):
        self.db = DashboardWindow()
        self.db.show()

    def close(self):
        sys.exit(0)

    def is_command_available(self,cmd):
        from shutil import which
        return which(cmd) is not None
