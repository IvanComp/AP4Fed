import os
import sys
import json
import re
import glob
import random
import pandas as pd
import seaborn as sns
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def random_pastel():
    return (
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5
    )

class DashboardWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Dashboard")
        self.setStyleSheet("background-color: white;")
        self.resize(1000, 800)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Persistent pastel colors
        self.color_f1 = random_pastel()
        self.color_tot = random_pastel()
        self.client_colors = {}
        self.clients = []

        # Model section
        lbl_mod = QLabel("Model")
        lbl_mod.setStyleSheet("font-weight: bold; font-size: 16px; color: black;")
        layout.addWidget(lbl_mod)
        self.model_area = QPlainTextEdit()
        self.model_area.setReadOnly(True)
        self.model_area.setStyleSheet("background-color: #f9f9f9; color: black;")
        layout.addWidget(self.model_area)

        # Model plots: F1 and Total Time
        h_model = QHBoxLayout()
        self.fig_f1, self.ax_f1 = plt.subplots()
        self.fig_f1.patch.set_facecolor('white')
        self.canvas_f1 = FigureCanvas(self.fig_f1)
        h_model.addWidget(self.canvas_f1)
        self.fig_tot, self.ax_tot = plt.subplots()
        self.fig_tot.patch.set_facecolor('white')
        self.canvas_tot = FigureCanvas(self.fig_tot)
        h_model.addWidget(self.canvas_tot)
        layout.addLayout(h_model)

        # Clients section
        lbl_cli = QLabel("Clients")
        lbl_cli.setStyleSheet("font-weight: bold; font-size: 16px; color: black;")
        layout.addWidget(lbl_cli)
        self.client_area = QPlainTextEdit()
        self.client_area.setReadOnly(True)
        self.client_area.setStyleSheet("background-color: #f9f9f9; color: black;")
        layout.addWidget(self.client_area)

        # Client plots: Training and Communication
        h_client = QHBoxLayout()
        self.fig_train, self.ax_train = plt.subplots()
        self.fig_train.patch.set_facecolor('white')
        self.canvas_train = FigureCanvas(self.fig_train)
        h_client.addWidget(self.canvas_train)
        self.fig_comm, self.ax_comm = plt.subplots()
        self.fig_comm.patch.set_facecolor('white')
        self.canvas_comm = FigureCanvas(self.fig_comm)
        h_client.addWidget(self.canvas_comm)
        layout.addLayout(h_client)

        # Timer to update
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)

    def update_data(self):
        perf_dir = os.path.join(os.path.dirname(__file__), 'Local', 'performance')
        files = sorted(glob.glob(os.path.join(perf_dir, 'FLwithAP_performance_metrics_round*.csv')))
        if not files:
            return

        # fixed clients
        df0 = pd.read_csv(files[0])
        if not self.clients:
            self.clients = df0['Client ID'].tolist()
            for cid in self.clients:
                self.client_colors[cid] = random_pastel()

        rounds, f1s, totals = [], [], []
        text_model = ''
        for f in files:
            df = pd.read_csv(f)
            last = df.dropna(subset=['Train Loss']).iloc[-1]
            rnd = int(re.search(r'round(\d+)', f).group(1))
            rounds.append(rnd)
            f1s.append(last['Val F1'])
            totals.append(last['Total Time of FL Round'])
            text_model += f"Round {rnd}: F1={last['Val F1']:.2f}, Total Round Time={last['Total Time of FL Round']:.0f}s\n"
        self.model_area.setPlainText(text_model)

        # plot F1
        self.ax_f1.clear()
        sns.lineplot(x=rounds, y=f1s, marker='o', ax=self.ax_f1, color=self.color_f1)
        self.ax_f1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_f1.set_title('F1 Score over Federated Learning Round', fontweight='bold')
        self.ax_f1.set_xlabel('Federated Learning Round')
        self.ax_f1.set_ylabel('F1 Score')
        self.canvas_f1.draw()

        # plot Total Time
        self.ax_tot.clear()
        sns.lineplot(x=rounds, y=totals, marker='o', ax=self.ax_tot, color=self.color_tot)
        self.ax_tot.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_tot.set_title('Total Round Time over Federated Learning Round', fontweight='bold')
        self.ax_tot.set_xlabel('Federated Learning Round')
        self.ax_tot.set_ylabel('Total Round Time (sec)')
        self.canvas_tot.draw()

        # clients text
        df_last = pd.read_csv(files[-1])
        text_cli = ''
        for cid in self.clients:
            row = df_last[df_last['Client ID']==cid].iloc[0]
            text_cli += f"{cid}: Training Time={row['Training Time']:.2f}s, Communication Time={row['Communication Time']:.2f}s\n"
        self.client_area.setPlainText(text_cli)

        # plot clients trends
        self.ax_train.clear()
        self.ax_comm.clear()
        for cid in self.clients:
            rds, tv, cv = [], [], []
            for f in files:
                df = pd.read_csv(f)
                sel = df[df['Client ID']==cid]
                if not sel.empty:
                    r = int(re.search(r'round(\d+)', f).group(1))
                    rds.append(r)
                    tv.append(sel['Training Time'].values[0])
                    cv.append(sel['Communication Time'].values[0])
            col = self.client_colors[cid]
            sns.lineplot(x=rds, y=tv, marker='o', ax=self.ax_train, label=cid, color=col)
            sns.lineplot(x=rds, y=cv, marker='o', ax=self.ax_comm, label=cid, color=col)
            self.ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_comm.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax,title in [(self.ax_train,'Training Time'),(self.ax_comm,'Communication Time')]:
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Round')
            ax.set_ylabel('Seconds')
            ax.legend()
        self.canvas_train.draw()
        self.canvas_comm.draw()

class SimulationPage(QWidget):
    def __init__(self, num_supernodes=None):
        super().__init__()
        self.setWindowTitle("Simulation Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label = QLabel("Running the Simulation...")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()

        # Output area
        self.output_area = QPlainTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QPlainTextEdit {
                background-color: black;
                font-family: Courier;
                font-size: 12px;
                color: white;
            }
        """)
        layout.addWidget(self.output_area)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.dashboard_button = QPushButton("📈 Real-Time Performance Analysis")
        self.dashboard_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_button.setStyleSheet("background-color:#2196f3;color:white;padding:8px 16px;border-radius:5px;font-size:14px;")
        self.dashboard_button.clicked.connect(self.open_dashboard)
        btn_layout.addWidget(self.dashboard_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Bottone per terminare la simulazione
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 5px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button, alignment=Qt.AlignCenter)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.start_simulation(num_supernodes)

    def open_dashboard(self):
        self.db = DashboardWindow()
        self.db.show()

    def start_simulation(self, num_supernodes):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, 'Local')
        docker_dir = os.path.join(base_dir, 'Docker')
        config_path = os.path.join(base_dir, 'configuration', 'config.json')

        simulation_type = "Local"
        num_rounds = 2

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    simulation_type = config.get("simulation_type", "Local")
                    num_rounds = config.get("rounds", 2)
                    if num_supernodes is None:
                        num_supernodes = config.get('clients', 2)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to read configuration file:\n{e}")
                if num_supernodes is None:
                    num_supernodes = 2
        else:
            QMessageBox.warning(self, "Warning", f"Configuration file not found at {config_path}. Using defaults.")
            if num_supernodes is None:
                num_supernodes = 2

        if simulation_type.lower() == "docker":
            if not os.path.exists(docker_dir):
                self.output_area.appendPlainText(f"The 'Docker' directory does not exist: {docker_dir}")
                return
            self.process.setWorkingDirectory(docker_dir)
            env = QProcessEnvironment.systemEnvironment()
            env.insert("NUM_ROUNDS", str(num_rounds))
            self.process.setProcessEnvironment(env)
            command = "docker-compose"
            args = ["up", "--scale", f"client={num_supernodes}"]
        else:
            if not os.path.exists(local_dir):
                self.output_area.appendPlainText(f"The 'Local' directory does not exist: {local_dir}")
                return
            self.process.setWorkingDirectory(local_dir)
            command = "flower-simulation"
            args = [
                "--server-app", "server:app",
                "--client-app", "client:app",
                "--num-supernodes", str(num_supernodes)
            ]

        if not self.is_command_available(command):
            self.output_area.appendPlainText(f"Command '{command}' not found. Ensure it is installed and in the system PATH.")
            return

        self.process.start(command, args)
        if not self.process.waitForStarted():
            self.output_area.appendPlainText("Failed to start the simulation process.")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')
        lines = stdout.splitlines()
        for line in lines:
            cleaned_line = self.remove_ansi_sequences(line)
            if "This is a deprecated feature." in cleaned_line or "entirely in future versions of Flower." in cleaned_line:
                continue
            if "WARNING" in cleaned_line or "warning" in cleaned_line:
                continue
            if "Client" in cleaned_line or "client" in cleaned_line:
                colored_line = f"<span style='color: #4caf50;'>{cleaned_line}</span>"
            elif "Server" in cleaned_line or "server" in cleaned_line:
                colored_line = f"<span style='color: #2196f3;'>{cleaned_line}</span>"
            else:
                colored_line = f"<span style='color: white;'>{cleaned_line}</span>"
            self.output_area.appendHtml(colored_line)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def remove_ansi_sequences(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def process_finished(self):
        self.output_area.appendPlainText("Simulation finished.")
        self.title_label.setText("Simulation Results")
        self.stop_button.setText("Close")
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.close_application)
        self.process = None

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")
            self.title_label.setText("Simulation Terminated")
            self.stop_button.setText("Close")
            self.stop_button.clicked.disconnect()
            self.stop_button.clicked.connect(self.close_application)
            if hasattr(self, 'loading_movie'):
                self.loading_movie.stop()

    def close_application(self):
        sys.exit(0)

    def is_command_available(self, command):
        from shutil import which
        return which(command) is not None