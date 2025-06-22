import os
import sys
import json
import re
import yaml
import copy
import glob
import random
import locale
import importlib.util
from flwr.common.logger import log
from logging import INFO
import pandas as pd
import seaborn as sns
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.segmentation import mark_boundaries, slic
import os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torchvision.transforms as T
from PIL import Image
import os
import torch
from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import subprocess
import ctypes 

def keep_awake():
    if sys.platform == "darwin":
        # macOS
        subprocess.Popen(["caffeinate", "-dimsu"])
    elif os.name == "nt":
        ES_CONTINUOUS        = 0x80000000
        ES_SYSTEM_REQUIRED   = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    else:
        try:
            subprocess.Popen([
                "systemd-inhibit",
                "--what=idle",
                "--why=Keep app awake",
                "--mode=block",
                "sleep",
                "infinity"
            ])
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
        random.random() * 0.5 + 0.5
    )

class DashboardWindow(QWidget):
    def __init__(self, simulation_type):
        super().__init__()
        self.base_dir     = os.path.dirname(os.path.abspath(__file__))
        self.simulation_type = simulation_type
        self.setWindowTitle("Live Dashboard")
        self.setStyleSheet("background-color: white;")
        self.resize(1200, 800)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        base_dir = os.path.dirname(__file__)
        subdir   = 'Docker' if simulation_type == 'Docker' else 'Local'
        cfg_path = os.path.join(self.base_dir, self.simulation_type, "configuration", "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.client_configs = {
            int(d["client_id"]): d
            for d in cfg.get("client_details", [])
        }
        model_name = ""
        dataset_name = ""

        with open(cfg_path, 'r') as cf:
            cfg = json.load(cf)
            if cfg.get("client_details"):
                first = cfg["client_details"][0]
                model_name = first.get("model")
                dataset_name = first.get("dataset")

        # Persistent pastel colors
        self.color_f1 = random_pastel()
        self.color_tot = random_pastel()
        self.client_colors = {}
        self.clients = []

        # Model section
        lbl_mod = QLabel(f"Model ({model_name})")
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
        lbl_cli = QLabel(f"Clients ({dataset_name})")
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
        base_dir = os.path.dirname(__file__)
        subdir  = 'Docker' if self.simulation_type == 'Docker' else 'Local'
        perf_dir = os.path.join(base_dir, subdir, 'performance')
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
        self.ax_f1.set_title('Accuracy over Federated Learning Round', fontweight='bold')
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

        self.ax_train.clear()
        self.ax_comm.clear()
        for cid in self.clients:
            rds, tv, cv = [], [], []
            for f in files:
                df = pd.read_csv(f)
                r = int(re.search(r'round(\d+)', f).group(1))
                row = df[(df['Client ID'] == cid) & (df['FL Round'] == r)]
                if row.empty:
                    continue
                rds.append(r)
                tv.append(row['Training Time'].values[0])
                cv.append(row['Communication Time'].values[0])
            col = self.client_colors[cid]
            sns.lineplot(x=rds, y=tv, marker='o', ax=self.ax_train, label=cid, color=col)
            sns.lineplot(x=rds, y=cv, marker='o', ax=self.ax_comm, label=cid, color=col)
            self.ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_comm.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax, title in [(self.ax_train, 'Training Time'), (self.ax_comm, 'Communication Time')]:
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Round')
            ax.set_ylabel('Training Time (sec)' if title=='Training Time' else 'Communication Time (sec)')
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
        self.dashboard_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dashboard_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_button.setStyleSheet("""
            QPushButton {
                background-color: #007ACC; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
            QPushButton:pressed {
                background-color: #004970;
            }
        """)
        self.dashboard_button.clicked.connect(self.open_dashboard)
        layout.addWidget(self.dashboard_button)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.XAI_button = QPushButton("Explainable AI")
        self.XAI_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.XAI_button.setCursor(Qt.PointingHandCursor)
        self.XAI_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                width: 200px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }      
        """)
        self.XAI_button.clicked.connect(self.open_XAI)
        layout.addWidget(self.XAI_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.start_simulation(num_supernodes)

    def open_dashboard(self):
        self.db = DashboardWindow(self.config['simulation_type'])
        self.db.show()
    
    def open_XAI(self):
        base = os.path.dirname(os.path.abspath(__file__))
        sim_type = self.config["simulation_type"]       
        dataset = self.config["client_details"][0]["dataset"]
        clients = [d["client_id"] for d in self.config["client_details"]]
        rounds  = list(range(1, self.config["rounds"] + 1))
        self.xai_win = XAIWindow(base, dataset, clients, rounds, sim_type)
        self.xai_win.show()

    def start_simulation(self, num_supernodes):
        # base_dir = dove sta simulation.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sim_type = self.config['simulation_type']
        rounds   = self.config['rounds']

        if sim_type == 'Docker':
            # — Lasci tutto com’era per Docker —
            work_dir = os.path.join(base_dir, 'Docker')
            dc_in    = os.path.join(work_dir, 'docker-compose.yml')
            dc_out   = os.path.join(work_dir, 'docker-compose.dynamic.yml')

            self.output_area.appendPlainText("Launching Docker Compose...")

            with open(dc_in, 'r') as f:
                compose = yaml.safe_load(f)

            server_svc = compose['services'].get('server')
            client_tpl = compose['services'].get('client')
            if not server_svc or not client_tpl:
                self.output_area.appendPlainText("Error: Missing server or client service in docker-compose.yml")
                return

            new_svcs = {'server': server_svc}
            for detail in self.config['client_details']:
                cid = detail['client_id']
                cpu = detail['cpu']
                ram = detail['ram']
                svc = copy.deepcopy(client_tpl)
                svc.pop('image', None)
                svc.pop('deploy', None)
                svc['container_name'] = f"Client{cid}"
                svc['cpus']           = cpu
                svc['mem_limit']      = f"{ram}g"
                env = svc.setdefault('environment', {})
                env['NUM_ROUNDS'] = str(rounds)
                env['NUM_CPUS']   = str(cpu)
                env['NUM_RAM']    = str(ram)
                env['CLIENT_ID']  = str(cid)
                new_svcs[f'client{cid}'] = svc

            compose['services'] = new_svcs
            with open(dc_out, 'w') as f:
                yaml.safe_dump(compose, f)

            cmd  = 'docker-compose'
            args = ['-f', dc_out, 'up']

            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)
            self.process.start(cmd, args)

            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Error: Docker Compose failed to start")
                return

        else:
            # ** Ramo locale: working dir = progetto/Local **
            work_dir = os.path.join(base_dir, 'Local')
            cmd      = 'flower-simulation'
            args     = [
                '--server-app', 'server:app',
                '--client-app', 'client:app',
                '--num-supernodes', str(num_supernodes),
            ]

            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)   
            self.process.start(cmd, args)

            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Local simulation failed to start")
          
    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        try:
            encoding = locale.getpreferredencoding(False)
            stdout = bytes(data).decode(encoding)
        except UnicodeDecodeError:
            stdout = bytes(data).decode('utf-8', errors='replace')

        for line in stdout.splitlines():
            cleaned = self.remove_ansi_sequences(line)
            stripped = cleaned.lstrip()

            if not stripped:
                continue
            if stripped.startswith('#'):
                continue
            if stripped.startswith('Network '):
                continue
            if stripped.startswith('Container '):
                continue
            if stripped.startswith('Attaching to'):
                continue
            if 'flower-super' in stripped:
                continue
            lower = stripped.lower()
            if any(key in lower for key in [
                'deprecated',
                'to view usage',
                'to view all available options',
                'warning',
                'entirely in future versions',
                'Files already downloaded and verified',
                'client Pulling',
            ]):
                continue
            if re.match(r'^[^|]+\|\s*$', cleaned):
                continue

            if 'client' in lower:
                html = f"<span style='color:#4caf50;'>{cleaned}</span>"
            elif 'server' in lower:
                html = f"<span style='color:#2196f3;'>{cleaned}</span>"
            else:
                html = f"<span style='color:white;'>{cleaned}</span>"

            self.output_area.appendHtml(html)

        self.output_area.verticalScrollBar().setValue(
            self.output_area.verticalScrollBar().maximum()
        )

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

    def close_application(self):
        sys.exit(0)

    def is_command_available(self, command):
        from shutil import which
        return which(command) is not None
    
class XAIWindow(QDialog):
    def __init__(self, base_dir, dataset_name, clients, rounds, sim_type):
        super().__init__()
        # attributi base
        self.base_dir = base_dir
        self.sim_type = sim_type
        self.dataset  = dataset_name
        self.clients  = clients

        # carico config.json dei client
        cfg_path = os.path.join(self.base_dir, self.sim_type, "configuration", "config.json")
        with open(cfg_path, "r") as f:
            full_cfg = json.load(f)
        self.client_configs = {int(c["client_id"]): c for c in full_cfg["client_details"]}

        # import dinamico di taskA.py
        taskA_path = os.path.join(self.base_dir, self.sim_type, "taskA.py")
        spec       = importlib.util.spec_from_file_location("taskA", taskA_path)
        taskA_mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(taskA_mod)
        self.NetA        = taskA_mod.Net
        self.load_data_A = taskA_mod.load_data

        # ricavo test_loader e preprocess
        first_cid       = self.clients[0]
        if "imagenet100" in self.dataset.lower():
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
            self.preprocess = Compose([
                Resize((256, 256)),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
            self.test_loader = None
        else:
            # tutti gli altri dataset (CIFAR, ecc.)
            _, test_loader = self.load_data_A(self.client_configs[first_cid])
            self.test_loader = test_loader
            self.preprocess  = test_loader.dataset.transform

        self.setWindowTitle("Explainable AI")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        # seleziona target
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Target:"))
        self.target_cb = QComboBox()
        self.target_cb.addItem("Server")
        for c in clients:
            self.target_cb.addItem(f"Client {c}")
        hl.addWidget(self.target_cb)

        # seleziona round
        hl.addWidget(QLabel("Round:"))
        self.round_sb = QSpinBox()
        self.round_sb.setMinimum(1)
        self.round_sb.setMaximum(max(rounds))
        self.round_sb.setValue(1)
        hl.addWidget(self.round_sb)

        # seleziona indice immagine
        hl.addWidget(QLabel("Sample idx:"))
        self.idx_sb = QSpinBox()
        self.idx_sb.setMinimum(0)
        if self.test_loader:
            max_idx = len(self.test_loader.dataset) - 1
        else:
            max_idx = len(self.image_paths) - 1
        self.idx_sb.setMaximum(max_idx)
        self.idx_sb.setValue(0)
        hl.addWidget(self.idx_sb)

        # button genera explain
        self.run_btn = QPushButton("Run LIME")
        self.run_btn.clicked.connect(self.run_explain)
        hl.addWidget(self.run_btn)

        layout.addLayout(hl)

        # area figure
        self.fig, self.axes = plt.subplots(1, 2)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # preview automatico
        rand_idx = random.randint(self.idx_sb.minimum(), self.idx_sb.maximum())
        self.idx_sb.setValue(rand_idx)
        self.run_explain()

    
            # 2.a) setto gli attributi base
        self.base_dir = base_dir
        self.sim_type = sim_type
        self.dataset  = dataset_name
        self.clients  = clients

        # 2.b) carico config.json per i client
        cfg_path = os.path.join(self.base_dir, self.sim_type, "configuration", "config.json")
        with open(cfg_path, "r") as f:
            full_cfg = json.load(f)
        self.client_configs = {
            int(c["client_id"]): c
            for c in full_cfg["client_details"]
        }

        # 2.c) import dinamico di taskA.py
        taskA_path = os.path.join(self.base_dir, self.sim_type, "taskA.py")
        spec       = importlib.util.spec_from_file_location("taskA", taskA_path)
        taskA_mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(taskA_mod)
        self.NetA        = taskA_mod.Net
        self.load_data_A = taskA_mod.load_data

        # 2.d) ricavo test_loader e preprocess giusti
        first_cid = self.clients[0]
        if "imagenet100" in self.dataset.lower():
            from pathlib import Path
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

            project_dir = Path(self.base_dir) / self.sim_type
            subset_dir  = project_dir / "data" / self.dataset / "test"
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"{subset_dir} non trovato")

            # elenco ordinato di tutti i file immagine
            self.image_paths = sorted(
                f
                for cls in sorted(subset_dir.iterdir()) if cls.is_dir()
                for f   in sorted(cls.iterdir())
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            )

            # stessa pipeline di trasformazione usata in training
            self.preprocess = Compose([
                Resize(256), CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225]),
            ])
            self.test_loader = None
        else:
            _, test_loader = self.load_data_A(self.client_configs[first_cid])
            self.test_loader = test_loader
            self.preprocess  = test_loader.dataset.transform
            self.image_paths  = []

    def load_model(self, target, rnd, client_id=None):
        work_dir = os.path.join(self.base_dir, self.sim_type)
        if target == "server":
            model_path = os.path.join(work_dir, "model_weights", "server", f"MW_round{rnd}.pt")
            cfg         = { "client_id": self.clients[0], **{} }
        else:
            model_path = os.path.join(work_dir, "model_weights", "clients", str(client_id), f"MW_round{rnd}.pt")
            cfg         = { "client_id": client_id,    **{} }

        taskA_path = os.path.join(work_dir, "taskA.py")
        spec       = importlib.util.spec_from_file_location("taskA", taskA_path)
        taskA_mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(taskA_mod)
        model      = taskA_mod.Net()  
        state      = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def load_image(self, idx, client_id=None):
        img_tensor, _ = self.test_loader.dataset[idx]
        return T.ToPILImage()(img_tensor)

    def run_explain(self):
        from PIL import Image
        import numpy as np
        import torch
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        # 1) modello e device
        idx = self.idx_sb.value()
        if self.target_cb.currentText() == "Server":
            model = self.load_model("server", self.round_sb.value())
        else:
            cid   = int(self.target_cb.currentText().split()[-1])
            model = self.load_model("client", self.round_sb.value(), client_id=cid)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device).eval()

        # 2) carica immagine
        if self.test_loader is None:
            # ImageNet100
            path  = self.image_paths[idx]
            img   = Image.open(path).convert("RGB")
            img_np = np.array(img) / 255.0
            preprocess_fn = self.preprocess
        else:
            # CIFAR o altro
            from torchvision.transforms import Normalize
            tensor, _ = self.test_loader.dataset[idx]
            # inverti l'ultima Normalize
            inv = None
            for t in reversed(self.preprocess.transforms):
                if isinstance(t, Normalize):
                    inv = Normalize(
                        mean=[-m/s for m,s in zip(t.mean,t.std)],
                        std =[1/s    for s   in t.std],
                    )
                    break
            un = inv(tensor) if inv else tensor
            img_np = un.cpu().numpy().transpose(1,2,0)
            img_np = np.clip(img_np, 0, 1)
            preprocess_fn = self.preprocess

        # 3) spiegazione LIME
        explainer = lime_image.LimeImageExplainer()
        def batch_predict(batch):
            batch_t = torch.stack([
                preprocess_fn(Image.fromarray((img*255).astype(np.uint8)))
                for img in batch
            ], dim=0).to(device)
            with torch.no_grad():
                out = model(batch_t)
                return torch.softmax(out, dim=1).cpu().numpy()

        exp = explainer.explain_instance(
            img_np, batch_predict,
            top_labels=5, hide_color=0, num_samples=1000
        )
        lbl = exp.top_labels[0]

        # 4) mask e boundary
        temp, mask = exp.get_image_and_mask(
            label=lbl,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        if temp.max() > 1:
            temp = temp / 255.0
        seg = mark_boundaries(temp, mask)

        # 5) disegna
        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].imshow(img_np)
        self.axes[0].set_title("Originale")
        self.axes[0].axis("off")

        self.axes[1].imshow(seg)
        self.axes[1].set_title("LIME Explanation")
        self.axes[1].axis("off")

        # 6) legenda
        import matplotlib.patches as mpatches
        pos = mpatches.Patch(color='green', label='Positive')
        neg = mpatches.Patch(color='red',   label='Negative')
        self.fig.legend(
            handles=[pos, neg],
            labels =['Positive', 'Negative'],
            loc    ='upper right',
            frameon=False
        )

        self.canvas.draw()





