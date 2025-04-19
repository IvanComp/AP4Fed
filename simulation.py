import os
import sys
import json
import re
import zipfile
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt5.QtGui import QMovie
from simulation_results import SimulationResults

class SimulationPage(QWidget):
    def __init__(self, num_supernodes=None):
        super().__init__()
        self.setWindowTitle("Simulation Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        #self.LLM_button = QPushButton("🤖 LLM Agent")
        #self.LLM_button.setCursor(Qt.PointingHandCursor)
        #self.LLM_button.setStyleSheet("""
        #    QPushButton {
        #       background-color: #1a69c9;
        #        color: white;
        #        font-size: 14px;
        #        padding: 8px 16px;
        #        border-radius: 5px;
        #        margin-top: 10px;
        #    }
        #    QPushButton:hover {
        #        background-color: #0066cc;
        #    }
        #    QPushButton:pressed {
        #        background-color: #004d99;
        #    }
        #""")
        #self.LLM_button.clicked.connect(self.LLM_Agent)
        #layout.addWidget(self.LLM_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        # Title layout con animazione e timer
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

        # Layout orizzontale per i bottoni "View Simulation Report" e "Download Model Weights"
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.view_report_button = QPushButton("📊 View Simulation Report")
        self.view_report_button.setCursor(Qt.PointingHandCursor)
        self.view_report_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 5px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)
        self.view_report_button.clicked.connect(self.open_simulation_results)
        self.view_report_button.hide()
        button_layout.addWidget(self.view_report_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.start_simulation(num_supernodes)

    def update_loading_animation(self):
        self.current_emoji_index = (self.current_emoji_index + 1) % len(self.loading_emojis)
        self.loading_label.setText(self.loading_emojis[self.current_emoji_index])

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
        self.view_report_button.show()
        self.stop_button.setText("Close")
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.close_application)
        self.process = None
        if hasattr(self, 'loading_movie'):
            self.loading_movie.stop()

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")
            self.title_label.setText("Simulation Terminated")
            self.view_report_button.hide()
            self.stop_button.setText("Close")
            self.stop_button.clicked.disconnect()
            self.stop_button.clicked.connect(self.close_application)
            if hasattr(self, 'loading_movie'):
                self.loading_movie.stop()

    #def LLM_Agent(self):
     #   QMessageBox.information(self, "LLM Agent", "LLM Agent is not implemented yet.")

    def open_simulation_results(self):
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Local",
            "performance",
            "FLwithAP_performance_metrics.csv"
        )
        self.results_window = SimulationResults(csv_path)
        self.results_window.show()

    def close_application(self):
        sys.exit(0)

    def is_command_available(self, command):
        from shutil import which
        return which(command) is not None
