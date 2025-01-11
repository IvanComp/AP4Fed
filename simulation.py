import os
import sys
import json  # Imported for reading the configuration file
import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QProcess
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

        self.title_label = QLabel("Running the Simulation...")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.title_label)

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

        self.view_report_button = QPushButton("View Simulation Report")
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
        layout.addWidget(self.view_report_button, alignment=Qt.AlignCenter)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        self.start_simulation(num_supernodes)

    def start_simulation(self, num_supernodes):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, 'Local')
        config_path = os.path.join(base_dir, 'configuration', 'config.json')

        if num_supernodes is None:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        num_supernodes = config.get('clients', 2)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to read configuration file:\n{e}")
                    num_supernodes = 2
            else:
                QMessageBox.warning(self, "Warning", f"Configuration file not found at {config_path}. Using default num_supernodes=2.")
                num_supernodes = 2

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
        # Suddividiamo in righe e le elaboriamo una a una
        lines = stdout.splitlines()
        for line in lines:
            cleaned_line = self.remove_ansi_sequences(line)

            # Filtra le righe che contengono i messaggi di deprecazione
            if "This is a deprecated feature." in cleaned_line or "entirely in future versions of Flower." in cleaned_line:
                continue

            # Se la riga contiene WARNING, la saltiamo
            if "WARNING" in cleaned_line or "warning" in cleaned_line:
                continue

            # Se la riga contiene "Client", la coloriamo di verde
            if "Client" in cleaned_line or "client" in cleaned_line:
                colored_line = f"<span style='color: #4caf50;'>{cleaned_line}</span>"
            # Se la riga contiene "Server", la coloriamo di blu
            elif "Server" in cleaned_line or "server" in cleaned_line:
                colored_line = f"<span style='color: #2196f3;'>{cleaned_line}</span>"
            # Altrimenti, lasciamo il colore bianco
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

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")

    def open_simulation_results(self):
        from simulation_results import SimulationResults
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