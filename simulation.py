import os
import json  # Imported for reading the configuration file
import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QProcess

class SimulationPage(QWidget):
    def __init__(self, num_supernodes=None):
        """
        Initializes the SimulationPage.

        Args:
            num_supernodes (int, optional): The number of supernodes based on the number of clients.
                                             Defaults to None, which will attempt to read from config.json or use a default value.
        """
        super().__init__()
        self.setWindowTitle("Simulation Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")

        # Layout setup
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Title
        title_label = QLabel("Simulation Running...")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)

        # Output area
        self.output_area = QPlainTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: Courier;
            font-size: 12px;
        """)
        layout.addWidget(self.output_area)

        # Stop button
        stop_button = QPushButton("Stop Simulation")
        stop_button.setCursor(Qt.PointingHandCursor)
        stop_button.setStyleSheet("""
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
        stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(stop_button, alignment=Qt.AlignCenter)

        # Initialize QProcess
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

        # Start the simulation
        self.start_simulation(num_supernodes)

    def start_simulation(self, num_supernodes):
        """
        Starts the simulation process with dynamic num_supernodes.

        Args:
            num_supernodes (int, optional): The number of supernodes. If None, attempts to read from config.json or uses default.
        """
        # Change working directory to 'Local'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, 'Local')

        # Path to the configuration JSON
        config_path = os.path.join(base_dir, 'configuration', 'config.json')

        # Determine num_supernodes
        if num_supernodes is None:
            # Attempt to read from config.json
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    client_details = config.get('client_details', [])
                    num_supernodes = len(client_details)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to read configuration file:\n{e}")
                    num_supernodes = 2  # Default value
            else:
                QMessageBox.warning(self, "Warning", f"Configuration file not found at {config_path}. Using default num_supernodes=2.")
                num_supernodes = 2  # Default value

        # Verify that the 'Local' directory exists
        if not os.path.exists(local_dir):
            self.output_area.appendPlainText(f"The 'Local' directory does not exist: {local_dir}")
            return

        # Set the working directory for the process
        self.process.setWorkingDirectory(local_dir)

        # Command to execute
        command = "flower-simulation"
        args = [
            "--server-app", "server:app",
            "--client-app", "client:app",
            "--num-supernodes", str(num_supernodes)  # Use the dynamic number of supernodes
        ]

        # Verify if the command is available
        if not self.is_command_available(command):
            self.output_area.appendPlainText(f"Command '{command}' not found. Ensure it is installed and in the system PATH.")
            return

        # Start the process
        self.process.start(command, args)

        if not self.process.waitForStarted():
            self.output_area.appendPlainText("Failed to start the simulation process.")

    def handle_stdout(self):
        """
        Handles the standard output from the simulation process.
        """
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')

        # Remove ANSI escape sequences
        cleaned_stdout = self.remove_ansi_sequences(stdout)

        self.output_area.appendPlainText(cleaned_stdout)
        # Scroll to the end
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

    def remove_ansi_sequences(self, text):
        """
        Removes ANSI escape sequences from text.

        Args:
            text (str): Text containing ANSI sequences.

        Returns:
            str: Cleaned text.
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def process_finished(self):
        """
        Handles the process finished signal.
        """
        self.output_area.appendPlainText("Simulation finished.")
        self.process = None

    def stop_simulation(self):
        """
        Stops the simulation process.
        """
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")

    def is_command_available(self, command):
        """
        Checks if a command is available in the system PATH.

        Args:
            command (str): The command to check.

        Returns:
            bool: True if available, False otherwise.
        """
        from shutil import which
        return which(command) is not None