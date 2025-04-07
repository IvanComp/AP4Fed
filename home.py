import os
import sys
import json  
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl, QSize
from presimulation import PreSimulationPage 
from recap_simulation import RecapSimulationPage  

user_choices = []

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AP4FED - Home Page")
        self.resize(800, 600)
        base_dir = os.path.dirname(__file__)
        logo_path = os.path.join(base_dir, "img/readme/logo.svg")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)
        layout.addStretch()

        logo_label = QLabel(self)
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("Logo not found")
            logo_label.setStyleSheet("color: red; font-size: 14px;")
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        description_label = QLabel("A Federated Learning Benchmarking Platform")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(description_label)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_start = QPushButton("Start a new project")
        button_start.setStyleSheet("""
            QPushButton {
                background-color: #70C284; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
                width: 200px;  /* Larghezza metà di Close */
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        button_start.setCursor(Qt.PointingHandCursor)
        button_start.clicked.connect(self.start_new_project)
        button_layout.addWidget(button_start)

        button_load = QPushButton("Load a .json Configuration")
        button_load.setStyleSheet("""
            QPushButton {
                background-color: #5c9bee; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
                width: 200px;  /* Larghezza metà di Close */
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #004d99;
            }
        """)
        button_load.setCursor(Qt.PointingHandCursor)
        button_load.clicked.connect(self.load_configuration)
        button_layout.addWidget(button_load)

        layout.addLayout(button_layout)
        button_close = QPushButton("Close")
        button_close.setStyleSheet("""
            QPushButton {
                background-color: #ee534f; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
                width: 430px;  /* Larghezza del pulsante Close */
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        button_close.setCursor(Qt.PointingHandCursor)
        button_close.clicked.connect(self.close_application)
        layout.addWidget(button_close, alignment=Qt.AlignCenter)
        layout.addStretch()
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignCenter)
        version_label = QLabel("1.5.0 version")
        version_label.setStyleSheet("font-size: 12px; color: black; margin: 5px;")
        footer_layout.addWidget(version_label)

        github_button = QPushButton()
        github_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        github_pixmap_path = os.path.join(base_dir, "img/github.png")
        if os.path.exists(github_pixmap_path):
            github_pixmap = QPixmap(github_pixmap_path).scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            github_icon = QIcon(github_pixmap)
            github_button.setIcon(github_icon)
        else:
            github_button.setText("GitHub")
        github_button.setCursor(Qt.PointingHandCursor)
        github_button.clicked.connect(self.open_github_link)
        footer_layout.addWidget(github_button)

        layout.addLayout(footer_layout)

        # Personalizza la finestra
        self.setStyleSheet("background-color: white;")

    def start_new_project(self):
        self.second_screen = SecondScreen()
        self.second_screen.show()
        self.close()

    def load_configuration(self):
        """
        Funzione per caricare un file di configurazione JSON.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Configuration File",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    loaded_config = json.load(f)

                if self.validate_configuration(loaded_config):
                    global user_choices
                    user_choices = [loaded_config]

                    self.recap_simulation_page = RecapSimulationPage(user_choices)
                    self.recap_simulation_page.show()

                    self.close()
                else:
                    QMessageBox.warning(self, "Invalid Configuration", "The loaded configuration is invalid.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"An error occurred while loading the file:\n{e}")

    def validate_configuration(self, config):
        """
        Valida la configurazione caricata.
        """
        expected_keys = {'simulation_type', 'rounds', 'clients', 'patterns', 'client_details'}

        if not isinstance(config, dict):
            return False

        for key in expected_keys:
            if key not in config:
                return False

        return True

    def close_application(self):
        # Mostra un popup di conferma
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("Are you sure you want to close the application?")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setObjectName("myMessageBox")

        # Stile del popup corretto
        msg_box.setStyleSheet("""
            #myMessageBox {
                background-color: white;
            }
            #myMessageBox QLabel {
                color: black;
                font-size: 14px;
            }
            #myMessageBox QPushButton {
                background-color: lightgray;
                color: black;
                font-size: 12px;
                padding: 5px;
                border-radius: 5px;
            }
            #myMessageBox QPushButton:hover {
                background-color: gray;
                color: white;
            }
        """)

        yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
        no_button = msg_box.addButton("No", QMessageBox.NoRole)
        yes_button.setCursor(Qt.PointingHandCursor)
        no_button.setCursor(Qt.PointingHandCursor)

        msg_box.exec_()

        if msg_box.clickedButton() == yes_button:
            self.close()

    def open_github_link(self):
        QDesktopServices.openUrl(QUrl("https://github.com/IvanComp/AP4Fed"))


class SecondScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AP4FED - New Project")
        self.resize(800, 600)

        base_dir = os.path.dirname(__file__)
        docker_path = os.path.join(base_dir, "img/docker.png")
        local_path = os.path.join(base_dir, "img/local.png")
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        local_button = QPushButton()
        local_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-size: 14px; 
                padding: 10px;
                border: 2px solid black;
                border-radius: 10px;
                width: 250px;
                height: 150px;
                text-align: center;
            }
            QPushButton::icon {
                subcontrol-origin: padding;
                subcontrol-position: top center; /* Posiziona l'icona sopra */
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        if os.path.exists(local_path):
            local_pixmap = QPixmap(local_path).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            local_icon = QIcon(local_pixmap)
            local_button.setIcon(local_icon)
            local_button.setIconSize(QSize(50, 50))
        local_button.setText("Create project Locally")
        local_button.setCursor(Qt.PointingHandCursor)
        local_button.clicked.connect(self.select_local)
        button_layout.addWidget(local_button)

        docker_button = QPushButton()
        docker_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-size: 14px; 
                padding: 10px;
                border: 2px solid black;
                border-radius: 10px;
                width: 250px;
                height: 150px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        if os.path.exists(docker_path):
            docker_icon = QIcon(docker_path)
            docker_button.setIcon(docker_icon)
            docker_button.setIconSize(QSize(50, 50))
        docker_button.setText("Create project with Docker")
        docker_button.setCursor(Qt.PointingHandCursor)
        docker_button.clicked.connect(self.select_docker) 
        button_layout.addWidget(docker_button)

        main_layout.addLayout(button_layout)
        self.setStyleSheet("background-color: white;")

    def select_docker(self):
        global user_choices
        user_choices.append({"simulation_type": "Docker"})
        self.open_presimulation_page()
        self.close()

    def select_local(self):
        global user_choices
        user_choices.append({"simulation_type": "Local"})
        self.open_presimulation_page()
        self.close()

    def open_presimulation_page(self):
        self.presimulation_page = PreSimulationPage(user_choices, self.show_home_page)
        self.presimulation_page.show()

    def show_home_page(self):
        self.hide()  
        self.show()  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomePage()
    window.show()
    sys.exit(app.exec_())