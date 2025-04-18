import os
import json
import random
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QHBoxLayout, QStyle, QGridLayout
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon

class RecapSimulationPage(QWidget):
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        self.setWindowTitle("AP4Fed")
        self.resize(1000, 800)
        # Remove global style that might interfere with QMessageBox
        self.setStyleSheet("background-color: white;")
        back_btn = QPushButton()
        back_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setIconSize(QSize(24, 24))
        back_btn.setFixedSize(36, 36)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-radius: 18px;
            }
        """)
        back_btn.clicked.connect(self.on_back)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # header con back‑button e titolo sulla stessa riga
        title_label = QLabel("List of Input Parameters for the Simulation")
        title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        header_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        header_layout.addWidget(title_label, stretch=1)

        layout.insertLayout(0, header_layout)

        # Scrollable area to display configuration details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)

        # Divide configuration into sections
        self.display_general_parameters(scroll_layout)
        self.display_clients(scroll_layout)
        self.display_patterns(scroll_layout)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter)
        buttons_layout.setSpacing(20)

        # "Run Simulation" Button
        run_button = QPushButton("Run Simulation")
        run_button.clicked.connect(self.run_simulation)
        run_button.setCursor(Qt.PointingHandCursor)
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #70C284; 
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
        buttons_layout.addWidget(run_button)

        # "Download .json Configuration" Button
        download_button = QPushButton("Download .json Configuration")
        download_button.clicked.connect(self.download_configuration)
        download_button.setCursor(Qt.PointingHandCursor)
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
            QPushButton:pressed {
                background-color: #c69500;
            }
        """)

        # Add standard PyQt5 icon
        json_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        download_button.setIcon(json_icon)
        download_button.setIconSize(QSize(24, 24))

        buttons_layout.addWidget(download_button)

        # "Close" Button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        buttons_layout.addWidget(close_button)

        layout.addLayout(buttons_layout)

    def on_back(self):
        self.close()
        self.home_page_callback()

    def display_general_parameters(self, layout):
        """
        Displays general parameters.
        """
        # Merge configurations
        merged_config = {}
        for choice in self.user_choices:
            if isinstance(choice, dict):
                merged_config.update(choice)

        # General Parameters Section
        general_label = QLabel("General Parameters")
        general_label.setAlignment(Qt.AlignLeft)
        general_label.setStyleSheet("color: black; font-size: 20px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(general_label)

        # Display general parameters
        general_params = {}
        for key in ['simulation_type', 'rounds']:
            if key in merged_config:
                general_params[key] = merged_config[key]

        # Clean up parameter names for display
        display_params = {}
        for key, value in general_params.items():
            display_key = key.replace('_', ' ').title()
            display_params[display_key] = value

        self.add_configuration_items(display_params, layout)

    def display_clients(self, layout):
        """
        Displays client information in cards.
        """
        # Merge configurations
        merged_config = {}
        for choice in self.user_choices:
            if isinstance(choice, dict):
                merged_config.update(choice)

        clients = merged_config.get('client_details', [])

        if clients:
            clients_label = QLabel("Clients")
            clients_label.setStyleSheet("color: black; font-size: 20px; font-weight: bold; margin-top: 20px;")
            layout.addWidget(clients_label)

            # Grid layout for client cards
            grid_layout = QGridLayout()
            grid_layout.setSpacing(10)
            max_columns = 6
            row = 0
            col = 0

            for idx, client in enumerate(clients):
                card = self.create_client_card(client)
                grid_layout.addWidget(card, row, col)
                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

            layout.addLayout(grid_layout)

    def create_client_card(self, client_info):
        """
        Creates a card for the client with a computer icon and its information.
        """
        card = QFrame()
        card.setFrameShape(QFrame.Box)
        card.setLineWidth(1)
        card.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")

        card_layout = QVBoxLayout()
        card_layout.setAlignment(Qt.AlignCenter)
        card.setLayout(card_layout)

        # Computer icon using standard icon
        pc_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        pc_icon_label = QLabel()
        pc_icon_label.setPixmap(pc_icon.pixmap(32, 32))
        card_layout.addWidget(pc_icon_label, alignment=Qt.AlignCenter)

        # Client information
        for key, value in client_info.items():
            display_key = key.replace('_', ' ').title()

            # Capitalize "CPU", "RAM", and "ID"
            words = display_key.split()
            words = [word.upper() if word.lower() in ['cpu', 'ram', 'id'] else word for word in words]
            display_key = ' '.join(words)

            info_label = QLabel(f"{display_key}: {value}")
            info_label.setStyleSheet("color: black; font-size: 12px;")
            card_layout.addWidget(info_label, alignment=Qt.AlignCenter)

        return card

    def display_patterns(self, layout):
        """
        Visualizza i pattern raggruppati in 4 categorie.
        Se un pattern è True mostra una spunta verde, se False mostra una X rossa.
        """
        # Carichiamo i dati dal file config.json
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'configuration', 'config.json')
            with open(config_path, 'r') as f:
                merged_config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config.json: {e}")
            return

        # Sezione Patterns
        patterns_label = QLabel("Patterns")
        patterns_label.setStyleSheet("color: black; font-size: 20px; font-weight: bold; margin-top: 20px;")
        layout.addWidget(patterns_label)

        # Otteniamo il dict dei pattern dal config
        all_patterns = merged_config.get('patterns', {})

        # Definiamo la struttura delle 4 categorie, ciascuna con una lista di (chiave_config, label_amichevole)
        categories = [
            ("Client Management", [
                ("client_registry", "Client Registry"),
                ("client_selector", "Client Selector"),
                ("client_cluster", "Client Cluster"),
            ]),
            ("Model Management", [
                ("message_compressor", "Message Compressor"),
                ("model_co-versioning_registry", "Model co-Versioning Registry"),
                ("model_replacement_trigger", "Model Replacement Trigger"),
                ("deployment_selector", "Deployment Selector"),
            ]),
            ("Model Training", [
                ("multi-task_model_trainer", "Multi-Task Model Trainer"),
                ("heterogeneous_data_handler", "Heterogeneous Data Handler"),
                ("incentive_registry", "Incentive Registry"),
            ]),
            ("Model Aggregation", [
                ("asynchronous_aggregator", "Asynchronous Aggregator"),
                ("decentralised_aggregator", "Decentralised Aggregator"),
                ("hierarchical_aggregator", "Hierarchical Aggregator"),
                ("secure_aggregator", "Secure Aggregator"),
            ])
        ]

        # Creiamo una griglia 2x2 per le 4 categorie
        categories_grid = QGridLayout()
        categories_grid.setSpacing(20)
        layout.addLayout(categories_grid)

        row, col = 0, 0
        for category_title, pattern_list in categories:
            # Crea un frame per la singola categoria
            cat_frame = QFrame()
            cat_frame.setFrameShape(QFrame.Box)
            cat_frame.setLineWidth(1)
            cat_frame.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")

            cat_layout = QVBoxLayout()
            cat_layout.setAlignment(Qt.AlignTop)
            cat_frame.setLayout(cat_layout)

            # Titolo della categoria
            cat_label = QLabel(category_title)
            cat_label.setStyleSheet("color: black; font-size: 14px; font-weight: bold; margin-top: 5px;")
            cat_layout.addWidget(cat_label, alignment=Qt.AlignCenter)

            # Elenco pattern
            for pattern_key, pattern_display_name in pattern_list:
                # Crea un layout orizzontale per l'icona e il nome
                pattern_layout = QHBoxLayout()
                pattern_layout.setAlignment(Qt.AlignLeft)

                # Icona verde (check) o rossa (X) PRIMA del nome pattern
                is_enabled = all_patterns.get(pattern_key, {}).get("enabled", False)  # Controllo sullo stato del pattern
                icon_label = QLabel()
                if is_enabled:
                    icon_label.setText("✔")  # Check unicode
                    icon_label.setStyleSheet("color: green; font-size: 14px; margin-right: 5px;")
                else:
                    icon_label.setText("✘")  # X unicode
                    icon_label.setStyleSheet("color: red; font-size: 14px; margin-right: 5px;")

                pattern_layout.addWidget(icon_label)

                # Nome pattern
                p_label = QLabel(pattern_display_name)
                p_label.setStyleSheet("color: black; font-size: 13px;")
                pattern_layout.addWidget(p_label)

                cat_layout.addLayout(pattern_layout)

            # Inseriamo il frame della categoria nella griglia 2x2
            categories_grid.addWidget(cat_frame, row, col)

            col += 1
            if col >= 2:
                col = 0
                row += 1

    def on_back(self):
        self.close()
        self.home_page_callback()

    def add_configuration_items(self, config, layout, indent=0):
        """
        Recursively adds configuration items to the layout.
        """
        for key, value in config.items():
            # Horizontal layout for each item
            item_layout = QHBoxLayout()
            item_layout.setAlignment(Qt.AlignLeft)
            # Indentation
            indent_str = ' ' * indent * 20  # 20 pixels per indentation level

            # Key label
            key_label = QLabel(f"{key}:")
            key_label.setStyleSheet(f"color: black; font-size: 14px; margin-left: {indent_str}px;")
            item_layout.addWidget(key_label)

            if isinstance(value, dict):
                # If the value is a dictionary, add its items recursively
                layout.addLayout(item_layout)
                self.add_configuration_items(value, layout, indent + 1)
            elif isinstance(value, list):
                # If the value is a list, display each item
                layout.addLayout(item_layout)
                for idx, item in enumerate(value, start=1):
                    self.add_configuration_items({f"Item {idx}": item}, layout, indent + 1)
            else:
                # Value label
                value_label = QLabel(str(value))
                value_label.setStyleSheet("color: black; font-size: 14px;")
                item_layout.addWidget(value_label)
                layout.addLayout(item_layout)

    def download_configuration(self):
        """
        Downloads the merged configuration as a JSON file.
        """
        # Merge configurations
        merged_config = {}
        for choice in self.user_choices:
            if isinstance(choice, dict):
                merged_config.update(choice)

        # Open a file dialog to save the file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if file_name:
            # Ensure the file has a .json extension
            if not file_name.endswith('.json'):
                file_name += '.json'
            try:
                with open(file_name, 'w') as f:
                    json.dump(merged_config, f, indent=4)

            except Exception as e:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Error")
                msg_box.setText(f"An error occurred while saving the file:\n{e}")
                
                # Imposta un timer per chiudere il box dopo 2 secondi
                QTimer.singleShot(2000, msg_box.close)
                
                # Mostra il box
                msg_box.exec_()
    
    def run_simulation(self):
        merged_config = {}
        for choice in self.user_choices:
            if isinstance(choice, dict):
                merged_config.update(choice)

        for c in merged_config.get("client_details", []):
            if c.get("data_distribution_type") == "Random":
                c["data_distribution_type"] = "IID" if random.random() < 0.5 else "non-IID"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, 'configuration')

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_file_name = 'config.json'
        config_file_path = os.path.join(config_dir, config_file_name)

        try:
            with open(config_file_path, 'w') as f:
                json.dump(merged_config, f, indent=4)

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText(f"Configuration saved to {config_file_path}")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: black;
                }
            """)

            ok_button = msg_box.addButton("OK", QMessageBox.AcceptRole)
            ok_button.setCursor(Qt.PointingHandCursor)
            ok_button.setStyleSheet("""
                QPushButton {
                    background-color: green;
                    color: white;
                    font-size: 12px;
                    padding: 8px 16px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #00b300;
                }
                QPushButton:pressed {
                    background-color: #008000;
                }
            """)

            #msg_box.exec_()
        except Exception as e:
            return 

        client_details = merged_config.get('client_details', [])
        num_supernodes = len(client_details)

        if num_supernodes == 0:
            QMessageBox.warning(self, "Warning", "No clients defined in the configuration. Using default num_supernodes=2.")
            num_supernodes = 2

        from simulation import SimulationPage
        self.simulation_page = SimulationPage(num_supernodes)
        self.simulation_page.show()
        self.close()