import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QLabel, QPushButton, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QHBoxLayout, QGridLayout,
    QComboBox, QScrollArea, QSizePolicy, QStyle, QMessageBox,
    QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize
from recap_simulation import RecapSimulationPage
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtSvg import QSvgWidget

class PreSimulationPage(QWidget):
    """
    Questa classe rappresenta la pagina di pre-simulazione dove l'utente può configurare
    le impostazioni generali della simulazione, come il numero di round, il numero di client,
    e selezionare i pattern architetturali da implementare.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()

        self.pattern_data = {
            # Client Management Category
            "Client Registry": {
                "category": "Client Management Category",
                "image": "img/patterns/Client-Management/clientregistry.png",  
                "description": "Maintains information about all participating client devices for client management.",
                "benefits": "Centralized tracking of client states; easier organization.",
                "drawbacks": "Requires overhead for maintaining the registry."
            },
            "Client Selector": {
                "category": "Client Management Category",
                "image": "img/patterns/Client-Management/clientselector.svg",
                "description": "Actively selects client devices for a specific training round based on predefined criteria to enhance model performance and system efficiency.",
                "benefits": "Ensures only the most relevant clients train each round, potentially improving performance.",
                "drawbacks": "May exclude important data from non-selected clients."
            },
            "Client Cluster": {
                "category": "Client Management Category",
                "image": "img/patterns/Client-Management/clientcluster.svg",
                "description": "Groups client devices based on their similarity (e.g., resources, data distribution) to improve model performance and training efficiency.",
                "benefits": "Allows specialized training; can handle different groups more effectively.",
                "drawbacks": "Additional overhead to manage cluster membership."
            },

            # Model Management Category
            "Message Compressor": {
                "category": "Model Management Category",
                "image": "img/patterns/Model-Management/message_compressor.png",
                "description": "Compresses and reduces the size of message data before each model exchange round to improve communication efficiency.",
                "benefits": "Reduces bandwidth usage; can speed up communication rounds.",
                "drawbacks": "Compression/decompression overhead might offset gains for large data."
            },
            "Model co-Versioning Registry": {
                "category": "Model Management Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },
            "Model Replacement Trigger": {
                "category": "Model Management Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },
            "Deployment Selector": {
                "category": "Model Management Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },

            # Model Training Category
            "Multi-Task Model Trainer": {
                "category": "Model Training Category",
                "image": "img/patterns/Model-Management/multi_task_model_trainer.svg",
                "description": "Utilizes data from related models on local devices to enhance efficiency.",
                "benefits": "Potential knowledge sharing among similar tasks; improved training.",
                "drawbacks": "Training logic may become more complex to handle multiple tasks."
            },
            "Heterogeneous Data Handler": {
                "category": "Model Training Category",
                "image": "img/patterns/Model-Management/heterogeneous_data_handler.svg",
                "description": "Addresses issues with non-IID and skewed data while preserving data privacy.",
                "benefits": "Better management of varied data distributions.",
                "drawbacks": "Requires more sophisticated data partitioning and handling logic."
            },
            "Incentive Registry": {
                "category": "Model Training Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },

            # Model Aggregation Category
            "Asynchronous Aggregator": {
                "category": "Model Aggregation Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },
            "Decentralised Aggregator": {
                "category": "Model Aggregation Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },
            "Hierarchical Aggregator": {
                "category": "Model Aggregation Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            },
            "Secure Aggregator": {
                "category": "Model Aggregation Category",
                "image": "TODO",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "TODO",
                "drawbacks": "TODO"
            }
        }

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: black;  /* Testo nero */
            }
            QLabel {
                color: black;  /* Testo delle etichette in nero */
            }
            QPushButton {
                background-color: green;  /* Colore dei bottoni */
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        self.setWindowTitle("Pre-Simulation Configuration")
        self.resize(800, 600)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Display the chosen configuration
        choice_label = QLabel(f"Type of Simulation: {self.user_choices[-1]['simulation_type']}")
        choice_label.setAlignment(Qt.AlignCenter)
        choice_label.setStyleSheet("font-size: 16px; color: #333;")
        layout.addWidget(choice_label)

        # General Settings GroupBox
        general_settings_group = QGroupBox("General Settings")
        general_settings_group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: black;  /* Colore del titolo in nero */
                font-size: 14px;
                font-weight: bold;
            }
        """)
        general_settings_layout = QFormLayout()

        # Input per il numero di round
        self.rounds_input = QSpinBox()
        self.rounds_input.setMinimum(1)
        self.rounds_input.setMaximum(100)
        self.rounds_input.setValue(3)
        general_settings_layout.addRow("Number of Rounds:", self.rounds_input)

        # Input per il numero di client
        self.clients_input = QSpinBox()
        self.clients_input.setMinimum(1)  # Minimo 1 client
        self.clients_input.setMaximum(100)
        self.clients_input.setValue(3)
        general_settings_layout.addRow("Number of Clients:", self.clients_input)

        # Verifica lo stato di Docker se la simulazione è Docker
        if self.user_choices[-1]["simulation_type"] == "Docker":
            # Docker status layout
            docker_status_layout = QHBoxLayout()
            docker_status_layout.setAlignment(Qt.AlignLeft)

            # Label "Docker Status:"
            docker_status_text_label = QLabel("Docker Status:")
            docker_status_text_label.setStyleSheet("font-size: 12px;")
            docker_status_text_label.setAlignment(Qt.AlignLeft)

            # Status label (Active or Not Active)
            self.docker_status_label = QLabel()
            self.docker_status_label.setStyleSheet("font-size: 12px;")
            self.docker_status_label.setAlignment(Qt.AlignLeft)

            # Update button with spinning wheel icon
            update_button = QPushButton()
            update_button.setCursor(Qt.PointingHandCursor)
            update_button.setToolTip("Update Status")
            update_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
            update_button.setIcon(update_icon)
            update_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)
            update_button.clicked.connect(self.update_docker_status)

            # Add widgets to the layout
            docker_status_layout.addWidget(docker_status_text_label)
            docker_status_layout.addWidget(self.docker_status_label)
            docker_status_layout.addWidget(update_button)
            docker_status_layout.addStretch()
            general_settings_layout.addRow(docker_status_layout)

            # Verifica iniziale dello stato di Docker
            self.check_docker_status()

        general_settings_group.setLayout(general_settings_layout)
        layout.addWidget(general_settings_group)

        # Selezione dei Pattern Architetturali
        patterns_label = QLabel("Select Architectural Patterns to Implement in the Simulation:")
        patterns_label.setAlignment(Qt.AlignLeft)
        patterns_label.setStyleSheet("font-size: 14px; color: #333; margin-top: 10px;")
        layout.addWidget(patterns_label)

        # Aggiungi i checkbox dei pattern divisi per macrocategorie in 2 colonne
        patterns_grid = QGridLayout()
        patterns_grid.setSpacing(10)  # Spaziatura tra gli elementi
        self.pattern_checkboxes = {}
        macrotopics = [
            ("Client Management Category", [
                "Client Registry: Maintains information about all participating client devices for client management.",
                "Client Selector: Actively selects client devices for a specific training round based on predefined criteria to enhance model performance and system efficiency.",
                "Client Cluster: Groups client devices based on their similarity in certain characteristics (e.g., resources, data distribution) to improve model performance and training efficiency."
            ]),
            ("Model Management Category", [
                "Message Compressor: Compresses and reduces the size of message data before each model exchange round to improve communication efficiency.",
                "Model co-Versioning Registry: Stores and aligns local models with the global model versions for tracking purposes.",
                "Model Replacement Trigger: Triggers model replacement when performance degradation is detected.",
                "Deployment Selector: Matches converging global models with suitable clients for task optimization."
            ]),
            ("Model Training Category", [
                "Multi-Task Model Trainer: Utilizes data from related models on local devices to enhance efficiency.",
                "Heterogeneous Data Handler: Addresses issues with non-IID and skewed data while maintaining data privacy.",
                "Incentive Registry: Measures and records client contributions and provides incentives."
            ]),
            ("Model Aggregation Category", [
                "Asynchronous Aggregator: Aggregates asynchronously to reduce latency.",
                "Decentralised Aggregator: Removes the central server to prevent single-point failures.",
                "Hierarchical Aggregator: Adds an edge layer for partial aggregation to improve efficiency.",
                "Secure Aggregator: Ensures security during aggregation."
            ])
        ]

        row, col = 0, 0
        for topic, patterns in macrotopics:
            topic_group = QGroupBox(topic)
            topic_group.setStyleSheet("""
                QGroupBox {
                    background-color: white;
                    border: 1px solid lightgray;
                    border-radius: 5px;
                    margin-top: 5px;
                }
                QGroupBox:title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 5px;
                    color: black;  /* Colore del titolo in nero */
                    font-size: 13px;  /* Ridotto a 13px */
                    font-weight: bold;
                }
            """)
            topic_group.setMinimumWidth(350)  # Larghezza minima per evitare che il testo vada a capo
            topic_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            topic_layout = QVBoxLayout()
            topic_layout.setSpacing(5)

            for pattern in patterns:
                # pattern contiene stringhe tipo "Client Registry: descrizione..."
                pattern_name = pattern.split(":")[0].strip()
                pattern_desc = pattern.split(":")[1].strip()  # La descrizione dopo i due punti

                # Layout orizzontale per (pulsante info + checkbox)
                pattern_layout = QHBoxLayout()
                pattern_layout.setSpacing(6)

                # Pulsante con icona info
                info_button = QPushButton()
                info_button.setCursor(Qt.PointingHandCursor)
                info_button.setToolTip(f"More info about {pattern_name}")
                info_icon = self.style().standardIcon(QStyle.SP_MessageBoxInformation)
                info_button.setCursor(Qt.PointingHandCursor)
                info_button.setIcon(info_icon)
                
                info_button.setFixedSize(24, 24)
                info_button.setIconSize(QSize(16, 16))
                info_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        padding: 0px;
                        margin: 0px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                """)

                # Funzione wrapper per recuperare i dati dal dizionario (self.pattern_data)
                def info_clicked(checked, pname=pattern_name):
                    # Se pattern_name esiste in self.pattern_data, mostriamo i dati custom
                    if pname in self.pattern_data:
                        data = self.pattern_data[pname]
                        cat_ = data["category"]
                        img_ = data["image"]
                        desc_ = data["description"]
                        ben_ = data["benefits"]
                        dr_  = data["drawbacks"]
                        self.show_pattern_info(pname, cat_, img_, desc_, ben_, dr_)
                    else:
                        # fallback (se non c'è)
                        self.show_pattern_info(
                            pname,
                            topic,
                            "img/fittizio.png",
                            pattern_desc,
                            "No custom benefits",
                            "No custom drawbacks"
                        )

                info_button.clicked.connect(info_clicked)

                # Creazione checkbox
                checkbox = QCheckBox(pattern_name)
                checkbox.setToolTip(pattern_desc)
                checkbox.setStyleSheet("""
                    QCheckBox {
                        color: black;
                        font-size: 12px;
                    }
                    QCheckBox:disabled {
                        color: black;
                    }
                """)

                # Abilita/disabilita pattern
                enabled_patterns = [
                    "Client Registry",
                    "Client Selector",
                    "Client Cluster",
                    "Message Compressor",
                    "Multi-Task Model Trainer",
                    "Heterogeneous Data Handler",
                ]
                if pattern_name not in enabled_patterns:
                    checkbox.setEnabled(False)

                # Gestione speciale di "Client Registry (Active by default)"
                if pattern_name == "Client Registry":
                    checkbox.setText("Client Registry (Active by Default)")
                    checkbox.setChecked(True)
                    def prevent_uncheck(state):
                        if state != Qt.Checked:
                            checkbox.blockSignals(True)
                            checkbox.setChecked(True)
                            checkbox.blockSignals(False)
                    checkbox.stateChanged.connect(prevent_uncheck)

                pattern_layout.addWidget(info_button)
                pattern_layout.addWidget(checkbox)
                topic_layout.addLayout(pattern_layout)

                # Salva riferimento al checkbox
                self.pattern_checkboxes[pattern_name] = checkbox

            topic_group.setLayout(topic_layout)
            patterns_grid.addWidget(topic_group, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        layout.addLayout(patterns_grid)

        # Bottone per Salvare e Continuare
        save_button = QPushButton("Save and Continue")
        save_button.setCursor(Qt.PointingHandCursor)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        save_button.clicked.connect(self.save_preferences_and_open_client_config)
        layout.addWidget(save_button)

    def check_docker_status(self):
        """
        Metodo per verificare lo stato di Docker.
        """
        try:
            subprocess.check_output(['docker', 'info'], stderr=subprocess.STDOUT)
            # Docker è attivo
            self.docker_status_label.setText("Active")
            self.docker_status_label.setStyleSheet("color: green; font-size: 12px;")
        except subprocess.CalledProcessError as e:
            # Docker non è attivo o non installato
            self.docker_status_label.setText("Not Active")
            self.docker_status_label.setStyleSheet("color: red; font-size: 12px;")
        except FileNotFoundError:
            # Comando docker non trovato
            self.docker_status_label.setText("Not Installed")
            self.docker_status_label.setStyleSheet("color: red; font-size: 12px;")

    def show_pattern_info(self, pattern_name, pattern_category, image_path, description, benefits, drawbacks):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{pattern_name} - {pattern_category}")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        layout.setAlignment(Qt.AlignTop)

        title_label = QLabel(f"{pattern_name} ({pattern_category})")
        title_label.setStyleSheet("color: black; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        # DEBUG: stampa il percorso e il risultato di os.path.exists
        print(f"[DEBUG] Trying to load image at: {image_path}")
        print(f"[DEBUG] os.path.exists(image_path) = {os.path.exists(os.path.join(base_dir, image_path))}")

        image_label = QLabel()   
        
        if os.path.exists(os.path.join(base_dir, image_path)):            
            image_path = os.path.join(base_dir, image_path)
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
        else:
            # Il file non esiste
            image_label.setText("Image not found")
            image_label.setStyleSheet("color: red;")
            image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(image_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: black; font-size: 13px; margin-top: 5px;")
        layout.addWidget(desc_label)

        benefits_label = QLabel(f"Benefits: {benefits}")
        benefits_label.setWordWrap(True)
        benefits_label.setStyleSheet("color: green; font-size: 12px; margin-top: 10px;")
        layout.addWidget(benefits_label)

        drawbacks_label = QLabel(f"Drawbacks: {drawbacks}")
        drawbacks_label.setWordWrap(True)
        drawbacks_label.setStyleSheet("color: red; font-size: 12px; margin-top: 5px;")
        layout.addWidget(drawbacks_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        button_box.setCursor(Qt.PointingHandCursor)

        button_box.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                border-radius: 5px;
                margin: 5px;   /* "Margine" esterno al pulsante */
                padding: 8px;  /* Spazio interno al pulsante */
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)

        layout.addWidget(button_box, alignment=Qt.AlignCenter)
        dialog.exec_()

    def update_docker_status(self):
        """
        Metodo chiamato quando si clicca il pulsante per aggiornare lo stato di Docker.
        """
        self.check_docker_status()

    def save_preferences_and_open_client_config(self):
        """
        Salva le preferenze e apre la pagina di configurazione dei client.
        """
        # Salva le configurazioni dei pattern selezionati
        patterns_data = {}
        for key in ["Client Registry", "Client Selector", "Client Cluster",
                    "Message Compressor", "Multi-Task Model Trainer", "Heterogeneous Data Handler"]:
            patterns_data[key.lower().replace(" ", "_")] = (
                self.pattern_checkboxes[key].isChecked() if key in self.pattern_checkboxes else False
            )

        simulation_config = {
            "rounds": self.rounds_input.value(),
            "clients": self.clients_input.value(),
            "patterns": patterns_data,
        }

        # Rimuovi la sezione Docker Configuration, dato che è stata eliminata

        self.user_choices.append(simulation_config)

        # Passa alla pagina di configurazione dei client
        self.client_config_page = ClientConfigurationPage(self.user_choices, self.home_page_callback)
        self.client_config_page.show()

        # Chiudi la finestra attuale
        self.close()

class ClientConfigurationPage(QWidget):
    """
    Questa classe rappresenta la pagina di configurazione dei client,
    con card verticali (una sotto l'altra), larghezza fissa
    e centratura orizzontale.
    Mantiene lo stile e le funzionalità precedenti.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.setWindowTitle("Client Configuration")
        self.resize(800, 600)
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        # Stile ripreso dal tuo codice precedente + card style
        self.setStyleSheet("""
            QWidget {
                background-color: white;  /* Sfondo bianco */
                color: black;
            }
            QLabel {
                color: black;
                background-color: transparent;  /* Rimuove lo sfondo bianco dai label */
            }
            QSpinBox, QComboBox {
                background-color: white;
                border: 1px solid gray;
                border-radius: 3px;
                height: 20px;  /* Riduce l'altezza dei widget */
                font-size: 12px;
            }
            QPushButton {
                height: 30px;  /* Riduce l'altezza del pulsante */
                background-color: green;
                color: white;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
            /* Stile per i QFrame che fungono da 'card' dei client */
            QFrame#ClientCard {
                background-color: #f0f0f0; /* come il tuo QGroupBox di prima */
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
        """)

        # Layout verticale principale
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # Titolo
        title_label = QLabel("Configure Each Client")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        main_layout.addWidget(title_label)

        # Area di scorrimento per gestire molti client
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("background-color: transparent;")
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(5)
        scroll_area.setWidget(scroll_widget)

        self.client_configs = []
        num_clients = self.user_choices[-1]["clients"]

        # Creazione di una "card" verticale per ogni client (con larghezza fissa)
        for client_id in range(1, num_clients + 1):
            card_widget, config_dict = self.create_client_card(client_id)

            # Usa un piccolo layout orizzontale per centrare la card
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(card_widget)
            scroll_layout.addLayout(hbox)

            self.client_configs.append(config_dict)

        # Pulsante di conferma in basso
        confirm_button = QPushButton("Confirm and Continue")
        confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #70C284;
                color: white;
                border-radius: 5px;
                margin: 5px;   /* "Margine" esterno al pulsante */
                padding: 8px;  /* Spazio interno al pulsante (simile a CSS 'padding') */
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.clicked.connect(self.save_client_configurations_and_continue)
        main_layout.addWidget(confirm_button, alignment=Qt.AlignCenter)

    def create_client_card(self, client_id):
        """
        Crea una card (QFrame) per un singolo client
        restituendo (card_widget, dict_config).
        """
        card = QFrame(objectName="ClientCard")  # lo stile #ClientCard verrà applicato
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(5)
        card.setLayout(card_layout)
        card.setStyleSheet("""
                QMessageBox {
                    background-color: lightgray;
                }
            """)

        # Larghezza fissa delle card
        card.setFixedWidth(600)

        # Icona del computer come in Recap
        pc_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        icon_label = QLabel()
        icon_label.setPixmap(pc_icon.pixmap(24, 24))
        icon_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(icon_label)

        # Titolo "Client X"
        client_title = QLabel(f"Client {client_id}")
        client_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        client_title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(client_title)

        # Layout per i campi di input
        # Riga 1: CPU, RAM
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(5)

        cpu_label = QLabel("CPU Allocation:")
        cpu_label.setStyleSheet("font-size: 12px; background:white")
        cpu_label.setAlignment(Qt.AlignLeft)
        cpu_input = QSpinBox()
        cpu_input.setRange(1, 16)
        cpu_input.setValue(1)
        cpu_input.setSuffix(" CPUs")
        cpu_input.setFixedWidth(80)

        ram_label = QLabel("RAM Allocation:")
        ram_label.setStyleSheet("font-size: 12px; background:white")
        ram_label.setAlignment(Qt.AlignLeft)
        ram_input = QSpinBox()
        ram_input.setRange(1, 128)
        ram_input.setValue(2)
        ram_input.setSuffix(" GB")
        ram_input.setFixedWidth(80)

        row1_layout.addWidget(cpu_label)
        row1_layout.addWidget(cpu_input)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(ram_label)
        row1_layout.addWidget(ram_input)
        card_layout.addLayout(row1_layout)

        # Riga 2: Dataset, Partition
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(5)

        dataset_label = QLabel("Testing Dataset:")
        dataset_label.setStyleSheet("font-size: 12px;; background:white")
        dataset_label.setAlignment(Qt.AlignLeft)
        dataset_combobox = QComboBox()
        dataset_combobox.addItems(["CIFAR-10", "FMNIST", "MIXED"])
        dataset_combobox.setFixedWidth(100)

        partition_label = QLabel("Dataset Partition:")
        partition_label.setStyleSheet("font-size: 12px; ; background:white")
        partition_label.setAlignment(Qt.AlignLeft)
        partition_combobox = QComboBox()
        partition_combobox.addItems(["IID", "non-IID", "Random"])
        partition_combobox.setFixedWidth(100)

        row2_layout.addWidget(dataset_label)
        row2_layout.addWidget(dataset_combobox)
        row2_layout.addSpacing(10)
        row2_layout.addWidget(partition_label)
        row2_layout.addWidget(partition_combobox)
        card_layout.addLayout(row2_layout)

        # Riferimenti ai campi
        config_dict = {
            "cpu_input": cpu_input,
            "ram_input": ram_input,
            "dataset_combobox": dataset_combobox,
            "data_distribution_type_combobox": partition_combobox
        }

        return card, config_dict

    def save_client_configurations_and_continue(self):
        """
        Salva le configurazioni dei client e passa alla pagina di riepilogo.
        """
        client_details = []
        for idx, config in enumerate(self.client_configs):
            client_info = {
                "client_id": idx + 1,
                "cpu": config["cpu_input"].value(),
                "ram": config["ram_input"].value(),
                "dataset": config["dataset_combobox"].currentText(),
                "data_distribution_type": config["data_distribution_type_combobox"].currentText()
            }
            client_details.append(client_info)

        self.user_choices[-1]["client_details"] = client_details

        # Procede alla pagina di riepilogo (RecapSimulationPage)
        self.recap_simulation_page = RecapSimulationPage(self.user_choices)
        self.recap_simulation_page.show()
        self.close()