import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QHBoxLayout, QGridLayout,
    QComboBox, QScrollArea, QSizePolicy, QStyle
)
from PyQt5.QtCore import Qt
from recap_simulation import RecapSimulationPage

class PreSimulationPage(QWidget):
    """
    Questa classe rappresenta la pagina di pre-simulazione dove l'utente può configurare
    le impostazioni generali della simulazione, come il numero di round, il numero di client,
    e selezionare i pattern architetturali da implementare.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
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
        self.rounds_input.setValue(10)
        general_settings_layout.addRow("Number of Rounds:", self.rounds_input)

        # Input per il numero di client
        self.clients_input = QSpinBox()
        self.clients_input.setMinimum(1)  # Minimo 1 client
        self.clients_input.setMaximum(100)
        self.clients_input.setValue(2)
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
                "Client Registry: Mantiene le informazioni di tutti i dispositivi client partecipanti per la gestione dei client.",
                "Client Selector: Seleziona attivamente i dispositivi client per un certo round di training secondo criteri predefiniti per aumentare le prestazioni del modello e l'efficienza del sistema.",
                "Client Cluster: Raggruppa i dispositivi client in base alla loro somiglianza di determinate caratteristiche (es. risorse, distribuzione dei dati) per aumentare le prestazioni del modello e l'efficienza del training."
            ]),
            ("Model Management Category", [
                "Message Compressor: Comprime e riduce la dimensione dei dati dei messaggi prima di ogni round di scambio del modello per aumentare l'efficienza della comunicazione.",
                "Model co-Versioning Registry: Memorizza e allinea i modelli locali con le versioni del modello globale per il tracciamento.",
                "Model Replacement Trigger: Attiva la sostituzione del modello quando viene rilevato un degrado delle prestazioni del modello.",
                "Deployment Selector: Abbina modelli globali convergenti a client idonei per l'ottimizzazione dei task."
            ]),
            ("Model Training Category", [
                "Multi-task Model Trainer: Utilizza dati da modelli correlati su dispositivi locali per migliorare l'efficienza.",
                "Heterogeneous Data Handler: Risolve problemi di dati non-IID e distorti mantenendo la privacy dei dati.",
                "Incentive Registry: Misura e registra i contributi dei client e fornisce incentivi."
            ]),
            ("Model Aggregation Category", [
                "Asynchronous Aggregator: Aggrega in modo asincrono per ridurre la latenza.",
                "Decentralised Aggregator: Rimuove il server centrale per prevenire guasti del punto singolo.",
                "Hierarchical Aggregator: Aggiunge un layer edge per l'aggregazione parziale per migliorare l'efficienza.",
                "Secure Aggregator: Garantisce la sicurezza durante l'aggregazione."
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
                pattern_name = pattern.split(":")[0]
                checkbox_label = pattern_name
                checkbox_tooltip = pattern.split(":")[1].strip()
                checkbox = QCheckBox(checkbox_label)
                checkbox.setToolTip(checkbox_tooltip)
                checkbox.setStyleSheet("""
                    QCheckBox {
                        color: black;
                        font-size: 12px;
                    }
                    QCheckBox:disabled {
                        color: black;
                    }
                """)
                # Patterns implementati
                enabled_patterns = [
                    "Client Registry",
                    "Client Selector",
                    "Client Cluster",
                    "Message Compressor",
                    "Multi-task Model Trainer"
                ]

                if pattern_name not in enabled_patterns:
                    checkbox.setEnabled(False)

                if pattern_name == "Client Registry":
                    # Modifica dell'etichetta per includere "(Active by default)"
                    checkbox.setText("Client Registry (Active by default)")
                    checkbox.setChecked(True)
                    # Non disabilitiamo il checkbox, ma impediamo che possa essere deselezionato
                    def prevent_uncheck(state):
                        if state != Qt.Checked:
                            checkbox.blockSignals(True)
                            checkbox.setChecked(True)
                            checkbox.blockSignals(False)
                    checkbox.stateChanged.connect(prevent_uncheck)
                    checkbox.setStyleSheet("""
                        QCheckBox {
                            color: black;
                            font-size: 12px;
                        }
                    """)

                self.pattern_checkboxes[pattern_name] = checkbox
                topic_layout.addWidget(checkbox)
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
                    "Message Compressor", "Multi-task Model Trainer"]:
            patterns_data[key.lower().replace(" ", "_")] = (
                self.pattern_checkboxes[key].isChecked() if key in self.pattern_checkboxes else False
            )

        simulation_config = {
            "rounds": self.rounds_input.value(),
            "clients": self.clients_input.value(),
            "patterns": patterns_data
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
    Questa classe rappresenta la pagina di configurazione dei client, dove l'utente può
    specificare CPU, RAM e dataset per ciascun client.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.setWindowTitle("Client Configuration")
        self.resize(800, 600)
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        self.setStyleSheet("""
            QWidget {
                background-color: white;  /* Sfondo bianco */
                color: black;
            }
            QLabel {
                color: black;
                background-color: transparent;  /* Rimuove lo sfondo bianco dai label */
            }
            QGroupBox {
                background-color: #f0f0f0;  /* Colore di sfondo dei box dei client */
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: black;
                font-size: 12px;
                font-weight: bold;
            }
            QSpinBox, QComboBox {
                background-color: white;
                border: 1px solid gray;
                border-radius: 3px;
                height: 20px;  /* Riduce l'altezza dei widget */
            }
            QPushButton {
                height: 30px;  /* Riduce l'altezza del pulsante */
            }
        """)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        self.setLayout(layout)

        title_label = QLabel("Configure Each Client")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        # Area di scorrimento per gestire molti client
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("background-color: transparent;")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(5)
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.client_configs = []

        num_clients = self.user_choices[-1]["clients"]

        for client_id in range(1, num_clients + 1):
            client_group = QGroupBox(f"Client {client_id}")
            client_group.setStyleSheet("""
                QGroupBox {
                    background-color: #f0f0f0;
                    border: 1px solid lightgray;
                    border-radius: 5px;
                    margin-top: 5px;
                    margin-bottom: 5px;
                }
                QGroupBox:title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 5px;
                    color: black;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
            client_layout = QGridLayout()
            client_layout.setContentsMargins(5, 5, 5, 5)
            client_layout.setHorizontalSpacing(10)
            client_layout.setVerticalSpacing(5)

            # CPU Input
            cpu_label = QLabel("CPU Allocation:")
            cpu_label.setStyleSheet("font-size: 12px;")
            cpu_input = QSpinBox()
            cpu_input.setMinimum(1)
            cpu_input.setMaximum(16)
            cpu_input.setValue(1)
            cpu_input.setSuffix(" CPUs")
            cpu_input.setFixedWidth(80)
            cpu_input.setStyleSheet("font-size: 12px;")

            # RAM Input
            ram_label = QLabel("RAM Allocation:")
            ram_label.setStyleSheet("font-size: 12px;")
            ram_input = QSpinBox()
            ram_input.setMinimum(1)
            ram_input.setMaximum(128)
            ram_input.setValue(2)
            ram_input.setSuffix(" GB")
            ram_input.setFixedWidth(80)
            ram_input.setStyleSheet("font-size: 12px;")

            # Dataset Selection
            dataset_label = QLabel("Testing Dataset:")
            dataset_label.setStyleSheet("font-size: 12px;")
            dataset_combobox = QComboBox()
            dataset_combobox.addItems(["CIFAR-10", "FMNIST"])
            dataset_combobox.setFixedWidth(120)
            dataset_combobox.setStyleSheet("font-size: 12px; background-color:white")

            # Aggiungi gli elementi al layout del client
            client_layout.addWidget(cpu_label, 0, 0, alignment=Qt.AlignLeft)
            client_layout.addWidget(cpu_input, 0, 1, alignment=Qt.AlignLeft)
            client_layout.addWidget(ram_label, 0, 2, alignment=Qt.AlignLeft)
            client_layout.addWidget(ram_input, 0, 3, alignment=Qt.AlignLeft)
            client_layout.addWidget(dataset_label, 0, 4, alignment=Qt.AlignLeft)
            client_layout.addWidget(dataset_combobox, 0, 5, alignment=Qt.AlignLeft)

            client_group.setLayout(client_layout)
            scroll_layout.addWidget(client_group)

            # Salva gli input per un successivo recupero
            self.client_configs.append({
                "cpu_input": cpu_input,
                "ram_input": ram_input,
                "dataset_combobox": dataset_combobox
            })

        # Bottone per Confermare e Continuare
        confirm_button = QPushButton("Confirm and Continue")
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 12px;
                padding: 5px 10px;
                border-radius: 5px;
                height: 30px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        confirm_button.clicked.connect(self.save_client_configurations_and_continue)
        layout.addWidget(confirm_button, alignment=Qt.AlignCenter)

    def save_client_configurations_and_continue(self):
        """
        Salva le configurazioni dei client e procede alla pagina di riepilogo.
        """
        client_details = []
        for idx, config in enumerate(self.client_configs):
            client_info = {
                "client_id": idx + 1,
                "cpu": config["cpu_input"].value(),
                "ram": config["ram_input"].value(),
                "dataset": config["dataset_combobox"].currentText()
            }
            client_details.append(client_info)

        self.user_choices[-1]["client_details"] = client_details

        # Procede alla pagina di riepilogo
        self.recap_simulation_page = RecapSimulationPage(self.user_choices)
        self.recap_simulation_page.show()

        # Chiude la finestra attuale
        self.close()