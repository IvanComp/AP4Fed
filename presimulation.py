import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QLabel, QPushButton, QSpinBox,
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
                "Multi-Task Model Trainer: Utilizza dati da modelli correlati su dispositivi locali per migliorare l'efficienza.",
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
                    "Multi-Task Model Trainer",
                    "Heterogeneous Data Handler",
                ]

                if pattern_name not in enabled_patterns:
                    checkbox.setEnabled(False)

                if pattern_name == "Client Registry":
                    # Modifica dell'etichetta per includere "(Active by default)"
                    checkbox.setText("Client Registry (Active by Default)")
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