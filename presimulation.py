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
from PyQt5.QtGui import QPixmap, QIcon

# ------------------------------------------------------------------------------------------
# Dialog specializzato per configurare i parametri di "Client Selector"
# ------------------------------------------------------------------------------------------
class ClientSelectorDialog(QDialog):
    def __init__(self, existing_params=None):
        super().__init__()
        self.setWindowTitle("Configure Client Selector")
        self.resize(400, 200)

        self.existing_params = existing_params or {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Selection Strategy
        self.strategy_label = QLabel("Selection Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Resource-Based", "Data-based", "Performance-based"])
        layout.addWidget(self.strategy_label)
        layout.addWidget(self.strategy_combo)

        # Selection Criteria
        self.criteria_label = QLabel("Selection Criteria:")
        self.criteria_combo = QComboBox()
        layout.addWidget(self.criteria_label)
        layout.addWidget(self.criteria_combo)

        # Collegamento per aggiornare le opzioni in base alla strategy
        self.strategy_combo.currentIndexChanged.connect(self.update_criteria_options)

        # Se esistevano parametri precedenti, li ricarichiamo
        if "selection_strategy" in self.existing_params:
            self.strategy_combo.setCurrentText(self.existing_params["selection_strategy"])
        self.update_criteria_options()  # Popoliamo le scelte

        if "selection_criteria" in self.existing_params:
            self.criteria_combo.setCurrentText(self.existing_params["selection_criteria"])

        # Bottoni Ok/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def update_criteria_options(self):
        """
        Aggiorna le opzioni di selection_criteria in base alla selection_strategy.
        """
        strategy = self.strategy_combo.currentText()
        self.criteria_combo.clear()

        if strategy == "Resource-Based":
            self.criteria_combo.addItems(["CPU", "RAM"])
        elif strategy == "Data-based":
            self.criteria_combo.addItems(["IID", "non-IID"])
        elif strategy == "Performance-based":
            self.criteria_combo.addItems(["CPU", "RAM"])

    def get_params(self):
        """
        Restituisce il dizionario con i parametri scelti.
        """
        return {
            "selection_strategy": self.strategy_combo.currentText(),
            "selection_criteria": self.criteria_combo.currentText()
        }

# ------------------------------------------------------------------------------------------
# Dialog generico per configurare due parametri di esempio per tutti gli altri pattern
# ------------------------------------------------------------------------------------------
class GenericPatternDialog(QDialog):
    def __init__(self, pattern_name, existing_params=None):
        super().__init__()
        self.setWindowTitle(f"Configure {pattern_name}")
        self.resize(400, 200)

        self.pattern_name = pattern_name
        self.existing_params = existing_params or {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Esempio: variable1 come SpinBox, variable2 come ComboBox
        self.var1_label = QLabel("Variable1:")
        self.var1_input = QSpinBox()
        self.var1_input.setRange(0, 999)
        self.var1_input.setValue(self.existing_params.get("variable1", 0))

        self.var2_label = QLabel("Variable2:")
        self.var2_input = QComboBox()
        self.var2_input.addItems(["OptionA", "OptionB", "OptionC"])
        if "variable2" in self.existing_params:
            self.var2_input.setCurrentText(self.existing_params["variable2"])

        layout.addWidget(self.var1_label)
        layout.addWidget(self.var1_input)
        layout.addWidget(self.var2_label)
        layout.addWidget(self.var2_input)

        # Bottoni Ok/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_params(self):
        return {
            "variable1": self.var1_input.value(),
            "variable2": self.var2_input.currentText()
        }

# ------------------------------------------------------------------------------------------
# Classe principale PreSimulationPage
# ------------------------------------------------------------------------------------------
class PreSimulationPage(QWidget):
    """
    Questa classe rappresenta la pagina di pre-simulazione dove l'utente può configurare
    le impostazioni generali della simulazione, come il numero di round, il numero di client,
    e selezionare (e configurare) i pattern architetturali da implementare.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()

        # Metadati su pattern (descrizione, immagine ecc.)
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
                "image": "img/patterns/Client-Management/clientselector.png",
                "description": "Actively selects client devices for a specific training round based on predefined criteria to enhance model performance and system efficiency.",
                "benefits": "Ensures only the most relevant clients train each round, potentially improving performance.",
                "drawbacks": "May exclude important data from non-selected clients."
            },
            "Client Cluster": {
                "category": "Client Management Category",
                "image": "img/patterns/Client-Management/clientcluster.png",
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

        super().__init__()
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        # Dizionario dove memorizziamo param per ogni pattern
        # Esempio: self.temp_pattern_config["Client Selector"] = {"enabled": True, "params": {...}}
        self.temp_pattern_config = {}

        self.setWindowTitle("Pre-Simulation Configuration")
        self.resize(800, 600)

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: black;
            }
            QLabel {
                color: black;
            }
            QPushButton {
                background-color: green;
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

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        self.setLayout(main_layout)

        choice_label = QLabel(f"Type of Simulation: {self.user_choices[-1]['simulation_type']}")
        choice_label.setAlignment(Qt.AlignCenter)
        choice_label.setStyleSheet("font-size: 16px; color: #333;")
        main_layout.addWidget(choice_label)

        # General Settings
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
                color: black;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        g_layout = QFormLayout()

        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 100)
        self.rounds_input.setValue(3)
        g_layout.addRow("Number of Rounds:", self.rounds_input)

        self.clients_input = QSpinBox()
        self.clients_input.setRange(1, 100)
        self.clients_input.setValue(3)
        g_layout.addRow("Number of Clients:", self.clients_input)

        if self.user_choices[-1]["simulation_type"] == "Docker":
            docker_status_layout = QHBoxLayout()
            docker_status_layout.setAlignment(Qt.AlignLeft)

            docker_status_label1 = QLabel("Docker Status:")
            docker_status_label1.setStyleSheet("font-size: 12px;")
            docker_status_label1.setAlignment(Qt.AlignLeft)

            self.docker_status_label = QLabel()
            self.docker_status_label.setStyleSheet("font-size: 12px;")
            self.docker_status_label.setAlignment(Qt.AlignLeft)

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

            docker_status_layout.addWidget(docker_status_label1)
            docker_status_layout.addWidget(self.docker_status_label)
            docker_status_layout.addWidget(update_button)
            docker_status_layout.addStretch()
            g_layout.addRow(docker_status_layout)

            self.check_docker_status()

        general_settings_group.setLayout(g_layout)
        main_layout.addWidget(general_settings_group)

        # Label "Select AP"
        patterns_label = QLabel("Select Architectural Patterns to Implement in the Simulation:")
        patterns_label.setAlignment(Qt.AlignLeft)
        patterns_label.setStyleSheet("font-size: 14px; color: #333; margin-top: 10px;")
        main_layout.addWidget(patterns_label)

        patterns_grid = QGridLayout()
        patterns_grid.setSpacing(10)
        self.pattern_checkboxes = {}

        # Macrocategorie
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
        for topic, patterns_list in macrotopics:
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
                    color: black;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            topic_layout = QVBoxLayout()
            topic_layout.setSpacing(5)

            for pattern_entry in patterns_list:
                pattern_name = pattern_entry.split(":")[0].strip()
                pattern_desc = pattern_entry.split(":")[1].strip()

                hl = QHBoxLayout()
                hl.setSpacing(6)

                info_button = QPushButton()
                info_button.setCursor(Qt.PointingHandCursor)
                info_icon = self.style().standardIcon(QStyle.SP_MessageBoxInformation)
                info_button.setIcon(info_icon)
                info_button.setFixedSize(24, 24)
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

                def info_clicked(checked, p=pattern_name):
                    if p in self.pattern_data:
                        data = self.pattern_data[p]
                        cat_ = data["category"]
                        img_ = data["image"]
                        desc_ = data["description"]
                        ben_ = data["benefits"]
                        dr_  = data["drawbacks"]
                        self.show_pattern_info(p, cat_, img_, desc_, ben_, dr_)
                    else:
                        self.show_pattern_info(p, topic, "img/fittizio.png", pattern_desc,
                                               "No custom benefits", "No custom drawbacks")

                info_button.clicked.connect(info_clicked)

                checkbox = QCheckBox(pattern_name)
                checkbox.setToolTip(pattern_desc)
                checkbox.setStyleSheet("QCheckBox { color: black; font-size: 12px; }")

                # Abilita/disabilita pattern
                # Se vuoi tutti cliccabili, commenta la riga sotto.
                # (Attuale: solo alcuni pattern abilitati)
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

                # Gestione "Client Registry (Active by default)"
                if pattern_name == "Client Registry":
                    checkbox.setText("Client Registry (Active by Default)")
                    checkbox.setChecked(True)
                    def prevent_uncheck(state):
                        if state != Qt.Checked:
                            checkbox.blockSignals(True)
                            checkbox.setChecked(True)
                            checkbox.blockSignals(False)
                    checkbox.stateChanged.connect(prevent_uncheck)

                # Pulsante "Configure" -> compare/abilitato solo se checkbox = True
                configure_button = QPushButton("Configure")
                configure_button.setCursor(Qt.PointingHandCursor)
                configure_button.setVisible(False)  # Inizialmente nascosto
                configure_button.setFixedWidth(80)

                def on_checkbox_state_changed(state, btn=configure_button, p=pattern_name):
                    """
                    Funzione per gestire il cambiamento di stato della checkbox.
                    """
                    btn.setVisible(state == Qt.Checked)  # Mostra/nasconde il pulsante "Configure"
                    if state == Qt.Checked:
                        # Aggiungi un valore di default se il pattern non è già presente
                        if p not in self.temp_pattern_config:
                            self.temp_pattern_config[p] = {
                                "enabled": True,
                                "params": {}
                            }
                    else:
                        # Se deselezionato, disabilita il pattern nella configurazione
                        if p in self.temp_pattern_config:
                            self.temp_pattern_config[p]["enabled"] = False

                # Collega lo stato della checkbox al pulsante specifico
                checkbox.stateChanged.connect(
                    lambda state, btn=configure_button, p=pattern_name: on_checkbox_state_changed(state, btn, p)
                )

                # Se la checkbox cambia stato, abilitiamo/disabilitiamo Configure
                def on_checkbox_state_changed(state, btn, p):
                    """
                    Funzione per gestire il cambiamento di stato della checkbox.
                    """
                    btn.setVisible(state == Qt.Checked)  # Mostra/nasconde il pulsante "Configure"
                    if state == Qt.Checked:
                        # Aggiungi il pattern alla configurazione temporanea se non esiste
                        if p not in self.temp_pattern_config:
                            self.temp_pattern_config[p] = {
                                "enabled": True,
                                "params": {}
                            }
                    else:
                        # Rimuovi il pattern dalla configurazione temporanea se deselezionato
                        if p in self.temp_pattern_config:
                            self.temp_pattern_config[p]["enabled"] = False

                hl.addWidget(info_button)
                hl.addWidget(checkbox)
                hl.addWidget(configure_button)

                topic_layout.addLayout(hl)
                self.pattern_checkboxes[pattern_name] = checkbox

            topic_group.setLayout(topic_layout)
            patterns_grid.addWidget(topic_group, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        main_layout.addLayout(patterns_grid)

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
        main_layout.addWidget(save_button)

    def check_docker_status(self):
        try:
            subprocess.check_output(['docker', 'info'], stderr=subprocess.STDOUT)
            self.docker_status_label.setText("Active")
            self.docker_status_label.setStyleSheet("color: green; font-size: 12px;")
        except subprocess.CalledProcessError:
            self.docker_status_label.setText("Not Active")
            self.docker_status_label.setStyleSheet("color: red; font-size: 12px;")
        except FileNotFoundError:
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
        full_path = os.path.join(base_dir, image_path)
        image_label = QLabel()
        if os.path.exists(full_path):
            pixmap = QPixmap(full_path)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
        else:
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
        layout.addWidget(button_box, alignment=Qt.AlignCenter)

        dialog.exec_()

    def save_preferences_and_open_client_config(self):
        """
        Al salvataggio, creiamo un dict patterns_data:
          pattern_key -> { "enabled": bool, "params": {...} }
        """
        patterns_data = {}

        relevant_patterns = [
            "Client Registry",
            "Client Selector",
            "Client Cluster",
            "Message Compressor",
            "Multi-Task Model Trainer",
            "Heterogeneous Data Handler",
            "Incentive Registry"  # se vuoi abilitare anche Incentive Registry
        ]

        for pat_name in relevant_patterns:
            cb_checked = (self.pattern_checkboxes[pat_name].isChecked()
                          if pat_name in self.pattern_checkboxes else False)

            if pat_name in self.temp_pattern_config:
                # Abbiamo già un dict param
                existing = self.temp_pattern_config[pat_name]
                existing["enabled"] = cb_checked
                patterns_data[pat_name.lower().replace(" ", "_")] = existing
            else:
                # Non abbiamo configurato => default
                patterns_data[pat_name.lower().replace(" ", "_")] = {
                    "enabled": cb_checked,
                    "params": {}
                }

        simulation_config = {
            "rounds": self.rounds_input.value(),
            "clients": self.clients_input.value(),
            "patterns": patterns_data
        }

        self.user_choices.append(simulation_config)

        # Passa alla pagina successiva
        self.client_config_page = ClientConfigurationPage(self.user_choices, self.home_page_callback)
        self.client_config_page.show()
        self.close()


class ClientConfigurationPage(QWidget):
    """
    Pagina di configurazione dei client, con card verticali e centratura orizzontale.
    """
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.setWindowTitle("Client Configuration")
        self.resize(1200, 1000)
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: black;
            }
            QLabel {
                color: black;
                background-color: transparent;
            }
            QSpinBox, QComboBox {
                background-color: white;
                border: 1px solid gray;
                border-radius: 3px;
                height: 20px;
                font-size: 12px;
            }
            QPushButton {
                height: 30px;
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
            QFrame#ClientCard {
                background-color: #f0f0f0;
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
        """)

        main_hlayout = QHBoxLayout()
        main_hlayout.setContentsMargins(10, 10, 10, 10)
        main_hlayout.setSpacing(10)
        self.setLayout(main_hlayout)

        # Colonna sinistra: Patterns selezionati
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(5)

        patterns_label = QLabel("Selected Patterns")
        patterns_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        left_layout.addWidget(patterns_label)

        selected_patterns_dict = self.user_choices[-1].get("patterns", {})
        # Filtriamo quelli con "enabled"=True
        for p_key, p_val in selected_patterns_dict.items():
            if isinstance(p_val, dict):
                if p_val.get("enabled", False):
                    pretty_name = p_key.replace("_", " ").title()
                    lbl = QLabel(f"• {pretty_name}")
                    lbl.setStyleSheet("font-size: 12px; color: black;")
                    left_layout.addWidget(lbl)
            else:
                # Caso fallback se fosse bool
                if p_val:
                    pretty_name = p_key.replace("_", " ").title()
                    lbl = QLabel(f"• {pretty_name}")
                    lbl.setStyleSheet("font-size: 12px; color: black;")
                    left_layout.addWidget(lbl)

        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, stretch=1)

        # Colonna destra: configurazione client
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(5)

        title_label = QLabel("Configure Each Client")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        right_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0,0,0,0)
        scroll_layout.setSpacing(5)
        scroll_area.setWidget(scroll_widget)

        self.client_configs = []
        num_clients = self.user_choices[-1]["clients"]

        for client_id in range(1, num_clients + 1):
            card_widget, config_dict = self.create_client_card(client_id)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(card_widget)
            scroll_layout.addLayout(hbox)
            self.client_configs.append(config_dict)

        confirm_button = QPushButton("Confirm and Continue")
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.clicked.connect(self.save_client_configurations_and_continue)
        right_layout.addWidget(confirm_button, alignment=Qt.AlignCenter)

        main_hlayout.addWidget(right_widget, stretch=3)

    def create_client_card(self, client_id):
        card = QFrame(objectName="ClientCard")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(5)
        card.setLayout(card_layout)

        card.setFixedWidth(600)

        pc_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        icon_label = QLabel()
        icon_label.setPixmap(pc_icon.pixmap(24, 24))
        icon_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(icon_label)

        client_title = QLabel(f"Client {client_id}")
        client_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        client_title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(client_title)

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
        dataset_label.setStyleSheet("font-size: 12px; background:white")
        dataset_label.setAlignment(Qt.AlignLeft)
        dataset_combobox = QComboBox()
        dataset_combobox.addItems(["CIFAR-10", "FMNIST", "MIXED"])
        dataset_combobox.setFixedWidth(100)
        dataset_combobox.setStyleSheet("background:white")

        partition_label = QLabel("Dataset Partition:")
        partition_label.setStyleSheet("font-size: 12px; background:white")
        partition_label.setAlignment(Qt.AlignLeft)
        partition_combobox = QComboBox()
        partition_combobox.addItems(["IID", "non-IID", "Random"])
        partition_combobox.setFixedWidth(100)
        partition_combobox.setStyleSheet("background:white")

        row2_layout.addWidget(dataset_label)
        row2_layout.addWidget(dataset_combobox)
        row2_layout.addSpacing(10)
        row2_layout.addWidget(partition_label)
        row2_layout.addWidget(partition_combobox)
        card_layout.addLayout(row2_layout)

        config_dict = {
            "cpu_input": cpu_input,
            "ram_input": ram_input,
            "dataset_combobox": dataset_combobox,
            "data_distribution_type_combobox": partition_combobox
        }

        return card, config_dict

    def save_client_configurations_and_continue(self):
        client_details = []
        for idx, cfg in enumerate(self.client_configs):
            client_info = {
                "client_id": idx + 1,
                "cpu": cfg["cpu_input"].value(),
                "ram": cfg["ram_input"].value(),
                "dataset": cfg["dataset_combobox"].currentText(),
                "data_distribution_type": cfg["data_distribution_type_combobox"].currentText()
            }
            client_details.append(client_info)

        self.user_choices[-1]["client_details"] = client_details

        self.recap_simulation_page = RecapSimulationPage(self.user_choices)
        self.recap_simulation_page.show()
        self.close()