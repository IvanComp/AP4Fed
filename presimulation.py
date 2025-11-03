import os
import json
import subprocess
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QLabel, QPushButton, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QHBoxLayout, QGridLayout,
    QComboBox, QScrollArea, QStyle, QMessageBox,
    QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize
from recap_simulation import RecapSimulationPage
from PyQt5.QtGui import QPixmap, QFont

# ------------------------------------------------------------------------------------------
# Dialog window for the "Client Selector" parameters
# ------------------------------------------------------------------------------------------
class ClientSelectorDialog(QDialog):
    def __init__(self, existing_params=None):
        super().__init__()
        self.setWindowTitle("AP4Fed")
        self.resize(400, 300) 
        self.existing_params = existing_params or {}
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        self.strategy_label = QLabel("Selection Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Resource-Based")  
        self.strategy_combo.addItem("SSIM-Based") 
        self.strategy_combo.addItem("Data-Based") 
        self.strategy_combo.addItem("Performance-based")
        self.strategy_combo.model().item(2).setEnabled(False)
        self.strategy_combo.model().item(3).setEnabled(False)
        layout.addWidget(self.strategy_label)
        layout.addWidget(self.strategy_combo)

        # Selection Criteria
        self.criteria_label = QLabel("Selection Criteria:")
        self.criteria_combo = QComboBox()
        layout.addWidget(self.criteria_label)
        layout.addWidget(self.criteria_combo)

        self.strategy_combo.currentIndexChanged.connect(self.update_criteria_options)

        # Selection Value
        self.value_label = QLabel("Minimum Value:")
        self.value_spinbox = QSpinBox()
        self.value_spinbox.setRange(1, 128)  
        self.value_spinbox.setValue(4) 
        layout.addWidget(self.value_label)
        layout.addWidget(self.value_spinbox)
        self.explanation_label = QLabel("The client should have at least a minimum value CPU or RAM based on the selected criteria.")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(self.explanation_label)

        if "selection_strategy" in self.existing_params:
            self.strategy_combo.setCurrentText(self.existing_params["selection_strategy"])
        self.update_criteria_options()

        if "selection_criteria" in self.existing_params:
            self.criteria_combo.setCurrentText(self.existing_params["selection_criteria"])
        if "selection_value" in self.existing_params:
            self.value_spinbox.setValue(self.existing_params["selection_value"])

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def update_criteria_options(self):
        strategy = self.strategy_combo.currentText()
        self.criteria_combo.clear()
        if strategy == "SSIM-Based":
                self.criteria_combo.addItems(["Min","Mid","Max"])
                self.value_label.hide()
                self.value_spinbox.hide()
                self.explanation_label.hide()
        if strategy == "Resource-Based":
            self.criteria_combo.addItems(["CPU", "RAM"])
        elif strategy == "Data-Based":
            self.criteria_combo.addItems(["IID", "non-IID"])
        elif strategy == "Performance-based":
            self.criteria_combo.addItems(["Accuracy", "Latency"])

    def on_back(self):
        self.close()
        self.home_page_callback()

    def get_params(self):
        return {
            "selection_strategy": self.strategy_combo.currentText(),
            "selection_criteria": self.criteria_combo.currentText(),
            "selection_value": self.value_spinbox.value()
        }

# ------------------------------------------------------------------------------------------
# Dialog window for the "Client Cluster" parameters
# ------------------------------------------------------------------------------------------
class ClientClusterDialog(QDialog):
    def __init__(self, existing_params=None):
        super().__init__()
        self.setWindowTitle("Configure Client Cluster")
        self.resize(400, 300) 
        self.existing_params = existing_params or {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Clustering Strategy
        self.strategy_label = QLabel("Clustering Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Resource-Based")  
        self.strategy_combo.addItem("Data-Based")  
        self.strategy_combo.addItem("Network-Based")
        self.strategy_combo.model().item(2).setEnabled(False)
        layout.addWidget(self.strategy_label)
        layout.addWidget(self.strategy_combo)

        self.criteria_label = QLabel("Clustering Criteria:")
        self.criteria_combo = QComboBox()
        layout.addWidget(self.criteria_label)
        layout.addWidget(self.criteria_combo)

        self.strategy_combo.currentIndexChanged.connect(self.update_criteria_options)
        self.value_label = QLabel("Minimum Value:")
        self.value_spinbox = QSpinBox()
        self.value_spinbox.setRange(1, 128) 
        self.value_spinbox.setValue(1)  
        layout.addWidget(self.value_label)
        layout.addWidget(self.value_spinbox)
        self.explanation_label = QLabel("The clients will be clustered based on the selected criteria and a minimum [VALUE] if applicable.")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(self.explanation_label)

        if "clustering_strategy" in self.existing_params:
            self.strategy_combo.setCurrentText(self.existing_params["clustering_strategy"])
        self.update_criteria_options()

        if "clustering_criteria" in self.existing_params:
            self.criteria_combo.setCurrentText(self.existing_params["clustering_criteria"])
        if "clustering_value" in self.existing_params:
            self.value_spinbox.setValue(self.existing_params["clustering_value"])

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def update_criteria_options(self):
        strategy = self.strategy_combo.currentText()
        self.criteria_combo.clear()
        if strategy == "Resource-Based":
            self.criteria_combo.addItems(["CPU", "RAM"])
        elif strategy == "Data-Based":
            self.criteria_combo.addItems(["IID", "non-IID"])
        elif strategy == "Network-Based":
            self.criteria_combo.addItems(["Latency", "Bandwidth"])

    def get_params(self):
        return {
            "clustering_strategy": self.strategy_combo.currentText(),
            "clustering_criteria": self.criteria_combo.currentText(),
            "clustering_value": self.value_spinbox.value()
        }

# ------------------------------------------------------------------------------------------
# Dialog window for the "Multi-Task Model Trainer" parameters
# ------------------------------------------------------------------------------------------
class MultiTaskModelTrainerDialog(QDialog):
    def __init__(self, existing_params=None):
        super().__init__()
        self.setWindowTitle("Configure Multi-Task Model Trainer")
        self.resize(400, 200)

        self.existing_params = existing_params or {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        self.m1_label = QLabel("Select Model M1:")
        self.m1_combo = QComboBox()
        self.m1_combo.addItems(["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST", "KMNIST", "ImageNet100"])
        layout.addWidget(self.m1_label)
        layout.addWidget(self.m1_combo)

        self.m2_label = QLabel("Select Model M2:")
        self.m2_combo = QComboBox()
        self.m2_combo.addItems(["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST", "KMNIST", "ImageNet100"])
        layout.addWidget(self.m2_label)
        layout.addWidget(self.m2_combo)

        if "model1" in self.existing_params:
            self.m1_combo.setCurrentText(self.existing_params["model1"])
        if "model2" in self.existing_params:
            self.m2_combo.setCurrentText(self.existing_params["model2"])

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept(self):
        if self.m1_combo.currentText() == self.m2_combo.currentText():
            QMessageBox.warning(self, "Configuration Error", 
                                "Model1 and Model2 cannot be the same.")
            return
        super().accept()

    def get_params(self):
        return {
            "model1": self.m1_combo.currentText(),
            "model2": self.m2_combo.currentText()
        }

# ------------------------------------------------------------------------------------------
# Generic Dialog Window 
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
# Main Class PreSimulationPage
# ------------------------------------------------------------------------------------------
class PreSimulationPage(QWidget):
    def __init__(self, user_choices, home_page_callback):
        super().__init__()

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

        self.pattern_data = {
            "Client Registry": {
                "category": "Client Management Category",
                "image": "img/patterns/clientregistry.png",
                "description": "Maintains information about all participating client devices for client management.",
                "benefits": "Centralized tracking of client states; easier organization.",
                "drawbacks": "Requires overhead for maintaining the registry."
            },
            "Client Selector": {
                "category": "Client Management Category",
                "image": "img/patterns/clientselector.png",
                "description": "Actively selects client devices for a specific training round based on predefined criteria to enhance model performance and system efficiency.",
                "benefits": "Ensures only the most relevant clients train each round, potentially improving performance.",
                "drawbacks": "May exclude important data from non-selected clients."
            },
            "Client Cluster": {
                "category": "Client Management Category",
                "image": "img/patterns/clientcluster.png",
                "description": "Groups client devices based on their similarity in certain characteristics (e.g., resources, data distribution) to improve model performance and training efficiency.",
                "benefits": "Allows specialized training; can handle different groups more effectively.",
                "drawbacks": "Additional overhead to manage cluster membership."
            },
            "Message Compressor": {
                "category": "Model Management Category",
                "image": "img/patterns/messagecompressor.png",
                "description": "Compresses and reduces the size of message data before each model exchange round to improve communication efficiency.",
                "benefits": "Reduces bandwidth usage; can speed up communication rounds.",
                "drawbacks": "Compression/decompression overhead might offset gains for large data."
            },
            "Model co-Versioning Registry": {
                "category": "Model Management Category",
                "image": "img/patterns/modelversioningregistry.png",
                "description": "It is designed to store both the current model version trained by each client device and the aggregated model version stored on the server in a Federated Learning process.",
                "benefits": "Enables reproducibility and consistent version tracking.",
                "drawbacks": "Extra storage cost is incurred to store all the local and global models."
            },
            "Model Replacement Trigger": {
                "category": "Model Management Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Deployment Selector": {
                "category": "Model Management Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Multi-Task Model Trainer": {
                "category": "Model Training Category",
                "image": "img/patterns/multitaskmodeltrainer.png",
                "description": "Utilizes data from related models on local devices to enhance efficiency.",
                "benefits": "Potential knowledge sharing among similar tasks; improved training.",
                "drawbacks": "Training logic may become more complex to handle multiple tasks."
            },
            "Heterogeneous Data Handler": {
                "category": "Model Training Category",
                "image": "img/patterns/heterogeneousdatahandler.png",
                "description": "Addresses issues with non-IID and skewed data while maintaining data privacy.",
                "benefits": "Better management of varied data distributions.",
                "drawbacks": "Requires more sophisticated data partitioning and handling logic."
            },
            "Incentive Registry": {
                "category": "Model Training Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Asynchronous Aggregator": {
                "category": "Model Aggregation Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Decentralised Aggregator": {
                "category": "Model Aggregation Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Hierarchical Aggregator": {
                "category": "Model Aggregation Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            },
            "Secure Aggregator": {
                "category": "Model Aggregation Category",
                "image": "",
                "description": "This Architectural Pattern is not yet implemented",
                "benefits": "",
                "drawbacks": ""
            }
        }

        super().__init__()
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback
        self.temp_pattern_config = {}
        self.setWindowTitle("AP4Fed")
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

        choice_label = QLabel(f"Input Parameters Setup")
        choice_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        choice_label.setAlignment(Qt.AlignCenter)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        header_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        header_layout.addWidget(choice_label, stretch=1)
        main_layout.insertLayout(0, header_layout)

        general_settings_group = QGroupBox("General Settings")
        general_settings_group.setStyleSheet(
            "QGroupBox::title { font-weight: bold; }"
        )
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
        bold_font = QFont()
        bold_font.setBold(True)

        rounds_label = QLabel("Number of Rounds:")
        rounds_label.setFont(bold_font)
        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 100)
        self.rounds_input.setValue(2)
        g_layout.addRow(rounds_label, self.rounds_input)
        clients_label = QLabel("Number of Clients:")
        clients_label.setFont(bold_font)
        self.clients_input = QSpinBox()
        self.clients_input.setRange(1, 128)
        self.clients_input.setValue(2)
        g_layout.addRow(clients_label, self.clients_input)

        label = QLabel("Type of Simulation:")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        self.sim_type_combo = QComboBox()
        self.sim_type_combo.addItems(["Docker","Local"])
        self.sim_type_combo.setFixedWidth(90)
        g_layout.addRow(label, self.sim_type_combo)

        label = QLabel("Type of Adaptation:")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        self.adaptation_combo = QComboBox()
        self.adaptation_combo.addItems(["None","Random","Single AI-Agent (Zero-Shot)","Single AI-Agent (Few-Shot)","Multiple AI-Agents (Voting-Based)","Multiple AI-Agents (Role-Based)","Multiple AI-Agents (Debate-Based)"])
        self.adaptation_combo.setFixedWidth(90)
        g_layout.addRow(label, self.adaptation_combo)

        self.llm_label = QLabel("LLM")
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(["llama3.2:3b","deepseek-r1:8b","gpt-oss:20b"])
        self.llm_label.hide()
        self.llm_combo.hide()

        def _toggle_llm_selector(text):
            vis = "single" in str(text).lower()
            self.llm_label.setVisible(vis)
            self.llm_combo.setVisible(vis)

        g_layout.addRow(self.llm_label, self.llm_combo)
        self.adaptation_combo.currentTextChanged.connect(_toggle_llm_selector)
        _toggle_llm_selector(self.adaptation_combo.currentText())

        docker_status_label = QLabel("Docker Status:")
        font = docker_status_label.font()
        font.setBold(True)
        docker_status_label.setFont(font)
        self.docker_status_label = QLabel()
        update_btn = QPushButton()
        update_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        update_btn.setCursor(Qt.PointingHandCursor)
        update_btn.clicked.connect(self.check_docker_status)
        update_btn.setStyleSheet("""
        QPushButton {
                background-color: white;
            }
        """)
        for w in (docker_status_label, self.docker_status_label, update_btn):
            w.setVisible(False)
        row = QHBoxLayout()
        row.addWidget(docker_status_label)
        row.addWidget(self.docker_status_label)
        row.addWidget(update_btn)
        row.addStretch()
        g_layout.addRow(row)

        def on_type_changed(text):
            show = (text == "Docker")
            for w in (docker_status_label, self.docker_status_label, update_btn):
                w.setVisible(show)
            if show:
                self.check_docker_status()
        self.sim_type_combo.currentTextChanged.connect(on_type_changed)

        general_settings_group.setLayout(g_layout)
        main_layout.addWidget(general_settings_group)

        patterns_label = QLabel("Select Architectural Patterns to be applied:")
        patterns_label.setAlignment(Qt.AlignLeft)
        patterns_label.setStyleSheet("font-size: 14px; color: black; margin-top: 10px;")
        main_layout.addWidget(patterns_label)

        patterns_grid = QGridLayout()
        patterns_grid.setSpacing(10)
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
                "Heterogeneous Data Handler: Addresses issues with non-IID and skewed data while maintaining data privacy.",
                "Multi-Task Model Trainer: Utilizes data from related models on local devices to enhance efficiency.",
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
        enabled_patterns = [
            "Client Registry",
            "Client Selector",
            "Client Cluster",
            "Message Compressor",
            "Model co-Versioning Registry",
            #"Multi-Task Model Trainer",
            "Heterogeneous Data Handler",
        ]

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
                if pattern_name == "Model co-Versioning Registry":
                    checkbox.setToolTip(pattern_desc)
                    checkbox.setStyleSheet("QCheckBox { color: black; font-size: 12px; }")

                if pattern_name not in enabled_patterns:
                    checkbox.setEnabled(False)
                    checkbox.setStyleSheet("QCheckBox { color: darkgray; font-size: 12px; }")

                if pattern_name == "Client Registry":
                    checkbox.setText("Client Registry (Active by Default)")
                    checkbox.setChecked(True)
                    def prevent_uncheck(state):
                        if state != Qt.Checked:
                            checkbox.blockSignals(True)
                            checkbox.setChecked(True)
                            checkbox.blockSignals(False)
                    checkbox.stateChanged.connect(prevent_uncheck)

                if pattern_name in ["Message Compressor", "Heterogeneous Data Handler", "Model co-Versioning Registry"]:
                    configure_button = None
                elif pattern_name in ["Client Selector", "Client Cluster", "Multi-Task Model Trainer"]:
                    configure_button = QPushButton("Configure")
                    configure_button.setCursor(Qt.PointingHandCursor)
                    configure_button.setStyleSheet("""
                        QPushButton {
                            background-color: #ffc107;
                            color: white;
                            font-size: 10px;
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
                    configure_button.setVisible(False)
                    configure_button.setFixedWidth(80)

                    configure_button.clicked.connect(lambda _, p=pattern_name: open_config(p))
                else:
                    configure_button = None

                def on_checkbox_state_changed(state, btn, p=pattern_name):
                    if p == "Multi-Task Model Trainer" and state == Qt.Checked:
                        if self.clients_input.value() < 4:
                            msg_box = QMessageBox(self)
                            msg_box.setWindowTitle("Configuration Error")
                            msg_box.setText("Multi-Task Model Trainer requires at least 4 clients.")
                            msg_box.setIcon(QMessageBox.Warning)

                            ok_button = msg_box.addButton("OK", QMessageBox.AcceptRole)
                            ok_button.setCursor(Qt.PointingHandCursor)
                            ok_button.setStyleSheet("""
                                QPushButton {
                                    background-color: green;
                                    color: white;
                                    font-size: 10px;
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

                            msg_box.exec_()

                            checkbox.blockSignals(True)
                            checkbox.setChecked(False)
                            checkbox.blockSignals(False)
                            return

                    if btn is not None:
                        btn.setVisible(state == Qt.Checked)
                    if state == Qt.Checked:
                        if p not in self.temp_pattern_config:
                            self.temp_pattern_config[p] = {
                                "enabled": True,
                                "params": {}
                            }
                    else:
                        if p in self.temp_pattern_config:
                            self.temp_pattern_config[p]["enabled"] = False

                checkbox.stateChanged.connect(
                    lambda state, btn=configure_button, p=pattern_name:
                    on_checkbox_state_changed(state, btn, p)
                )

                def open_config(p_name):
                    if p_name == "Client Selector":
                        existing_params = self.temp_pattern_config.get(p_name, {}).get("params", {})
                        dlg = ClientSelectorDialog(existing_params)
                        if dlg.exec_() == QDialog.Accepted:
                            new_params = dlg.get_params()
                            self.temp_pattern_config[p_name] = {
                                "enabled": True,
                                "params": new_params
                            }
                    elif p_name == "Client Cluster":
                        existing_params = self.temp_pattern_config.get(p_name, {}).get("params", {})
                        dlg = ClientClusterDialog(existing_params)
                        if dlg.exec_() == QDialog.Accepted:
                            new_params = dlg.get_params()
                            self.temp_pattern_config[p_name] = {
                                "enabled": True,
                                "params": new_params
                            }
                    elif p_name == "Multi-Task Model Trainer":
                        existing_params = self.temp_pattern_config.get(p_name, {}).get("params", {})
                        dlg = MultiTaskModelTrainerDialog(existing_params)
                        if dlg.exec_() == QDialog.Accepted:
                            new_params = dlg.get_params()
                            self.temp_pattern_config[p_name] = {
                                "enabled": True,
                                "params": new_params
                            }
                    else:
                        QMessageBox.information(self, "Not Implemented", f"The configuration for {p_name} is not implemented yet.")

                hl.addWidget(info_button)
                hl.addWidget(checkbox)
                if configure_button is not None:
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

    def update_docker_status(self):
        self.check_docker_status()

    def on_back(self):
        self.close()
        self.home_page_callback()

    def show_pattern_info(self, pattern_name, pattern_category, image_path, description, benefits, drawbacks):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{pattern_name} - {pattern_category}")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        layout.setAlignment(Qt.AlignTop)

        title_label = QLabel(f"{pattern_name}")
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
            image_label.setText("Architectural Pattern not Implemented!")
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
        button_box.setCursor(Qt.PointingHandCursor)
        button_box.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 10px;
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
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box, alignment=Qt.AlignCenter)

        dialog.exec_()

    def save_preferences_and_open_client_config(self):
        config_needed_patterns = ["Client Selector", "Client Cluster", "Multi-Task Model Trainer"]
        for p_name in config_needed_patterns:
            if p_name in self.pattern_checkboxes and self.pattern_checkboxes[p_name].isChecked():
                if p_name not in self.temp_pattern_config or not self.temp_pattern_config[p_name]["params"]:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Configuration Needed")
                    msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setText(f"Please configure '{p_name}' before continuing.")

                    ok_button = msg_box.addButton("OK", QMessageBox.AcceptRole)
                    ok_button.setCursor(Qt.PointingHandCursor)
                    ok_button.setStyleSheet("""
                        QPushButton {
                            background-color: green;
                            color: white;
                            font-size: 10px;
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

                    msg_box.exec_()
                    return

        patterns_data = {}
        relevant_patterns = [
            "Client Registry",
            "Client Selector",
            "Client Cluster",
            "Message Compressor",
            "Model co-Versioning Registry",
            "Multi-Task Model Trainer",
            "Heterogeneous Data Handler",
        ]

        for pat_name in relevant_patterns:
            cb_checked = (self.pattern_checkboxes[pat_name].isChecked()
                          if pat_name in self.pattern_checkboxes else False)

            if pat_name in self.temp_pattern_config:
                existing = self.temp_pattern_config[pat_name]
                existing["enabled"] = cb_checked
                patterns_data[pat_name.lower().replace(" ", "_")] = existing
            else:
                patterns_data[pat_name.lower().replace(" ", "_")] = {
                    "enabled": cb_checked,
                    "params": {}
                }

        simulation_config = {
            "simulation_type": self.sim_type_combo.currentText(),
            "rounds": self.rounds_input.value(),
            "clients": self.clients_input.value(),
            "adaptation": self.adaptation_combo.currentText(),
            "LLM": self.llm_combo.currentText(),
            "patterns": patterns_data,
            "client_details": []
        }

        self.user_choices.append(simulation_config)
        self.client_config_page = ClientConfigurationPage(self.user_choices, home_page_callback=self.show)
        self.client_config_page.show()
        self.close()

class ClientConfigurationPage(QWidget):
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.setWindowTitle("AP4Fed")
        self.resize(1000, 800)
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
                background-color: #f9f9f9;
                border: 1px solid lightgray;
                border-radius: 5px;
                padding: 10px;
                margin: 5px;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        back_btn = QPushButton()
        back_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
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

        title_label = QLabel("Clients Configuration")
        title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        header_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        header_layout.addWidget(title_label, stretch=1)
        main_layout.insertLayout(0, header_layout)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(grid_layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self.client_configs = []
        num_clients = self.user_choices[-1]["clients"]

        for index in range(num_clients):
            card_widget, config_dict = self.create_client_card(index + 1)
            row = index // 3
            col = index % 3
            grid_layout.addWidget(card_widget, row, col)
            self.client_configs.append(config_dict)

        max_columns = min(num_clients, 3)
        fixed_width = max_columns * 300 + (max_columns - 1) * grid_layout.spacing()
        scroll_widget.setFixedWidth(fixed_width)

        confirm_button = QPushButton("Confirm and Continue")
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #c69500;
            }
        """)
        confirm_button.clicked.connect(self.save_client_configurations_and_continue)
        main_layout.addWidget(confirm_button)

        copy_button = QPushButton("Copy Client 1 to each Client")
        copy_button.setCursor(Qt.PointingHandCursor)
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                height: 30px;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
            QPushButton:pressed {
                background-color: #004970;
            }
        """)
        copy_button.clicked.connect(self.copy_to_each_client)
        main_layout.addWidget(copy_button)

    def on_back(self):
        self.close()
        self.home_page_callback()

    def copy_to_each_client(self):
        first = self.client_configs[0]
        cpu = first["cpu_input"].value()
        ram = first["ram_input"].value()
        ds = first["dataset_combobox"].currentText()
        part = first["partition_combobox"].currentText()
        pers = first["persistence_combobox"].currentText()
        delay = first["delay_combobox"].currentText()
        model = first["model_combobox"].currentText()
        epochs = first["epochs_spinbox"].value()

        for cfg in self.client_configs:
            cfg["cpu_input"].setValue(cpu)
            cfg["ram_input"].setValue(ram)

            idx_ds = cfg["dataset_combobox"].findText(ds)
            if idx_ds >= 0:
                cfg["dataset_combobox"].setCurrentIndex(idx_ds)

            idx_part = cfg["partition_combobox"].findText(part)
            if idx_part >= 0:
                cfg["partition_combobox"].setCurrentIndex(idx_part)

            idx_pers = cfg["persistence_combobox"].findText(pers)
            if idx_pers >= 0:
                cfg["persistence_combobox"].setCurrentIndex(idx_pers)

            idx_delay = cfg["delay_combobox"].findText(delay)
            if idx_delay >= 0:
                cfg["delay_combobox"].setCurrentIndex(idx_delay)

            idx_model = cfg["model_combobox"].findText(model)
            if idx_model >= 0:
                cfg["model_combobox"].setCurrentIndex(idx_model)

            cfg["epochs_spinbox"].setValue(epochs)

    def create_client_card(self, client_id):
        card = QFrame(objectName="ClientCard")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 9, 9)
        card_layout.setSpacing(5)
        card.setLayout(card_layout)

        fixed_width = 305
        fixed_height = 420
        card.setFixedWidth(fixed_width)
        card.setFixedHeight(fixed_height)

        pc_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        pc_icon_label = QLabel()
        pc_icon_label.setPixmap(pc_icon.pixmap(80, 80))
        card_layout.addWidget(pc_icon_label, alignment=Qt.AlignCenter)

        client_title = QLabel(f"Client {client_id}")
        client_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        client_title.setAlignment(Qt.AlignCenter)
        client_title.setContentsMargins(0, 0, 0, 10)
        card_layout.addWidget(client_title)

        cpu_label = QLabel("CPU Allocation:")
        cpu_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        cpu_label.setAlignment(Qt.AlignLeft)
        cpu_input = QSpinBox()
        cpu_input.setRange(1, 16)
        cpu_input.setValue(5)
        cpu_input.setSuffix(" CPUs")
        cpu_input.setFixedWidth(160)
        cpu_layout = QHBoxLayout()
        cpu_layout.setSpacing(16) 
        cpu_layout.addWidget(cpu_label)
        cpu_layout.addWidget(cpu_input)
        card_layout.addLayout(cpu_layout)

        ram_label = QLabel("RAM Allocation:")
        ram_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        ram_label.setAlignment(Qt.AlignLeft)
        ram_input = QSpinBox()
        ram_input.setRange(1, 128)
        ram_input.setValue(2)
        ram_input.setSuffix(" GB")
        ram_input.setFixedWidth(160)
        ram_layout = QHBoxLayout()
        ram_layout.setSpacing(14)
        ram_layout.addWidget(ram_label)
        ram_layout.addWidget(ram_input)
        card_layout.addLayout(ram_layout)

        dataset_label = QLabel("Testing Dataset:")
        dataset_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        dataset_label.setAlignment(Qt.AlignLeft)
        dataset_combobox = QComboBox()
        dataset_combobox.addItems(["CIFAR-10","CIFAR-100","FashionMNIST","MNIST", "KMNIST", "OXFORDIIITPET","ImageNet100"])
        dataset_combobox.setFixedWidth(160)
        dataset_layout = QHBoxLayout()
        dataset_layout.setSpacing(12)
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(dataset_combobox)
        card_layout.addLayout(dataset_layout)

        partition_label = QLabel("Data Distribution:")
        partition_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        partition_label.setAlignment(Qt.AlignLeft)
        partition_combobox = QComboBox()
        partition_combobox.addItems(["IID", "non-IID", "Random"])
        partition_combobox.setFixedWidth(160)
        partition_layout = QHBoxLayout()
        partition_layout.addWidget(partition_label)
        partition_layout.addWidget(partition_combobox)
        card_layout.addLayout(partition_layout)

        persistence_label = QLabel("Data Persistence:")
        persistence_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        persistence_label.setAlignment(Qt.AlignLeft)
        persistence_combobox = QComboBox()
        persistence_combobox.addItems(["Same Data", "New Data", "Remove Data"])
        persistence_combobox.setFixedWidth(160)
        persistence_layout = QHBoxLayout()
        persistence_layout.addWidget(persistence_label)
        persistence_layout.addWidget(persistence_combobox)
        card_layout.addLayout(persistence_layout)

        delay_label = QLabel("Delay Injection:")
        delay_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        delay_label.setAlignment(Qt.AlignLeft)
        delay_combobox = QComboBox()
        delay_combobox.addItems(["No","Yes","Random"])
        delay_combobox.setFixedWidth(160)
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(delay_label)
        delay_layout.setSpacing(17) 
        delay_layout.addWidget(delay_combobox)
        card_layout.addLayout(delay_layout)

        model_group = QGroupBox("Model Training Settings")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

        model_group_layout = QVBoxLayout()
        model_group.setLayout(model_group_layout)  

        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-size: 12px;")
        model_label.setAlignment(Qt.AlignLeft)
        model_combobox = QComboBox()
        model_combobox.setFixedWidth(160)

        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_combobox)
        model_group_layout.addLayout(model_layout)

        epochs_label = QLabel("Epochs:")
        epochs_label.setStyleSheet("font-size: 12px;")
        epochs_label.setAlignment(Qt.AlignLeft)
        epochs_spinbox = QSpinBox()
        epochs_spinbox.setRange(1, 100)
        epochs_spinbox.setValue(1)
        epochs_spinbox.setFixedWidth(60)

        epochs_layout = QHBoxLayout()
        epochs_layout.setSpacing(17)
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(epochs_spinbox)
        model_group_layout.addLayout(epochs_layout)

        card_layout.addWidget(model_group)

        def update_model_options():
            models_list = [
                "CNN 16k", "CNN 64k","CNN 256k","alexnet", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                "densenet121", "densenet161", "densenet169", "densenet201",
                "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
                "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
                "googlenet", "inception_v3",
                "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
                "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
                "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_16gf", "regnet_x_32gf", "regnet_x_3_2gf", "regnet_x_8gf",
                "regnet_y_400mf", "regnet_y_800mf", "regnet_y_128gf", "regnet_y_16gf", "regnet_y_1_6gf", "regnet_y_32gf", "regnet_y_3_2gf", "regnet_y_8gf",
                "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "resnext50_32x4d", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
                "squeezenet1_0", "squeezenet1_1",
                "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
                "wide_resnet50_2", "wide_resnet101_2",
                "swin_t", "swin_s", "swin_b",
                "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"
            ]
            model_combobox.clear()
            model_combobox.addItems(models_list)

        dataset_combobox.currentIndexChanged.connect(update_model_options)
        update_model_options() 

        config_dict = {
            "cpu_input": cpu_input,
            "ram_input": ram_input,
            "dataset_combobox": dataset_combobox,
            "persistence_combobox": persistence_combobox,
            "partition_combobox": partition_combobox,
            "delay_combobox": delay_combobox,
            "model_combobox": model_combobox,
            "epochs_spinbox": epochs_spinbox
        }

        return card, config_dict

    def save_client_configurations_and_continue(self):
        client_details = []
        for idx, cfg in enumerate(self.client_configs):
            _dist = cfg['partition_combobox'].currentText()
            if _dist == 'Random':
                _dist = 'IID' if random.random() < 0.5 else 'non-IID'
            _delay = cfg['delay_combobox'].currentText()
            if _delay == 'Random':
                _delay = 'Yes' if random.random() < 0.5 else 'No'
            client_info = {
                "client_id": idx + 1,
                "cpu": cfg["cpu_input"].value(),
                "ram": cfg["ram_input"].value(),
                "dataset": cfg["dataset_combobox"].currentText(),
                "data_distribution_type": cfg["partition_combobox"].currentText(),
                "data_persistence_type": cfg["persistence_combobox"].currentText(),
                "delay_combobox": cfg["delay_combobox"].currentText(),
                "model": cfg["model_combobox"].currentText(),
                "epochs": cfg["epochs_spinbox"].value()
            }
            client_details.append(client_info)

        self.user_choices[-1]["client_details"] = client_details
        self.save_configuration_to_file()
        self.recap_simulation_page = RecapSimulationPage(self.user_choices, home_page_callback=self.show)
        self.recap_simulation_page.show()
        self.close()

    def save_configuration_to_file(self):
        try:
            cfg = self.user_choices[-1]
            for c in cfg.get('client_details', []):
                if c.get('data_distribution_type') == 'Random':
                    c['data_distribution_type'] = 'IID' if random.random() < 0.5 else 'non-IID'
                if c.get('delay_combobox') == 'Random':
                    c['delay_combobox'] = 'Yes' if random.random() < 0.5 else 'No'
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sim_type = self.user_choices[-1].get("simulation_type")
            if sim_type.lower() == "docker":
                config_dir = os.path.join(base_dir, 'Docker', 'configuration')
            else:
                config_dir = os.path.join(base_dir, 'Local', 'configuration')
            os.makedirs(config_dir, exist_ok=True)
            config_file_path = os.path.join(config_dir, 'config.json')

            with open(config_file_path, 'w') as f:
                json.dump(self.user_choices[-1], f, indent=4)
        except Exception as e:
            error_box = QMessageBox(self)
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Error")
            error_box.setText(f"An error occurred while saving the configuration: {e}")
            error_box.exec_()