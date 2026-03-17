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

        # Explainer Type (for SSIM-based)
        self.explainer_label = QLabel("Explainer Type:")
        self.explainer_combo = QComboBox()
        self.explainer_combo.addItems([
            "GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", 
            "AblationCAM", "XGradCAM", "EigenCAM", "FullGrad", "All"
        ])
        layout.addWidget(self.explainer_label)
        layout.addWidget(self.explainer_combo)
        self.explainer_label.hide()
        self.explainer_combo.hide()

        if "selection_strategy" in self.existing_params:
            self.strategy_combo.setCurrentText(self.existing_params["selection_strategy"])
        self.update_criteria_options()

        if "selection_criteria" in self.existing_params:
            self.criteria_combo.setCurrentText(self.existing_params["selection_criteria"])
        if "explainer_type" in self.existing_params:
            self.explainer_combo.setCurrentText(self.existing_params["explainer_type"])
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
                self.explainer_label.show()
                self.explainer_combo.show()
        else:
                self.explainer_label.hide()
                self.explainer_combo.hide()
                self.value_label.show()
                self.value_spinbox.show()
                self.explanation_label.show()

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
        params = {
            "selection_strategy": self.strategy_combo.currentText(),
            "selection_criteria": self.criteria_combo.currentText(),
            "selection_value": self.value_spinbox.value()
        }
        if self.strategy_combo.currentText() == "SSIM-Based":
            params["explainer_type"] = self.explainer_combo.currentText()
        return params

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
        g_layout = QGridLayout()
        g_layout.setHorizontalSpacing(24)
        g_layout.setVerticalSpacing(14)
        bold_font = QFont()
        bold_font.setBold(True)

        rounds_label = QLabel("Number of Rounds:")
        rounds_label.setFont(bold_font)
        self.rounds_input = QSpinBox()
        self.rounds_input.setRange(1, 100)
        self.rounds_input.setValue(2)

        clients_label = QLabel("Number of Clients:")
        clients_label.setFont(bold_font)
        self.clients_input = QSpinBox()
        self.clients_input.setRange(1, 128)
        self.clients_input.setValue(2)

        clients_per_round_label = QLabel("Clients per Round:")
        clients_per_round_label.setFont(bold_font)
        self.clients_per_round_input = QSpinBox()
        self.clients_per_round_input.setRange(1, 128)
        self.clients_per_round_input.setValue(2)
        self.clients_input.valueChanged.connect(self._sync_clients_per_round_range)
        self._sync_clients_per_round_range(self.clients_input.value())

        simulation_label = QLabel("Type of Simulation:")
        font = simulation_label.font()
        font.setBold(True)
        simulation_label.setFont(font)
        self.sim_type_combo = QComboBox()
        self.sim_type_combo.addItems(["Local","Docker"])
        self.sim_type_combo.setMinimumWidth(160)

        adaptation_label = QLabel("Type of Adaptation:")
        font = adaptation_label.font()
        font.setBold(True)
        adaptation_label.setFont(font)
        self.adaptation_combo = QComboBox()
        self.adaptation_combo.addItems(["None","Random","Expert-Driven","Single AI-Agent (Zero-Shot)","Single AI-Agent (Few-Shot)","Multiple AI-Agents (Voting-Based)","Multiple AI-Agents (Role-Based)","Multiple AI-Agents (Debate-Based)"])
        self.adaptation_combo.setMinimumWidth(280)

        self.llm_label = QLabel("LLM")
        self.llm_label.setFont(bold_font)
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(["llama3.2:3b","deepseek-r1:8b","gpt-oss:20b"])
        self.llm_combo.setMinimumWidth(180)
        self.llm_label.hide()
        self.llm_combo.hide()

        def _toggle_llm_selector(text):
            vis = "single" in str(text).lower()
            self.llm_label.setVisible(vis)
            self.llm_combo.setVisible(vis)

        def add_setting(row, col, setting_label, setting_widget):
            field = QWidget()
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(6)
            field_layout.addWidget(setting_label)
            field_layout.addWidget(setting_widget)
            g_layout.addWidget(field, row, col)

        add_setting(0, 0, rounds_label, self.rounds_input)
        add_setting(0, 1, clients_label, self.clients_input)
        add_setting(0, 2, clients_per_round_label, self.clients_per_round_input)
        add_setting(1, 0, simulation_label, self.sim_type_combo)
        add_setting(1, 1, adaptation_label, self.adaptation_combo)
        add_setting(1, 2, self.llm_label, self.llm_combo)

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
        docker_status_row = QWidget()
        docker_status_row.setLayout(row)
        g_layout.addWidget(docker_status_row, 2, 0, 1, 3)

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
            "clients_per_round": min(self.clients_per_round_input.value(), self.clients_input.value()),
            "adaptation": self.adaptation_combo.currentText(),
            "LLM": self.llm_combo.currentText(),
            "patterns": patterns_data,
            "client_generation_mode": "manual",
            "client_profiles": [],
            "client_details": []
        }

        self.user_choices.append(simulation_config)
        self.client_config_page = ClientConfigurationPage(self.user_choices, home_page_callback=self.show)
        self.client_config_page.show()
        self.close()

    def _sync_clients_per_round_range(self, total_clients):
        total_clients = max(1, int(total_clients))
        self.clients_per_round_input.setMaximum(total_clients)
        if self.clients_per_round_input.value() > total_clients:
            self.clients_per_round_input.setValue(total_clients)

class ClientConfigurationPage(QWidget):
    def __init__(self, user_choices, home_page_callback):
        super().__init__()
        self.setWindowTitle("AP4Fed")
        self.resize(1000, 800)
        self.user_choices = user_choices
        self.home_page_callback = home_page_callback
        self.current_config = self.user_choices[-1]
        self.saved_manual_values = self.current_config.get("client_details", [])
        self.saved_profile_values = self.current_config.get("client_profiles", [])

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

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(16)

        mode_label = QLabel("Configuration Mode:")
        self.config_mode_combo = QComboBox()
        self.config_mode_combo.addItems(["Manual", "Profile-Based"])
        stored_mode = str(self.current_config.get("client_generation_mode", "manual")).strip().lower()
        self.config_mode_combo.setCurrentText("Profile-Based" if stored_mode == "profile_based" else "Manual")
        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(self.config_mode_combo)

        self.profile_count_label = QLabel("Number of Profiles:")
        self.profile_count_input = QSpinBox()
        self.profile_count_input.setRange(1, 12)
        self.profile_count_input.setValue(max(1, len(self.saved_profile_values)) if self.saved_profile_values else 2)
        controls_layout.addWidget(self.profile_count_label)
        controls_layout.addWidget(self.profile_count_input)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.cards_grid_layout = QGridLayout()
        self.cards_grid_layout.setSpacing(20)
        self.cards_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.cards_scroll_area = QScrollArea()
        self.cards_scroll_area.setWidgetResizable(True)
        self.cards_scroll_widget = QWidget()
        self.cards_scroll_widget.setLayout(self.cards_grid_layout)
        self.cards_scroll_area.setWidget(self.cards_scroll_widget)
        main_layout.addWidget(self.cards_scroll_area)

        self.client_configs = []
        self.profile_configs = []
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

        self.copy_button = QPushButton("Copy Client 1 to each Client")
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.setStyleSheet("""
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
        self.copy_button.clicked.connect(self.copy_to_each_client)
        main_layout.addWidget(self.copy_button)

        self.profile_count_input.valueChanged.connect(self.rebuild_cards)
        self.config_mode_combo.currentIndexChanged.connect(self.on_configuration_mode_changed)
        self.on_configuration_mode_changed()

    def on_back(self):
        self.close()
        self.home_page_callback()

    def copy_to_each_client(self):
        configs = self.profile_configs if self.is_profile_mode() else self.client_configs
        if not configs:
            return

        first = configs[0]
        cpu = first["cpu_input"].value()
        ram = first["ram_input"].value()
        ds = first["dataset_combobox"].currentText()
        part = first["partition_combobox"].currentText()
        pers = first["persistence_combobox"].currentText()
        delay = first["delay_combobox"].currentText()
        model = first["model_combobox"].currentText()
        epochs = first["epochs_spinbox"].value()
        share = first.get("share_spinbox").value() if "share_spinbox" in first else None

        for cfg in configs:
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
            if share is not None and "share_spinbox" in cfg:
                cfg["share_spinbox"].setValue(share)

    def is_profile_mode(self):
        return self.config_mode_combo.currentText() == "Profile-Based"

    def on_configuration_mode_changed(self):
        is_profile = self.is_profile_mode()
        self.profile_count_label.setVisible(is_profile)
        self.profile_count_input.setVisible(is_profile)
        self.copy_button.setText("Copy Profile 1 to each Profile" if is_profile else "Copy Client 1 to each Client")
        self.rebuild_cards()

    def rebuild_cards(self):
        self.saved_manual_values = self.capture_manual_values()
        self.saved_profile_values = self.capture_profile_values()

        while self.cards_grid_layout.count():
            item = self.cards_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.client_configs = []
        self.profile_configs = []

        if self.is_profile_mode():
            count = self.profile_count_input.value()
            for index in range(count):
                existing = self.saved_profile_values[index] if index < len(self.saved_profile_values) else None
                card_widget, config_dict = self.create_profile_card(index + 1, existing)
                row = index // 3
                col = index % 3
                self.cards_grid_layout.addWidget(card_widget, row, col)
                self.profile_configs.append(config_dict)
        else:
            num_clients = self.user_choices[-1]["clients"]
            for index in range(num_clients):
                existing = self.saved_manual_values[index] if index < len(self.saved_manual_values) else None
                card_widget, config_dict = self.create_client_card(index + 1, existing)
                row = index // 3
                col = index % 3
                self.cards_grid_layout.addWidget(card_widget, row, col)
                self.client_configs.append(config_dict)

        max_cards = len(self.profile_configs) if self.is_profile_mode() else len(self.client_configs)
        max_columns = min(max(max_cards, 1), 3)
        fixed_width = max_columns * 300 + (max_columns - 1) * self.cards_grid_layout.spacing()
        self.cards_scroll_widget.setFixedWidth(fixed_width)

    def capture_manual_values(self):
        captured = []
        for idx, cfg in enumerate(getattr(self, "client_configs", []), start=1):
            captured.append(self._serialize_card_values(cfg, idx))
        return captured

    def capture_profile_values(self):
        captured = []
        for idx, cfg in enumerate(getattr(self, "profile_configs", []), start=1):
            profile_data = self._serialize_card_values(cfg, idx)
            profile_data["share_percent"] = cfg["share_spinbox"].value()
            captured.append(profile_data)
        return captured

    def _serialize_card_values(self, cfg, index):
        return {
            "client_id": index,
            "cpu": cfg["cpu_input"].value(),
            "ram": cfg["ram_input"].value(),
            "dataset": cfg["dataset_combobox"].currentText(),
            "data_distribution_type": cfg["partition_combobox"].currentText(),
            "data_persistence_type": cfg["persistence_combobox"].currentText(),
            "delay_combobox": cfg["delay_combobox"].currentText(),
            "model": cfg["model_combobox"].currentText(),
            "epochs": cfg["epochs_spinbox"].value(),
        }

    def create_client_card(self, client_id, existing=None):
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
        cpu_input.setValue(int((existing or {}).get("cpu", 5)))
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
        ram_input.setValue(int((existing or {}).get("ram", 2)))
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
        epochs_spinbox.setValue(int((existing or {}).get("epochs", 1)))
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

        if existing:
            self._set_combo_text(dataset_combobox, existing.get("dataset"))
            self._set_combo_text(partition_combobox, existing.get("data_distribution_type"))
            self._set_combo_text(persistence_combobox, existing.get("data_persistence_type"))
            self._set_combo_text(delay_combobox, existing.get("delay_combobox"))
            self._set_combo_text(model_combobox, existing.get("model"))

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

    def create_profile_card(self, profile_id, existing=None):
        card, config_dict = self.create_client_card(profile_id, existing)
        layout = card.layout()

        share_label = QLabel("Population Share:")
        share_label.setStyleSheet("font-size: 12px; background:#f9f9f9")
        share_label.setAlignment(Qt.AlignLeft)
        share_spinbox = QSpinBox()
        share_spinbox.setRange(0, 100)
        default_share = int((existing or {}).get("share_percent", max(1, round(100 / max(self.profile_count_input.value(), 1)))))
        share_spinbox.setValue(default_share)
        share_spinbox.setSuffix(" %")
        share_spinbox.setFixedWidth(160)
        share_layout = QHBoxLayout()
        share_layout.addWidget(share_label)
        share_layout.addWidget(share_spinbox)
        layout.insertLayout(2, share_layout)

        title_label = layout.itemAt(1).widget()
        if isinstance(title_label, QLabel):
            title_label.setText(f"Profile {profile_id}")

        config_dict["share_spinbox"] = share_spinbox
        return card, config_dict

    def _set_combo_text(self, combo, value):
        if value is None:
            return
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _normalize_profile_counts(self, total_clients, profiles):
        positive_profiles = [profile for profile in profiles if profile.get("share_percent", 0) > 0]
        if not positive_profiles:
            return []

        if total_clients <= 0:
            return []

        if total_clients < len(positive_profiles):
            ranked = sorted(positive_profiles, key=lambda p: p["share_percent"], reverse=True)
            counts = {id(p): 0 for p in positive_profiles}
            for profile in ranked[:total_clients]:
                counts[id(profile)] += 1
            return [counts[id(profile)] for profile in profiles]

        counts = {id(profile): 1 for profile in positive_profiles}
        remaining = total_clients - len(positive_profiles)
        weight_sum = sum(profile["share_percent"] for profile in positive_profiles)
        raw = [
            (profile["share_percent"] / weight_sum) * remaining if weight_sum > 0 else 0.0
            for profile in positive_profiles
        ]
        floors = [int(value) for value in raw]
        fractions = [value - floor for value, floor in zip(raw, floors)]

        for profile, floor in zip(positive_profiles, floors):
            counts[id(profile)] += floor

        leftover = remaining - sum(floors)
        order = sorted(range(len(positive_profiles)), key=lambda idx: fractions[idx], reverse=True)
        for idx in order[:leftover]:
            counts[id(positive_profiles[idx])] += 1

        return [counts.get(id(profile), 0) for profile in profiles]

    def _expand_profiles_to_clients(self, total_clients, profiles):
        counts = self._normalize_profile_counts(total_clients, profiles)
        client_details = []
        next_client_id = 1
        for profile, count in zip(profiles, counts):
            for _ in range(count):
                client_details.append({
                    "client_id": next_client_id,
                    "cpu": profile["cpu"],
                    "ram": profile["ram"],
                    "dataset": profile["dataset"],
                    "data_distribution_type": profile["data_distribution_type"],
                    "data_persistence_type": profile["data_persistence_type"],
                    "delay_combobox": profile["delay_combobox"],
                    "model": profile["model"],
                    "epochs": profile["epochs"],
                })
                next_client_id += 1
        return client_details

    def save_client_configurations_and_continue(self):
        total_clients = int(self.user_choices[-1]["clients"])

        if self.is_profile_mode():
            profiles = []
            for idx, cfg in enumerate(self.profile_configs, start=1):
                profiles.append({
                    "profile_id": idx,
                    "share_percent": cfg["share_spinbox"].value(),
                    "cpu": cfg["cpu_input"].value(),
                    "ram": cfg["ram_input"].value(),
                    "dataset": cfg["dataset_combobox"].currentText(),
                    "data_distribution_type": cfg["partition_combobox"].currentText(),
                    "data_persistence_type": cfg["persistence_combobox"].currentText(),
                    "delay_combobox": cfg["delay_combobox"].currentText(),
                    "model": cfg["model_combobox"].currentText(),
                    "epochs": cfg["epochs_spinbox"].value()
                })
            if sum(profile["share_percent"] for profile in profiles) <= 0:
                QMessageBox.warning(self, "Invalid Profiles", "At least one profile must have a share percentage greater than 0.")
                return
            client_details = self._expand_profiles_to_clients(total_clients, profiles)
            self.user_choices[-1]["client_generation_mode"] = "profile_based"
            self.user_choices[-1]["client_profiles"] = profiles
        else:
            client_details = []
            for idx, cfg in enumerate(self.client_configs):
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
            self.user_choices[-1]["client_generation_mode"] = "manual"
            self.user_choices[-1]["client_profiles"] = []

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
