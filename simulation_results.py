import sys
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QToolButton, QApplication, 
    QStyle, QLabel, QWidget, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QMovie

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Helvetica',
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

PASTEL_COLORS = [
    '#AEC6CF', '#FFB347', '#77DD77', '#CFCFC4', '#FFD1DC',
    '#B39EB5', '#FF6961', '#FDFD96', '#CB99C9', '#C23B22'
]

class SingleChartWidget(QWidget):
    def __init__(self, parent=None, title="Chart", figure=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        title_layout = QHBoxLayout()
        layout.addLayout(title_layout)

        title_layout.addStretch()
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()

        self.download_button = QToolButton()
        self.download_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.download_button.setToolTip("Save this chart as SVG")
        self.download_button.setCursor(Qt.PointingHandCursor)
        self.download_button.setStyleSheet("padding: 5px;")
        title_layout.addWidget(self.download_button)

        self.canvas = FigureCanvas(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

class SimulationResults(QDialog):
    def __init__(self, data_file=None):
        super().__init__()
        self.setWindowTitle("Simulation Results")
        self.setStyleSheet("background-color: white;")
        self.resize(1600, 1000)

        # Carica i dati CSV se fornito
        self.data = None
        if data_file and os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
        
        # Determina il percorso corrente.
        try:
            current_dir = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            current_dir = os.getcwd()
        
        config = {}
        config_paths = [
            os.path.join(current_dir, 'configuration', 'config.json'),
            os.path.join(current_dir, 'config.json')
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = json.load(f)
                break
        if not config:
            print("DEBUG: config.json non trovato nei percorsi:", config_paths)
        
        # Ottieni il numero di client:
        clients = config.get("clients", None)
        if clients is None and "client_details" in config:
            clients = len(config["client_details"])
        if clients is None:
            clients = "N/A"
        
        # Ottieni il dataset: se non presente, prendi dal primo client
        dataset = config.get("dataset", None)
        if dataset is None and "client_details" in config and len(config["client_details"]) > 0:
            dataset = config["client_details"][0].get("dataset", "N/A")
        if dataset is None:
            dataset = "N/A"
        
        # Ottieni il modello dal primo client in "client_details"
        model = None
        if "client_details" in config and len(config["client_details"]) > 0:
            model = config["client_details"][0].get("model", None)
        if model is None:
            model = "N/A"
        
        print(f"DEBUG: clients = {clients}, dataset = {dataset}, model = {model}")
        
        # Layout principale
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Titolo principale
        title_label = QLabel("Simulation Report")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Informazioni della simulazione (numero client, dataset, modello)
        info_text = f"Clients: {clients} | Dataset: {dataset} | Model: {model}"
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: black; font-size: 18px; font-weight: bold;")
        main_layout.addWidget(info_label)
        
        # Stato della simulazione ed il timer
        simulation_layout = QHBoxLayout()
        main_layout.addLayout(simulation_layout)
        self.simulation_label = QLabel("Simulation Running...")
        self.simulation_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        simulation_layout.addWidget(self.simulation_label)
        self.loading_gif = QLabel()
        self.movie = QMovie("path_to_your_loading.gif")
        self.loading_gif.setMovie(self.movie)
        self.movie.start()
        simulation_layout.addWidget(self.loading_gif)
        self.timer_label = QLabel("0 min : 0 s")
        self.timer_label.setStyleSheet("font-size: 16px;")
        simulation_layout.addWidget(self.timer_label)
        simulation_layout.addStretch()
        self.elapsed_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)
        
        # Griglia per i grafici (2 righe x 3 colonne)
        self.grid = QGridLayout()
        main_layout.addLayout(self.grid)
        
        # Se il numero di client è maggiore di 10, mostra un messaggio di avviso
        if self.data is not None and "Client ID" in self.data.columns and self.data["Client ID"].nunique() > 10:
            warning_label = QLabel("Preview is not available for more than 10 clients. Please refer to the .csv report.")
            warning_label.setAlignment(Qt.AlignCenter)
            warning_label.setStyleSheet("font-size: 16px; font-weight: bold; color: black;")
            self.grid.addWidget(warning_label, 0, 0, 1, 3)
        else:
            # Grafico 1: Average Times per Client
            fig1 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
            ax1 = fig1.add_subplot(111)
            chart1 = SingleChartWidget(title="Average Time per Client", figure=fig1)
            self.plot_average_times_per_client(ax1)
            chart1.download_button.clicked.connect(lambda _, f=fig1: self.save_figure_svg(f, "average_time_per_client"))
            self.add_chart_to_grid(chart1, 0, 0)
            
            # Grafico 2: Communication Time per FL Round
            fig2 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
            ax2 = fig2.add_subplot(111)
            chart2 = SingleChartWidget(title="Communication Time per Federated Learning Round", figure=fig2)
            self.plot_communication_time_per_round(ax2)
            chart2.download_button.clicked.connect(lambda _, f=fig2: self.save_figure_svg(f, "communication_time_per_round"))
            self.add_chart_to_grid(chart2, 0, 1)
            
            # Grafico 3: Training Time per FL Round
            fig3 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
            ax3 = fig3.add_subplot(111)
            chart3 = SingleChartWidget(title="Training Time per Federated Learning Round", figure=fig3)
            self.plot_training_time_per_round(ax3)
            chart3.download_button.clicked.connect(lambda _, f=fig3: self.save_figure_svg(f, "training_time_per_round"))
            self.add_chart_to_grid(chart3, 0, 2)
        
        # Seconda riga di grafici
        # Grafico 4: F1 Score per FL Round
        fig4 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax4 = fig4.add_subplot(111)
        chart4 = SingleChartWidget(title="F1 Score per Federated Learning Round", figure=fig4)
        self.plot_val_f1_per_round(ax4)
        chart4.download_button.clicked.connect(lambda _, f=fig4: self.save_figure_svg(f, "val_f1_per_round"))
        self.add_chart_to_grid(chart4, 1, 0)
        
        # Grafico 5: Validation Loss per FL Round
        fig5 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax5 = fig5.add_subplot(111)
        chart5 = SingleChartWidget(title="Validation Loss per Federated Learning Round", figure=fig5)
        self.plot_val_loss_per_round(ax5)
        chart5.download_button.clicked.connect(lambda _, f=fig5: self.save_figure_svg(f, "val_loss_per_round"))
        self.add_chart_to_grid(chart5, 1, 1)
        
        # Grafico 6: Validation Accuracy per FL Round
        fig6 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax6 = fig6.add_subplot(111)
        chart6 = SingleChartWidget(title="Validation Accuracy per Federated Learning Round", figure=fig6)
        self.plot_val_accuracy_per_round(ax6)
        chart6.download_button.clicked.connect(lambda _, f=fig6: self.save_figure_svg(f, "val_accuracy_per_round"))
        self.add_chart_to_grid(chart6, 1, 2)
        
        # Pulsanti in basso
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        
        self.close_button = QPushButton("Close")
        self.close_button.setCursor(Qt.PointingHandCursor)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        self.close_button.clicked.connect(self.close_project)
        button_layout.addWidget(self.close_button)
        
        self.download_button = QPushButton("Download .csv Report")
        self.download_button.setCursor(Qt.PointingHandCursor)
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)
        json_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        self.download_button.setIcon(json_icon)
        self.download_button.setIconSize(QSize(24, 24))
        self.download_button.clicked.connect(self.download_report)
        button_layout.addWidget(self.download_button)
    
    def add_chart_to_grid(self, chart_widget, row, col):
        self.grid.addWidget(chart_widget, row, col)
    
    # Funzioni dei grafici della prima riga
    def plot_average_times_per_client(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["Client ID", "Training Time", "Communication Time", "Total Client Time"]
        if all(col in self.data.columns for col in required_columns):
            grouped = self.data.groupby("Client ID", as_index=False)[
                ["Training Time", "Communication Time", "Total Client Time"]
            ].mean()
            import numpy as np
            x = grouped["Client ID"].astype(str)
            training = grouped["Training Time"]
            communication = grouped["Communication Time"]
            total = grouped["Total Client Time"]
            ind = np.arange(len(x))
            width = 0.25
            ax.bar(ind - width, training, width, label="Training Time", color=PASTEL_COLORS[0])
            ax.bar(ind, communication, width, label="Communication Time", color=PASTEL_COLORS[1])
            ax.bar(ind + width, total, width, label="Total Time", color=PASTEL_COLORS[2])
            ax.set_xlabel("Client ID", fontsize=10)
            ax.set_ylabel("Time", fontsize=10)
            ax.set_title("Average Time per Client", fontsize=10)
            ax.set_xticks(ind)
            ax.set_xticklabels(x, fontsize=8, rotation=45)
            ax.legend(fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
        else:
            ax.set_title("Missing columns (Client ID, Training Time, etc.)", fontsize=10)
    
    def plot_communication_time_per_round(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["FL Round", "Client ID", "Communication Time"]
        if all(col in self.data.columns for col in required_columns):
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", 
                                        values="Communication Time", aggfunc='mean')
            pivot.sort_index(inplace=True)
            pivot = pivot.fillna(method='ffill')
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1,
                        label=str(column),
                        color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
            ax.set_xlabel("Federated Learning Round", fontsize=10)
            ax.set_ylabel("Communication Time (s)", fontsize=10)
            ax.set_title("Communication Time per Federated Learning Round", fontsize=10)
            ax.legend(fontsize=8, title_fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Communication Time)", fontsize=10)
    
    def plot_training_time_per_round(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["FL Round", "Client ID", "Training Time"]
        if all(col in self.data.columns for col in required_columns):
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Training Time", aggfunc='mean')
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1, 
                        label=f"{column}",
                        color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
            ax.set_xlabel("Federated Learning Round", fontsize=10)
            ax.set_ylabel("Training Time (s)", fontsize=10)
            ax.set_title("Training Time per Federated Learning Round", fontsize=10)
            ax.legend(fontsize=8, title_fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Training Time)", fontsize=10)
    
    # Funzioni dei grafici della seconda riga
    def plot_val_f1_per_round(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["FL Round", "Client ID", "Val F1"]
        if all(col in self.data.columns for col in required_columns):
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Val F1", aggfunc='mean')
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1,
                        label="F1 Score",
                        color="#84A59E")
            ax.set_xlabel("Federated Learning Round", fontsize=10)
            ax.set_ylabel("F1 Score", fontsize=10)
            ax.set_title("F1 Score per Federated Learning Round", fontsize=10)
            ax.legend(fontsize=8, title_fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8)
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Val F1)", fontsize=10)
    
    def plot_val_loss_per_round(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["FL Round", "Client ID", "Val Loss"]
        if all(col in self.data.columns for col in required_columns):
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Val Loss", aggfunc='mean')
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1,
                        label="Val. Loss",
                        color="#F28582")
            ax.set_xlabel("Federated Learning Round", fontsize=10)
            ax.set_ylabel("Val. Loss", fontsize=10)
            ax.set_title("Validation Loss per Federated Learning Round", fontsize=10)
            ax.legend(fontsize=8, title_fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8)
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Val Loss)", fontsize=10)
    
    def plot_val_accuracy_per_round(self, ax):
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return
        required_columns = ["FL Round", "Client ID", "Val Accuracy"]
        if all(col in self.data.columns for col in required_columns):
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Val Accuracy", aggfunc='mean')
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1,
                        label="Val. Accuracy",
                        color="#84A59E")
            ax.set_xlabel("Federated Learning Round", fontsize=10)
            ax.set_ylabel("Val. Accuracy", fontsize=10)
            ax.set_title("Validation Accuracy per Federated Learning Round", fontsize=10)
            ax.legend(fontsize=8, title_fontsize=8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8)
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Val Accuracy)", fontsize=10)
    
    # Funzioni per salvare e chiudere
    def save_figure_svg(self, figure, default_name="chart"):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart as SVG",
            f"{default_name}.svg",
            "SVG Files (*.svg);;All Files (*)",
            options=options
        )
        if file_name:
            figure.savefig(file_name, format='svg')
    
    def close_project(self):
        sys.exit(0)
    
    def download_report(self):
        if self.data is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV Report",
                "",
                "CSV Files (*.csv);;All Files (*)",
                options=options
            )
            if file_name:
                self.data.to_csv(file_name, index=False)
    
    def update_timer(self):
        self.elapsed_time += 1
        minutes = self.elapsed_time // 60
        seconds = self.elapsed_time % 60
        self.timer_label.setText(f"{minutes} min : {seconds} s")
    
    def finish_simulation(self):
        self.timer.stop()
        self.movie.stop()
        self.simulation_label.setText("Simulation Completed")
