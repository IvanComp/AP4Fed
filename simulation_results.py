# simulation_results.py

import sys
import os
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
from PyQt5.QtCore import Qt

class SingleChartWidget(QWidget):
    """
    Un widget personalizzato che contiene:
      - Un titolo (QLabel) con accanto un QToolButton per il download
      - Un canvas Matplotlib per il grafico
    """
    def __init__(self, parent=None, title="Chart", figure=None):
        super().__init__(parent)

        # Layout principale verticale (titolo+icona in alto, poi il canvas)
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        # Riga orizzontale con: LABEL TITOLO + QToolButton (icona di download)
        title_layout = QHBoxLayout()
        layout.addLayout(title_layout)

        self.title_label = QLabel(title)
        # Font un po' più grande o grassetto se vuoi
        # self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(self.title_label)

        self.download_button = QToolButton()
        self.download_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.download_button.setToolTip("Save this chart as SVG")
        self.download_button.setCursor(Qt.PointingHandCursor)
        title_layout.addWidget(self.download_button)

        # Canvas Matplotlib
        self.canvas = FigureCanvas(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

class SimulationResults(QDialog):
    def __init__(self, data_file=None):
        super().__init__()
        self.setWindowTitle("Simulation Results")
        self.setStyleSheet("background-color: white;")
        self.resize(1400, 900)  # Un po' più grande di 1200x800

        # Se esiste data_file, carichiamo i dati.
        self.data = None
        if data_file and os.path.exists(data_file):
            # Se il tuo file è Excel, usa pd.read_excel
            self.data = pd.read_csv(data_file)

        # Layout principale verticale
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Griglia 3x3 per i 9 grafici
        self.grid = QGridLayout()
        main_layout.addLayout(self.grid)

        # Creiamo 9 figure separate (ognuna con il proprio SingleChartWidget)
        self.figures = []
        self.chart_widgets = []

        # -- Riga 1: Dedicata ai Clients ---------------------
        # In ciascun grafico confrontiamo training time, communication time, total round time
        # su base "Client ID" (o come si chiama la colonna del tuo CSV).
        # Qui diamo solo un esempio di come potresti gestire i dati.

        # Primo grafico (riga=0,col=0)
        fig1 = plt.Figure(figsize=(4,3), facecolor="white")
        ax1 = fig1.add_subplot(111)
        chart1 = SingleChartWidget(title="(1,1) Clients - Times", figure=fig1)
        # Disegno
        self.plot_clients_times(ax1, kind="line")  
        # Callback download
        chart1.download_button.clicked.connect(lambda _, f=fig1: self.save_figure_svg(f, "(1,1)_clients_times"))
        self.add_chart_to_grid(chart1, 0, 0)

        # Secondo grafico (riga=0,col=1)
        fig2 = plt.Figure(figsize=(4,3), facecolor="white")
        ax2 = fig2.add_subplot(111)
        chart2 = SingleChartWidget(title="(1,2) Clients - Times (Bar)", figure=fig2)
        self.plot_clients_times(ax2, kind="bar")  
        chart2.download_button.clicked.connect(lambda _, f=fig2: self.save_figure_svg(f, "(1,2)_clients_times_bar"))
        self.add_chart_to_grid(chart2, 0, 1)

        # Terzo grafico (riga=0,col=2)
        fig3 = plt.Figure(figsize=(4,3), facecolor="white")
        ax3 = fig3.add_subplot(111)
        chart3 = SingleChartWidget(title="(1,3) Clients - Times (Stacked?)", figure=fig3)
        self.plot_clients_times(ax3, kind="stacked")  
        chart3.download_button.clicked.connect(lambda _, f=fig3: self.save_figure_svg(f, "(1,3)_clients_times_stacked"))
        self.add_chart_to_grid(chart3, 0, 2)

        # -- Riga 2: Dedicata ai tempi (FL Rounds) -----------
        # Stessi 3 parametri (training time, comm. time, total round time),
        # ma su base round (x-axis = round).
        
        fig4 = plt.Figure(figsize=(4,3), facecolor="white")
        ax4 = fig4.add_subplot(111)
        chart4 = SingleChartWidget(title="(2,1) Rounds - Times", figure=fig4)
        self.plot_rounds_times(ax4, kind="line")
        chart4.download_button.clicked.connect(lambda _, f=fig4: self.save_figure_svg(f, "(2,1)_rounds_times"))
        self.add_chart_to_grid(chart4, 1, 0)

        fig5 = plt.Figure(figsize=(4,3), facecolor="white")
        ax5 = fig5.add_subplot(111)
        chart5 = SingleChartWidget(title="(2,2) Rounds - Times (Bar)", figure=fig5)
        self.plot_rounds_times(ax5, kind="bar")
        chart5.download_button.clicked.connect(lambda _, f=fig5: self.save_figure_svg(f, "(2,2)_rounds_times_bar"))
        self.add_chart_to_grid(chart5, 1, 1)

        fig6 = plt.Figure(figsize=(4,3), facecolor="white")
        ax6 = fig6.add_subplot(111)
        chart6 = SingleChartWidget(title="(2,3) Rounds - Times (Mix)", figure=fig6)
        self.plot_rounds_times(ax6, kind="mix")
        chart6.download_button.clicked.connect(lambda _, f=fig6: self.save_figure_svg(f, "(2,3)_rounds_times_mix"))
        self.add_chart_to_grid(chart6, 1, 2)

        # -- Riga 3: Dedicata alla qualità del modello -------
        # Confrontiamo Val Loss, Val Accuracy, Val F1 su base round.
        
        fig7 = plt.Figure(figsize=(4,3), facecolor="white")
        ax7 = fig7.add_subplot(111)
        chart7 = SingleChartWidget(title="(3,1) Model Quality (Line)", figure=fig7)
        self.plot_model_quality(ax7, kind="line")
        chart7.download_button.clicked.connect(lambda _, f=fig7: self.save_figure_svg(f, "(3,1)_model_quality_line"))
        self.add_chart_to_grid(chart7, 2, 0)

        fig8 = plt.Figure(figsize=(4,3), facecolor="white")
        ax8 = fig8.add_subplot(111)
        chart8 = SingleChartWidget(title="(3,2) Model Quality (Bar)", figure=fig8)
        self.plot_model_quality(ax8, kind="bar")
        chart8.download_button.clicked.connect(lambda _, f=fig8: self.save_figure_svg(f, "(3,2)_model_quality_bar"))
        self.add_chart_to_grid(chart8, 2, 1)

        fig9 = plt.Figure(figsize=(4,3), facecolor="white")
        ax9 = fig9.add_subplot(111)
        chart9 = SingleChartWidget(title="(3,3) Model Quality (Mix)", figure=fig9)
        self.plot_model_quality(ax9, kind="mix")
        chart9.download_button.clicked.connect(lambda _, f=fig9: self.save_figure_svg(f, "(3,3)_model_quality_mix"))
        self.add_chart_to_grid(chart9, 2, 2)

        # -- Layout finale con i pulsanti in basso -----------
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
        self.download_button.clicked.connect(self.download_report)
        button_layout.addWidget(self.download_button)

    # -------------------- FUNZIONI DI SUPPORTO ----------------------------

    def add_chart_to_grid(self, chart_widget, row, col):
        """Inserisce un SingleChartWidget nella griglia row,col."""
        self.grid.addWidget(chart_widget, row, col)

    def plot_clients_times(self, ax, kind="line"):
        """
        Confronta su un unico grafico:
          - Training Time
          - Communication Time
          - Total Round Time
        raggruppando per 'Client ID' (o come si chiama da te).
        """
        if self.data is None:
            ax.set_title("No data")
            return

        # Adatta i nomi delle colonne al tuo CSV
        if all(col in self.data.columns for col in ["Client ID","Training Time","Communication Time","Total Round Time"]):
            grouped = self.data.groupby("Client ID", as_index=False)[
                ["Training Time","Communication Time","Total Round Time"]
            ].mean()
            x_vals = grouped["Client ID"]

            if kind == "line":
                ax.plot(x_vals, grouped["Training Time"], marker='o', label="Training")
                ax.plot(x_vals, grouped["Communication Time"], marker='o', label="Comm")
                ax.plot(x_vals, grouped["Total Round Time"], marker='o', label="Total")
            elif kind == "bar":
                # Esempio di bar affiancate
                width = 0.25
                import numpy as np
                idx = np.arange(len(x_vals))
                ax.bar(idx - width, grouped["Training Time"], width=width, label="Training")
                ax.bar(idx, grouped["Communication Time"], width=width, label="Comm")
                ax.bar(idx + width, grouped["Total Round Time"], width=width, label="Total")
                ax.set_xticks(idx)
                ax.set_xticklabels(x_vals.astype(str))
            elif kind == "stacked":
                # Esempio di bar stacked
                import numpy as np
                idx = np.arange(len(x_vals))
                ax.bar(idx, grouped["Training Time"], label="Training")
                ax.bar(idx, grouped["Communication Time"], bottom=grouped["Training Time"], label="Comm")
                bottom_sum = grouped["Training Time"] + grouped["Communication Time"]
                ax.bar(idx, grouped["Total Round Time"], bottom=bottom_sum, label="Total")
                ax.set_xticks(idx)
                ax.set_xticklabels(x_vals.astype(str))

            ax.set_xlabel("Clients")
            ax.set_ylabel("Time")
            ax.legend()
            ax.set_title("Clients Times")

        else:
            ax.set_title("Missing columns (Client ID, Training Time, etc.)")

    def plot_rounds_times(self, ax, kind="line"):
        """
        Confronta in un unico grafico:
          - Training Time
          - Communication Time
          - Total Round Time
        raggruppando per 'FL Round'.
        """
        if self.data is None:
            ax.set_title("No data")
            return

        # Adatta i nomi delle colonne
        if all(col in self.data.columns for col in ["FL Round","Training Time","Communication Time","Total Round Time"]):
            grouped = self.data.groupby("FL Round", as_index=False)[
                ["Training Time","Communication Time","Total Round Time"]
            ].mean()

            x_vals = grouped["FL Round"]
            # Se vuoi forzare i tick a numeri naturali:
            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_vals.astype(int))

            if kind == "line":
                ax.plot(x_vals, grouped["Training Time"], marker='o', label="Training")
                ax.plot(x_vals, grouped["Communication Time"], marker='o', label="Comm")
                ax.plot(x_vals, grouped["Total Round Time"], marker='o', label="Total")
            elif kind == "bar":
                import numpy as np
                idx = np.arange(len(x_vals))
                width = 0.25
                ax.bar(idx - width, grouped["Training Time"], width=width, label="Training")
                ax.bar(idx, grouped["Communication Time"], width=width, label="Comm")
                ax.bar(idx + width, grouped["Total Round Time"], width=width, label="Total")
                ax.set_xticks(idx)
                ax.set_xticklabels(x_vals.astype(int))
            elif kind == "mix":
                # Esempio "mix": line per Training, bar per Communication e Total
                import numpy as np
                idx = np.arange(len(x_vals))
                width = 0.3
                ax.plot(x_vals, grouped["Training Time"], marker='o', label="Training (line)")
                ax.bar(idx, grouped["Communication Time"], width=width, label="Comm (bar)")
                ax.bar(idx+width, grouped["Total Round Time"], width=width, label="Total (bar)")
                ax.set_xticks(idx+width/2)
                ax.set_xticklabels(x_vals.astype(int))

            ax.set_xlabel("FL Rounds")
            ax.set_ylabel("Time")
            ax.legend()
            ax.set_title("Rounds Times")
        else:
            ax.set_title("Missing columns (FL Round, Training Time, etc.)")

    def plot_model_quality(self, ax, kind="line"):
        """
        Confronta Val Loss, Val Accuracy, Val F1 su base 'FL Round'.
        """
        if self.data is None:
            ax.set_title("No data")
            return

        # Adatta i nomi delle colonne
        if all(col in self.data.columns for col in ["FL Round","Val Loss","Val Accuracy","Val F1"]):
            grouped = self.data.groupby("FL Round", as_index=False)[
                ["Val Loss","Val Accuracy","Val F1"]
            ].mean()
            x_vals = grouped["FL Round"]
            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_vals.astype(int))

            if kind == "line":
                ax.plot(x_vals, grouped["Val Loss"], marker='o', label="Val Loss")
                ax.plot(x_vals, grouped["Val Accuracy"], marker='o', label="Val Accuracy")
                ax.plot(x_vals, grouped["Val F1"], marker='o', label="Val F1")
            elif kind == "bar":
                import numpy as np
                idx = np.arange(len(x_vals))
                width = 0.25
                ax.bar(idx - width, grouped["Val Loss"], width=width, label="Val Loss")
                ax.bar(idx, grouped["Val Accuracy"], width=width, label="Val Accuracy")
                ax.bar(idx + width, grouped["Val F1"], width=width, label="Val F1")
                ax.set_xticks(idx)
                ax.set_xticklabels(x_vals.astype(int))
            elif kind == "mix":
                import numpy as np
                idx = np.arange(len(x_vals))
                width = 0.3
                ax.plot(x_vals, grouped["Val Loss"], marker='o', label="Val Loss (line)")
                ax.bar(idx, grouped["Val Accuracy"], width=width, label="Val Accuracy (bar)")
                ax.bar(idx+width, grouped["Val F1"], width=width, label="Val F1 (bar)")
                ax.set_xticks(idx+width/2)
                ax.set_xticklabels(x_vals.astype(int))

            ax.set_xlabel("FL Rounds")
            ax.set_ylabel("Quality Metric")
            ax.legend()
            ax.set_title("Model Quality")
        else:
            ax.set_title("Missing columns (FL Round, Val Loss, Val Accuracy, Val F1)")

    def save_figure_svg(self, figure, default_name="chart"):
        """
        Salva la singola figura (un subplot) in formato SVG, chiedendo dove salvare.
        """
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
