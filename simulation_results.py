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
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QMovie

# Imposta i parametri di Matplotlib globalmente
plt.rcParams.update({
    'font.size': 10,               # Dimensione del font predefinita
    'font.family': 'Helvetica',    # Sostituisci con un font esistente
    'axes.titlesize': 10,          # Uniforma la dimensione del titolo dell'asse
    'axes.labelsize': 10,          # Dimensione delle etichette degli assi
    'legend.fontsize': 8,          # Dimensione della legenda
    'xtick.labelsize': 8,          # Dimensione dei tick dell'asse x
    'ytick.labelsize': 8           # Dimensione dei tick dell'asse y
})

# Definisci una palette di colori pastello
PASTEL_COLORS = [
    '#AEC6CF', '#FFB347', '#77DD77', '#CFCFC4', '#FFD1DC',
    '#B39EB5', '#FF6961', '#FDFD96', '#CB99C9', '#C23B22'
]

class SingleChartWidget(QWidget):
    """
    Un widget personalizzato che contiene:
      - Un titolo (QLabel) centrato in alto
      - Un QToolButton per il download posizionato alla destra del titolo
      - Un canvas Matplotlib per il grafico
    """
    def __init__(self, parent=None, title="Chart", figure=None):
        super().__init__(parent)

        # Layout principale verticale (titolo+icona in alto, poi il canvas)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Ridotti i margini per aumentare lo spazio
        self.setLayout(layout)

        # Riga orizzontale con: QLabel TITOLO (centrato) + QToolButton (icona di download)
        title_layout = QHBoxLayout()
        layout.addLayout(title_layout)

        # Spacer iniziale per spingere il titolo al centro
        title_layout.addStretch()

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)  # Centra il testo nel QLabel
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")  # Stile del titolo
        title_layout.addWidget(self.title_label)

        # Spacer tra il titolo e il pulsante di download
        title_layout.addStretch()

        self.download_button = QToolButton()
        self.download_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.download_button.setToolTip("Save this chart as SVG")
        self.download_button.setCursor(Qt.PointingHandCursor)
        self.download_button.setStyleSheet("padding: 5px;")  # Padding per migliorare l'aspetto
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
        self.resize(1600, 1000)  # Un po' più grande di 1200x800

        # Se esiste data_file, carichiamo i dati.
        self.data = None
        if data_file and os.path.exists(data_file):
            # Se il tuo file è Excel, usa pd.read_excel
            self.data = pd.read_csv(data_file)

        # Layout principale verticale
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # --- Aggiunta della sezione di simulazione in esecuzione ---
        simulation_layout = QHBoxLayout()
        main_layout.addLayout(simulation_layout)

        # Etichetta "Simulation Running..."
        self.simulation_label = QLabel("Simulation Running...")
        self.simulation_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        simulation_layout.addWidget(self.simulation_label)

        # Gif animata
        self.loading_gif = QLabel()
        self.movie = QMovie("path_to_your_loading.gif")  # Sostituisci con il percorso della tua gif
        self.loading_gif.setMovie(self.movie)
        self.movie.start()
        simulation_layout.addWidget(self.loading_gif)

        # Timer
        self.timer_label = QLabel("0 min : 0 s")
        self.timer_label.setStyleSheet("font-size: 16px;")
        simulation_layout.addWidget(self.timer_label)

        # Spacer per allineare gli elementi a sinistra
        simulation_layout.addStretch()

        self.elapsed_time = 0  # Tempo trascorso in secondi
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Aggiorna ogni secondo

        # Griglia 3x3 per i 9 grafici
        self.grid = QGridLayout()
        main_layout.addLayout(self.grid)

        # Creiamo 3 figure separate (ognuna con il proprio SingleChartWidget)
        self.figures = []
        self.chart_widgets = []

        # -- Grafico 1: Tempo Medio per Client ID ---------------------
        fig1 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax1 = fig1.add_subplot(111)
        chart1 = SingleChartWidget(title="Average Times per Client", figure=fig1)
        self.plot_average_times_per_client(ax1)
        chart1.download_button.clicked.connect(lambda _, f=fig1: self.save_figure_svg(f, "average_times_per_client"))
        self.add_chart_to_grid(chart1, 0, 0)

        # -- Grafico 2: Tempo di Comunicazione vs FL Rounds -------
        fig2 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax2 = fig2.add_subplot(111)
        chart2 = SingleChartWidget(title="Communication Time per FL Round", figure=fig2)
        self.plot_communication_time_per_round(ax2)
        chart2.download_button.clicked.connect(lambda _, f=fig2: self.save_figure_svg(f, "communication_time_per_round"))
        self.add_chart_to_grid(chart2, 0, 1)

        # -- Grafico 3: Tempo di Training vs FL Rounds ------------
        fig3 = plt.Figure(figsize=(6, 4), facecolor="white", constrained_layout=True)
        ax3 = fig3.add_subplot(111)
        chart3 = SingleChartWidget(title="Training Time per FL Round", figure=fig3)
        self.plot_training_time_per_round(ax3)
        chart3.download_button.clicked.connect(lambda _, f=fig3: self.save_figure_svg(f, "training_time_per_round"))
        self.add_chart_to_grid(chart3, 0, 2)

        # -- Grafici 4-9: Placeholder per futuri implementazioni --
        for row in range(1, 3):
            for col in range(3):
                fig = plt.Figure(figsize=(6,4), facecolor="white", constrained_layout=True)
                ax = fig.add_subplot(111)
                chart = SingleChartWidget(title=f"Grafico {row*3 + col +1}", figure=fig)
                # Placeholder: nessuna funzione di plotting
                ax.text(0.5, 0.5, "In attesa di implementazione", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_axis_off()
                chart.download_button.setEnabled(False)  # Disabilita il pulsante di download
                self.add_chart_to_grid(chart, row, col)

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
        # Aggiungi l'icona del file sul pulsante download the csv report
        json_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        self.download_button.setIcon(json_icon)
        self.download_button.setIconSize(QSize(24, 24))
        
        self.download_button.clicked.connect(self.download_report)
        button_layout.addWidget(self.download_button)

    # -------------------- FUNZIONI DI SUPPORTO ----------------------------
    
    def add_chart_to_grid(self, chart_widget, row, col):
        """Inserisce un SingleChartWidget nella griglia row,col."""
        self.grid.addWidget(chart_widget, row, col)

    def plot_average_times_per_client(self, ax):
        """
        Grafico 1: Tempo Medio per Client ID
        Y: Tempo (Training Time, Communication Time, Total Client Time)
        X: Client ID
        """
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return

        required_columns = ["Client ID", "Training Time", "Communication Time", "Total Client Time"]
        if all(col in self.data.columns for col in required_columns):
            # Calcola la media dei tempi per ogni client
            grouped = self.data.groupby("Client ID", as_index=False)[
                ["Training Time", "Communication Time", "Total Client Time"]
            ].mean()

            x = grouped["Client ID"].astype(str)
            training = grouped["Training Time"]
            communication = grouped["Communication Time"]
            total = grouped["Total Client Time"]

            # Indici per le barre
            import numpy as np
            ind = np.arange(len(x))  # L'indice dei client
            width = 0.25  # Larghezza delle barre

            # Assegna colori pastello
            ax.bar(ind - width, training, width, label="Training Time", color=PASTEL_COLORS[0])
            ax.bar(ind, communication, width, label="Communication Time", color=PASTEL_COLORS[1])
            ax.bar(ind + width, total, width, label="Total Time", color=PASTEL_COLORS[2])

            ax.set_xlabel("Client ID", fontsize=10)
            ax.set_ylabel("Time", fontsize=10)
            ax.set_title("Average Times per Client", fontsize=10)  # Uniformato a 10
            ax.set_xticks(ind)
            ax.set_xticklabels(x, fontsize=8, rotation=45)
            ax.legend(fontsize=8)

            # Rimuovi le griglie di sfondo
            ax.grid(False)

            # Imposta le dimensioni dei tick
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Utilizza constrained_layout per evitare sovrapposizioni
            # Non è necessario chiamare tight_layout() qui
        else:
            ax.set_title("Missing columns (Client ID, Training Time, etc.)", fontsize=10)

    def plot_communication_time_per_round(self, ax):
        """
        Grafico 2: Tempo di Comunicazione vs FL Rounds
        Y: Tempo di Comunicazione
        X: FL Rounds
        Tipo di Grafico: Line plot multiplo per ogni client
        """
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return

        required_columns = ["FL Round", "Client ID", "Communication Time"]
        if all(col in self.data.columns for col in required_columns):
            # Pivot dei dati per avere FL Round come indice e Client ID come colonne
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Communication Time", aggfunc='mean')

            # Plot delle linee per ogni client con colori pastello
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1, label=str(column), color=PASTEL_COLORS[i % len(PASTEL_COLORS)])

            ax.set_xlabel("FL Rounds", fontsize=10)
            ax.set_ylabel("Communication Time", fontsize=10)
            ax.set_title("Communication Time per FL Round", fontsize=10)  # Uniformato a 10
            ax.legend(title="Client ID", fontsize=8, title_fontsize=8)
            
            # Rimuovi le griglie di sfondo
            ax.grid(False)

            # Imposta le dimensioni dei tick
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Imposta i tick dell'asse X a valori interi
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8, rotation=45)

            # Utilizza constrained_layout per evitare sovrapposizioni
            # Non è necessario chiamare tight_layout() qui
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Communication Time)", fontsize=10)

    def plot_training_time_per_round(self, ax):
        """
        Grafico 3: Tempo di Training vs FL Rounds
        Y: Tempo di Training
        X: FL Rounds
        Tipo di Grafico: Line plot multiplo per ogni client
        """
        if self.data is None:
            ax.set_title("No data", fontsize=10)
            return

        required_columns = ["FL Round", "Client ID", "Training Time"]
        if all(col in self.data.columns for col in required_columns):
            # Pivot dei dati per avere FL Round come indice e Client ID come colonne
            pivot = self.data.pivot_table(index="FL Round", columns="Client ID", values="Training Time", aggfunc='mean')

            # Plot delle linee per ogni client con colori pastello
            for i, column in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[column], marker='o', linewidth=1, label=str(column), color=PASTEL_COLORS[i % len(PASTEL_COLORS)])

            ax.set_xlabel("FL Rounds", fontsize=10)
            ax.set_ylabel("Training Time", fontsize=10)
            ax.set_title("Training Time per FL Round", fontsize=10)  # Uniformato a 10
            ax.legend(title="Client ID", fontsize=8, title_fontsize=8)
            
            # Rimuovi le griglie di sfondo
            ax.grid(False)

            # Imposta le dimensioni dei tick
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Imposta i tick dell'asse X a valori interi
            if not pivot.index.is_integer():
                ax.set_xticks(pivot.index.astype(int))
            else:
                ax.set_xticks(pivot.index)
            ax.set_xticklabels(pivot.index.astype(int), fontsize=8, rotation=45)

            # Utilizza constrained_layout per evitare sovrapposizioni
            # Non è necessario chiamare tight_layout() qui
        else:
            ax.set_title("Missing columns (FL Round, Client ID, Training Time)", fontsize=10)

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

    # --- Funzione per aggiornare il timer ---
    def update_timer(self):
        self.elapsed_time += 1
        minutes = self.elapsed_time // 60
        seconds = self.elapsed_time % 60
        self.timer_label.setText(f"{minutes} min : {seconds} s")

    # --- Funzione per fermare il timer e la gif alla fine della simulazione ---
    def finish_simulation(self):
        self.timer.stop()
        self.movie.stop()
        self.simulation_label.setText("Simulation Completed")