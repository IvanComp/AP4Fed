
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "simulation_type": "Local",
    "rounds": 10,
    "clients": 3,
    "adaptation": "None",
    "patterns": {
        "client_registry": {"enabled": True, "params": {}},
        "client_selector": {"enabled": False, "params": {"selection_criteria": "CPU", "selection_value": "2"}},
        "client_cluster": {"enabled": False, "params": {"clustering_criteria": "CPU", "selection_value": "2"}},
        "message_compressor": {"enabled": False, "params": {}},
        "model_co-versioning_registry": {"enabled": False, "params": {}},
        "multi-task_model_trainer": {"enabled": False, "params": {}},
        "heterogeneous_data_handler": {"enabled": False, "params": {}}
    },
    "client_details": [
        {"client_id": 1, "cpu": 2, "data_distribution_type": "IID", "data_persistence_type": "Same Data", "delay_combobox": "No"},
        {"client_id": 2, "cpu": 2, "data_distribution_type": "IID", "data_persistence_type": "Same Data", "delay_combobox": "No"},
        {"client_id": 3, "cpu": 2, "data_distribution_type": "IID", "data_persistence_type": "Same Data", "delay_combobox": "No"}
    ]
}

PATTERNS_LIST = [
    "client_registry",
    "client_selector",
    "client_cluster",
    "message_compressor",
    "model_co-versioning_registry",
    "multi-task_model_trainer",
    "heterogeneous_data_handler"
]

TASKS = ["Image Classification", "Text Classification"]

TASK_DATASETS = {
    "Image Classification": ["CIFAR10", "MNIST", "FashionMNIST", "KMNIST", "ImageNet100", "OXFORDIIITPET"],
    "Text Classification": ["AG_NEWS"]
}

TASK_MODELS = {
    "Image Classification": ["CNN 16k", "CNN 64k", "CNN 256k", "resnet18", "resnet50", "mobilenet_v2"],
    "Text Classification": ["MLP"]
}

DISTR_TYPES = ["IID", "Non-IID"]



class ConfiguratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LightAP4FED Configurator")
        self.root.geometry("1000x850")
        
        self.config = self.load_config()
        self.pattern_vars = {}
        self.pattern_params = {} # Store pattern-specific parameters
        
        self.create_widgets()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    # Merge with default to ensure missing keys are present
                    if "patterns" not in data: data["patterns"] = DEFAULT_CONFIG["patterns"]
                    if "client_details" not in data: data["client_details"] = DEFAULT_CONFIG["client_details"]
                    return data
            except:
                pass
        return DEFAULT_CONFIG

    def create_widgets(self):
        # Create a container frame
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas, padding="20")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(self.scrollable_frame, text="AP4FED Configuration", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Simulation Settings
        settings_frame = ttk.LabelFrame(self.scrollable_frame, text="Simulation Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

        ttk.Label(settings_frame, text="Rounds:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.rounds_var = tk.StringVar(value=str(self.config.get("rounds", 10)))
        ttk.Entry(settings_frame, textvariable=self.rounds_var, width=10).grid(row=0, column=1, sticky=tk.W)

        # Client Management Buttons
        mgr_frame = ttk.Frame(settings_frame)
        mgr_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(mgr_frame, text="+ Add Client", command=self.add_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(mgr_frame, text="- Remove Client", command=self.remove_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(mgr_frame, text="Bulk Add (5)", command=lambda: self.bulk_add(5)).pack(side=tk.LEFT, padx=5)
        
        self.clients_var = tk.StringVar(value=str(len(self.config.get("client_details", []))))
        ttk.Label(settings_frame, text="Total Clients:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Label(settings_frame, textvariable=self.clients_var).grid(row=2, column=1, sticky=tk.W)

        # Global Client Settings
        client_frame = ttk.LabelFrame(self.scrollable_frame, text="Global Client Settings", padding="10")
        client_frame.pack(fill=tk.X, pady=5)
        
        # ... (rest of the fields in scrollable_frame)
        default_ds = "CIFAR10"
        default_model = "CNN 16k"
        
        if self.config.get("client_details") and len(self.config["client_details"]) > 0:
            c0 = self.config["client_details"][0]
            default_ds = c0.get("dataset", default_ds)
            default_model = c0.get("model", default_model)

        ttk.Label(client_frame, text="ML Task:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        initial_task = "Image Classification"
        if default_ds in TASK_DATASETS.get("Text Classification", []):
            initial_task = "Text Classification"
        self.task_var = tk.StringVar(value=initial_task)
        self.task_cb = ttk.Combobox(client_frame, textvariable=self.task_var, values=TASKS, state="readonly")
        self.task_cb.grid(row=0, column=1, sticky=tk.W)
        self.task_cb.bind("<<ComboboxSelected>>", self.update_options)

        ttk.Label(client_frame, text="Dataset:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.dataset_var = tk.StringVar(value=default_ds)
        self.ds_cb = ttk.Combobox(client_frame, textvariable=self.dataset_var, state="readonly")
        self.ds_cb.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(client_frame, text="Model:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value=default_model)
        self.model_cb = ttk.Combobox(client_frame, textvariable=self.model_var, state="readonly")
        self.model_cb.grid(row=2, column=1, sticky=tk.W)

        self.client_config_frame = ttk.LabelFrame(self.scrollable_frame, text="Individual Client Overrides", padding="10")
        self.client_config_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.client_config_frame, text="Select Client:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.selected_client_idx = tk.IntVar(value=1)
        self.client_selector_cb = ttk.Combobox(self.client_config_frame, textvariable=self.selected_client_idx, state="readonly")
        self.client_selector_cb.grid(row=0, column=1, sticky=tk.W)
        self.client_selector_cb.bind("<<ComboboxSelected>>", self.on_client_selected)

        ttk.Label(self.client_config_frame, text="CPU:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.client_cpu_var = tk.StringVar()
        ttk.Entry(self.client_config_frame, textvariable=self.client_cpu_var, width=5).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(self.client_config_frame, text="Data Distr:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.client_distr_var = tk.StringVar()
        self.client_dist_cb = ttk.Combobox(self.client_config_frame, textvariable=self.client_distr_var, values=DISTR_TYPES, state="readonly")
        self.client_dist_cb.grid(row=2, column=1, sticky=tk.W)

        ttk.Label(self.client_config_frame, text="Persistence:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.client_persist_var = tk.StringVar()
        self.client_persist_cb = ttk.Combobox(self.client_config_frame, textvariable=self.client_persist_var, values=["Same Data", "New Data", "Remove Data"], state="readonly")
        self.client_persist_cb.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(self.client_config_frame, text="Network Delay:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.client_delay_var = tk.StringVar()
        self.client_delay_cb = ttk.Combobox(self.client_config_frame, textvariable=self.client_delay_var, values=["No", "Yes"], state="readonly")
        self.client_delay_cb.grid(row=4, column=1, sticky=tk.W)

        ttk.Button(self.client_config_frame, text="Apply Override to Client", command=self.apply_to_client).grid(row=5, column=0, columnspan=2, pady=5)

        self.update_options(None)
        if default_ds in self.ds_cb['values']: self.ds_cb.set(default_ds)
        if default_model in self.model_cb['values']: self.model_cb.set(default_model)

        patterns_frame = ttk.LabelFrame(self.scrollable_frame, text="Architectural Patterns", padding="10")
        patterns_frame.pack(fill=tk.X, pady=5)

        # Pattern Controls
        row = 0
        for pattern in PATTERNS_LIST:
            enabled = self.config.get("patterns", {}).get(pattern, {}).get("enabled", False)
            var = tk.BooleanVar(value=enabled)
            self.pattern_vars[pattern] = var
            clean_name = pattern.replace("_", " ").title()
            ttk.Checkbutton(patterns_frame, text=clean_name, variable=var).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Additional params for specific patterns
            if pattern == "client_selector":
                p_conf = self.config.get("patterns", {}).get(pattern, {}).get("params", {})
                
                ttk.Label(patterns_frame, text="Criteria:").grid(row=row, column=1, sticky=tk.W)
                crit_var = tk.StringVar(value=p_conf.get("selection_criteria", "CPU"))
                cb = ttk.Combobox(patterns_frame, textvariable=crit_var, values=["CPU", "RAM"], width=10, state="readonly")
                cb.grid(row=row, column=2, sticky=tk.W)
                
                ttk.Label(patterns_frame, text="Threshold:").grid(row=row, column=3, sticky=tk.W)
                val_var = tk.StringVar(value=str(p_conf.get("selection_value", "2")))
                ent = ttk.Entry(patterns_frame, textvariable=val_var, width=5)
                ent.grid(row=row, column=4, sticky=tk.W)
                
                self.pattern_params[pattern] = {"selection_criteria": crit_var, "selection_value": val_var}

            elif pattern == "client_cluster":
                p_conf = self.config.get("patterns", {}).get(pattern, {}).get("params", {})
                
                ttk.Label(patterns_frame, text="Criteria:").grid(row=row, column=1, sticky=tk.W)
                crit_var = tk.StringVar(value=p_conf.get("clustering_criteria", "CPU"))
                cb = ttk.Combobox(patterns_frame, textvariable=crit_var, values=["CPU", "RAM"], width=10, state="readonly")
                cb.grid(row=row, column=2, sticky=tk.W)
                
                ttk.Label(patterns_frame, text="Threshold:").grid(row=row, column=3, sticky=tk.W)
                val_var = tk.StringVar(value=str(p_conf.get("selection_value", "2")))
                ent = ttk.Entry(patterns_frame, textvariable=val_var, width=5)
                ent.grid(row=row, column=4, sticky=tk.W)
                
                self.pattern_params[pattern] = {"clustering_criteria": crit_var, "selection_value": val_var}

            row += 1

        # Fixed Buttons at the bottom
        btn_frame = ttk.Frame(self.root, padding="10")
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        save_btn = ttk.Button(btn_frame, text="Save Config File", command=self.save_only)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Styling the save button to make it more obvious
        style = ttk.Style()
        style.configure("Save.TButton", font=("Helvetica", 10, "bold"))
        save_btn.configure(style="Save.TButton")

        ttk.Button(btn_frame, text="Save & Exit", command=self.save_and_exit).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Exit without Saving", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        self.update_client_list()

    def add_client(self):
        details = self.config.get("client_details", [])
        new_id = len(details) + 1
        details.append({
            "client_id": new_id,
            "cpu": 2,
            "data_distribution_type": "IID",
            "data_persistence_type": "Same Data",
            "delay_combobox": "No"
        })
        self.config["client_details"] = details
        self.update_client_list()
        self.client_selector_cb.set(new_id)
        self.on_client_selected(None)

    def remove_client(self):
        details = self.config.get("client_details", [])
        if details:
            details.pop()
            self.config["client_details"] = details
            self.update_client_list()
            if details:
                self.client_selector_cb.set(len(details))
                self.on_client_selected(None)

    def bulk_add(self, count):
        for _ in range(count):
            self.add_client()

    def update_client_list(self):
        details = self.config.get("client_details", [])
        n = len(details)
        self.clients_var.set(str(n))
        self.client_selector_cb['values'] = list(range(1, n + 1))
        
        if n > 0:
            if not self.client_selector_cb.get():
                self.client_selector_cb.current(0)
            self.on_client_selected(None)

    def on_client_selected(self, event):
        idx = self.selected_client_idx.get() - 1
        if 0 <= idx < len(self.config["client_details"]):
            client = self.config["client_details"][idx]
            self.client_cpu_var.set(str(client.get("cpu", 2)))
            self.client_dist_cb.set(client.get("data_distribution_type", "IID"))
            self.client_persist_cb.set(client.get("data_persistence_type", "Same Data"))
            self.client_delay_cb.set(client.get("delay_combobox", "No"))

    def apply_to_client(self):
        idx = self.selected_client_idx.get() - 1
        if 0 <= idx < len(self.config["client_details"]):
            try:
                self.config["client_details"][idx]["cpu"] = int(self.client_cpu_var.get())
                self.config["client_details"][idx]["data_distribution_type"] = self.client_distr_var.get()
                self.config["client_details"][idx]["data_persistence_type"] = self.client_persist_var.get()
                self.config["client_details"][idx]["delay_combobox"] = self.client_delay_var.get()
                messagebox.showinfo("Success", f"Override applied to Client {idx+1}")
            except:
                messagebox.showerror("Error", "Invalid CPU value.")

    def update_options(self, event):
        task = self.task_var.get()
        self.ds_cb['values'] = TASK_DATASETS.get(task, [])
        self.model_cb['values'] = TASK_MODELS.get(task, [])
        if self.ds_cb['values']: self.ds_cb.current(0)
        if self.model_cb['values']: self.model_cb.current(0)
        self.update_client_list()

    def save_only(self):
        conf = self.get_current_config()
        if conf:
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(conf, f, indent=4)
                messagebox.showinfo("Success", "Configuration saved to config.json")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    def get_current_config(self):
        try:
            rounds = int(self.rounds_var.get())
            n_clients = int(self.clients_var.get())
            dataset = self.dataset_var.get()
            model = self.model_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric values.")
            return None

        new_config = {
            "simulation_type": "Local",
            "rounds": rounds,
            "clients": n_clients,
            "adaptation": "None",
            "patterns": {},
            "client_details": []
        }
        
        for p in PATTERNS_LIST:
            params = {}
            if p in self.pattern_params:
                for k, v in self.pattern_params[p].items():
                    try:
                        params[k] = v.get()
                    except:
                        params[k] = v
            else:
                params = self.config.get("patterns", {}).get(p, {}).get("params", {})

            new_config["patterns"][p] = {
                "enabled": self.pattern_vars[p].get(),
                "params": params
            }

        for i in range(n_clients):
            # Use specifically overridden values or defaults from config
            client = self.config["client_details"][i] if i < len(self.config["client_details"]) else {"cpu": 2, "data_distribution_type": "IID"}
            new_config["client_details"].append({
                "client_id": i + 1,
                "cpu": client.get("cpu", 2),
                "ram": 2,
                "dataset": dataset,
                "data_distribution_type": client.get("data_distribution_type", "IID"),
                "data_persistence_type": client.get("data_persistence_type", "Same Data"),
                "delay_combobox": client.get("delay_combobox", "No"),
                "model": model,
                "epochs": 1
            })
        return new_config

    def save_and_exit(self):
        conf = self.get_current_config()
        if conf:
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(conf, f, indent=4)
                self.root.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfiguratorApp(root)
    root.mainloop()
