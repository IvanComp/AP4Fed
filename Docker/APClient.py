import platform
from datetime import datetime

import psutil


class ClientRegistry:
    def __init__(self):
        self.registry = {}

    def get_client_info(self):
        """Gathers system information about the client."""
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),  # Physical cores
            "cpu_threads": psutil.cpu_count(logical=True),  # Logical threads
            "ram_total": psutil.virtual_memory().total / (1024 ** 3),  # Total RAM in GB
            "ram_available": psutil.virtual_memory().available / (1024 ** 3),  # Available RAM in GB
            "python_version": platform.python_version(),
        }
        return info

    def register_client(self, cid, model_type):
        """Registers a new client with its cid and assigned model_type."""
        if cid in self.registry:
            print(f"[WARNING] Client {cid} is already registered.")
            return

        client_info = self.get_client_info()
        self.registry[cid] = {
            'model_type': model_type,
            'system_info': client_info,  # Save the system info
            'active': True,
            'last_update': datetime.now()
        }

    def get_client_model(self, cid):
        """Returns the model assigned to the client (taskA or taskB) based on cid."""
        client = self.registry.get(cid)
        print(" Client: {client}")
        if client:
            return client['model_type']
        else:
            print(f"[ERROR] Client {cid} is not registered.")
            return None

    def update_client(self, cid, status, last_update=None):
        """Updates the status of the client."""
        if cid in self.registry:
            self.registry[cid]['active'] = status
            self.registry[cid]['last_update'] = last_update or datetime.now()
        else:
            print(f"[WARNING] Attempt to update an unregistered client: {cid}")

    def get_active_clients(self):
        """Returns the list of active clients."""
        active_clients = {cid: info for cid, info in self.registry.items() if info['active']}
        print(f"Active clients: {list(active_clients.keys())}")
        return active_clients

    def is_registered(self, cid):
        """Check if a client is already registered."""
        return cid in self.registry

    def print_clients_info(self):
        """Prints the list of clients with their respective system information."""
        active_clients = self.get_active_clients()
        if active_clients:
            for i, (cid, client_info) in enumerate(active_clients.items(), 1):
                system_info = client_info['system_info']
                print(f"Client {i}: {cid}")
                for key, value in system_info.items():
                    print(f"  {key}: {value}")
                print() 
        else:
            print("No active clients registered.")