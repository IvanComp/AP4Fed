# Client Selector Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/1ClientSelector.ipynb) of the Client Selector architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /With Client Selector  
├── /Without Client Selector      
```

# Input Parameters

In the following, there are the input parameters for the Client Selector architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 4 | Number of Clients. Container name: **clienthigh** = High-Spec clients, <br> **clientlow** = Low-Spec clients|
| `n_CPU` | 2 for High-Spec clients,<br> 1 for Low-Spec clients | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Selection Strategy` | Resource-based | The Selection Strategy will include/exclude clients based on their computational capabilities |
| `Selection Criteria` | Number of CPU > 1 | The Selector Criteria will evaluate clients based on their number of CPU |

# How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Client Selector_ or _Without Client Selector_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration A (1 Server, 4 Clients with "High" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=4 --scale clientlow=0

#For Configuration B (1 Server, 3 Clients A with "High" specifications, 1 Client with "Low" specifications) -- Without Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=3 --scale clientlow=1

#For Configuration C (1 Server, 3 Clients A with "High" specifications, 1 Client A with "Low" specifications) -- With Client Selector pattern
NUM_ROUNDS=10 docker-compose up --scale clienthigh=3 --scale clientlow=1
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


# How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale clienthigh=15 --scale clientlow=5
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.
