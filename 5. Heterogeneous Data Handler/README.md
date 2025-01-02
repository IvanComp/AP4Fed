# Client Cluster Architectural Pattern

This folder contains scripts, data, and [experimental results](https://github.com/IvanComp/AP4Fed/blob/main/Experiments%20Results/2ClientCluster.ipynb) of the Client Cluster architectural pattern.

It is possible to replicate the same experiments proposed in the paper or run the experiments by considering different input parameters (see [section](#how-to-run-custom-experiments)).

```bash
$ tree .
.
├── /1. With Client Cluster  
├── /2. Without Client Cluster      
```

# Input Parameters

In the following, there are the input parameters for the Client Cluster architectural pattern.

| Parameter | Experiment Default Value | Description | 
| --- | --- | --- | 
| `NUM_ROUNDS` | 10 | Number of Federated Learning rounds. |
| `nS` | 1 | Number of Server. Container name: server|
| `nC` | 8 | Number of Clients. Container name: **clienta** = IID clients, <br> **clientb** = non-IID clients|
| `n_CPU` | 1 | Number of physical CPU cores allocated to each container |
| `RAM` | 2GB | Memory capacity allocated to each container |
| `Clustering Strategy` | Data-based | The Clustering Strategy will groups clients based on their data distribution type |
| `Clustering Criteria` | IID vs non-IID | The Clustering Criteria will cluster clients based on IID or non-IID data |

# How to run the Paper's Experiments

The commands for the experiments proposed in the paper are defined in the following.
To reproduce the results presented in the paper please, follow these steps:

1. Run Docker Desktop
2. Open the _With Client Cluster_ or _Without Client Cluster_ folders based on the type of experiment.
3. Open the terminal and enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker Compose Configuration. Please chose one of the following configurations based on the folder:

#For Configuration 4a4b (1 Server, 4 Clients with IID, 4 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=4 --scale clientb=4

#For Configuration 5a3b (1 Server, 5 Clients with IID, 3 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=5 --scale clientb=3

#For Configuration 6a2b (1 Server, 6 Clients with IID, 2 Clients with non-IID) 
NUM_ROUNDS=10 docker-compose up --scale clienta=6 --scale clientb=2
```
4. The process of Federated Learning will start, and the progress of each round will be displayed in the terminal.
   <br> Note that **different values may be observed because of the stochastic nature of the simulation**. 


# How to run Custom Experiments

Users can also customize input parameters to investigate the architectural pattern performance considering different settings.
All of the input parameters can be varied by changing the corresponding values from the command line before starting the project, For example:

```bash
# Custom Configuration:
NUM_ROUNDS=50 docker-compose up --scale clienta=15 --scale clientb=5
```

Note that changing CPU and RAM parameters requires access to the docker-compose file, where these settings can be manually adjusted.
