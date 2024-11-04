# Client Cluster Architectural Pattern

This folder contains models, scripts, data, and results of the Client Cluster architectural pattern.

# How to run Experiments

To reproduce results presented in the paper, users can run new simulation. 
In this case, they should note that slightly different values may be observed due to the stochastic simulation. Please, follow these steps:

1.
2.

# Customize Input Parameters

Users can also tune input parameters of provided models to study different applications designed with the same patter.
Please, follow these steps to study your own applications with our framework:

In the __2. Client Cluster/__ folder with 'docker-compose' file, enter the following command:

```bash
#Build Docker images
docker compose build

#Launch Docker images (Server, 2 Clients A, 2 Clients B)
NUM_ROUNDS=10 docker-compose up --scale clienta=2 --scale clientb=2
```