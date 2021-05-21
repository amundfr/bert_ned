
# BERT for Named Entity Disambiguation — bert_ned

## Make

There are **shorthands** for the preceding Docker and Wharfer commands in a Makefile. Navigate to the directory with this repository and type `make help` for a list of instructions. 

## Docker

You can build and run containers with specific volumes or names if you disregard the 'make' instructions. 

Run the image on a machine with a Cuda-enabled GPU to train the model or use the model for inference. 

Build the Docker image with GPU (7.27GB):

```bash
docker build -t bert_ned_cpu .
```

For completeness, there is a CPU version of the Dockerfile (`Dockerfile.CPU`). The CPU container works for the data preparation scripts, but the model runs poorly on a CPU.

Build without GPU (3.19GB):

```bash
docker build -f Dockerfile.CPU -t bert_ned_gpu .
```

## Docker run
To run the Docker image on an AD machine using files from /nfs/:

```bash
docker run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data \
           -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data \
           -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models \
           -it --name bert_ned bert_ned_gpu
```

Please note: accessing files over NFS makes some of the operations quite slow. You can also copy the directories in /nfs/students/amund-faller-raheim/master_project_bert_ned to a local directory and mount those to the docker container.

## Run scripts in container

Instructions should appear **once the container is running**. Type 'make help' to see a list of actions. (This is from a second makefile `Makefile_scripts`.)

You can use 'make' to **run scripts** in the container. For example:

```bash
make full
```

Which runs the script `bert_ned_full_pipeline.py`. This script can do the full process, from data generation to training and evaluation. 

When running the scripts, make sure that the settings in `config.ini` are correct for your environment (or the container, by default). **Leave the settings as provided to reproduce results.**
