
# BERT for Named Entity Disambiguation — bert_ned

## Docker build

Build the Docker image:

```bash
docker build -t bert_ned .
```

## Docker run
To run the Docker image:

```bash
docker run -v /some/local/directory/with/data:/bert_ned/data \
           -it --name bert_ned bert_ned
```

Or if you want to provide a previously saved model, and directly run the script:

```bash
docker run -v /some/local/directory/with/data:/bert_ned/data \
           -v /some/local/directory/with/models:/bert_ned/models \
           -it --name bert_ned bert_ned \
           /bin/bash -c "python bert_ned_full_pipeline.py"
```
