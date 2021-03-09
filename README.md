
# BERT for Entity Linking — bert_el

## Docker build

Build the Docker image:

```bash
docker build -t bert_el .
```

## Docker run
To run the Docker image:

```bash
docker run -v /some/local/directory/with/data:/bert_el/data \
           -it --name bert_el bert_el
```

Or if you want to provide a previously saved model, and directly run the script:

```bash
docker run -v /some/local/directory/with/data:/bert_el/data \
           -v /some/local/directory/with/models:/bert_el/models \
           -it --name bert_el bert_el \
           /bin/bash -c "python bert_el_full_pipeline.py"
```
