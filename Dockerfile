FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainers="Amund Faller Raheim raheim@informatik.uni-freiburg.de"

RUN mkdir bert_ned

WORKDIR /bert_ned

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy full build context to image
COPY . /bert_ned

# Build with
# docker build -t bert_ned .
# Run with
# docker run -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_ned/ex_data -v /some/local/directory/with/data:/bert_ned/data -v /some/local/directory/with/models:/bert_ned/models -it --name bert_ned bert_ned /bin/bash -c "python bert_ned_full_pipeline.py"
