FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainers="Amund Faller Raheim raheim@informatik.uni-freiburg.de"

RUN mkdir bert_el

WORKDIR /bert_el

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy full build context to image
COPY . /bert_el

# Build with
# docker build -t bert_el .
# Run with
# docker run -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_el/ex_data -v /some/local/directory/with/data:/bert_el/data -v /some/local/directory/with/models:/bert_el/models -it --name bert_el bert_el /bin/bash -c "python bert_el_full_pipeline.py"
