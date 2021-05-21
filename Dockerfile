FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainers="Amund Faller Raheim raheim@informatik.uni-freiburg.de"

RUN mkdir bert_ned

WORKDIR /bert_ned

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy from build context to image
COPY lib /bert_ned/lib
COPY src /bert_ned/src
COPY tests /bert_ned/tests
COPY bert_ned_full_pipeline.py .
COPY generate_input_data.py .
COPY config.ini .
COPY Makefile_scripts /bert_ned/Makefile
COPY .bashrc /root/

# Build image with
# docker build -t bert_ned .
# Run container with
# docker run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -it --name bert_ned bert_ned
