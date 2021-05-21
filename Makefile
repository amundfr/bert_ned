help:
	@echo ""
	@echo "Available make commands are:"
	@echo " help                  : prints this message"
	@echo " wharfer-build         : builds image using Wharfer"
	@echo " wharfer-run           : runs container using Wharfer"
	@echo " wharfer-run-full      : runs container using Wharfer, and immediately calls main script"
	@echo " wharfer-run-unittest  : runs container using Wharfer, and runs the unittests"
	@echo " wharfer-all           : build and run with Wharfer, call script"
	@echo " docker-build          : builds image using Docker"
	@echo " docker-run            : runs container using Docker"
	@echo " docker-run-full       : runs container using Docker, and immediately calls main script"
	@echo " docker-run-unittest   : runs container using Docker, and runs the unittests"
	@echo " docker-all            : build and run with Docker, call script"
	@echo ""

wharfer-build:
	wharfer build -t bert_ned .

wharfer-run:
	wharfer run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned

wharfer-run-full:
	wharfer run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned /bin/sh -c "python bert_ned_full_pipeline.py"

wharfer-run-unittest:
	wharfer run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned /bin/sh -c "python3 -m unittest tests/*.py"


docker-build:
	docker build -t bert_ned .

docker-run:
	docker run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned

docker-run-full:
	docker run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned /bin/sh -c "python bert_ned_full_pipeline.py"

docker-run-unittest:
	docker run -v /nfs/students/amund-faller-raheim/master_project_bert_ned/models:/bert_ned/models -v /nfs/students/amund-faller-raheim/master_project_bert_ned/data:/bert_ned/data -v /nfs/students/amund-faller-raheim/master_project_bert_ned/ex_data:/bert_ned/ex_data -it bert_ned /bin/sh -c "python3 -m unittest tests/*.py"


wharfer-all: wharfer-build wharfer-run-full

build: wharfer-build

run: wharfer-run

all: wharfer-all