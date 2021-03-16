help:
	@echo ""
	@echo "Available make commands are:"
	@echo " help                  : prints this message"
	@echo " run                   : runs the python script (recommended inside container, try build first)"
	@echo " clean                 : removes files and folders created during run"
	@echo " wharfer-build         : builds image using Wharfer"
	@echo " docker-build          : builds image using Docker"
	@echo " wharfer-run-container : runs image using Wharfer"
	@echo " docker-run-container  : runs image using Docker"
	@echo " wharfer-run           : runs image using Wharfer, and immediately calls script"
	@echo " docker-run            : runs image using Docker, and immediately calls script"
	@echo " wharfer-all           : build and run with Wharfer"
	@echo " docker-all            : build and run with Docker"
	@echo ""

all: clean run

clean:
	echo "NOTE: This is meant to be run inside of a docker container, or locally."
	echo "If ./data or ./model are mounted to a Docker container, they cannot be removed."
	echo "In that case, simply restart the container without the folders mounted."
	rm -r ./data
	rm -r ./model

wharfer-build:
	wharfer build -t raheim/bert_el .

build: wharfer-build

docker-build:
	docker build -t raheim/bert_el .

wharfer-run-container:
	wharfer run -v /local/data/raheim/models:/bert_el/models -v /local/data/raheim/data:/bert_el/data -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_el/ex_data -it raheim/bert_el

docker-run-container:
	docker run -v /local/data/raheim/models:/bert_el/models -v /local/data/raheim/data:/bert_el/data -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_el/ex_data -it raheim/bert_el


wharfer-run:
	wharfer run -v /local/data/raheim/models:/bert_el/models -v /local/data/raheim/data:/bert_el/data -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_el/ex_data -it raheim/bert_el /bin/sh -c "python bert_el_full_pipeline.py"

docker-run:
	docker run -v /local/data/raheim/models:/bert_el/models -v /local/data/raheim/data:/bert_el/data -v /nfs/students/matthias-hertel/wiki_entity_linker:/bert_el/ex_data -it raheim/bert_el /bin/sh -c "python bert_el_full_pipeline.py"

wharfer-all: wharfer-build wharfer-run

all: wharfer-all

docker-all: docker-run docker-build

run:
	python bert_el_full_pipeline.py
