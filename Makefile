IMAGE_NAME=facade-sam
USER_FLAG=--user $(shell id -u):$(shell id -g)

build:
	docker build -t $(IMAGE_NAME) .

download-models:
	bash scripts/download_models.sh

run:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		$(IMAGE_NAME)

shell:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		--entrypoint /bin/bash \
		$(IMAGE_NAME)

view:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		facade-sam \
		python3 app/view_meshes.py

clean:
	sudo rm -rf output/masks/* output/meshes/* output/per_class/* \
		output/combined_scene.obj output/sionna_scene.json \
		output/mask_overlay.png
	@echo "output cleared"