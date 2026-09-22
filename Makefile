.PHONY: install download preprocess train evaluate deploy serve predict monitor pipeline clean

install:
	pip install -r requirements.txt

download:
	python -m src.data.data_download

preprocess:
	python -m src.data.data_preprocess

train:
	python -m src.model.model_train

evaluate:
	python -m src.evaluation.model_evaluate

deploy:
	python -m src.deployment.model_deploy

serve:
	python -m src.serving.serve

predict:
	python -m src.inference.predict

monitor:
	python -m src.monitoring.monitor_pipeline

# This target runs every stage in order, from download through deploy.
pipeline:
	python -m src.pipeline

clean:
	rm -rf data/raw/* data/processed/* models/output/* models/scalers/* \
		models/registry/latest/* evaluation/* monitoring/*
