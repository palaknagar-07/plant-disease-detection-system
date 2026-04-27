.PHONY: setup app train evaluate predict

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt

app:
	streamlit run main.py

train:
	$(PYTHON) -m src.train --config configs/default.json

evaluate:
	$(PYTHON) -m src.evaluate --config configs/default.json

predict:
	$(PYTHON) -m src.predict "$(IMAGE)" --config configs/default.json
