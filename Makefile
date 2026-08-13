.PHONY: setup data experiments peeking cuped lalonde app test docker clean
PY := .venv/bin/python

setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt pytest

data:            ## LaLonde/NSW from NBER -- public, no credentials
	$(PY) -m src.abcausal.data

experiments: peeking cuped lalonde

peeking:
	$(PY) -m src.abcausal.experiments.peeking

cuped:
	$(PY) -m src.abcausal.experiments.cuped_gain

lalonde: data
	$(PY) -m src.abcausal.experiments.lalonde

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest tests/ -q

docker:
	docker build -t ab-causal .

clean:
	rm -rf reports/*.csv
