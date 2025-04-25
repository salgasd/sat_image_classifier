clean:
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "mypy_cache" | xargs rm -rf
	rm -f .coverage

format:
	pre-commit run -a

create_venv:
	python3 -m venv venv

install_reqs_dev: create_venv
	./venv/bin/python3 -m pip install -r requirements_dev.txt

install_reqs_infer: create_venv
	./venv/bin/python3 -m pip install -r requirements.txt

get_data:
	wget -O data.zip "https://www.dropbox.com/scl/fi/nm2lm7ms8z6l44qag1gb9/data.zip?rlkey=vrsj8dqb60lqhxuw4f93iiumi&st=qmck4tsg&dl=0"
	unzip data.zip
	rm data.zip

train:
	PYTHONPATH=. ./venv/bin/python3 src/train.py
