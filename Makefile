# Full pipeline: make all   (about 30 minutes on an Apple M-series GPU)
PY := .venv/bin/python

.PHONY: setup data audit split prepare train-resnet train-cnn evaluate gradcam leaky cifar test all

setup:
	uv venv --python 3.13 .venv && uv pip install --python $(PY) -r requirements.txt && uv pip install --python $(PY) -e .

data:            # FracAtlas, CC BY 4.0, 338 MB (figshare, version 7)
	mkdir -p data/raw && curl -L -o data/raw/FracAtlas.zip https://ndownloader.figshare.com/files/65518038
	cd data/raw && unzip -q -o FracAtlas.zip && rm FracAtlas.zip

audit:   ; $(PY) scripts/01_audit.py
split:   ; $(PY) scripts/02_split.py
prepare: ; $(PY) scripts/03_prepare.py

train-resnet: ; $(PY) scripts/04_train.py --model resnet18 --run-name resnet18
train-cnn:    ; $(PY) scripts/04_train.py --model simple_cnn --run-name simple_cnn --epochs 40 --patience 8 --freeze-epochs 0 --lr-head 1e-3

evaluate: ; $(PY) scripts/05_evaluate.py resnet18 && $(PY) scripts/05_evaluate.py simple_cnn
gradcam:  ; $(PY) scripts/06_gradcam.py resnet18
leaky:    ; $(PY) scripts/07_leaky_protocol.py
cifar:    ; $(PY) cifar10/train_cifar.py
test:     ; $(PY) -m pytest -q

all: audit split prepare train-resnet train-cnn evaluate gradcam leaky
