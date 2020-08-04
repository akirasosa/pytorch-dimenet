# DimeNet PyTorch

This repository is [DimeNet](https://arxiv.org/abs/2003.03123) PyTorch version which is ported from the original [TensorFlow repo](https://github.com/klicperajo/dimenet).

## Getting Started

```
# Download processed QM9 data.
./download-data.sh

# Train model to predict mu.
cd src
python run_train.py params/001.yaml
```

## Results

Epochs 800 is used.

|Target|Unit|MAE|
|---|---|---|
| mu | Debye | 0.0285 |
| U0 | meV | TODO |


## Differences from original

* Use RAdam as an optimizer.
* Use Mish as an activation.
* The number of layers and n_hidden in OutputBlock might be different.
* The loss func might be different.
* Data splitting might be different.

## Cite

```
@inproceedings{klicpera_dimenet_2020,
  title = {Directional Message Passing for Molecular Graphs},
  author = {Klicpera, Johannes and Gro{\ss}, Janek and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year = {2020}
}
```
