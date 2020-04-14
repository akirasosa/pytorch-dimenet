# DimeNet PyTorch

This repository is [DimeNet](https://arxiv.org/abs/2003.03123) PyTorch version which is ported from the original [TensorFlow repo](https://github.com/klicperajo/dimenet).

## Getting Started

```
# Download processed QM9 data.
./download-data.sh

# Train model to predict U0.
cd src
python run_train.py
```

## Differences from original

* Use Ranger as an optimizer.
* Use Mish as an activation.
* Use flat cos anneal learning rate schedule.
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
