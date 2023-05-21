# BIRIS

[![arXiv](https://img.shields.io/badge/arXiv-2206.04436-b31b1b.svg)](https://arxiv.org/abs/2209.07074)

This is the official implementation for On the Reuse Bias in Off-Policy Reinforcement Learning (Accepted in IJCAI 2023).

## Usage

First, you need to install corresponding packages by

```
conda create -n BIRIS python=3.8
conda activate BIRIS
pip install -r requirements.txt
```

### Gym-MiniGrid
The code for MiniGrid is in the fold MiniGrid, thus you can train the code by

```
cd MiniGrid
python main.py --sample_algorithm IS --use_biris True --buffer_size 40 --env MiniGrid-Empty-5x5-v0
```

You can choose sample_algorithm=IS or WIS, use_biris=True or False, buffer_size=30, 40, or 50, env=MiniGrid-Empty-5x5-v0, MiniGrid-Empty-Random-5x5-v0, MiniGrid-Empty-6x6-v0, MiniGrid-Empty-Random-6x6-v0, MiniGrid-Empty-8x8-v0, or MiniGrid-Empty-16x16-v0, to reproduce the results in the paper.
