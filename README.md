# Latent-variable Advantage-weighted Policy Optimization for Offline Reinforcement Learning

This is a pytorch implementation of paper [Latent-variable advantage-weighted policy optimization for offline reinforcement learning (LAPO)](https://arxiv.org/pdf/2203.08949.pdf) on [D4RL](https://github.com/rail-berkeley/d4rl) dataset.

![LAPO-framwork](https://github.com/pcchenxi/LAPO-offlienRL/blob/main/figs/LAPO.jpg)

## Requirements

- python=3.7.11
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.10.0

## Scripts for D4RL dataset

Maze2d: maze2d-umaze/medium/large-v1
```shell
$ python main_d4rl.py --env_name maze2d-umaze-v1 --kl_beta 0.3 --plot
```

Antmaze: antmaze-umaze/medium/large-diverse-v1
```shell
$ python main_d4rl.py --env_name antmaze-umaze-diverse-v1 --doubleq_min 0.7 --plot
```

Mujoco locomotion: hopper/walker2d/halfcheetah-random/medium/expert-v2
```shell
$ python main_d4rl.py --env_name hopper-random-v2
```

Kitchen: kitchen-complete/partial/mixed-v0
```shell
$ python main_d4rl.py --env_name kitchen-complete-v0
```

## Expected results

You will get following results using --seed: 123(red) 456(green) 789(blue)

![LAPO-framwork](https://github.com/pcchenxi/LAPO-offlienRL/blob/main/figs/result_3seeds.jpg)

## Citing
If you find this code useful, please cite our paper:
```
@article{chen2022latent,
  title={Latent-Variable Advantage-Weighted Policy Optimization for Offline RL},
  author={Chen, Xi and Ghadirzadeh, Ali and Yu, Tianhe and Gao, Yuan and Wang, Jianhao and Li, Wenzhe and Liang, Bin and Finn, Chelsea and Zhang, Chongjie},
  journal={arXiv preprint arXiv:2203.08949},
  year={2022}
}
```

## Note
+ If you have any questions, please contact me: pcchenxi@gmail.com
