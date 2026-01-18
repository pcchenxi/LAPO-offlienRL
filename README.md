# Latent-variable Advantage-weighted Policy Optimization for Offline Reinforcement Learning

This is a pytorch implementation of paper [Latent-variable advantage-weighted policy optimization for offline reinforcement learning (LAPO)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html) on [D4RL](https://github.com/rail-berkeley/d4rl) dataset.

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
@article{chen2022lapo,
  title={Lapo: Latent-variable advantage-weighted policy optimization for offline reinforcement learning},
  author={Chen, Xi and Ghadirzadeh, Ali and Yu, Tianhe and Wang, Jianhao and Gao, Alex Yuan and Li, Wenzhe and Bin, Liang and Finn, Chelsea and Zhang, Chongjie},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={36902--36913},
  year={2022}
}
```

## Note
+ If you have any questions, please contact me: pcchenxi@gmail.com
