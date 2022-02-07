#!/bin/bash

python main_d4rl.py --env_name antmaze-umaze-diverse-v1 --doubleq_min 0.7 --plot --ExpID 1
python main_d4rl.py --env_name antmaze-medium-diverse-v1 --doubleq_min 0.7 --plot --ExpID 1
python main_d4rl.py --env_name antmaze-large-diverse-v1 --doubleq_min 0.7 --plot --ExpID 1

python main_d4rl.py --env_name maze2d-umaze-v1 --kl_beta 0.3 --plot --ExpID 1
python main_d4rl.py --env_name maze2d-medium-v1 --kl_beta 0.3 --plot --ExpID 1
python main_d4rl.py --env_name maze2d-large-v1 --kl_beta 0.3 --plot --ExpID 1

python main_d4rl.py --env_name hopper-random-v2 --ExpID 1
python main_d4rl.py --env_name hopper-medium-v2 --ExpID 1
python main_d4rl.py --env_name hopper-expert-v2 --ExpID 1

python main_d4rl.py --env_name walker2d-random-v2 --ExpID 1
python main_d4rl.py --env_name walker2d-medium-v2 --ExpID 1
python main_d4rl.py --env_name walker2d-expert-v2 --ExpID 1

python main_d4rl.py --env_name halfcheetah-random-v2 --ExpID 1
python main_d4rl.py --env_name halfcheetah-medium-v2 --ExpID 1
python main_d4rl.py --env_name halfcheetah-expert-v2 --ExpID 1

python main_d4rl.py --env_name kitchen-complete-v0 --ExpID 1
python main_d4rl.py --env_name kitchen-partial-v0 --ExpID 1
python main_d4rl.py --env_name kitchen-mixed-v0 --ExpID 1
