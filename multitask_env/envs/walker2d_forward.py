import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import pickle
from pathlib import Path

class Walker2dForwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, backward=False, task_one_hot_id=None):
        self.backward = backward
        self.task_one_hot_id = task_one_hot_id
        self.step_count = 0
        # self._max_episode_steps = 1000
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        
    def step(self, a):
        if not hasattr(self, "init_zpos"):
            self.init_zpos = self.sim.data.get_body_xpos('torso')[2]
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        if not self.backward:
            reward_run = ((posafter - posbefore) / self.dt)
        else:
            reward_run = -((posafter - posbefore) / self.dt)
        reward_ctrl = -1e-3 * np.square(a).sum()
        reward_jump = 10.0*(height - self.init_zpos)
        reward = alive_bonus + reward_run + reward_ctrl
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        # self.step_count += 1
        # if self.step_count == self._max_episode_steps:
        #     done = True
        #     self.step_count = 0

        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_jump=reward_jump)

    def step_obs_act(self, obs, action):
        next_obs = []
        for i in range(obs.shape[0]):
            qpos, qvel = obs[i, :self.sim.data.qpos.shape[0]-1], obs[i, self.sim.data.qpos.shape[0]-1:]
            qpos = np.concatenate(([0.], qpos))
            self.set_state(qpos, qvel)
            # self.render()
            self.do_simulation(action[i], self.frame_skip)
            ob = self._get_obs()
            next_obs.append(ob)
        return np.array(next_obs)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        if self.task_one_hot_id is not None:
            return np.concatenate([qpos[1:],
                                np.clip(qvel, -10, 10),
                                self.task_one_hot_id]).ravel()
        return np.concatenate([qpos[1:],
                                np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.init_zpos = self.sim.data.get_body_xpos('torso')[2]
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_normalized_score(self, score):
        min_score = 0.5454125 * 1000
        max_score = 4.3112516 * 1000

        return (score - min_score)/(max_score-min_score)

    def get_dataset(self):
        home = str(Path.home())
        with open(home + "/.d4rl/datasets/multi-walker2d-forward-v1.pkl", 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset