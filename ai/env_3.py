# == 입력 ==
# 모터 상태(10) + 목표 위치(3) + 손끝 위치(3) + [목표 위치 - 손끝 위치](3) + 목표 각도(2) + 손끝 각도(2) + [목표 각도 - 손끝 각도](2)
# Roll
# Pitch
# == 보상 ==
# 손 끝과 목표 지점이 가까울 수록
# 적은 step일 수록

import os
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class CarArmEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, render_mode=None, **kwargs):
        model_path = os.path.join(os.path.dirname(__file__), "car_arm.xml")

        # 초기화
        EzPickle.__init__(self, render_mode, **kwargs)
        super().__init__(
            model_path,
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode,
            **kwargs,
        )

        # 액션 공간
        self.action_space = Box(
            low=-1,
            high=1,
            shape=(5,),
            dtype=np.float32,
        )

        # 관측 공간
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32,
        )

        # 변수
        self.max_step = 500
        self.current_step = 0
        self.qpos_min = np.array(
            [-2.0944, -1.5708, 0.0, -2.0944, -2.0944], dtype=np.float32
        )
        self.qpos_max = np.array(
            [2.0944, 1.5708, 3.14159, 2.0944, 2.0944], dtype=np.float32
        )
        self.low_angle = np.array(-2.0944, dtype=np.float32)
        self.high_angle = np.array(2.0944, dtype=np.float32)
        self.min_rad_sq = np.array(0.15**2, dtype=np.float32)
        self.max_rad_sq = np.array(0.30**2, dtype=np.float32)

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_rot = np.zeros(2, dtype=np.float32)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1

        # 관측 정보 업데이트
        obs = self._get_obs()

        # 거리
        ee_pos = self.data.site("end_effector").xpos
        target_pos = self.data.site("target_site").xpos
        distance = np.linalg.norm(target_pos - ee_pos)

        # 각도
        ee_mat = self.data.site("end_effector").xmat
        ee_pitch = np.arcsin(np.clip(-ee_mat[6], -1.0, 1.0))
        ee_roll = np.arctan2(ee_mat[7], ee_mat[8])
        ee_rot = np.array([ee_roll, ee_pitch], dtype=np.float32)

        rel_rot = self.target_rot - ee_rot
        rel_rot = (rel_rot + np.pi) % (2 * np.pi) - np.pi
        rot_dist = np.linalg.norm(rel_rot)

        # 동작 최소화
        action_cost = -0.01 * np.sum(np.square(action))

        # 보상
        reward = -distance - 0.1 * rot_dist + action_cost

        # 종료
        terminated = bool(distance < 0.02 and rot_dist < 0.1)

        # 강제 종료
        truncated = bool(self.current_step >= self.max_step)

        info = {
            "step": self.current_step,
            "success": terminated,
            "dist": distance,
            "rot_dist": rot_dist,
        }

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.current_step = 0

        # 모터 설정
        qpos = self.np_random.uniform(self.qpos_min, self.qpos_max)
        qvel = np.zeros(5)

        self.set_state(qpos, qvel)

        # 목표 설정
        angle = self.np_random.uniform(self.low_angle, self.high_angle)
        radius = np.sqrt(self.np_random.uniform(self.min_rad_sq, self.max_rad_sq))
        self.target_pos = np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                self.np_random.uniform(0, 0.2),
            ],
            dtype=np.float32,
        )

        target_roll = self.np_random.uniform(-1.5708, 1.5708)
        target_pitch = self.np_random.uniform(-1.0472, 1.0472)
        self.target_rot = np.array([target_roll, target_pitch], dtype=np.float32)

        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, [target_roll, target_pitch, 0], "xyz")

        self.model.body("target").pos[:] = self.target_pos
        self.model.body("target").quat[:] = target_quat

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        target_pos = self.data.site("target_site").xpos.copy()
        ee_pos = self.data.site("end_effector").xpos.copy()
        rel_pos = target_pos - ee_pos

        ee_mat = self.data.site("end_effector").xmat
        ee_pitch = np.arcsin(np.clip(-ee_mat[6], -1.0, 1.0))
        ee_roll = np.arctan2(ee_mat[7], ee_mat[8])
        ee_rot = np.array([ee_roll, ee_pitch], dtype=np.float32)

        target_rot = self.target_rot
        rel_rot = target_rot - ee_rot
        rel_rot = (rel_rot + np.pi) % (2 * np.pi) - np.pi

        return np.concatenate(
            [
                qpos,
                qvel,
                target_pos,
                ee_pos,
                rel_pos,
                target_rot,
                ee_rot,
                rel_rot,
            ]
        ).astype(np.float32)
