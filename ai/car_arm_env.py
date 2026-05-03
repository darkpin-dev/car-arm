import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os


class CarArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        model_path = os.path.join(os.path.dirname(__file__), "car_arm.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.renderer = None
        self.viewer = None

        self.max_steps = 500
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # qpos(5) + qvel(5) + rel_pos(3) + rel_angle_err(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self._prev_dist = 0.0
        self._prev_angle_err = 0.0
        self.goal_angle = np.zeros(2)
        self.difficulty = 0.0

    def set_difficulty(self, difficulty_level):
        self.difficulty = np.clip(difficulty_level, 0.0, 1.0)

    def _get_angle_from_vector(self, vector):
        x, y, z = vector
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(z, np.sqrt(x**2 + y**2) + 1e-6)
        return np.array([yaw, pitch])

    def _get_ee_angle(self):
        ee_forward = self.data.site_xmat[0].reshape(3, 3)[:, 2]
        return self._get_angle_from_vector(ee_forward)

    def _get_obs(self):
        qpos     = self.data.qpos[:5]
        qvel     = self.data.qvel[:5]
        ee_pos   = self.data.site_xpos[0]
        tgt_pos  = self.data.site_xpos[1]
        rel_pos  = tgt_pos - ee_pos

        ee_angle  = self._get_ee_angle()
        rel_angle = self.goal_angle - ee_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        return np.concatenate([qpos, qvel, rel_pos, rel_angle]).astype(np.float32)

    def _get_info(self, obs=None):
        """obs를 인자로 받아 이중 계산을 방지합니다."""
        if obs is None:
            obs = self._get_obs()

        ee_pos  = self.data.site_xpos[0]
        tgt_pos = self.data.site_xpos[1]
        distance  = float(np.linalg.norm(ee_pos - tgt_pos))
        angle_err = float(np.linalg.norm(obs[-2:]))

        dist_thresh  = 0.05 - (0.03 * self.difficulty)
        angle_thresh = 0.5  - (0.35 * self.difficulty)
        is_success   = (distance < dist_thresh) and (angle_err < angle_thresh)

        return {
            "distance":   distance,
            "angle_err":  angle_err,
            "is_success": is_success,
            "difficulty": self.difficulty,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # 목표 위치 랜덤 설정
        target_x = self.np_random.uniform(0.15, 0.25)
        target_y = (
            self.np_random.uniform( 0.15,  0.25) if self.np_random.random() > 0.5
            else self.np_random.uniform(-0.25, -0.15)
        )
        target_z  = self.np_random.uniform(-0.1, 0.15)
        target_pos = np.array([target_x, target_y, target_z])

        base_angle      = self._get_angle_from_vector(target_pos)
        angle_noise     = self.np_random.uniform(-1.0, 1.0, size=2)
        goal_angle_raw  = base_angle + angle_noise
        self.goal_angle = (goal_angle_raw + np.pi) % (2 * np.pi) - np.pi

        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[target_id] = target_pos
        mujoco.mj_forward(self.model, self.data)

        obs  = self._get_obs()
        info = self._get_info(obs)
        self._prev_dist      = info["distance"]
        self._prev_angle_err = info["angle_err"]

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.step_count += 1
        self.data.ctrl[:5] = action
        mujoco.mj_step(self.model, self.data)

        obs  = self._get_obs()
        info = self._get_info(obs)
        dist      = info["distance"]
        angle_err = info["angle_err"]

        # ══════════════════════════════════════════════════════════════
        # 보상 함수 (SAC_2 vs SAC_4 분석 결과 반영)
        #
        #  변경 내역:
        #  ┌─────────────────────┬─────────┬─────────┬────────────┐
        #  │ 항목                │ SAC_2   │ SAC_4   │ SAC_5(현재)│
        #  ├─────────────────────┼─────────┼─────────┼────────────┤
        #  │ 거리 potential      │  50x    │  50x    │  50x       │
        #  │ 각도 potential      │  10x    │  30x  ↑ │  30x    ✅ │
        #  │ 거리 penalty        │   1.0   │   2.0 ↑ │   1.0   ✅ │
        #  │ 각도 penalty        │   0.5   │   2.0 ↑ │   1.0   ✅ │
        #  │ 진동 억제 penalty   │  없음   │  0.2    │   0.1   ✅ │
        #  │ 각도 노이즈 스케일  │ 고정    │ diff비례│ diff비례✅ │
        #  └─────────────────────┴─────────┴─────────┴────────────┘
        #
        #  근거:
        #  - 각도 potential 30x 유지 → 각도 학습 신호 강도 보존
        #  - 거리 penalty 1.0 복원  → SAC_4의 reward 음수화 방지
        #  - 각도 penalty 1.0       → SAC_2(0.5)보다 강하되 SAC_4(2.0)보다 약하게
        #                             (중간값: 각도 중요성 반영 + reward 붕괴 방지)
        #  - 진동 억제 0.1          → SAC_4(0.2)보다 약하게, 신호 간섭 최소화
        # ══════════════════════════════════════════════════════════════

        # 1. Potential-based 방향 신호
        reward  = (self._prev_dist      - dist)      * 50.0
        reward += (self._prev_angle_err - angle_err) * 30.0
        self._prev_dist      = dist
        self._prev_angle_err = angle_err

        # 2. Dense penalty (SAC_2 수준 복원)
        reward -= dist      * 1.0
        reward -= angle_err * 1.0   # SAC_2(0.5)보다 강하되 SAC_4(2.0)보다 완만

        # 3. 진동 억제 penalty (5cm 이내, 약하게)
        if dist < 0.05:
            qvel = self.data.qvel[:5]
            reward -= float(np.sum(np.square(qvel))) * 0.1

        # 4. 행동 부드러움 penalty
        reward -= 0.01 * float(np.sum(np.square(action)))

        # 5. 성공 보너스 (에피소드 내 최강 신호)
        terminated = bool(info["is_success"])
        if terminated:
            reward += 200.0

        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            import mujoco_viewer
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.render()

        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None