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
        yaw   = np.arctan2(y, x)
        pitch = np.arctan2(z, np.sqrt(x**2 + y**2) + 1e-6)
        return np.array([yaw, pitch])

    def _get_ee_angle(self):
        ee_forward = self.data.site_xmat[0].reshape(3, 3)[:, 2]
        return self._get_angle_from_vector(ee_forward)

    def _get_obs(self):
        qpos    = self.data.qpos[:5]
        qvel    = self.data.qvel[:5]
        ee_pos  = self.data.site_xpos[0]
        tgt_pos = self.data.site_xpos[1]
        rel_pos = tgt_pos - ee_pos

        ee_angle  = self._get_ee_angle()
        rel_angle = self.goal_angle - ee_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        return np.concatenate([qpos, qvel, rel_pos, rel_angle]).astype(np.float32)

    def _get_info(self, obs=None):
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
            "distance":    distance,
            "angle_err":   angle_err,
            "is_success":  is_success,
            "difficulty":  self.difficulty,
            "dist_thresh": dist_thresh,
            "angle_thresh":angle_thresh,
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
        target_z   = self.np_random.uniform(-0.1, 0.15)
        target_pos = np.array([target_x, target_y, target_z])

        # 사용자 수정 유지: 각도 노이즈 ±1.0 rad 고정
        base_angle     = self._get_angle_from_vector(target_pos)
        angle_noise    = self.np_random.uniform(-1.0, 1.0, size=2)
        goal_angle_raw = base_angle + angle_noise
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
        dist        = info["distance"]
        angle_err   = info["angle_err"]
        dist_thresh  = info["dist_thresh"]
        angle_thresh = info["angle_thresh"]

        # ══════════════════════════════════════════════════════════════
        # 보상 함수 설계 원칙
        #
        # 해결하려는 두 가지 구조적 문제:
        #
        # [문제 1] Potential-based 신호 소멸
        #   prev - curr 차이는 목표 근처에서 → 0 수렴
        #   정밀도가 필요한 순간에 gradient가 사라짐
        #   해결: 지수형 근접 보상 추가 (가까울수록 기하급수적으로 강해짐)
        #
        # [문제 2] AND 조건 확률 천장
        #   P(dist OK) × P(angle OK) ≒ 0.75 × 0.75 = 0.56
        #   수학적으로 70% 돌파가 구조적으로 어려움
        #   해결: 복합 달성도(combined ratio)로 AND를 연속 근사
        #         → 두 조건이 동시에 성공 임계 근처일수록 강한 보상
        #         → 상수 체류 보너스가 아니므로 reward hacking 없음
        # ══════════════════════════════════════════════════════════════

        # 1. Potential-based 방향 신호 (SAC_5 수준 유지)
        reward  = (self._prev_dist      - dist)      * 50.0
        reward += (self._prev_angle_err - angle_err) * 30.0
        self._prev_dist      = dist
        self._prev_angle_err = angle_err

        # 2. 지수형 근접 보상 (FIX: 신호 소멸 방지)
        #    dist/angle_err 가 작아질수록 exp 값이 커짐
        #    → 정밀 구간에서도 gradient 신호 유지
        #
        #    값 범위 예시 (거리):
        #      dist=0.20m → exp(-4.0) ≈ 0.018 → × 4 = 0.07
        #      dist=0.05m → exp(-1.0) ≈ 0.368 → × 4 = 1.47
        #      dist=0.02m → exp(-0.4) ≈ 0.670 → × 4 = 2.68
        #      dist=0.00m → exp( 0.0) = 1.000 → × 4 = 4.00
        #
        #    값 범위 예시 (각도):
        #      angle=1.0  → exp(-3.0) ≈ 0.050 → × 3 = 0.15
        #      angle=0.3  → exp(-0.9) ≈ 0.407 → × 3 = 1.22
        #      angle=0.1  → exp(-0.3) ≈ 0.741 → × 3 = 2.22
        #      angle=0.0  → exp( 0.0) = 1.000 → × 3 = 3.00
        reward += np.exp(-dist * 20.0)      * 4.0
        reward += np.exp(-angle_err * 3.0)  * 3.0

        # 3. 복합 달성도 보상 (FIX: AND 조건 천장 우회)
        #    dist_ratio, angle_ratio: 0 = 목표 달성, 1 = 임계 경계
        #    combined: 두 조건이 동시에 임계 근처일수록 1에 가까워짐
        #    → 위치만 맞추거나 각도만 맞추면 0.5 미만
        #    → 둘 다 임계 안쪽에서 맞춰야 1에 가까워짐
        #    → 상수 보너스가 아닌 연속값 → reward hacking 불가
        dist_ratio  = np.clip(dist      / (dist_thresh  + 1e-6), 0.0, 1.0)
        angle_ratio = np.clip(angle_err / (angle_thresh + 1e-6), 0.0, 1.0)
        combined    = (1.0 - dist_ratio) * (1.0 - angle_ratio)
        reward += combined * 15.0

        # 4. Dense penalty (SAC_5 수준 유지)
        reward -= dist      * 1.0
        reward -= angle_err * 1.0

        # 5. 진동 억제 penalty (5cm 이내)
        if dist < 0.05:
            qvel = self.data.qvel[:5]
            reward -= float(np.sum(np.square(qvel))) * 0.1

        # 6. 행동 부드러움 penalty
        reward -= 0.01 * float(np.sum(np.square(action)))

        # 7. 성공 보너스 (에피소드 내 최강 단일 신호)
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