import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from car_arm_env import CarArmEnv
from curriculum_callback import CurriculumCallback


def make_env(rank, seed=0):
    def _init():
        env = CarArmEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train():
    N_ENVS     = 12
    BATCH_SIZE = 1024
    RUN_NAME   = "SAC"

    _tmp_env = CarArmEnv(render_mode=None)
    ACTION_DIM = _tmp_env.action_space.shape[0]
    _tmp_env.close()
    print(f"Action dim: {ACTION_DIM}")

    # ── 학습 환경 ──────────────────────────────────────────────────────
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── 평가 환경 ──────────────────────────────────────────────────────
    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)
    # ✅ FIX 5: training=False 유지, sync는 EvalCallback에서 처리
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ── 장치 설정 ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"장치: {device}")

    # ── 디렉토리 ─────────────────────────────────────────────────────
    log_dir   = f"./logs/{RUN_NAME}/"
    model_dir = f"./models/{RUN_NAME}/"
    best_dir  = f"./models/{RUN_NAME}/best/"
    os.makedirs(model_dir, exist_ok=True)

    # ── 모델 ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=[256, 256, 256],
        activation_fn=torch.nn.ReLU,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=2_000_000,
        batch_size=BATCH_SIZE,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        learning_starts=10_000,
        ent_coef="auto",
        target_entropy=-float(ACTION_DIM),
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/",
        device=device,
    )

    # ── 콜백 ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=model_dir,
        name_prefix=RUN_NAME,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    curriculum_cb = CurriculumCallback(
        eval_env=eval_env,
        window_size=200,
        threshold_up=0.70,
        threshold_down=0.30,
        difficulty_step=0.1,
        verbose=1,
    )

    # ── 학습 ─────────────────────────────────────────────────────────
    print(f"{RUN_NAME} | lr=3e-4 | grad_steps=4 | envs={N_ENVS} | Curriculum ON")
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_cb, eval_cb, curriculum_cb],
        tb_log_name=RUN_NAME,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    # ── 저장 ─────────────────────────────────────────────────────────
    model.save(f"{model_dir}/{RUN_NAME}_final")
    train_env.save(f"{model_dir}/vec_normalize_stats.pkl")
    model.save_replay_buffer(f"{model_dir}/replay_buffer.pkl")
    print(f"완료! {model_dir}에 저장됨")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()