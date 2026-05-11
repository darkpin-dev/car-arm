import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import utils

def train():
    setting = utils.train_setting()

    n_envs = 12

    env = SubprocVecEnv([lambda: setting.env(render_mode=None) for _ in range(n_envs)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        device=setting.device,
    )

    custom_logger = configure(setting.logs_dir, ["stdout", "tensorboard"])
    model.set_logger(custom_logger)

    eval_env = DummyVecEnv([lambda: setting.env(render_mode=None)])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=setting.models_dir,
        log_path=setting.models_dir,
        eval_freq=max(50000 // n_envs, 1),
        deterministic=True,
    )

    model.learn(
        total_timesteps=setting.step,
        callback=eval_callback,
        progress_bar=True
    )
    model.save(os.path.join(setting.models_dir, "final_model"))


if __name__ == "__main__":
    train()
