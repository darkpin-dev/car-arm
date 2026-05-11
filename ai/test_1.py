import os
import time
import mujoco.viewer
from stable_baselines3 import SAC
import utils


def test():
    set = utils.test_setting()

    model_path = os.path.join(set.models_dir, "best_model")
    model = SAC.load(model_path)

    env = set.env(render_mode="rgb")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(10):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward

                done = terminated or truncated

                viewer.sync()
                time.sleep(1 / 100)

            print(f"에피소드 {ep + 1} 결과 | 총 보상: {total_reward:.2f} | 진행 스텝: {info.get('step', 0)} | 성공 여부: {info.get('success', False)}")

    env.close()


if __name__ == "__main__":
    test()
