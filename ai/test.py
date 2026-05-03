import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from car_arm_env import CarArmEnv

def test():
    RUN_NAME = "SAC"
    
    # 모델 및 정규화 통계 파일 경로 설정
    # 학습 코드에서 best_model은 ./models/SAC/best/ 에 저장됩니다.
    model_path = f"./models/{RUN_NAME}/best/best_model.zip" 
    stats_path = f"./models/{RUN_NAME}/vec_normalize_stats.pkl"
    
    if not os.path.exists(stats_path):
        print(f"오류: 정규화 파일({stats_path})을 찾을 수 없습니다.")
        # 최종 모델 경로 폴더에서 찾도록 폴백 (학습 코드의 모델 저장 경로 고려)
        stats_path = f"./models/{RUN_NAME}/vec_normalize_stats.pkl"
        
    # 1. 환경 생성 (단일 환경을 DummyVecEnv로 감싸기)
    def make_env():
        return CarArmEnv(render_mode="human")

    # 벡터화된 환경 생성
    env = DummyVecEnv([make_env])
    env.env_method("set_difficulty", 1.0)

    # 2. 학습 시 사용된 VecNormalize 상태 불러오기
    try:
        env = VecNormalize.load(stats_path, env)
        # 테스트 시에는 통계를 업데이트하지 않도록 설정
        env.training = False 
        env.norm_reward = False
        print(f"정규화 통계 '{stats_path}' 로드 성공!")
    except Exception as e:
        print(f"정규화 통계 로드 실패: {e}")
        return

    # 3. 학습된 SAC 모델 불러오기
    try:
        model = SAC.load(model_path, env=env)
        print(f"모델 '{model_path}' 로드 성공!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        # best_model이 없을 경우 final 모델 시도
        try:
            final_path = f"./models/{RUN_NAME}/{RUN_NAME}_final.zip"
            model = SAC.load(final_path, env=env)
            print(f"대체 모델 '{final_path}' 로드 성공!")
        except Exception as e2:
            print(f"최종 모델 로드 실패: {e2}")
            return

    # 4. 테스트 루프
    num_episodes = 10
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        # VecEnv를 사용하므로 done은 리스트/배열 형태입니다.
        while not done:
            # SAC는 predict 시 deterministic=True 권장
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done_arr, info_arr = env.step(action)
            
            # DummyVecEnv는 배열을 반환하므로 첫 번째 환경([0])의 데이터 추출
            total_reward += reward[0]
            done = done_arr[0]
            info = info_arr[0]
            
            time.sleep(0.01) 

        # 환경의 info 딕셔너리에 'distance' 키가 존재한다고 가정
        distance = info.get('distance', 0.0)
        angle_err = info.get('angle_err', 0.0)
        is_success = info.get('is_success', False)
        print(f"[{"성공" if is_success else "실패"}] 에피소드 {episode + 1} 총 보상: {total_reward:.2f}, 거리: {distance:.4f}, 각도: {angle_err:.4f}")

    print("\n모든 테스트가 완료되었습니다.")
    env.close()

if __name__ == "__main__":
    test()