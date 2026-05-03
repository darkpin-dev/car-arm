import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    성공률 기반 커리큘럼 학습 콜백.

    - 난이도 변경 시 히스토리를 초기화하여 연속 상승을 방지합니다.
    - 새 난이도에서 window_size개 에피소드를 새로 쌓은 뒤 다시 판정합니다.

    Args:
        eval_env:        평가 환경 (VecEnv). None이면 학습 환경에만 반영.
        window_size:     성공률 계산에 사용할 에피소드 수 (기본값: 200).
        threshold_up:    성공률 ≥ 이 값 → 난이도 +difficulty_step (기본값: 0.70).
        threshold_down:  성공률 <  이 값 → 난이도 -difficulty_step (기본값: 0.30).
        difficulty_step: 한 번에 변경되는 난이도 크기 (기본값: 0.1).
        verbose:         0=무음, 1=난이도 변경 시 출력 (기본값: 1).
    """

    def __init__(
        self,
        eval_env=None,
        window_size: int = 200,
        threshold_up: float = 0.70,
        threshold_down: float = 0.30,
        difficulty_step: float = 0.1,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.window_size = window_size
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.difficulty_step = difficulty_step

        self.current_difficulty: float = 0.0
        self._episode_results: list[float] = []
        self._total_episodes: int = 0

    def _set_difficulty(self, difficulty: float) -> None:
        self.training_env.env_method("set_difficulty", difficulty)
        if self.eval_env is not None:
            self.eval_env.env_method("set_difficulty", difficulty)

    def _maybe_update_difficulty(self) -> None:
        # 아직 window_size만큼 쌓이지 않았으면 대기
        if len(self._episode_results) < self.window_size:
            return

        success_rate = float(np.mean(self._episode_results))

        new_difficulty = self.current_difficulty
        if success_rate >= self.threshold_up:
            new_difficulty = round(
                min(1.0, self.current_difficulty + self.difficulty_step), 4
            )
        elif success_rate < self.threshold_down:
            new_difficulty = round(
                max(0.0, self.current_difficulty - self.difficulty_step), 4
            )

        if new_difficulty != self.current_difficulty:
            direction = "↑ 증가" if new_difficulty > self.current_difficulty else "↓ 감소"
            self.current_difficulty = new_difficulty
            self._set_difficulty(self.current_difficulty)


            self._episode_results.clear()

            if self.verbose >= 1:
                print(
                    f"\n[Curriculum {direction}] "
                    f"난이도: {self.current_difficulty:.2f}  |  "
                    f"성공률: {success_rate:.1%}  |  "
                    f"누적 에피소드: {self._total_episodes}"
                )
        else:
            self._episode_results.pop(0)

    def _on_training_start(self) -> None:
        self._set_difficulty(self.current_difficulty)
        if self.verbose >= 1:
            print(
                f"[Curriculum] 커리큘럼 학습 시작  |  "
                f"초기 난이도: {self.current_difficulty:.2f}  |  "
                f"상승 임계: ≥{self.threshold_up:.0%}  |  "
                f"하강 임계: <{self.threshold_down:.0%}  |  "
                f"윈도우: {self.window_size}개 에피소드"
            )

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals["dones"]
        infos: list[dict] = self.locals["infos"]

        for done, info in zip(dones, infos):
            if not done:
                continue

            is_success = info.get("is_success", None)
            if is_success is None:
                is_success = info.get("episode", {}).get("is_success", False)

            self._episode_results.append(float(is_success))
            self._total_episodes += 1
            self._maybe_update_difficulty()

        # TensorBoard 로깅
        if self._total_episodes > 0:
            self.logger.record("curriculum/difficulty", self.current_difficulty)
            self.logger.record(
                "curriculum/success_rate_recent",
                float(np.mean(self._episode_results)) if self._episode_results else 0.0,
            )
            self.logger.record("curriculum/total_episodes", self._total_episodes)
            self.logger.record(
                "curriculum/window_fill",
                len(self._episode_results) / self.window_size,
            )

        return True