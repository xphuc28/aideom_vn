from __future__ import annotations

from src.bai11_q_learning import VietnamEconomyEnv, train_q_learning


def test_bai11_q_shape_is_correct():
    result = train_q_learning(n_episodes=20, seed=123)

    assert result["Q"].shape == (3, 3, 3, 3, 5)


def test_bai11_rewards_length_matches_episodes():
    n_episodes = 25
    result = train_q_learning(n_episodes=n_episodes, seed=123)

    assert len(result["rewards"]) == n_episodes


def test_bai11_step_returns_valid_state():
    env = VietnamEconomyEnv(seed=123)
    env.reset(seed=123, initial_state=("medium", "medium", "low", "medium"))
    next_state, reward, done, info = env.step(1)

    assert len(next_state) == 4
    assert all(0 <= value <= 2 for value in next_state)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert {"delta_GDP", "delta_unemployment", "cyber_risk", "emission", "reward"}.issubset(info)
