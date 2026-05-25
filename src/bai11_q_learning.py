"""Bai 11: Q-learning for adaptive economic policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MODULE_TITLE = "Bài 11 - Q-learning cho chính sách kinh tế thích nghi"

LEVELS = ["low", "medium", "high"]
LEVEL_TO_INDEX = {name: index for index, name in enumerate(LEVELS)}
STATE_COLUMNS = ["GDP_growth", "Digital_index", "AI_capacity", "Unemployment_risk"]

ACTION_ALLOCATIONS = {
    0: [0.70, 0.10, 0.10, 0.10],
    1: [0.40, 0.25, 0.15, 0.20],
    2: [0.25, 0.45, 0.15, 0.15],
    3: [0.20, 0.20, 0.45, 0.15],
    4: [0.30, 0.20, 0.10, 0.40],
}

ACTION_NAMES = {
    0: "a0 Truyền thống",
    1: "a1 Cân bằng",
    2: "a2 Số hóa nhanh",
    3: "a3 AI dẫn dắt",
    4: "a4 Bao trùm",
}

SAMPLE_STATES = {
    "VN 2026": ("medium", "medium", "low", "medium"),
    "GDP low, D low, AI low, U high": ("low", "low", "low", "high"),
    "GDP high, D high, AI high, U low": ("high", "high", "high", "low"),
    "GDP medium, D high, AI medium, U medium": ("medium", "high", "medium", "medium"),
    "GDP low, D medium, AI high, U high": ("low", "medium", "high", "high"),
}


def module_status() -> str:
    """Return a short implementation status for the Streamlit page."""
    return "Đã triển khai MDP 81 trạng thái, 5 hành động và Q-learning cho chính sách thích nghi."


def action_name(action: int) -> str:
    """Return the human-readable name for an action."""
    return ACTION_NAMES[int(action)]


def state_to_tuple(state) -> tuple[int, int, int, int]:
    """Convert state labels or integer tuples to a 4-index state tuple."""
    if len(state) != 4:
        raise ValueError("State phải có 4 yếu tố.")
    if all(isinstance(value, str) for value in state):
        return tuple(LEVEL_TO_INDEX[value] for value in state)
    return tuple(int(value) for value in state)


def state_to_labels(state) -> tuple[str, str, str, str]:
    """Convert an integer state tuple to level labels."""
    indices = state_to_tuple(state)
    return tuple(LEVELS[index] for index in indices)


@dataclass
class VietnamEconomyEnv:
    """Small custom MDP environment for adaptive economic policy."""

    horizon: int = 12
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.state = (1, 1, 1, 1)
        self.t = 0

    def reset(self, seed: int | None = None, initial_state=None):
        """Reset the environment and return the initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        if initial_state is None:
            self.state = tuple(int(value) for value in self.rng.integers(0, 3, size=4))
        else:
            self.state = state_to_tuple(initial_state)
        return self.state

    def _policy_effects(self, action: int) -> dict[str, float]:
        gdp, digital, ai, unemployment = self.state
        traditional, digital_share, ai_share, inclusive = ACTION_ALLOCATIONS[int(action)]
        noise = self.rng.normal(0.0, 0.03, size=4)

        delta_gdp = (
            0.35
            + 0.22 * gdp
            + 1.10 * traditional
            + 1.25 * digital_share * (1.0 + (2 - digital) / 3.0)
            + 1.45 * ai_share * (1.0 + (2 - ai) / 3.0)
            + 0.55 * inclusive
            - 0.30 * unemployment
            + noise[0]
        )
        delta_unemployment = (
            0.32 * ai_share * (1.0 + ai / 3.0)
            + 0.10 * digital_share
            - 0.55 * inclusive
            - 0.16 * traditional
            + 0.12 * (1 if gdp == 0 else 0)
            + noise[1]
        )
        cyber_risk = max(
            0.0,
            0.12 + 0.52 * ai_share + 0.32 * digital_share + 0.07 * ai - 0.14 * inclusive + noise[2],
        )
        emission = max(
            0.0,
            0.10 + 0.38 * traditional + 0.20 * digital_share + 0.24 * ai_share - 0.12 * inclusive + 0.04 * gdp + noise[3],
        )
        reward = 0.40 * delta_gdp - 0.25 * delta_unemployment - 0.20 * cyber_risk - 0.15 * emission
        return {
            "delta_GDP": float(delta_gdp),
            "delta_unemployment": float(delta_unemployment),
            "cyber_risk": float(cyber_risk),
            "emission": float(emission),
            "reward": float(reward),
        }

    @staticmethod
    def _move_level(level: int, signal: float, up_threshold: float, down_threshold: float) -> int:
        if signal > up_threshold:
            return min(2, level + 1)
        if signal < down_threshold:
            return max(0, level - 1)
        return level

    def step(self, action: int):
        """Apply one action and return next_state, reward, done, info."""
        if int(action) not in ACTION_ALLOCATIONS:
            raise ValueError("Action không hợp lệ.")

        gdp, digital, ai, unemployment = self.state
        effects = self._policy_effects(int(action))
        traditional, digital_share, ai_share, inclusive = ACTION_ALLOCATIONS[int(action)]

        next_gdp = self._move_level(gdp, effects["delta_GDP"], 1.15, 0.65)
        next_digital = self._move_level(digital, digital_share + 0.25 * effects["delta_GDP"], 0.55, 0.18)
        next_ai = self._move_level(ai, ai_share + 0.20 * digital_share, 0.48, 0.15)
        unemployment_signal = effects["delta_unemployment"] - 0.08 * traditional - 0.18 * inclusive
        if unemployment_signal > 0.14:
            next_unemployment = min(2, unemployment + 1)
        elif unemployment_signal < -0.12:
            next_unemployment = max(0, unemployment - 1)
        else:
            next_unemployment = unemployment

        self.state = (next_gdp, next_digital, next_ai, next_unemployment)
        self.t += 1
        done = self.t >= self.horizon
        return self.state, effects["reward"], done, effects


def _smooth_rewards(rewards: list[float], window: int = 100) -> np.ndarray:
    series = pd.Series(rewards, dtype=float)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def extract_policy(Q: np.ndarray) -> pd.DataFrame:
    """Extract the greedy policy from a Q table."""
    rows = []
    for gdp in range(3):
        for digital in range(3):
            for ai in range(3):
                for unemployment in range(3):
                    state = (gdp, digital, ai, unemployment)
                    action = int(np.argmax(Q[state]))
                    labels = state_to_labels(state)
                    rows.append(
                        {
                            "GDP_growth": labels[0],
                            "Digital_index": labels[1],
                            "AI_capacity": labels[2],
                            "Unemployment_risk": labels[3],
                            "action": action,
                            "action_name": action_name(action),
                            "q_value": float(Q[state + (action,)]),
                        }
                    )
    return pd.DataFrame(rows)


def _policy_action(policy_type: str, state: tuple[int, int, int, int], Q: np.ndarray | None, rng) -> int:
    if policy_type == "learned":
        if Q is None:
            raise ValueError("Q is required for learned policy evaluation.")
        return int(np.argmax(Q[state]))
    if policy_type == "always_a1":
        return 1
    if policy_type == "always_a3":
        return 3
    if policy_type == "random":
        return int(rng.integers(0, 5))
    raise ValueError("policy_type phải là learned, always_a1, always_a3 hoặc random.")


def evaluate_policy(
    policy_type: str,
    Q: np.ndarray | None = None,
    n_episodes: int = 300,
    horizon: int = 12,
    seed: int = 2026,
) -> dict[str, float]:
    """Evaluate learned and baseline policies."""
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for episode in range(n_episodes):
        env = VietnamEconomyEnv(horizon=horizon, seed=int(rng.integers(0, 1_000_000)))
        state = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        done = False
        while not done:
            action = _policy_action(policy_type, state, Q, rng)
            state, reward, done, _info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)

    return {
        "policy_type": policy_type,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }


def train_q_learning(
    n_episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    seed: int = 42,
    epsilon_start: float = 0.30,
    epsilon_end: float = 0.03,
    horizon: int = 12,
) -> dict[str, object]:
    """Train Q-learning over the 81-state, 5-action MDP."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5), dtype=float)
    rewards = []

    for episode in range(int(n_episodes)):
        env = VietnamEconomyEnv(horizon=horizon, seed=int(rng.integers(0, 1_000_000)))
        state = env.reset(seed=int(rng.integers(0, 1_000_000)))
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / max(n_episodes * 0.35, 1))
        total_reward = 0.0
        done = False

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, 5))
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done, _info = env.step(action)
            best_next = float(np.max(Q[next_state]))
            old_value = Q[state + (action,)]
            Q[state + (action,)] = old_value + alpha * (reward + gamma * best_next - old_value)
            state = next_state
            total_reward += reward

        rewards.append(float(total_reward))

    smoothed_rewards = _smooth_rewards(rewards)
    policy_df = extract_policy(Q)
    comparison_df = pd.DataFrame(
        [
            evaluate_policy("learned", Q=Q, seed=seed + 11),
            evaluate_policy("always_a1", Q=Q, seed=seed + 12),
            evaluate_policy("always_a3", Q=Q, seed=seed + 13),
            evaluate_policy("random", Q=Q, seed=seed + 14),
        ]
    )

    return {
        "rewards": rewards,
        "smoothed_rewards": smoothed_rewards,
        "Q": Q,
        "policy_df": policy_df,
        "comparison_df": comparison_df,
    }
