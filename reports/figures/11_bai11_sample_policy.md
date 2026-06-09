| sample_state                            | state_labels                    |   recommended_action | action_name     | allocation             |   q_value |
|:----------------------------------------|:--------------------------------|---------------------:|:----------------|:-----------------------|----------:|
| VN 2026                                 | medium | medium | low | medium  |                    1 | a1 Cân bằng     | [0.4, 0.25, 0.15, 0.2] |   14.8443 |
| GDP low, D low, AI low, U high          | low | low | low | high          |                    1 | a1 Cân bằng     | [0.4, 0.25, 0.15, 0.2] |   13.3117 |
| GDP high, D high, AI high, U low        | high | high | high | low        |                    0 | a0 Truyền thống | [0.7, 0.1, 0.1, 0.1]   |   14.8604 |
| GDP medium, D high, AI medium, U medium | medium | high | medium | medium |                    0 | a0 Truyền thống | [0.7, 0.1, 0.1, 0.1]   |   14.7627 |
| GDP low, D medium, AI high, U high      | low | medium | high | high      |                    0 | a0 Truyền thống | [0.7, 0.1, 0.1, 0.1]   |   10.0224 |