## TODOs

- Purge or invalidate legacy Neo4j summaries without cache metadata.
- Add explicit release-date calendars for macro series and transcripts.
- Pin embedding/LLM model versions and store prompt + model hashes.
- Add walk-forward evaluation and holdout test splits.
- Add baseline strategies (buy-and-hold, random, SMA crossover).
- Log action distribution and position sizing per method.
- Improve DQN stability (reward scaling, target updates, replay balance).
- Pre-train RL models (DQN/Q-Learning) on 50+ years of diverse market data (e.g., S&P 500) to learn "priors" before fine-tuning on specific assets. Runing models from scratch on <200 quarterly samples is insufficient for learning general trading logic.
