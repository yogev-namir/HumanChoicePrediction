import wandb
YOUR_WANDB_USERNAME = "yogev-avishag"
project = "NLP2024_PROJECT_yogev-avishag"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: Dropout Rate test 08.06.2024",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 2))},
        "features": {"values": ["EFs"]},
        "dropout": {"values": [0.6, 0.4, 0.2, 0.0]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
