# Collecting Play Data on the DROID

The following code supports:

* Object goes out of bounds (recover)
* Connection interruption (restart)
* Unexpected failure (stop)

## Getting Started

Instructions on Running the Franka Robot：https://github.com/irom-princeton/wiki/wiki/Franka-Arm-DROID

# Hybrid Supervisor Usage

## 1. Start Robot Server (NUC)
```bash
ssh 172.16.0.5 # password is " "
tmux new -s robot_server
conda activate polymetis-local
cd droid

python scripts/server/run_server.py
# detach with: Ctrl-b d
```

## 2. Run Supervisor (Laptop)
```bash
python controller.py
```
Automatically:
- runs main.py
- restarts on camera/robot errors
- restarts robot server via tmux if needed
- stops after too many failures

## 3. Monitor Logs
Supervisor logs: shown locally
```bash
ssh 172.16.0.5
tmux attach -t robot_server

# detach: Ctrl-b d
```


## 4. Stop Everything
```bash
# Stop supervisor locally: Ctrl-C

ssh 172.16.0.5
tmux kill-session -t robot_server
```

<!-- ```bash
# Starting the play-data collection
python examples/droid/main.py --remote_host=0.0.0.0 --remote_port=8000 --external_camera=right --long_data_collection --date_str=2025_12_04_test

python 
``` -->

### Expected Behavior

1. When reset happen X times in a row: remove safety filter for next Y iteractions (until no reset) and give opportunity for robot to recover object.

2. When program (main.py) terminates unexpectedly: re-try after sleep. 

3. Safety filter (always running): if observation looks vastly different from expected, quit the program.