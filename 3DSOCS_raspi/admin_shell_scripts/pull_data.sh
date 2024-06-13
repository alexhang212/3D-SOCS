#!/bin/bash

# Create a new session named "pullsess"
tmux new-session -d -s pullsess


# Array of IP addresses and corresponding destination directories
ip_addresses=("xxx.xxx.xxx.xxx" "xxx.xxx.xxx.xxx" "xxx.xxx.xxx.xxx" "xxx.xxx.xxx.xxx" "xxx.xxx.xxx.xxx" "xxx.xxx.xxx.xxx")
dest_directories=("CM4_1" "CM4_2" "CM4_3" "CM4_4" "CM4_5" "CM4_6")
username="pi"

# Create an array to store the PIDs of rsync commands
rsync_pids=()

# Create a window for the 3x3 grid
tmux split-window -h -t pullsess:0
tmux split-window -h -t pullsess:0
tmux split-window -v -t pullsess:0.0
tmux split-window -v -t pullsess:0.1
tmux split-window -v -t pullsess:0.2
tmux split-window -v -t pullsess:0.3

# Loop through the IP addresses and destination directories to open panes and execute rsync
for ((i = 0; i < ${#ip_addresses[@]}; i++)); do
    ip="${ip_addresses[i]}"
    dest_dir="/your/destination/directory/$(date +'%Y_%m_%d')/${dest_directories[i]}"
    
    #if shutdown argument
    if [ "$1" == "shutdown" ]; then
    tmux send-keys -t pullsess:0.$i "run_rsync $ip $dest_dir && ssh $username@$ip sudo shutdown" Enter
    else
    tmux send-keys -t pullsess:0.$i "run_rsync $ip $dest_dir" Enter
    fi
    # Store the PID of the rsync command in the array
    rsync_pids+=($!)
done


# Set pane synchronization
tmux set-window-option -t pullsess:0 synchronize-panes on

# Set pane layout for the 3x3 grid window
tmux select-layout -t pullsess:0 tiled

# Attach to session named "pullsess"
tmux attach -t pullsess



