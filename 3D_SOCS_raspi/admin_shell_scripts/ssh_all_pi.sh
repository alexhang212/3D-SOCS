#!/bin/bash

username="pi"

# create a new session named "newsess"
tmux new-session -d -s newsess

#ssh into lead CM4
tmux send-keys -t newsess "ssh $username@xxx.xxx.xxx.xxx" Enter
tmux set-window-option -t newsess:0 synchronize-panes on

# create a new window for follower CM4
tmux new-window -t newsess:1
tmux split-window -h -t newsess:1
tmux split-window -h -t newsess:1
tmux split-window -v -t newsess:1.0
tmux split-window -v -t newsess:1.1
tmux split-window -v -t newsess:1.2
tmux split-window -v -t newsess:1.3

ssh_commands=(
    "ssh $username@xxx.xxx.xxx.xxx"
    "ssh $username@xxx.xxx.xxx.xxx"
    "ssh $username@xxx.xxx.xxx.xxx"
    "ssh $username@xxx.xxx.xxx.xxx"
    "ssh $username@xxx.xxx.xxx.xxx"
    "ssh $username@xxx.xxx.xxx.xxx"
)

#ssh into all follower CM4
for i in {0..6}; do
    tmux send-keys -t newsess:2.$i "${ssh_commands[i]}" Enter
done
tmux set-window-option -t newsess:1 synchronize-panes on

# set pane layout for the grid window
tmux select-layout -t newsess:2 tiled

# attach
tmux attach -t newsess
