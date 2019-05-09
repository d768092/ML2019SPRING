#! /usr/bin/bash
python3 w2v.py "$1" "$3" "$4"
python3 train_LSTM2.py "$1" "$2" "$4"
python3 train_GRU.py "$1" "$2" "$4"
python3 train_conv.py "$1" "$2" "$4"
