#! /usr/bin/bash
wget https://www.csie.ntu.edu.tw/~b06902067/LSTM2-2.model
wget https://www.csie.ntu.edu.tw/~b06902067/GRU-2.model
wget https://www.csie.ntu.edu.tw/~b06902067/conv.model
python3 test.py "$1" "$2" "$3"
