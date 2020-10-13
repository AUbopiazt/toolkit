#!/usr/bin/env bash
 
VERSION1=$1
VERSION2=$2
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-$VERSION1.$VERSION2 /usr/local/cuda
nvcc --version

