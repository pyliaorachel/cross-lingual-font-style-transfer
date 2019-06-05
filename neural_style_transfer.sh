#!/bin/bash
#C_DIR = "data/pingfang/val/*"
#S_IMG = "data/chalk/e.png"

for c in data/pingfang/val/*
do
        python -m project.src.style_transfer.neural_style_transfer.train --content $c --style data/chalk/e.png --output project/output/neural/val/  --imsize 128 --epochs 150 --log-epochs 5
done
