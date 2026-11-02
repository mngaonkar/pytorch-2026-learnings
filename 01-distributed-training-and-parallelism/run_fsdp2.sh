#!/bin/sh

torchrun --nproc_per_node=1 fsdp2_train.py
