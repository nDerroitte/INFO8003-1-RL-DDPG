#!/bin/sh

python3 run.py
cp actor_wS.h5 safe/safework3/
cp critic_wS.h5 safe/safework3/

python3 run.py
python3 run.py
