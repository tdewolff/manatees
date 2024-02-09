#!/bin/bash

ssh taco@fourier -t "eval \"\$(bin/micromamba shell hook --shell bash)\"; cd src/manatees; micromamba activate manatees 2>/dev/null; python manatees.py $@"
