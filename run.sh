#!/bin/bash

ssh taco@fourier /bin/bash <<EOT
    eval "\$(bin/micromamba shell hook --shell bash)"
    cd src/manatees
    micromamba activate manatees 2>/dev/null
    nohup python -u manatees.py $@ > out.log | tail -F out.log &
EOT
