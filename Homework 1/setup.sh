#!/usr/bin/env bash

set -e

if [ ! -d "./venv" ]; then
    python3 -m venv venv
fi

export R_LIBS_USER=~/.r

r_setup_script=$(cat <<-END
    if (!require("tourr")) install.packages("tourr", repos="https://cran.uni-muenster.de/");
    if (!require("gifski")) install.packages("gifski", repos="https://cran.uni-muenster.de/");
END
)

. ./venv/bin/activate
pip3 install -r requirements.txt
Rscript -e $r_setup_script
