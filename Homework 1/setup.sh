#!/usr/bin/env bash

set -e

if [ ! -d "./venv" ]; then
    python3 -m venv venv
fi


r_setup_script=$(cat <<-END
    .libPaths(c("~/.r", .libPaths()));
    install.packages("IRkernel", repos="https://cran.uni-muenster.de/");
    install.packages("tourr", repos="https://cran.uni-muenster.de/");
    IRkernel::installspec();
END
)

. ./venv/bin/activate
pip3 install -r requirements.txt
Rscript -e $r_setup_script
