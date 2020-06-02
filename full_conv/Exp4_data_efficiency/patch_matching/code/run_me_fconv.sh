#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs_fconv"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"


( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet_fconv.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --n-triplets 50000 --training-set yosemite --experiment-name=liberty_train/ $@ | tee -a "$DATALOGS/log_HardNet_Lib_fconv.log"

)




( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet_fconv.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --n-triplets 100000 --training-set yosemite --experiment-name=liberty_train/ $@ | tee -a "$DATALOGS/log_HardNet_Lib_fconv.log"

)




( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet_fconv.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --n-triplets 250000 --training-set yosemite --experiment-name=liberty_train/ $@ | tee -a "$DATALOGS/log_HardNet_Lib_fconv.log"

)




( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)

( # Run the code
    cd "$RUNPATH"
    python ./code/HardNet_fconv.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --n-triplets 500000 --training-set yosemite --experiment-name=liberty_train/ $@ | tee -a "$DATALOGS/log_HardNet_Lib_fconv.log"
    #python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=liberty_train_with_aug/  $@ | tee -a "$DATALOGS/log_HardNetPlus_Lib.log"
)
