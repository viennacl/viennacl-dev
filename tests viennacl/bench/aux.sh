#!/bin/bash

#auxilary script to quickly bench new changes

FLOAT="float"
DOUBLE="double"
FAST="fast"

# enable fast mode which skips transposed cases in matrix-matrix multiplication
if [[ $1 = $FAST || $2 = $FAST || $3 = $FAST ]]; then
    FAST="fast"
else
    FAST=""
fi

# bench with foat entries
if [[ $1 = $FLOAT || $2 = $FLOAT || $3 = $FLOAT ]]; then

    printf "starting benchmark with floats!\n"
    
    # overwrite previous data
    printf "" > float_aux_data

    for i in {1000..2000..50}
    do
        printf %s "$(echo $i) " >> float_aux_data
        printf %s "$(../build/bench_viennacl_avx2 $i float $FAST)" >> float_aux_data
        printf "\n" >> float_aux_data
    done
fi

# bench with double entries
if [[ $1 = $DOUBLE || $2 = $DOUBLE ||  $3 = $DOUBLE ]]; then

    printf "starting benchmark with doubles!\n"
    
    # overwrite previous data
    printf "" > double_aux_data

    for i in {1000..2000..50}
    do
        printf %s "$(echo $i) " >> double_aux_data
        printf %s "$(../build/bench_viennacl_avx2 $i double $FAST)" >> double_aux_data
        printf "\n" >> double_aux_data
    done
fi
