#!/bin/bash

# This script calls the OpenBLAS, Eigen and ViennaCL (with avx) benchmarks.
# The according executables should be in '../build' ! Output is generated in './' .
# Up to two arguments (double and/or float) specify with what data-type the matrices should be filled.

FLOAT="float"
DOUBLE="double"

# bench with foat entries
if [[ $1 = $FLOAT || $2 = $FLOAT ]]; then

    # overwrite previous data
    printf "" > float_data_all

    for i in {100..3000..100}
    do
        printf %s "$(echo $i) " >> float_data_all
        printf %s "$(../build/bench_viennacl_avx2 $i float)" >> float_data_all
        printf %s "$(../build/bench_openblas $i float)" >> float_data_all
        printf %s "$(../build/bench_eigen $i float)" >> float_data_all
        printf "\n" >> float_data_all
    done

    printf %s "$(cat float_data_all.conf|gnuplot)"
fi

# bench with double entries
if [[ $1 = $DOUBLE ||  $2 = $DOUBLE ]]; then

    # overwrite previous data
    printf "" > double_data_all

    for i in {100..3000..100}
    do
        printf %s "$(echo $i) " >> double_data_all
        printf %s "$(../build/bench_viennacl_avx2 $i double)" >> double_data_all
        printf %s "$(../build/bench_openblas $i double)" >> double_data_all
        printf %s "$(../build/bench_eigen $i double)" >> double_data_all
        printf "\n" >> double_data_all
    done

    printf %s "$(cat double_data_all.conf|gnuplot)"
fi
