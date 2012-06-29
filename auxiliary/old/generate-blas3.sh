#!/bin/bash

############### Step 1: Matrix-Matrix products #########################

# all column-major:
./generate-blas3-prod-align1 0 0 0 0 0 > matrix_prod_col_col_col/align1/prod_AA.cl
./generate-blas3-prod-align1 0 0 0 0 1 > matrix_prod_col_col_col/align1/prod_AT.cl
./generate-blas3-prod-align1 0 0 0 1 0 > matrix_prod_col_col_col/align1/prod_TA.cl
./generate-blas3-prod-align1 0 0 0 1 1 > matrix_prod_col_col_col/align1/prod_TT.cl

# C row-major, others column-major:
./generate-blas3-prod-align1 0 0 1 0 0 > matrix_prod_col_col_row/align1/prod_AA.cl
./generate-blas3-prod-align1 0 0 1 0 1 > matrix_prod_col_col_row/align1/prod_AT.cl
./generate-blas3-prod-align1 0 0 1 1 0 > matrix_prod_col_col_row/align1/prod_TA.cl
./generate-blas3-prod-align1 0 0 1 1 1 > matrix_prod_col_col_row/align1/prod_TT.cl

# B row-major, others column-major:
./generate-blas3-prod-align1 0 1 0 0 0 > matrix_prod_col_row_col/align1/prod_AA.cl
./generate-blas3-prod-align1 0 1 0 0 1 > matrix_prod_col_row_col/align1/prod_AT.cl
./generate-blas3-prod-align1 0 1 0 1 0 > matrix_prod_col_row_col/align1/prod_TA.cl
./generate-blas3-prod-align1 0 1 0 1 1 > matrix_prod_col_row_col/align1/prod_TT.cl

# A column-major, others row-major:
./generate-blas3-prod-align1 0 1 1 0 0 > matrix_prod_col_row_row/align1/prod_AA.cl
./generate-blas3-prod-align1 0 1 1 0 1 > matrix_prod_col_row_row/align1/prod_AT.cl
./generate-blas3-prod-align1 0 1 1 1 0 > matrix_prod_col_row_row/align1/prod_TA.cl
./generate-blas3-prod-align1 0 1 1 1 1 > matrix_prod_col_row_row/align1/prod_TT.cl

# A row-major, others column-major:
./generate-blas3-prod-align1 1 0 0 0 0 > matrix_prod_row_col_col/align1/prod_AA.cl
./generate-blas3-prod-align1 1 0 0 0 1 > matrix_prod_row_col_col/align1/prod_AT.cl
./generate-blas3-prod-align1 1 0 0 1 0 > matrix_prod_row_col_col/align1/prod_TA.cl
./generate-blas3-prod-align1 1 0 0 1 1 > matrix_prod_row_col_col/align1/prod_TT.cl

# A row-major, B column-major, C row-major:
./generate-blas3-prod-align1 1 0 1 0 0 > matrix_prod_row_col_row/align1/prod_AA.cl
./generate-blas3-prod-align1 1 0 1 0 1 > matrix_prod_row_col_row/align1/prod_AT.cl
./generate-blas3-prod-align1 1 0 1 1 0 > matrix_prod_row_col_row/align1/prod_TA.cl
./generate-blas3-prod-align1 1 0 1 1 1 > matrix_prod_row_col_row/align1/prod_TT.cl

# A, B row-major, C column-major:
./generate-blas3-prod-align1 1 1 0 0 0 > matrix_prod_row_row_col/align1/prod_AA.cl
./generate-blas3-prod-align1 1 1 0 0 1 > matrix_prod_row_row_col/align1/prod_AT.cl
./generate-blas3-prod-align1 1 1 0 1 0 > matrix_prod_row_row_col/align1/prod_TA.cl
./generate-blas3-prod-align1 1 1 0 1 1 > matrix_prod_row_row_col/align1/prod_TT.cl

# all row-major
./generate-blas3-prod-align1 1 1 1 0 0 > matrix_prod_row_row_row/align1/prod_AA.cl
./generate-blas3-prod-align1 1 1 1 0 1 > matrix_prod_row_row_row/align1/prod_AT.cl
./generate-blas3-prod-align1 1 1 1 1 0 > matrix_prod_row_row_row/align1/prod_TA.cl
./generate-blas3-prod-align1 1 1 1 1 1 > matrix_prod_row_row_row/align1/prod_TT.cl


############### Step 2: Matrix-Matrix triangular solver #########################

# all col-major
./generate-blas3-solve-align1 0 0 0 0 0 0 > matrix_solve_col_col/align1/lower_solve.cl
./generate-blas3-solve-align1 0 0 0 0 0 1 > matrix_solve_col_col/align1/unit_lower_solve.cl
./generate-blas3-solve-align1 0 0 0 0 1 0 > matrix_solve_col_col/align1/upper_solve.cl
./generate-blas3-solve-align1 0 0 0 0 1 1 > matrix_solve_col_col/align1/unit_upper_solve.cl
./generate-blas3-solve-align1 0 0 0 1 0 0 > matrix_solve_col_col/align1/lower_trans_solve.cl
./generate-blas3-solve-align1 0 0 0 1 0 1 > matrix_solve_col_col/align1/unit_lower_trans_solve.cl
./generate-blas3-solve-align1 0 0 0 1 1 0 > matrix_solve_col_col/align1/upper_trans_solve.cl
./generate-blas3-solve-align1 0 0 0 1 1 1 > matrix_solve_col_col/align1/unit_upper_trans_solve.cl
./generate-blas3-solve-align1 0 0 1 0 0 0 > matrix_solve_col_col/align1/trans_lower_solve.cl
./generate-blas3-solve-align1 0 0 1 0 0 1 > matrix_solve_col_col/align1/trans_unit_lower_solve.cl
./generate-blas3-solve-align1 0 0 1 0 1 0 > matrix_solve_col_col/align1/trans_upper_solve.cl
./generate-blas3-solve-align1 0 0 1 0 1 1 > matrix_solve_col_col/align1/trans_unit_upper_solve.cl
./generate-blas3-solve-align1 0 0 1 1 0 0 > matrix_solve_col_col/align1/trans_lower_trans_solve.cl
./generate-blas3-solve-align1 0 0 1 1 0 1 > matrix_solve_col_col/align1/trans_unit_lower_trans_solve.cl
./generate-blas3-solve-align1 0 0 1 1 1 0 > matrix_solve_col_col/align1/trans_upper_trans_solve.cl
./generate-blas3-solve-align1 0 0 1 1 1 1 > matrix_solve_col_col/align1/trans_unit_upper_trans_solve.cl

# A col-major, B row_major
./generate-blas3-solve-align1 0 1 0 0 0 0 > matrix_solve_col_row/align1/lower_solve.cl
./generate-blas3-solve-align1 0 1 0 0 0 1 > matrix_solve_col_row/align1/unit_lower_solve.cl
./generate-blas3-solve-align1 0 1 0 0 1 0 > matrix_solve_col_row/align1/upper_solve.cl
./generate-blas3-solve-align1 0 1 0 0 1 1 > matrix_solve_col_row/align1/unit_upper_solve.cl
./generate-blas3-solve-align1 0 1 0 1 0 0 > matrix_solve_col_row/align1/lower_trans_solve.cl
./generate-blas3-solve-align1 0 1 0 1 0 1 > matrix_solve_col_row/align1/unit_lower_trans_solve.cl
./generate-blas3-solve-align1 0 1 0 1 1 0 > matrix_solve_col_row/align1/upper_trans_solve.cl
./generate-blas3-solve-align1 0 1 0 1 1 1 > matrix_solve_col_row/align1/unit_upper_trans_solve.cl
./generate-blas3-solve-align1 0 1 1 0 0 0 > matrix_solve_col_row/align1/trans_lower_solve.cl
./generate-blas3-solve-align1 0 1 1 0 0 1 > matrix_solve_col_row/align1/trans_unit_lower_solve.cl
./generate-blas3-solve-align1 0 1 1 0 1 0 > matrix_solve_col_row/align1/trans_upper_solve.cl
./generate-blas3-solve-align1 0 1 1 0 1 1 > matrix_solve_col_row/align1/trans_unit_upper_solve.cl
./generate-blas3-solve-align1 0 1 1 1 0 0 > matrix_solve_col_row/align1/trans_lower_trans_solve.cl
./generate-blas3-solve-align1 0 1 1 1 0 1 > matrix_solve_col_row/align1/trans_unit_lower_trans_solve.cl
./generate-blas3-solve-align1 0 1 1 1 1 0 > matrix_solve_col_row/align1/trans_upper_trans_solve.cl
./generate-blas3-solve-align1 0 1 1 1 1 1 > matrix_solve_col_row/align1/trans_unit_upper_trans_solve.cl

# A row-major, B col-major
./generate-blas3-solve-align1 1 0 0 0 0 0 > matrix_solve_row_col/align1/lower_solve.cl
./generate-blas3-solve-align1 1 0 0 0 0 1 > matrix_solve_row_col/align1/unit_lower_solve.cl
./generate-blas3-solve-align1 1 0 0 0 1 0 > matrix_solve_row_col/align1/upper_solve.cl
./generate-blas3-solve-align1 1 0 0 0 1 1 > matrix_solve_row_col/align1/unit_upper_solve.cl
./generate-blas3-solve-align1 1 0 0 1 0 0 > matrix_solve_row_col/align1/lower_trans_solve.cl
./generate-blas3-solve-align1 1 0 0 1 0 1 > matrix_solve_row_col/align1/unit_lower_trans_solve.cl
./generate-blas3-solve-align1 1 0 0 1 1 0 > matrix_solve_row_col/align1/upper_trans_solve.cl
./generate-blas3-solve-align1 1 0 0 1 1 1 > matrix_solve_row_col/align1/unit_upper_trans_solve.cl
./generate-blas3-solve-align1 1 0 1 0 0 0 > matrix_solve_row_col/align1/trans_lower_solve.cl
./generate-blas3-solve-align1 1 0 1 0 0 1 > matrix_solve_row_col/align1/trans_unit_lower_solve.cl
./generate-blas3-solve-align1 1 0 1 0 1 0 > matrix_solve_row_col/align1/trans_upper_solve.cl
./generate-blas3-solve-align1 1 0 1 0 1 1 > matrix_solve_row_col/align1/trans_unit_upper_solve.cl
./generate-blas3-solve-align1 1 0 1 1 0 0 > matrix_solve_row_col/align1/trans_lower_trans_solve.cl
./generate-blas3-solve-align1 1 0 1 1 0 1 > matrix_solve_row_col/align1/trans_unit_lower_trans_solve.cl
./generate-blas3-solve-align1 1 0 1 1 1 0 > matrix_solve_row_col/align1/trans_upper_trans_solve.cl
./generate-blas3-solve-align1 1 0 1 1 1 1 > matrix_solve_row_col/align1/trans_unit_upper_trans_solve.cl

# all row-major
./generate-blas3-solve-align1 1 1 0 0 0 0 > matrix_solve_row_row/align1/lower_solve.cl
./generate-blas3-solve-align1 1 1 0 0 0 1 > matrix_solve_row_row/align1/unit_lower_solve.cl
./generate-blas3-solve-align1 1 1 0 0 1 0 > matrix_solve_row_row/align1/upper_solve.cl
./generate-blas3-solve-align1 1 1 0 0 1 1 > matrix_solve_row_row/align1/unit_upper_solve.cl
./generate-blas3-solve-align1 1 1 0 1 0 0 > matrix_solve_row_row/align1/lower_trans_solve.cl
./generate-blas3-solve-align1 1 1 0 1 0 1 > matrix_solve_row_row/align1/unit_lower_trans_solve.cl
./generate-blas3-solve-align1 1 1 0 1 1 0 > matrix_solve_row_row/align1/upper_trans_solve.cl
./generate-blas3-solve-align1 1 1 0 1 1 1 > matrix_solve_row_row/align1/unit_upper_trans_solve.cl
./generate-blas3-solve-align1 1 1 1 0 0 0 > matrix_solve_row_row/align1/trans_lower_solve.cl
./generate-blas3-solve-align1 1 1 1 0 0 1 > matrix_solve_row_row/align1/trans_unit_lower_solve.cl
./generate-blas3-solve-align1 1 1 1 0 1 0 > matrix_solve_row_row/align1/trans_upper_solve.cl
./generate-blas3-solve-align1 1 1 1 0 1 1 > matrix_solve_row_row/align1/trans_unit_upper_solve.cl
./generate-blas3-solve-align1 1 1 1 1 0 0 > matrix_solve_row_row/align1/trans_lower_trans_solve.cl
./generate-blas3-solve-align1 1 1 1 1 0 1 > matrix_solve_row_row/align1/trans_unit_lower_trans_solve.cl
./generate-blas3-solve-align1 1 1 1 1 1 0 > matrix_solve_row_row/align1/trans_upper_trans_solve.cl
./generate-blas3-solve-align1 1 1 1 1 1 1 > matrix_solve_row_row/align1/trans_unit_upper_trans_solve.cl

