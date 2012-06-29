#!/bin/bash

rm *.*~

rm scalar/align1/*.cl~

rm vector/align1/*.cl~
rm vector/align4/*.cl~
rm vector/align16/*.cl~

rm matrix_row/align1/*.cl~
rm matrix_row/align16/*.cl~
rm matrix_col/align1/*.cl~
rm matrix_col/align16/*.cl~

rm matrix_prod_row_row_row/align1/*.cl~
rm matrix_prod_row_row_col/align1/*.cl~
rm matrix_prod_row_col_row/align1/*.cl~
rm matrix_prod_row_col_col/align1/*.cl~

rm matrix_prod_col_row_row/align1/*.cl~
rm matrix_prod_col_row_col/align1/*.cl~
rm matrix_prod_col_col_row/align1/*.cl~
rm matrix_prod_col_col_col/align1/*.cl~

rm matrix_solve_row_row/align1/*.cl~
rm matrix_solve_row_col/align1/*.cl~
rm matrix_solve_col_row/align1/*.cl~
rm matrix_solve_col_col/align1/*.cl~


rm compressed_matrix/align1/*.cl~
rm compressed_matrix/align4/*.cl~
rm compressed_matrix/align8/*.cl~

rm coordinate_matrix/align1/*.cl~
