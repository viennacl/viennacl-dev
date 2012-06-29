#ifndef VIENNACL_LINALG_KERNELS_SPAI_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_SPAI_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const spai_align1_assemble_blocks = 
"float get_element(__global const unsigned int * row_indices,\n"
"					 __global const unsigned int * column_indices,\n"
"					 __global const float * elements,\n"
"					 unsigned int row,\n"
"					 unsigned int col\n"
"					 )\n"
"{\n"
"	unsigned int row_end = row_indices[row+1];\n"
"	for(unsigned int i = row_indices[row]; i < row_end; ++i){\n"
"		if(column_indices[i] == col)\n"
"			return elements[i];\n"
"		if(column_indices[i] > col)\n"
"			return 0.0;\n"
"	}\n"
"	return 0.0;						\n"
"}\n"
"void block_assembly(__global const unsigned int * row_indices,\n"
"					__global const unsigned int * column_indices, \n"
"					__global const float * elements,\n"
"					__global const unsigned int * matrix_dimensions,\n"
"					__global const unsigned int * set_I,\n"
"					__global const unsigned int * set_J, \n"
"					unsigned int matrix_ind,\n"
"					__global float * com_A_I_J)\n"
"{\n"
"	unsigned int row_n = matrix_dimensions[2*matrix_ind];\n"
"	unsigned int col_n = matrix_dimensions[2*matrix_ind + 1];\n"
"	\n"
"	for(unsigned int i = 0; i < col_n; ++i){\n"
"				//start row index\n"
"				for(unsigned int j = 0; j < row_n; j++){\n"
"					com_A_I_J[ i*row_n + j] = get_element(row_indices, column_indices, elements, set_I[j], set_J[i]);\n"
"				}\n"
"			}\n"
"						\n"
"}\n"
"__kernel void assemble_blocks(\n"
"          __global const unsigned int * row_indices,\n"
"          __global const unsigned int * column_indices, \n"
"          __global const float * elements,\n"
"          __global const unsigned int * set_I,\n"
"  		  __global const unsigned int * set_J,\n"
" 		  __global const unsigned int * i_ind,\n"
"		  __global const unsigned int * j_ind,\n"
"	      __global const unsigned int * block_ind,\n"
"	      __global const unsigned int * matrix_dimensions,\n"
"		  __global float * com_A_I_J,\n"
"		  __global unsigned int * g_is_update,\n"
"                   unsigned int  block_elems_num) \n"
"{ \n"
"  	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			\n"
"            block_assembly(row_indices, column_indices, elements, matrix_dimensions, set_I + i_ind[i], set_J + j_ind[i], i, com_A_I_J + block_ind[i]);\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_assemble_blocks

const char * const spai_align1_block_q_mult = 
"void custom_dot_prod(__global float * A, unsigned int row_n, __local float * v, unsigned int ind, float *res){\n"
"            *res = 0.0;\n"
"            for(unsigned int j = ind; j < row_n; ++j){\n"
"                if(j == ind){\n"
"                    *res += v[j];\n"
"                }else{\n"
"                    *res += A[j + ind*row_n]*v[j];\n"
"                }\n"
"            }\n"
"        }\n"
"void apply_q_trans_vec(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __local float * y){\n"
"            float inn_prod = 0;\n"
"            for(unsigned int i = 0; i < col_n; ++i){\n"
"                custom_dot_prod(R, row_n, y, i, &inn_prod);\n"
"                for(unsigned int j = i; j < row_n; ++j){\n"
"                    if(i == j){\n"
"                        y[j] -= b_v[ i]*inn_prod;\n"
"                    }\n"
"                    else{\n"
"                        y[j] -= b_v[ i]*inn_prod*R[ j + i*row_n];\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"void q_mult(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __local float * R_u, unsigned int col_n_u){\n"
"				for(unsigned int i = get_local_id(0); i < col_n_u; i+= get_local_size(0)){\n"
"					apply_q_trans_vec(R, row_n, col_n, b_v, R_u + row_n*i);\n"
"				}				\n"
"}\n"
"void matrix_from_global_to_local(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){\n"
"	for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){\n"
"		for(unsigned int j = 0; j < row_n; ++j){\n"
"			l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j];\n"
"		}\n"
"	}\n"
"}\n"
"void matrix_from_local_to_global(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){\n"
"	for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){\n"
"		for(unsigned int j = 0; j < row_n; ++j){\n"
"			g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j];\n"
"		}\n"
"	}\n"
"}\n"
"__kernel void block_q_mult(__global float * global_R,\n"
"  __global unsigned int * block_ind,\n"
"  __global float * global_R_u,\n"
"  __global unsigned int *block_ind_u,\n"
"  __global float * b_v,\n"
"  __global unsigned int * start_bv_inds,\n"
"  __global unsigned int * matrix_dimensions,\n"
"  __global unsigned int * matrix_dimensions_u,\n"
"  __global unsigned int * g_is_update,\n"
"  __local  float * local_R_u,\n"
"    unsigned int  block_elems_num){\n"
"		for(unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){\n"
"	        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && (g_is_update[i] > 0)){\n"
"				//matrix_from_global_to_local(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);\n"
"				matrix_from_global_to_local(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i+ 1], block_ind_u[i]);\n"
"				barrier(CLK_LOCAL_MEM_FENCE);\n"
"	            q_mult(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v + start_bv_inds[i], local_R_u, \n"
"	 				   matrix_dimensions_u[2*i + 1]);\n"
"				barrier(CLK_LOCAL_MEM_FENCE);\n"
"	            matrix_from_local_to_global(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1], block_ind_u[i]);\n"
"	        }\n"
"	    }\n"
"}\n"
; //spai_align1_block_q_mult

const char * const spai_align1_block_least_squares = 
"void custom_dot_prod_ls(__global float * A, unsigned int row_n, __global float * v, unsigned int ind, float *res){\n"
"            *res = 0.0;\n"
"            for(unsigned int j = ind; j < row_n; ++j){\n"
"                if(j == ind){\n"
"                    *res += v[ j];\n"
"                }else{\n"
"                    *res += A[ j + ind*row_n]*v[ j];\n"
"                }\n"
"            }\n"
"        }\n"
"void backwardSolve(__global float * R,  unsigned int row_n, unsigned int col_n, __global float * y, __global float * x){\n"
"	for (int i = col_n-1; i >= 0 ; i--) {\n"
"		x[ i] = y[ i];\n"
"		for (int j = i+1; j < col_n; ++j) {\n"
"			x[ i] -= R[ i + j*row_n]*x[ j];\n"
"		}\n"
"		x[i] /= R[ i + i*row_n];\n"
"	}\n"
"	\n"
"}\n"
"		\n"
"void apply_q_trans_vec_ls(__global float * R, unsigned int row_n, unsigned int col_n, __global const float * b_v,  __global float * y){\n"
"            float inn_prod = 0;\n"
"            for(unsigned int i = 0; i < col_n; ++i){\n"
"                custom_dot_prod_ls(R, row_n, y, i, &inn_prod);\n"
"                for(unsigned int j = i; j < row_n; ++j){\n"
"                    if(i == j){\n"
"                        y[ j] -= b_v[ i]*inn_prod;\n"
"                    }\n"
"                    else{\n"
"                        y[j] -= b_v[ i]*inn_prod*R[ j +i*row_n];\n"
"                    }\n"
"                }\n"
"                //std::cout<<y<<std::endl;\n"
"            }\n"
"        }\n"
"void ls(__global float * R, unsigned int row_n, unsigned int col_n, __global float * b_v, __global float * m_v, __global float * y_v){\n"
"	\n"
"	apply_q_trans_vec_ls(R, row_n, col_n, b_v, y_v);\n"
"	//m_new - is m_v now\n"
"	backwardSolve(R, row_n, col_n, y_v, m_v);\n"
"}\n"
"__kernel void block_least_squares(\n"
"          __global float * global_R,\n"
"		  __global unsigned int * block_ind,\n"
"          __global float * b_v,\n"
"	      __global unsigned int * start_bv_inds,\n"
"		  __global float * m_v,\n"
"		  __global float * y_v,\n"
"		  __global unsigned int * start_y_inds,\n"
"	      __global unsigned int * matrix_dimensions,\n"
"	      __global unsigned int * g_is_update,\n"
"          //__local  float * local_R,\n"
"            unsigned int  block_elems_num) \n"
"{ \n"
"  	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			\n"
"            ls(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v +start_bv_inds[i], m_v + start_bv_inds[i], y_v + start_y_inds[i] );\n"
"			\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_block_least_squares

const char * const spai_align1_block_qr_assembly = 
"void assemble_upper_part(__global float * R_q,\n"
" 						unsigned int row_n_q, unsigned int col_n_q, __global float * R_u, \n"
"						unsigned int row_n_u, unsigned int col_n_u,\n"
"						unsigned int col_n, unsigned int diff){\n"
"            for(unsigned int i = 0; i < col_n_q; ++i){\n"
"                for(unsigned int j = 0; j < diff; ++j){\n"
"					R_q[ i*row_n_q + j] = R_u[ i*row_n_u + j + col_n ];\n"
"                }\n"
"            }\n"
"        }\n"
"void assemble_lower_part(__global float * R_q, unsigned int row_n_q, unsigned int col_n_q, __global float * R_u_u, \n"
"						 unsigned int row_n_u_u, unsigned int col_n_u_u, \n"
"						 unsigned int diff){\n"
"	for(unsigned int i = 0; i < col_n_u_u; ++i){\n"
"		for(unsigned int j = 0; j < row_n_u_u; ++j){\n"
"			R_q[i*row_n_q + j + diff] = R_u_u[i*row_n_u_u + j];\n"
"		}\n"
"	}	\n"
"}\n"
"void assemble_qr_block(__global float * R_q, unsigned int row_n_q, unsigned int col_n_q, __global float * R_u, unsigned int row_n_u,\n"
"						unsigned int col_n_u, __global float * R_u_u, unsigned int row_n_u_u, unsigned int col_n_u_u, unsigned int col_n){\n"
"						unsigned int diff = row_n_u - col_n;\n"
"						assemble_upper_part(R_q, row_n_q, col_n_q, R_u, row_n_u, col_n_u, col_n, diff);\n"
"						if(diff > 0){\n"
"							assemble_lower_part(R_q, row_n_q, col_n_q, R_u_u, row_n_u_u, col_n_u_u, diff);\n"
"						}\n"
"}\n"
"__kernel void block_qr_assembly(\n"
"          __global unsigned int * matrix_dimensions,\n"
"	      __global float * R_u,\n"
"		  __global unsigned int * block_ind_u,\n"
"		  __global unsigned int * matrix_dimensions_u,\n"
"		  __global float * R_u_u,\n"
"	      __global unsigned int * block_ind_u_u,\n"
"		  __global unsigned int * matrix_dimensions_u_u,\n"
"		  __global float * R_q,\n"
"		  __global unsigned int * block_ind_q,\n"
"		  __global unsigned int * matrix_dimensions_q,\n"
"		  __global unsigned int * g_is_update,\n"
"          //__local  float * local_R_q,\n"
"            unsigned int  block_elems_num) \n"
"{ \n"
"  	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			//\n"
"            assemble_qr_block(R_q + block_ind_q[i], matrix_dimensions_q[2*i], matrix_dimensions_q[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], \n"
"							matrix_dimensions_u[2*i + 1], R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1], matrix_dimensions[2*i + 1]);\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_block_qr_assembly

const char * const spai_align1_block_qr_assembly_1 = 
"void assemble_upper_part_1(__global float * R_q, unsigned int row_n_q, unsigned int col_n_q, __global float * R_u, \n"
"						 unsigned int row_n_u, unsigned int col_n_u,\n"
"						 unsigned int col_n, unsigned int diff){\n"
"            for(unsigned int i = 0; i < col_n_q; ++i){\n"
"                for(unsigned int j = 0; j < diff; ++j){\n"
"					R_q[ i*row_n_q + j] = R_u[i*row_n_u + j + col_n ];\n"
"                }\n"
"            }\n"
"        }\n"
"void assemble_qr_block_1(__global float * R_q,  unsigned int row_n_q, unsigned int col_n_q, __global float * R_u, unsigned int row_n_u,\n"
"						unsigned int col_n_u, unsigned int col_n){\n"
"						unsigned int diff = row_n_u - col_n;\n"
"						assemble_upper_part_1(R_q, row_n_q, col_n_q, R_u, row_n_u, col_n_u, col_n, diff);\n"
"}\n"
"__kernel void block_qr_assembly_1(\n"
"          __global unsigned int * matrix_dimensions,\n"
"	      __global float * R_u,\n"
"		  __global unsigned int * block_ind_u,\n"
"		  __global unsigned int * matrix_dimensions_u,\n"
"		  __global float * R_q,\n"
"		  __global unsigned int * block_ind_q,\n"
"		  __global unsigned int * matrix_dimensions_q,\n"
"		  __global unsigned int * g_is_update,\n"
"          //__local  float * local_R_q,\n"
"            unsigned int  block_elems_num) \n"
"{ \n"
"  	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"            assemble_qr_block_1(R_q + block_ind_q[i], matrix_dimensions_q[2*i], matrix_dimensions_q[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], \n"
"							matrix_dimensions_u[2*i + 1], matrix_dimensions[2*i + 1]);\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_block_qr_assembly_1

const char * const spai_align1_block_r_assembly = 
"void assemble_r(__global float * gR, unsigned int row_n_r, unsigned int col_n_r, __global float * R, \n"
"				unsigned int row_n, unsigned int col_n)\n"
"{\n"
"  for(unsigned int i = 0; i < col_n; ++i){\n"
"     for(unsigned int j = 0; j < row_n; ++j){\n"
"		gR[i*row_n_r + j] = R[i*row_n + j ];\n"
"     }\n"
"  }\n"
"}\n"
"void assemble_r_u(__global float * gR,\n"
" 				  unsigned int row_n_r, unsigned int col_n_r, __global float * R_u, unsigned int row_n_u, unsigned int col_n_u, \n"
"				  unsigned int col_n)\n"
"{\n"
"	for(unsigned int i = 0; i < col_n_u; ++i){\n"
"		for(unsigned int j = 0; j < col_n; ++j){\n"
"			gR[ (i+col_n)*row_n_r + j] = R_u[ i*row_n_u + j];\n"
"		}\n"
"	}				\n"
"}\n"
"void assemble_r_u_u(__global float * gR,  unsigned int row_n_r, unsigned int col_n_r, __global float * R_u_u, unsigned int row_n_u_u, \n"
"					unsigned int col_n_u_u, unsigned int col_n)\n"
"{\n"
"	for(unsigned int i = 0; i < col_n_u_u; ++i){\n"
"		for(unsigned int j = 0; j < row_n_u_u; ++j){\n"
"			gR[(col_n+i)*row_n_r + j + col_n] = R_u_u[i*row_n_u_u + j];\n"
"		}\n"
"	}					\n"
"}\n"
"void assemble_r_block(__global float * gR, unsigned int row_n_r, unsigned int col_n_r, __global float * R, unsigned int row_n, \n"
"				unsigned int col_n, __global float * R_u, unsigned int row_n_u, unsigned int col_n_u, __global float * R_u_u, \n"
"				unsigned int row_n_u_u, unsigned int col_n_u_u){\n"
"				assemble_r(gR, row_n_r, col_n_r, R, row_n, col_n);				\n"
"				assemble_r_u(gR, row_n_r, col_n_r, R_u, row_n_u, col_n_u, col_n);\n"
"				assemble_r_u_u(gR, row_n_r, col_n_r, R_u_u, row_n_u_u, col_n_u_u, col_n);\n"
"}\n"
"__kernel void block_r_assembly(\n"
"          					__global float * R,\n"
"	      					__global unsigned int * block_ind,\n"
"		  					__global unsigned int * matrix_dimensions,\n"
"		  					__global float * R_u,\n"
"	      					__global unsigned int * block_ind_u,\n"
"		  					__global unsigned int * matrix_dimensions_u,\n"
"		  					__global float * R_u_u,\n"
"		  					__global unsigned int * block_ind_u_u,\n"
"		  					__global unsigned int * matrix_dimensions_u_u,\n"
"		  					__global float * g_R,\n"
"		  					__global unsigned int * block_ind_r,\n"
"		  					__global unsigned int * matrix_dimensions_r,\n"
"						    __global unsigned int * g_is_update,\n"
"          					//__local  float * local_gR,\n"
"            				unsigned int  block_elems_num) \n"
"{ \n"
"  	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			\n"
"            assemble_r_block(g_R + block_ind_r[i], matrix_dimensions_r[2*i], matrix_dimensions_r[2*i + 1], R + block_ind[i], matrix_dimensions[2*i], \n"
"							matrix_dimensions[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1],\n"
"							R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1]);\n"
"			\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_block_r_assembly

const char * const spai_align1_block_bv_assembly = 
"void assemble_bv(__global float * g_bv_r, __global float * g_bv, unsigned int col_n){\n"
"	for(unsigned int i = 0; i < col_n; ++i){\n"
"		g_bv_r[i] = g_bv[ i];\n"
"	}\n"
"}\n"
"void assemble_bv_block(__global float * g_bv_r, __global float * g_bv, unsigned int col_n,\n"
" 					   __global float * g_bv_u, unsigned int col_n_u)\n"
"{\n"
"	assemble_bv(g_bv_r, g_bv, col_n);\n"
"	assemble_bv(g_bv_r + col_n, g_bv_u, col_n_u);\n"
"						\n"
"}\n"
"__kernel void block_bv_assembly(__global float * g_bv,\n"
"						__global unsigned int * start_bv_ind,\n"
"						__global unsigned int * matrix_dimensions,\n"
"						__global float * g_bv_u,\n"
"						__global unsigned int * start_bv_u_ind,\n"
"						__global unsigned int * matrix_dimensions_u,\n"
"						__global float * g_bv_r,\n"
"						__global unsigned int * start_bv_r_ind,\n"
"						__global unsigned int * matrix_dimensions_r,\n"
"						__global unsigned int * g_is_update,\n"
"						//__local  float * local_gb,\n"
"						unsigned int  block_elems_num)\n"
"{ \n"
"	for(unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){\n"
"		if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			assemble_bv_block(g_bv_r + start_bv_r_ind[i], g_bv + start_bv_ind[i], matrix_dimensions[2*i + 1], g_bv_u + start_bv_u_ind[i], matrix_dimensions_u[2*i + 1]);\n"
"		}\n"
"	}\n"
"}\n"
; //spai_align1_block_bv_assembly

const char * const spai_align1_block_qr = 
"void dot_prod(__local const float* A, unsigned int n, unsigned int beg_ind, float* res){\n"
"    *res = 0;\n"
"    for(unsigned int i = beg_ind; i < n; ++i){\n"
"        *res += A[(beg_ind-1)*n + i]*A[(beg_ind-1)*n + i];\n"
"    }\n"
"}\n"
" \n"
"void vector_div(__global float* v, unsigned int beg_ind, float b, unsigned int n){\n"
"    for(unsigned int i = beg_ind; i < n; ++i){\n"
"        v[i] /= b;\n"
"    }\n"
"}\n"
"void copy_vector(__local const float* A, __global float* v, const unsigned int beg_ind, const unsigned int n){\n"
"    for(unsigned int i = beg_ind; i < n; ++i){\n"
"        v[i] = A[(beg_ind-1)*n + i];\n"
"    }\n"
"}\n"
" \n"
" \n"
"void householder_vector(__local const float* A, unsigned int j, unsigned int n, __global float* v, __global float* b){\n"
"    float sg;\n"
"    dot_prod(A, n, j+1, &sg); \n"
"    copy_vector(A, v, j+1, n);\n"
"    float mu;\n"
"    v[j] = 1.0;\n"
"    //print_contigious_vector(v, v_start_ind, n);\n"
"    if(sg == 0){\n"
"        *b = 0;\n"
"    }\n"
"    else{\n"
"        mu = sqrt(A[j*n + j]*A[ j*n + j] + sg);\n"
"        if(A[ j*n + j] <= 0){\n"
"            v[j] = A[ j*n + j] - mu;\n"
"        }else{\n"
"            v[j] = -sg/(A[ j*n + j] + mu);\n"
"        }\n"
"		*b = 2*(v[j]*v[j])/(sg + v[j]*v[j]);\n"
"        //*b = (2*v[j]*v[j])/(sg + (v[j])*(v[j]));\n"
"        vector_div(v, j, v[j], n);\n"
"        //print_contigious_vector(v, v_start_ind, n);\n"
"    }\n"
"}\n"
"void custom_inner_prod(__local const float* A, __global float* v, unsigned int col_ind, unsigned int row_num, unsigned int start_ind, float* res){\n"
"    for(unsigned int i = start_ind; i < row_num; ++i){\n"
"        *res += A[col_ind*row_num + i]*v[i];  \n"
"    }\n"
"}\n"
"// \n"
"void apply_householder_reflection(__local float* A,  unsigned int row_n, unsigned int col_n, unsigned int iter_cnt, __global float* v, float b){\n"
"    float in_prod_res;\n"
"    for(unsigned int i= iter_cnt + get_local_id(0); i < col_n; i+=get_local_size(0)){\n"
"        in_prod_res = 0.0;\n"
"        custom_inner_prod(A, v, i, row_n, iter_cnt, &in_prod_res);\n"
"        for(unsigned int j = iter_cnt; j < row_n; ++j){\n"
"            A[ i*row_n + j] -= b*in_prod_res* v[j];\n"
"        }\n"
"    }\n"
"    \n"
"}\n"
"void store_householder_vector(__local float* A,  unsigned int ind, unsigned int n, __global float* v){\n"
"    for(unsigned int i = ind; i < n; ++i){\n"
"        A[ (ind-1)*n + i] = v[i];\n"
"    }\n"
"}\n"
"void single_qr( __local float* R, __global unsigned int* matrix_dimensions, __global float* b_v, __global float* v, unsigned int matrix_ind){\n"
"    				//matrix_dimensions[0] - number of rows\n"
"       				//matrix_dimensions[1] - number of columns\n"
"	unsigned int col_n = matrix_dimensions[2*matrix_ind + 1];\n"
"	unsigned int row_n = matrix_dimensions[2*matrix_ind];\n"
"	\n"
"	if((col_n == row_n)&&(row_n == 1)){\n"
"		b_v[0] = 0.0;\n"
"	    return;\n"
"	}\n"
"       for(unsigned int i = 0; i < col_n; ++i){\n"
"				if(get_local_id(0) == 0){\n"
"               		householder_vector(R, i, row_n, v, b_v + i);\n"
"				}\n"
"				barrier(CLK_LOCAL_MEM_FENCE);\n"
"               	apply_householder_reflection(R, row_n, col_n, i, v, b_v[i]);\n"
"                barrier(CLK_LOCAL_MEM_FENCE);\n"
"				if(get_local_id(0) == 0){\n"
"               		if(i < matrix_dimensions[2*matrix_ind]){\n"
"                   		store_householder_vector(R, i+1, row_n, v);\n"
"               		}\n"
"				}\n"
"           }\n"
"}\n"
"void matrix_from_global_to_local_qr(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){\n"
"	for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){\n"
"		for(unsigned int j = 0; j < row_n; ++j){\n"
"			l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j];\n"
"		}\n"
"	}\n"
"}\n"
"void matrix_from_local_to_global_qr(__global float* g_M, __local float* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){\n"
"	for(unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){\n"
"		for(unsigned int j = 0; j < row_n; ++j){\n"
"			g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j];\n"
"		}\n"
"	}\n"
"}\n"
"__kernel void block_qr(\n"
"			__global float* R, 	 \n"
"			__global unsigned int* matrix_dimensions, \n"
"			__global float* b_v, \n"
"			__global float* v, \n"
"			__global unsigned int* start_matrix_inds, \n"
"			__global unsigned int* start_bv_inds, \n"
"			__global unsigned int* start_v_inds,\n"
"			__global unsigned int * g_is_update,  \n"
"			__local float* local_buff_R,\n"
"			unsigned int block_elems_num){\n"
"    for(unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){\n"
"        if((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){\n"
"			matrix_from_global_to_local_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);\n"
"			barrier(CLK_LOCAL_MEM_FENCE);\n"
"            single_qr(local_buff_R, matrix_dimensions, b_v + start_bv_inds[i], v + start_v_inds[i], i);\n"
"			barrier(CLK_LOCAL_MEM_FENCE);\n"
"            matrix_from_local_to_global_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]);\n"
"        }\n"
"    }\n"
"}\n"
; //spai_align1_block_qr

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif
