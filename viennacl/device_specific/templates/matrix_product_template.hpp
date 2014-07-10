#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_PRODUCT_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_MATRIX_PRODUCT_HPP

/* =========================================================================
Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                ViennaCL - The Vienna Computing Library
                            -----------------

Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

(A list of authors and contributors can be found in the PDF manual)

License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file viennacl/generator/matrix_product.hpp
*
* Kernel template for the matrix product operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/mapped_objects.hpp"
#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/tree_parsing/read_write.hpp"
#include "viennacl/device_specific/tree_parsing/filter.hpp"
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/forwards.h"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

namespace device_specific{


class matrix_product_template : public template_base{

public:
    struct parameters : public template_base::parameters
    {
      parameters(char A_trans, char B_trans
                 , unsigned int simd_width
                 , unsigned int local_size_0, unsigned int KL, unsigned int local_size_1
                 , unsigned int ms, unsigned int ks, unsigned int ns
                 , bool use_A_local, bool use_B_local
                 , unsigned int local_fetch_0, unsigned int local_fetch_1): template_base::parameters(simd_width, local_size_0, local_size_1, 1),
                                             A_trans(A_trans), B_trans(B_trans), kL(KL), mS(ms), kS(ks), nS(ns), use_A_local(use_A_local), use_B_local(use_B_local),
                                             local_fetch_0(local_fetch_0), local_fetch_1(local_fetch_1),
                                              mL(ms*local_size_0), nL(ns*local_size_1){}

      const char A_trans;
      const char B_trans;

      unsigned int kL;

      unsigned int mS;
      unsigned int kS;
      unsigned int nS;

      bool use_A_local;
      bool use_B_local;

      unsigned int local_fetch_0;
      unsigned int local_fetch_1;

      unsigned int mL;
      unsigned int nL;
    };

private:

    unsigned int n_lmem_elements() const
    {
        unsigned int N = 0;
        if(p_.use_A_local)
            N += p_.kL * (p_.mL+1);
        if(p_.use_B_local)
            N += p_.nL * (p_.kL+1);
        return N;
    }

    void check_invalid_impl(viennacl::ocl::device const & /*device*/) const
    {
        static const unsigned int alignment = 128;
        bool res = false;
        res |= alignment % p_.mL > 0;
        res |= alignment % p_.kL > 0;
        res |= alignment % p_.nL > 0;
        res |= (p_.mS % p_.simd_width) > 0;
        res |= (p_.nS % p_.simd_width) > 0;
        res |= p_.mS > p_.mL;
        res |= p_.nS > p_.nL;
        res |= p_.kS > p_.kL;
        res |= (!(p_.A_trans=='N' && p_.B_trans=='T') && p_.simd_width>1);
        if(p_.use_A_local)
        {
            unsigned int bound1 = (p_.A_trans=='N')?p_.kL:p_.mL;
            unsigned int bound0 = (p_.A_trans=='N')?p_.mL:p_.kL;

            res |= p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0;
            res |= p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0;
        }
        if(p_.use_B_local)
        {
            unsigned int bound1 = (p_.B_trans=='T')?p_.kL:p_.nL;
            unsigned int bound0 = (p_.B_trans=='T')?p_.nL:p_.kL;

            res |= p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0;
            res |= p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0;
        }

        if(p_.use_A_local || p_.use_B_local)
            res |= ((p_.local_fetch_0*p_.local_fetch_1) !=(p_.local_size_0*p_.local_size_1));
    }

    void configure_impl(vcl_size_t /*kernel_id*/, viennacl::ocl::context & /*context*/, statements_container const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg) const
    {
        using namespace device_specific::utils;

        scheduler::statement::container_type const & array = statements.data().front().array();
        vcl_size_t root_idx = statements.data().front().root();

        //set M, N
        scheduler::statement_node const & root = array[root_idx];
        vcl_size_t M = call_on_matrix(root.lhs, internal_size1_fun());
        vcl_size_t N = call_on_matrix(root.lhs, internal_size2_fun());

        //set ND range
        k.global_work_size(0, M/p_.mS);
        k.global_work_size(1, N/p_.nS);

        //set arguments
        //M,N
        k.arg(n_arg++, cl_uint(M));
        k.arg(n_arg++, cl_uint(N));

        //K
        vcl_size_t A1, A2, B1, B2;
        std::vector<vcl_size_t> idx;
        traverse(statements.data().front(), root_idx, tree_parsing::filter(&is_matrix_product, idx), false);
        scheduler::statement_node const * A = &array[idx[0]];
        while(A->lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
          A = &array[A->lhs.node_index];
        A1 = call_on_matrix(A->lhs,internal_size1_fun());
        A2 = call_on_matrix(A->lhs,internal_size2_fun());


        scheduler::statement_node const * B = &array[idx[0]];
        if(B->rhs.type_family==scheduler::MATRIX_TYPE_FAMILY)
        {
          B1 = call_on_matrix(B->rhs,internal_size1_fun());
          B2 = call_on_matrix(B->rhs,internal_size2_fun());
        }
        else
        {
          B = &array[B->rhs.node_index];
          while(B->lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            B = &array[B->lhs.node_index];
          B1 = call_on_matrix(B->lhs,internal_size1_fun());
          B2 = call_on_matrix(B->lhs,internal_size2_fun());
        }

        if(A1==B1 || A1==B2)
           k.arg(n_arg++, cl_uint(A1));
        else
           k.arg(n_arg++, cl_uint(A2));




    }

    void add_kernel_arguments(statements_container const & /*statements*/, std::string & arguments_string) const
    {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
        arguments_string += generate_value_kernel_argument("unsigned int", "K");
    }


    static bool is_matrix_product(scheduler::statement_node const & node) { return node.op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE; }

    void set_simd_widths(scheduler::statement const & s, mapping_type const & m)
    {
      std::vector<vcl_size_t> idx;
      tree_parsing::traverse(s, s.root(), tree_parsing::filter(&is_matrix_product, idx), false);
      tree_parsing::traverse(s, idx[0], set_simd_width_traversal<mapped_matrix>(p_.simd_width, m), true);

    }

    void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mappings) const
    {
        using namespace tree_parsing;

        //////////////////
        /// INIT
        /// //////////////
        scheduler::statement const & s = statements.data().front();
        mapping_type const & mapping = mappings.front();

        std::vector<vcl_size_t> idx;
        traverse(s, s.root(), filter(&is_matrix_product, idx), false);
        vcl_size_t prod_idx = idx[0];
        scheduler::statement_node const * prod_node = &s.array()[prod_idx];

        mapped_matrix * C = (mapped_matrix*)(mapping.at(std::make_pair(s.root(),LHS_NODE_TYPE)).get());

        mapped_matrix * A;
        if(prod_node->lhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            A = (mapped_matrix *)mapping.at(std::make_pair(prod_node->lhs.node_index,LHS_NODE_TYPE)).get();
        else
            A = (mapped_matrix *)mapping.at(std::make_pair(prod_idx, LHS_NODE_TYPE)).get();

        mapped_matrix * B;
        if(prod_node->rhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            B = (mapped_matrix *)mapping.at(std::make_pair(prod_node->rhs.node_index, LHS_NODE_TYPE)).get();
        else
            B = (mapped_matrix *)mapping.at(std::make_pair(prod_idx,RHS_NODE_TYPE)).get();


        mapped_matrix_product * prod = (mapped_matrix_product *)mapping.at(std::make_pair(prod_idx, PARENT_NODE_TYPE)).get();

        if(p_.simd_width>1)
        {
            stream << A->ld() << "/=" << p_.simd_width << ";" << std::endl;
            stream << B->ld() << "/=" << p_.simd_width << ";" << std::endl;
        }


        std::string C_scalartype = C->scalartype();
        std::string A_scalartype = p_.use_A_local?A->scalartype():utils::simd_scalartype(A->scalartype(), p_.simd_width);
        std::string B_scalartype = p_.use_B_local?B->scalartype():utils::simd_scalartype(B->scalartype(), p_.simd_width);

        //////////////////
        /// DECLARATIONS
        /// //////////////


        ///Result Values
        stream << C_scalartype << " " << "rC[" << p_.mS << "][" << p_.nS <<"]  = {(" << C->scalartype() << ")0};" << std::endl;
        stream << A_scalartype << " " << "rA[" << p_.kS << "][" << (p_.use_A_local?p_.mS:p_.mS/p_.simd_width) << "];" << std::endl;
        stream << B_scalartype << " " << "rB[" << p_.kS << "][" << (p_.use_B_local?p_.nS:p_.nS/p_.simd_width) <<"];" << std::endl;
        stream << std::endl;

        if(p_.use_A_local)
            stream << "__local " << A->scalartype() << " lA[" << p_.kL * (p_.mL + 1) << "];" << std::endl;
        if(p_.use_B_local)
            stream << "__local " << B->scalartype() << " lB[" << p_.kL * (p_.nL + 1) << "];" << std::endl;
        stream << std::endl;

        stream << "uint gidx = get_group_id(0);" << std::endl;
        stream << "uint gidy = get_group_id(1);" << std::endl;
        stream << "uint idx = get_local_id(0);" << std::endl;
        stream << "uint idy = get_local_id(1);" << std::endl;
        if(p_.use_A_local || p_.use_B_local){
            stream << std::endl;
            stream << "uint idt = " << p_.local_size_0 << "*idy + idx;" << std::endl;
            stream << "uint idxT = idt % " << p_.local_fetch_0 << ";" << std::endl;
            stream << "uint idyT = idt / " << p_.local_fetch_0 << ";" << std::endl;
        }
        stream << std::endl;

        if(p_.use_A_local)
        {
            if(p_.A_trans=='N')
                stream << A->name() << " +=  gidx*" << p_.mL/p_.simd_width << "+ idxT + idyT*" << A->ld()  << ";" << std::endl;
            else
                stream << A->name() << " +=  gidx*" << p_.mL/p_.simd_width << "*" << A->ld() << "+ idxT + idyT*" << A->ld()  << ";" << std::endl;
        }
        else
        {
            if(p_.A_trans=='N')
                stream << A->name() << " += gidx*" << p_.mL/p_.simd_width << "+ idx" << ";" << std::endl;
            else
                stream << A->name() << " += (gidx*" << p_.mL/p_.simd_width << "+ idx)*" << A->ld() << ";" << std::endl;
        }

        if(p_.use_B_local)
        {
            if(p_.B_trans=='T')
                stream << B->name() << " +=  gidy*" << p_.nL/p_.simd_width << "+ idxT + idyT*" << B->ld()  << ";" << std::endl;
            else
                stream << B->name() << " +=  gidy*" << p_.nL/p_.simd_width << "*" << B->ld() << "+ idxT + idyT*" << B->ld()  << ";" << std::endl;
        }
        else
        {
            if(p_.B_trans=='T')
                stream << B->name() << " +=  gidy*" << p_.nL/p_.simd_width << "+ idy;" << std::endl;
            else
                stream << B->name() << " += (gidy*" << p_.nL/p_.simd_width << "+ idy)*" << B->ld() << ";" << std::endl;
        }

        stream << std::endl;

        stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << p_.kL << "){" << std::endl;
        stream.inc_tab();

        if(p_.use_A_local){
            if(p_.A_trans=='N')
                stream << "__local " << A->scalartype() << "* plA = lA + idyT*" << p_.mL+1 << "+" << p_.simd_width << "*idxT;" << std::endl;
            else
                stream << "__local " << A->scalartype() << "* plA = lA + idxT*" << p_.mL+1 << "+ idyT;" << std::endl;
        }


        if(p_.use_B_local)
        {
            if(p_.B_trans=='T')
                stream << "__local " << B->scalartype() << "* plB = lB + idyT*" << p_.nL+1 << "+" << p_.simd_width << "*idxT;" << std::endl;
            else
                stream << "__local " << B->scalartype() << "* plB = lB + idxT*" << p_.nL+1 << "+ idyT;" << std::endl;
        }


        if(p_.use_A_local || p_.use_B_local)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        ///Fetch LHS to Local Memory
        if(p_.use_A_local)
        {
            unsigned int bound1 = (p_.A_trans=='N')?p_.kL:p_.mL;
            unsigned int bound0 = (p_.A_trans=='N')?p_.mL:p_.kL;
            for(unsigned int k = 0 ; k < bound1 ; k += p_.local_fetch_1){
                for(unsigned int m = 0 ; m < bound0 ; m += p_.local_fetch_0*p_.simd_width){
                    unsigned int offset = (p_.A_trans=='N')?(k*(p_.mL+1)+m):(m*(p_.mL+1)+k);
                    if(p_.simd_width==1)
                        stream << "plA[" << offset << "] = " << A->name() << "[" << m/p_.simd_width <<  "+"  << k << "*" << A->ld() << "];" << std::endl;
                    else
                        stream << "vstore" << p_.simd_width << "(" <<  A->name() << "[" << m/p_.simd_width <<  "+"  << k << "*" << A->ld() << "],0,plA+" << offset << ");" << std::endl;
                }
            }
        }

        ///Fetch RHS to Local Memory
        if(p_.use_B_local)
        {
            unsigned int bound1 = (p_.B_trans=='T')?p_.kL:p_.nL;
            unsigned int bound0 = (p_.B_trans=='T')?p_.nL:p_.kL;
            for(unsigned int k = 0 ; k < bound1 ; k += p_.local_fetch_1){
                for(unsigned int n = 0 ; n < bound0 ; n += p_.local_fetch_0*p_.simd_width){
                    unsigned int offset = (p_.B_trans=='T')?k*(p_.nL+1) + n:n*(p_.nL+1) + k;
                    if(p_.simd_width==1)
                        stream << "plB[" << offset << "] = " << B->name() << "[" << n/p_.simd_width <<  "+"  << k << "*" << B->ld() << "];" << std::endl;
                    else
                        stream << "vstore"  << p_.simd_width << "(" <<  B->name() << "[" << n/p_.simd_width <<  "+"  << k << "*" << B->ld() << "],0,plB+" << offset << ");" << std::endl;
                }
            }
        }

        if(p_.use_A_local || p_.use_B_local)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        stream << "uint offA = " << p_.simd_width << "*idx;" << std::endl;
        stream << "uint offB = " << p_.simd_width << "*idy;" << std::endl;

        //stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int k = 0 ; k < " << p_.kL << "; k+=" << p_.kS << "){" << std::endl;
        stream.inc_tab();

        ///Fetch LHS to registers
        for(unsigned int kk = 0 ; kk < p_.kS ; ++kk){
            for(unsigned int mm = 0 ; mm < p_.mS/p_.simd_width ; ++mm){
                if(p_.use_A_local)
                    for(unsigned int ss = 0 ; ss < p_.simd_width ; ++ss)
                        stream << "rA[" << kk << "][" << mm*p_.simd_width + ss << "] = lA[offA + " << mm*p_.local_size_0*p_.simd_width + ss + kk*(p_.mL+1) << "];" << std::endl;
                else
                    if(p_.A_trans=='N')
                        stream << "rA[" << kk << "][" << mm << "] = " << A->name() << "[" << mm*p_.local_size_0 << "+" << kk << "*" << A->ld() << "];" << std::endl;
                    else
                        stream << "rA[" << kk << "][" << mm << "] = " << A->name() << "[" << kk << "+" << mm*p_.local_size_0 << "*" << A->ld() << "];" << std::endl;
            }
        }


            ///Fetch RHS to registers
        for(unsigned int kk = 0 ; kk < p_.kS ; ++kk){
            for(unsigned int nn=0 ; nn < p_.nS/p_.simd_width ; ++nn){
                if(p_.use_B_local)
                    for(unsigned int ss = 0 ; ss < p_.simd_width ; ++ss)
                        stream << "rB[" << kk << "][" << nn*p_.simd_width + ss << "] = lB[offB + " << nn*p_.local_size_1*p_.simd_width + ss + kk*(p_.nL+1) << "];" << std::endl;
                else
                    if(p_.B_trans=='T')
                        stream << "rB[" << kk << "][" << nn << "] = " << B->name() << "[" << nn*p_.local_size_1 << " + " << kk << "*" << B->ld() << "];" << std::endl;
                    else
                        stream << "rB[" << kk << "][" << nn << "] = " << B->name() << "[" << kk << "+" << nn*p_.local_size_1 << "*" << B->ld() << "];" << std::endl;

            }
        }

        ///Increment pointers
        if(p_.use_A_local)
            stream << "offA += " << p_.kS*(p_.mL+1) << ";" << std::endl;
        else
            if(p_.A_trans=='N')
                stream << A->name() << " += " << p_.kS << "*" << A->ld() << ";" << std::endl;
            else
                stream << A->name() << " += " << p_.kS << ";" << std::endl;

        if(p_.use_B_local)
            stream << "offB += " << p_.kS*(p_.nL+1) << ";" << std::endl;
        else
            if(p_.B_trans=='T')
                stream << B->name() << " += " << p_.kS << "*" << B->ld() << ";" << std::endl;
            else
                stream << B->name() << " += " << p_.kS << ";" << std::endl;


        for(unsigned int kk = 0 ; kk < p_.kS ; ++kk)
            for(unsigned int nn=0 ; nn < p_.nS ; ++nn)
                for(unsigned int mm=0 ; mm < p_.mS ; ++mm)
                {
                    std::string res_str, lhs_str, rhs_str;
                    res_str = "rC[" + tools::to_string(mm) + "][" + tools::to_string(nn) + "]";
                    if(p_.use_A_local || p_.simd_width==1)
                        lhs_str = "rA[" + tools::to_string(kk) + "][" + tools::to_string(mm) + "]";
                    else
                        lhs_str = "rA[" + tools::to_string(kk) + "][" + tools::to_string(mm/p_.simd_width) + "].s" + tools::to_string(mm%p_.simd_width);
                    if(p_.use_B_local || p_.simd_width==1)
                        rhs_str = "rB[" + tools::to_string(kk) + "]["+tools::to_string(nn)+"]";
                    else
                        rhs_str = "rB[" + tools::to_string(kk) + "]["+tools::to_string(nn/p_.simd_width)+"].s"+tools::to_string(nn%p_.simd_width);
                    stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
                }


        stream.dec_tab();
        stream << "}" << std::endl;

        if(p_.use_A_local){
            if(p_.A_trans=='N')
                stream << A->name() << " += " << p_.kL << "*" << A->ld() << ";" << std::endl;
            else
                stream << A->name() << " += " << p_.kL << ";" << std::endl;
        }

        if(p_.use_B_local){
            if(p_.B_trans=='T')
                stream << B->name() << " += " << p_.kL << "*" << B->ld() << ";" << std::endl;
            else
                stream << B->name() << " += " << p_.kL << ";" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;


        if(C->row_major())
        {
          stream << C->name() << "+= gidx*" << p_.mL << "*" << C->ld() << ";" << std::endl;
          stream << C->name() << "+= idx*" << p_.simd_width << "*" << C->ld() << ";" << std::endl;
          stream << C->name() << "+= gidy*" << p_.nL << ";" << std::endl;
          stream << C->name() << "+= idy*" << p_.simd_width << ";" << std::endl;
          for(unsigned int n=0 ; n < p_.nS ; ++n){
              for(unsigned int m=0 ; m < p_.mS ; ++m){
                  std::string j = tools::to_string((m/p_.simd_width)*(p_.local_size_0*p_.simd_width) + m%p_.simd_width);
                  prod->access_name("rC["+tools::to_string(m)+"]["+tools::to_string(n)+"]");
                  std::string str;
                  traverse(s, s.root(), evaluate_expression_traversal(index_tuple(j, "N", "0", "M"), 0, str, mapping), false);
                  stream << str << ";" << std::endl;
              }
              if((n+1)%p_.simd_width>0)
                  stream << C->name() << "+=1;" << std::endl;
              else
                  stream << C->name() << "+=" << (p_.local_size_1*p_.simd_width) - (p_.simd_width-1) << ";" << std::endl;
          }

        }
        else
        {
          stream << C->name() << "+= gidx*" << p_.mL << ";" << std::endl;
          stream << C->name() << "+= idx*" << p_.simd_width << ";" << std::endl;
          stream << C->name() << "+= gidy*" << p_.nL << "*" << C->ld() << ";" << std::endl;
          stream << C->name() << "+= idy*" << p_.simd_width << "*" << C->ld() << ";" << std::endl;
          for(unsigned int m=0 ; m < p_.mS ; ++m)
          {
              for(unsigned int n=0 ; n < p_.nS ; ++n)
              {
                  std::string j = tools::to_string((n/p_.simd_width)*(p_.local_size_1*p_.simd_width) + n%p_.simd_width);
                  prod->access_name("rC["+tools::to_string(m)+"]["+tools::to_string(n)+"]");
                  std::string str;
                  traverse(s, s.root(), evaluate_expression_traversal(index_tuple("0", "M", j, "N"), 0, str, mapping), false);
                  stream << str << ";" << std::endl;
              }
              if((m+1)%p_.simd_width>0)
                  stream << C->name() << "+=1;" << std::endl;
              else
                  stream << C->name() << "+=" << (p_.local_size_0*p_.simd_width) - (p_.simd_width-1) << ";" << std::endl;
          }
        }


    }

public:
    matrix_product_template(matrix_product_template::parameters const & parameters, std::string const & kernel_prefix) : template_base(parameters, kernel_prefix, BIND_TO_HANDLE), p_(parameters){ }

private:
    matrix_product_template::parameters const & p_;
};

}

}

#endif
