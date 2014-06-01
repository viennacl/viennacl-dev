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
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/forwards.h"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

namespace device_specific{


class matrix_product_template : public template_base{

public:
    /** @brief The user constructor */
    matrix_product_template(const char * scalartype, char A_trans, char B_trans
                   , unsigned int simd_width
                   , unsigned int local_size_0, unsigned int KL, unsigned int local_size_1
                   , unsigned int ms, unsigned int ks, unsigned int ns
                   , bool use_A_local, bool use_B_local
                   , unsigned int local_fetch_0, unsigned int local_fetch_1) : template_base(scalartype, simd_width,local_size_0, local_size_1,1)
    , A_trans_(A_trans), B_trans_(B_trans), kL_(KL), mS_(ms), kS_(ks), nS_(ns)
    , use_A_local_(use_A_local), use_B_local_(use_B_local), local_fetch_0_(local_fetch_0), local_fetch_1_(local_fetch_1)
    , mL_(ms*local_size_0), nL_(ns*local_size_1){ }

    void configure_range_enqueue_arguments(unsigned int kernel_id, viennacl::ocl::kernel & k, unsigned int & n_arg) const
    {
        //set M, N
        scheduler::statement_node const & root = statements.front().first.array()[statements.front().second];
        unsigned int M = utils::call_on_matrix(root.lhs, utils::internal_size1_fun());
        unsigned int N = utils::call_on_matrix(root.lhs, utils::internal_size2_fun());

        //set ND range
        configure_local_sizes(k, kernel_id);
        k.global_work_size(0, M/mS_);
        k.global_work_size(1, N/nS_);

        //set arguments
        //M,N
        k.arg(n_arg++, cl_uint(M));
        k.arg(n_arg++, cl_uint(N));

        //K
        scheduler::statement::container_type const & exprs = statements.back().first.array();

        scheduler::statement_node const * prod_node = NULL;
        for(scheduler::statement::container_type::const_iterator it = exprs.begin() ; it != exprs.end() ; ++it)
            if(it->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                prod_node = &(*it);

        if(prod_node->lhs.type_family==scheduler::MATRIX_TYPE_FAMILY)
            k.arg(n_arg++, cl_uint(utils::call_on_matrix(prod_node->lhs, utils::internal_size2_fun())));
        else if(prod_node->lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY

                &&exprs[prod_node->lhs.node_index].op.type==scheduler::OPERATION_UNARY_TRANS_TYPE)
            k.arg(n_arg++, cl_uint(utils::call_on_matrix(exprs[prod_node->lhs.node_index].lhs, utils::internal_size1_fun())));
        else
            assert(false && bool("unexpected expression tree"));
    }

    void add_kernel_arguments(std::string & arguments_string) const
    {
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
        arguments_string += generate_value_kernel_argument("unsigned int", "K");
    }

    virtual void init(scheduler::statement const & statement, mapping_type const & mapping)
    {
        scheduler::statement_node const & root_node = statement.array()[statement.root()];
        scheduler::statement::container_type const & exprs = statement.array();
        scheduler::statement_node const * prod_node = NULL;
        for(scheduler::statement::container_type::const_iterator it = exprs.begin() ; it != exprs.end() ; ++it)
            if(it->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                prod_node = &(*it);


        C_ = (mapped_matrix*)(mapping.at(std::make_pair(&root_node,tree_parsing::LHS_NODE_TYPE)).get());
        if(prod_node->lhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            A_ = (mapped_matrix *)mapping.at(std::make_pair(&exprs[prod_node->lhs.node_index],tree_parsing::LHS_NODE_TYPE)).get();
        else
            A_ = (mapped_matrix *)mapping.at(std::make_pair(prod_node, tree_parsing::LHS_NODE_TYPE)).get();

        if(prod_node->rhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            B_ = (mapped_matrix *)mapping.at(std::make_pair(&exprs[prod_node->rhs.node_index], tree_parsing::LHS_NODE_TYPE)).get();
        else
            B_ = (mapped_matrix *)mapping.at(std::make_pair(prod_node,tree_parsing::RHS_NODE_TYPE)).get();


        prod_ = (mapped_matrix_product *)mapping.at(std::make_pair(prod_node, tree_parsing::PARENT_NODE_TYPE)).get();

        C_->set_simd_width(1);
        A_->set_simd_width(simd_width_);
        B_->set_simd_width(simd_width_);

    }

private:


    unsigned int lmem_used(unsigned int scalartype_size) const
    {
        unsigned int lmem_used = 0;
        if(use_A_local_)
            lmem_used += kL_ * (mL_+1) * scalartype_size;
        if(use_B_local_)
            lmem_used += nL_ * (kL_ + 1) * scalartype_size;
        return lmem_used;
    }

    bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const
    {
        static const unsigned int alignment = 128;
        bool res = false;
        res |= alignment % mL_ > 0;
        res |= alignment % kL_ > 0;
        res |= alignment % nL_ > 0;
        res |= (mS_ % simd_width_) > 0;
        res |= (nS_ % simd_width_) > 0;
        res |= (!(A_trans_=='N' && B_trans_=='T') && simd_width_>1);
        if(use_A_local_){
            unsigned int bound1 = (A_trans_=='N')?kL_:mL_;
            unsigned int bound0 = (A_trans_=='N')?mL_:kL_;

            res |= local_fetch_1_>0 && (bound1 % local_fetch_1_)> 0;
            res |= local_fetch_0_>0 && (bound0 % (local_fetch_0_*simd_width_)) > 0;
        }
        if(use_B_local_){
            unsigned int bound1 = (B_trans_=='T')?kL_:nL_;
            unsigned int bound0 = (B_trans_=='T')?nL_:kL_;

            res |= local_fetch_1_>0 && (bound1 % local_fetch_1_)> 0;
            res |= local_fetch_0_>0 && (bound0 % (local_fetch_0_*simd_width_)) > 0;
        }

        if(use_A_local_ || use_B_local_)
            res |= ((local_fetch_0_*local_fetch_1_) !=(local_size_0_*local_size_1_));
        return res;
    }

    void core(unsigned int /*kernel_id*/, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const {
    {

        //////////////////
        /// INIT
        /// //////////////

        if(simd_width_>1){
            stream << A_->ld() << "/=" << simd_width_ << ";" << std::endl;
            stream << B_->ld() << "/=" << simd_width_ << ";" << std::endl;
        }


        std::string C_scalartype = C_->scalartype();
        std::string A_scalartype = use_A_local_?A_->scalartype():A_->simd_scalartype();
        std::string B_scalartype = use_B_local_?B_->scalartype():B_->simd_scalartype();

        //////////////////
        /// DECLARATIONS
        /// //////////////


        ///Result Values
        stream << C_scalartype << " " << "rC[" << mS_ << "][" << nS_ <<"]  = {(" << C_->scalartype() << ")0};" << std::endl;
        stream << A_scalartype << " " << "rA[" << kS_ << "][" << (use_A_local_?mS_:mS_/simd_width_) << "];" << std::endl;
        stream << B_scalartype << " " << "rB[" << kS_ << "][" << (use_B_local_?nS_:nS_/simd_width_) <<"];" << std::endl;
        stream << std::endl;

        if(use_A_local_)
            stream << "__local " << A_->scalartype() << " lA[" << kL_ * (mL_ + 1) << "];" << std::endl;
        if(use_B_local_)
            stream << "__local " << B_->scalartype() << " lB[" << kL_ * (nL_ + 1) << "];" << std::endl;
        stream << std::endl;

        stream << "uint gidx = get_group_id(0);" << std::endl;
        stream << "uint gidy = get_group_id(1);" << std::endl;
        stream << "uint idx = get_local_id(0);" << std::endl;
        stream << "uint idy = get_local_id(1);" << std::endl;
        if(use_A_local_ || use_B_local_){
            stream << std::endl;
            stream << "uint idt = " << local_size_0_ << "*idy + idx;" << std::endl;
            stream << "uint idxT = idt % " << local_fetch_0_ << ";" << std::endl;
            stream << "uint idyT = idt / " << local_fetch_0_ << ";" << std::endl;
        }
        stream << std::endl;

        if(use_A_local_){
            if(A_trans_=='N')
                stream << A_->name() << " +=  gidx*" << mL_/simd_width_ << "+ idxT + idyT*" << A_->ld()  << ";" << std::endl;
            else
                stream << A_->name() << " +=  gidx*" << mL_/simd_width_ << "*" << A_->ld() << "+ idxT + idyT*" << A_->ld()  << ";" << std::endl;
        }
        else{
            if(A_trans_=='N')
                stream << A_->name() << " += gidx*" << mL_/simd_width_ << "+ idx" << ";" << std::endl;
            else
                stream << A_->name() << " += (gidx*" << mL_/simd_width_ << "+ idx)*" << A_->ld() << ";" << std::endl;
        }

        if(use_B_local_){
            if(B_trans_=='T')
                stream << B_->name() << " +=  gidy*" << nL_/simd_width_ << "+ idxT + idyT*" << B_->ld()  << ";" << std::endl;
            else
                stream << B_->name() << " +=  gidy*" << nL_/simd_width_ << "*" << B_->ld() << "+ idxT + idyT*" << B_->ld()  << ";" << std::endl;
        }
        else{
            if(B_trans_=='T')
                stream << B_->name() << " +=  gidy*" << nL_/simd_width_ << "+ idy;" << std::endl;
            else
                stream << B_->name() << " += (gidy*" << nL_/simd_width_ << "+ idy)*" << B_->ld() << ";" << std::endl;
        }

        stream << std::endl;

        stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << kL_ << "){" << std::endl;
        stream.inc_tab();

        if(use_A_local_){
            if(A_trans_=='N')
                stream << "__local " << A_->scalartype() << "* plA = lA + idyT*" << mL_+1 << "+" << simd_width_ << "*idxT;" << std::endl;
            else
                stream << "__local " << A_->scalartype() << "* plA = lA + idxT*" << mL_+1 << "+ idyT;" << std::endl;
        }


        if(use_B_local_){
            if(B_trans_=='T')
                stream << "__local " << B_->scalartype() << "* plB = lB + idyT*" << nL_+1 << "+" << simd_width_ << "*idxT;" << std::endl;
            else
                stream << "__local " << B_->scalartype() << "* plB = lB + idxT*" << nL_+1 << "+ idyT;" << std::endl;

        }


        if(use_A_local_ || use_B_local_)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        ///Fetch LHS to Local Memory
        if(use_A_local_)
        {
            unsigned int bound1 = (A_trans_=='N')?kL_:mL_;
            unsigned int bound0 = (A_trans_=='N')?mL_:kL_;
            for(unsigned int k = 0 ; k < bound1 ; k += local_fetch_1_){
                for(unsigned int m = 0 ; m < bound0 ; m += local_fetch_0_*simd_width_){
                    unsigned int offset = (A_trans_=='N')?(k*(mL_+1)+m):(m*(mL_+1)+k);
                    if(simd_width_==1)
                        stream << "plA[" << offset << "] = " << A_->name() << "[" << m/simd_width_ <<  "+"  << k << "*" << A_->ld() << "];" << std::endl;
                    else
                        stream << "vstore" << simd_width_ << "(" <<  A_->name() << "[" << m/simd_width_ <<  "+"  << k << "*" << A_->ld() << "],0,plA+" << offset << ");" << std::endl;
                }
            }
        }

        ///Fetch RHS to Local Memory
        if(use_B_local_)
        {
            unsigned int bound1 = (B_trans_=='T')?kL_:nL_;
            unsigned int bound0 = (B_trans_=='T')?nL_:kL_;
            for(unsigned int k = 0 ; k < bound1 ; k += local_fetch_1_){
                for(unsigned int n = 0 ; n < bound0 ; n += local_fetch_0_*simd_width_){
                    unsigned int offset = (B_trans_=='T')?k*(nL_+1) + n:n*(nL_+1) + k;
                    if(simd_width_==1)
                        stream << "plB[" << offset << "] = " << B_->name() << "[" << n/simd_width_ <<  "+"  << k << "*" << B_->ld() << "];" << std::endl;
                    else
                        stream << "vstore"  << simd_width_ << "(" <<  B_->name() << "[" << n/simd_width_ <<  "+"  << k << "*" << B_->ld() << "],0,plB+" << offset << ");" << std::endl;
                }
            }
        }

        if(use_A_local_ || use_B_local_)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        stream << "uint offA = " << simd_width_ << "*idx;" << std::endl;
        stream << "uint offB = " << simd_width_ << "*idy;" << std::endl;

        //stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int k = 0 ; k < " << kL_ << "; k+=" << kS_ << "){" << std::endl;
        stream.inc_tab();

        ///Fetch LHS to registers
        for(unsigned int kk = 0 ; kk < kS_ ; ++kk){
            for(unsigned int mm = 0 ; mm < mS_/simd_width_ ; ++mm){
                if(use_A_local_)
                    for(unsigned int ss = 0 ; ss < simd_width_ ; ++ss)
                        stream << "rA[" << kk << "][" << mm*simd_width_ + ss << "] = lA[offA + " << mm*local_size_0_*simd_width_ + ss + kk*(mL_+1) << "];" << std::endl;
                else
                    if(A_trans_=='N')
                        stream << "rA[" << kk << "][" << mm << "] = " << A_->name() << "[" << mm*local_size_0_ << "+" << kk << "*" << A_->ld() << "];" << std::endl;
                    else
                        stream << "rA[" << kk << "][" << mm << "] = " << A_->name() << "[" << kk << "+" << mm*local_size_0_ << "*" << A_->ld() << "];" << std::endl;
            }
        }


            ///Fetch RHS to registers
        for(unsigned int kk = 0 ; kk < kS_ ; ++kk){
            for(unsigned int nn=0 ; nn < nS_/simd_width_ ; ++nn){
                if(use_B_local_)
                    for(unsigned int ss = 0 ; ss < simd_width_ ; ++ss)
                        stream << "rB[" << kk << "][" << nn*simd_width_ + ss << "] = lB[offB + " << nn*local_size_1_*simd_width_ + ss + kk*(nL_+1) << "];" << std::endl;
                else
                    if(B_trans_=='T')
                        stream << "rB[" << kk << "][" << nn << "] = " << B_->name() << "[" << nn*local_size_1_ << " + " << kk << "*" << B_->ld() << "];" << std::endl;
                    else
                        stream << "rB[" << kk << "][" << nn << "] = " << B_->name() << "[" << kk << "+" << nn*local_size_1_ << "*" << B_->ld() << "];" << std::endl;

            }
        }

        ///Increment pointers
        if(use_A_local_)
            stream << "offA += " << kS_*(mL_+1) << ";" << std::endl;
        else
            if(A_trans_=='N')
                stream << A_->name() << " += " << kS_ << "*" << A_->ld() << ";" << std::endl;
            else
                stream << A_->name() << " += " << kS_ << ";" << std::endl;

        if(use_B_local_)
            stream << "offB += " << kS_*(nL_+1) << ";" << std::endl;
        else
            if(B_trans_=='T')
                stream << B_->name() << " += " << kS_ << "*" << B_->ld() << ";" << std::endl;
            else
                stream << B_->name() << " += " << kS_ << ";" << std::endl;


        for(unsigned int kk = 0 ; kk < kS_ ; ++kk){
            for(unsigned int nn=0 ; nn < nS_ ; ++nn){
                for(unsigned int mm=0 ; mm < mS_ ; ++mm){
                    std::string res_str, lhs_str, rhs_str;
                    res_str = "rC[" + tools::to_string(mm) + "][" + tools::to_string(nn) + "]";
                    if(use_A_local_ || simd_width_==1)
                        lhs_str = "rA[" + tools::to_string(kk) + "][" + tools::to_string(mm) + "]";
                    else
                        lhs_str = "rA[" + tools::to_string(kk) + "][" + tools::to_string(mm/simd_width_) + "].s" + tools::to_string(mm%simd_width_);
                    if(use_B_local_ || simd_width_==1)
                        rhs_str = "rB[" + tools::to_string(kk) + "]["+tools::to_string(nn)+"]";
                    else
                        rhs_str = "rB[" + tools::to_string(kk) + "]["+tools::to_string(nn/simd_width_)+"].s"+tools::to_string(nn%simd_width_);
                    stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
                }
            }
        }




        stream.dec_tab();
        stream << "}" << std::endl;

        if(use_A_local_){
            if(A_trans_=='N')
                stream << A_->name() << " += " << kL_ << "*" << A_->ld() << ";" << std::endl;
            else
                stream << A_->name() << " += " << kL_ << ";" << std::endl;
        }

        if(use_B_local_){
            if(B_trans_=='T')
                stream << B_->name() << " += " << kL_ << "*" << B_->ld() << ";" << std::endl;
            else
                stream << B_->name() << " += " << kL_ << ";" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;


        if(C_->interpret_as_transposed()==false){
            stream << C_->name() << "+= gidx*" << mL_ << ";" << std::endl;
            stream << C_->name() << "+= idx*" << simd_width_ << ";" << std::endl;
            stream << C_->name() << "+= gidy*" << nL_ << "*" << C_->ld() << ";" << std::endl;
            stream << C_->name() << "+= idy*" << simd_width_ << "*" << C_->ld() << ";" << std::endl;
            for(unsigned int m=0 ; m < mS_ ; ++m){
                for(unsigned int n=0 ; n < nS_ ; ++n){
                    std::string j = tools::to_string((n/simd_width_)*(local_size_1_*simd_width_) + n%simd_width_);
                    prod_->access_name("rC["+tools::to_string(m)+"]["+tools::to_string(n)+"]");
                    std::string str;
                    tree_parsing::traverse(statements.front().first, statements.front().second, tree_parsing::evaluate_expression_traversal(index_tuple("0", "M", j, "N"), -1, str, mapping[0]), false);
                    stream << str << ";" << std::endl;
                }
                if((m+1)%simd_width_>0)
                    stream << C_->name() << "+=1;" << std::endl;
                else
                    stream << C_->name() << "+=" << (local_size_0_*simd_width_) - (simd_width_-1) << ";" << std::endl;
            }
        }
        else{
            stream << C_->name() << "+= gidx*" << mL_ << "*" << C_->ld() << ";" << std::endl;
            stream << C_->name() << "+= idx*" << simd_width_ << "*" << C_->ld() << ";" << std::endl;
            stream << C_->name() << "+= gidy*" << nL_ << ";" << std::endl;
            stream << C_->name() << "+= idy*" << simd_width_ << ";" << std::endl;
            for(unsigned int n=0 ; n < nS_ ; ++n){
                for(unsigned int m=0 ; m < mS_ ; ++m){
                    std::string j = tools::to_string((m/simd_width_)*(local_size_0_*simd_width_) + m%simd_width_);
                    prod_->access_name("rC["+tools::to_string(m)+"]["+tools::to_string(n)+"]");
                    std::string str;
                    tree_parsing::traverse(statements.front().first, statements.front().second, tree_parsing::evaluate_expression_traversal(index_tuple("0", "N", j, "M"), -1, str, mapping[0]), false);
                    stream << str << ";" << std::endl;
                }
                if((n+1)%simd_width_>0)
                    stream << C_->name() << "+=1;" << std::endl;
                else
                    stream << C_->name() << "+=" << (local_size_1_*simd_width_) - (simd_width_-1) << ";" << std::endl;
            }
        }


    }

private:
    const char A_trans_;
    const char B_trans_;

    unsigned int kL_;

    unsigned int mS_;
    unsigned int kS_;
    unsigned int nS_;

    bool use_A_local_;
    bool use_B_local_;

    unsigned int local_fetch_0_;
    unsigned int local_fetch_1_;

    unsigned int mL_;
    unsigned int nL_;

    mapped_matrix_product * prod_;
    mapped_matrix * A_;
    mapped_matrix * B_;
    mapped_matrix * C_;
};

}

}

#endif
