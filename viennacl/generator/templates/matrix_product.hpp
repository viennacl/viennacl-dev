#ifndef VIENNACL_GENERATOR_TEMPLATES_MATRIX_PRODUCT_HPP
#define VIENNACL_GENERATOR_TEMPLATES_MATRIX_PRODUCT_HPP

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

#include "viennacl/generator/mapped_objects.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/tree_parsing/fetch.hpp"
#include "viennacl/generator/tree_parsing/elementwise_expression.hpp"
#include "viennacl/forwards.h"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

namespace generator{

class matrix_product : public profile_base{

    bool is_slow_impl(viennacl::ocl::device const &) const { return false; }

    std::size_t lmem_used(std::size_t scalartype_size) const {
        std::size_t lmem_used = 0;
        if(use_lhs_shared_)
            lmem_used += KL_ * (ML_+1) * scalartype_size;
        if(use_rhs_shared_)
            lmem_used += NL_ * (KL_ + 1) * scalartype_size;
        return lmem_used;
    }

    virtual void print(std::ostream & s) const{
        s << "{vector_type, local_size1, cache_width, local_size2, ms, ks, ns, use_lhs_shared, use_rhs_shared} = {"
          << simd_width_ << ","
          << ls0_ << ", "
          << KL_ << ", "
          << ls1_ << ", "
          << ms_ << ", "
          << ks_ << ", "
          << ns_ << ", "
          << use_lhs_shared_ << ", " << use_rhs_shared_ << "}" ;
    }


    bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const{
        static const unsigned int alignment = 128;
        return  DIM_XA_*DIM_YA_ != (ls0_*ls1_)
                || DIM_XB_*DIM_YB_ != (ls0_*ls1_)
                || ML_ % DIM_XA_ > 0
                || KL_ % DIM_YA_> 0

                || KL_ % DIM_YB_ > 0
                || NL_ % DIM_XB_ > 0

                || alignment % ML_ > 0
                || alignment % KL_ > 0
                || alignment % NL_ > 0
                || (ms_ % simd_width_) > 0
                || (ns_ % simd_width_) > 0;
    }

public:
    /** @brief The user constructor */
    matrix_product(unsigned int vectorization
                   , std::size_t local_size1, std::size_t cache_width, std::size_t local_size2
                   , unsigned int ms, unsigned int ks, unsigned int ns
                   , bool use_lhs_shared, bool use_rhs_shared) : profile_base(vectorization,local_size1, local_size2,1){
        ls0_ = local_size1;
        ls1_ = local_size2;
        KL_=cache_width;
        ML_= ms*local_size1;
        NL_=ns*local_size2;
        ms_ = ms;
        ks_=ks;
        ns_=ns;
        use_lhs_shared_ = use_lhs_shared;
        use_rhs_shared_ = use_rhs_shared;
        DIM_XA_ = 32;
        DIM_YA_ = 4;
        DIM_XB_ = 32;
        DIM_YB_ = 4;
    }

    static std::string csv_format() {
        return "Vec,LSize1,CacheWidth,LSize2,mS,kS,nS,NumGroups";
    }

    std::string csv_representation() const{
        std::ostringstream oss;
        oss << simd_width_
            << "," << ls0_
            << "," << KL_
            << "," << ls1_
            << "," << ms_
            << "," << ks_
            << "," << ns_
            << "," << use_lhs_shared_
            << "," << use_rhs_shared_;
        return oss.str();
    }

    virtual void set_simd_width(statements_type::value_type const & statement_pair, mapping_type & mapping) const{
        for(mapping_type::const_iterator iit = mapping.begin() ; iit != mapping.end() ; ++iit)
            if(mapped_handle * p = dynamic_cast<mapped_handle *>(iit->second.get()))
                p->set_simd_width(1);

        scheduler::statement::container_type const & exprs = statement_pair.first.array();

        scheduler::statement_node const * prod_node = NULL;
        for(scheduler::statement::container_type::const_iterator it = exprs.begin() ; it != exprs.end() ; ++it)
            if(it->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                prod_node = &(*it);

        mapped_matrix * lhs = NULL;
        if(prod_node->lhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            lhs = (mapped_matrix *)mapping[std::make_pair(&exprs[prod_node->lhs.node_index],tree_parsing::LHS_NODE_TYPE)].get();
        else
            lhs = (mapped_matrix *)mapping[std::make_pair(prod_node, tree_parsing::LHS_NODE_TYPE)].get();

        mapped_matrix * rhs = NULL;
        if(prod_node->rhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            rhs = (mapped_matrix *)mapping[std::make_pair(&exprs[prod_node->rhs.node_index], tree_parsing::LHS_NODE_TYPE)].get();
        else
            rhs = (mapped_matrix *)mapping[std::make_pair(prod_node,tree_parsing::RHS_NODE_TYPE)].get();

        lhs->set_simd_width(simd_width_);
        rhs->set_simd_width(simd_width_);
    }

    void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const {
        //set M, N
        scheduler::statement_node const & first_node = statements.front().second;
        unsigned int M = utils::call_on_matrix(first_node.lhs, utils::internal_size1_fun());
        unsigned int N = utils::call_on_matrix(first_node.lhs, utils::internal_size2_fun());

        //set ND range
        configure_local_sizes(k, kernel_id);
        k.global_work_size(0, M/ms_);
        k.global_work_size(1, N/ns_);

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

    static std::string size1() { return "M";  }
    static std::string size2() { return "K"; }
    static std::string size3() { return "N"; }

    void add_kernel_arguments(statements_type  const & /*statements*/, std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
        arguments_string += generate_value_kernel_argument("unsigned int", "K");
    }

private:

    void transform_block(mapped_matrix const & mat, bool store_shared
                         , unsigned int & large_block_1, unsigned int & large_block_2
                         , unsigned int & small_block_1, unsigned int & small_block_2) const {
        if(mat.interpret_as_transposed()){
            large_block_2/=simd_width_;
            if(!store_shared)
                small_block_2/=simd_width_;
        }
        else{
            large_block_1/=simd_width_;
            if(!store_shared)
                small_block_1/=simd_width_;
        }
    }


    std::string helper_variable(utils::kernel_generation_stream & stream
                                , bool store_in_register
                                , std::string const & type
                                , std::string const & name
                                , std::string const & expr) const {
        if(!store_in_register)
            return expr;
        stream << type << " " << name << " = " << expr << ";" << std::endl;
        return name;
    }

    void core(std::size_t /*kernel_id*/, utils::kernel_generation_stream& stream, expression_descriptor descriptor, statements_type const & statements, std::vector<mapping_type> const & mapping) const {


        bool strided = true;

        //////////////////
        /// INIT
        /// //////////////

        mapped_matrix const * assigned = static_cast<mapped_matrix const *>(mapping.at(0).at(std::make_pair(&statements.front().second,tree_parsing::LHS_NODE_TYPE)).get());
        mapped_matrix_product* prod = NULL;
        mapped_matrix const * lhs = NULL;
        mapped_matrix const * rhs = NULL;

        scheduler::statement::container_type const & exprs = statements.back().first.array();

        scheduler::statement_node const * prod_node = NULL;
        for(scheduler::statement::container_type::const_iterator it = exprs.begin() ; it != exprs.end() ; ++it)
            if(it->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
                prod_node = &(*it);

        prod = (mapped_matrix_product *)mapping.at(0).at(std::make_pair(prod_node, tree_parsing::PARENT_NODE_TYPE)).get();

        if(prod_node->lhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            lhs = (mapped_matrix const *)mapping.at(0).at(std::make_pair(&exprs[prod_node->lhs.node_index],tree_parsing::LHS_NODE_TYPE)).get();
        else
            lhs = (mapped_matrix const *)mapping.at(0).at(std::make_pair(prod_node, tree_parsing::LHS_NODE_TYPE)).get();

        if(prod_node->rhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY)
            rhs = (mapped_matrix const *)mapping.at(0).at(std::make_pair(&exprs[prod_node->rhs.node_index], tree_parsing::LHS_NODE_TYPE)).get();
        else
            rhs = (mapped_matrix const *)mapping.at(0).at(std::make_pair(prod_node,tree_parsing::RHS_NODE_TYPE)).get();


        if(simd_width_>1){
            stream << lhs->ld() << "/=" << simd_width_ << ";" << std::endl;
            stream << rhs->ld() << "/=" << simd_width_ << ";" << std::endl;
        }



        unsigned int ms_res = ms_, ns_res = ns_;
        unsigned int ml_lhs = ML_, cache_width_lhs = KL_, ms_lhs = ms_, ks_lhs = ks_;
        unsigned int cache_width_rhs = KL_, nl_rhs = NL_, ks_rhs = ks_, ns_rhs = ns_;

        transform_block(*lhs,use_lhs_shared_,ml_lhs,cache_width_lhs,ms_lhs,ks_lhs);
        transform_block(*rhs,use_rhs_shared_,cache_width_rhs,nl_rhs,ks_rhs,ns_rhs);

        std::string C_scalartype = assigned->scalartype();
        std::string A_scalartype = use_lhs_shared_?lhs->scalartype():lhs->simd_scalartype();
        std::string B_scalartype = use_rhs_shared_?rhs->scalartype():rhs->simd_scalartype();

        //////////////////
        /// DECLARATIONS
        /// //////////////


        ///Result Values
        stream << C_scalartype << " " << "rC[" << ms_ << "][" << ns_ <<"]  = {(" << assigned->scalartype() << ")0};" << std::endl;
        stream << A_scalartype << " " << "rA[" << ms_lhs << "];" << std::endl;
        stream << B_scalartype << " " << "rB[" << ns_rhs <<"];" << std::endl;

        if(simd_width_>1 && (use_lhs_shared_ || use_rhs_shared_))
            stream << lhs->simd_scalartype() << " tmpreg;" << std::endl;
        stream << std::endl;


        if(use_lhs_shared_)
            stream << "__local " << lhs->scalartype() << " lA[" << KL_ << "][" << ML_ + 1 << "];" << std::endl;
        if(use_rhs_shared_)
            stream << "__local " << rhs->scalartype() << " lB[" << NL_ << "][" << KL_ + 1 << "];" << std::endl;
        stream << std::endl;


        stream << "uint gidx = get_group_id(0);" << std::endl;
        stream << "uint gidy = get_group_id(1);" << std::endl;
        stream << "uint idx = get_local_id(0);" << std::endl;
        stream << "uint idy = get_local_id(1);" << std::endl;
        if(use_lhs_shared_ || use_rhs_shared_){
          stream << std::endl;
          stream << "uint idt = " << ls0_ << "*idy + idx;" << std::endl;
          stream << "uint idxA = idt % " << DIM_XA_ << ";" << std::endl;
          stream << "uint idyA = idt / " << DIM_XA_ << ";" << std::endl;
          stream << "uint idxB = idt % " << DIM_XB_ << ";" << std::endl;
          stream << "uint idyB = idt / " << DIM_XB_ << ";" << std::endl;
        }
        stream << std::endl;

        stream << "uint offset_x = gidx*" << ML_ << "+ idx*" << simd_width_ << ";" << std::endl;
        stream << "uint offset_y = gidy*" << NL_ << "+ idy*" << simd_width_ << ";" << std::endl;

        stream << std::endl;

        if(use_lhs_shared_)
            stream << lhs->name() << " +=  gidx*" << ML_ << "+ idxA*" << simd_width_ << "+ idyA*" << lhs->ld()  << ";" << std::endl;
        else
            stream << lhs->name() << " +=  offset_x/" << simd_width_  << ";" << std::endl;

        if(use_rhs_shared_)
          stream << rhs->name() << " +=  gidy*" << NL_ << "+ idxB*" << simd_width_ << "+ idyB*" << rhs->ld()  << ";" << std::endl;
        else
          stream << rhs->name() << " +=  offset_y/" << simd_width_  << ";" << std::endl;

        stream << std::endl;


        stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << KL_ << "){" << std::endl;
        stream.inc_tab();

        if(use_lhs_shared_ || use_rhs_shared_)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        ///Fetch LHS to Local Memory
        if(use_lhs_shared_)
        {
            for(unsigned int k = 0 ; k < KL_ ; k += DIM_YA_){
                for(unsigned int m = 0 ; m < ML_ ; m += DIM_XA_*simd_width_){
                    if(simd_width_>1){
                        stream << "tmpreg = " << lhs->name() << "[" << m/simd_width_ <<  "+"  << k << "*" << lhs->ld() << "];" << std::endl;
                        for(unsigned int s = 0 ; s < simd_width_ ; ++s)
                            stream << "lA[idyA + " << k << "][" << simd_width_ << "*idxA + " << m + s << "] = tmpreg.s" << s << ";" << std::endl;
                    }
                    else{
                        stream << "lA[idyA + " << k << "][idxA + " << m << "] = " << lhs->name() << "[" << m <<  "+"  << k << "*" << lhs->ld() << "];" << std::endl;
                    }
                }
            }
        }

        ///Fetch RHS to Local Memory
        if(use_rhs_shared_)
        {
            for(unsigned int k = 0 ; k < KL_ ; k += DIM_YB_){
                for(unsigned int n = 0 ; n < NL_ ; n += DIM_XB_*simd_width_){
                    if(simd_width_>1){
                        stream << "tmpreg = " << rhs->name() << "[" << n/simd_width_ <<  "+"  << k << "*" << rhs->ld() << "];" << std::endl;
                        for(unsigned int s = 0 ; s < simd_width_ ; ++s)
                            stream << "lB[" << simd_width_ << "*idxB + " << n + s << "][idyB + " << k << "] = tmpreg.s" << s << ";" << std::endl;
                    }
                    else{
                        stream << "lB[idxB + " << n << "][idyB + " << k << "] = " << rhs->name() << "[" << n <<  "+"  << k << "*" << rhs->ld() << "];" << std::endl;
                    }
                }
            }
        }

        if(use_lhs_shared_ || use_rhs_shared_)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int k = 0 ; k < " << KL_ << "; ++k){" << std::endl;
        stream.inc_tab();

        ///Fetch LHS to registers
        for(unsigned int m = 0 ; m < ms_/simd_width_ ; ++m){
            if(use_lhs_shared_)
                for(unsigned int s = 0 ; s < simd_width_ ; ++s)
                    stream << "rA[" << m*simd_width_ + s << "] = lA[k][" << simd_width_ << "*idx + " << m*ls0_*simd_width_ + s << "];" << std::endl;
            else
                stream << "rA[" << m << "] = " << lhs->name() << "[" << m*ls0_ << "];" << std::endl;
        }


        ///Fetch RHS to registers
        for(unsigned int n=0 ; n < ns_/simd_width_ ; ++n){
            if(use_rhs_shared_)
                for(unsigned int s = 0 ; s < simd_width_ ; ++s)
                    stream << "rB[" << n*simd_width_ + s << "] = lB[" << simd_width_ << "*idy + " << n*ls1_*simd_width_ + s << "][k];" << std::endl;
            else
                stream << "rB[" << n << "] = " << rhs->name() << "[" << n*ls1_ << "];" << std::endl;
        }

        for(unsigned int m=0 ; m < ms_ ; ++m){
            for(unsigned int n=0 ; n < ns_ ; ++n){
                std::string res_str, lhs_str, rhs_str;
                res_str = "rC[" + utils::to_string(m) + "][" + utils::to_string(n) + "]";
                if(!lhs->interpret_as_transposed()){
                    if(use_lhs_shared_ || simd_width_==1)
                        lhs_str = "rA[" + utils::to_string(m) + "]";
                    else
                        lhs_str = "rA[" + utils::to_string(m/simd_width_) + "].s" + utils::to_string(m%simd_width_);
                }
                if(rhs->interpret_as_transposed()){
                    if(use_rhs_shared_ || simd_width_==1)
                        rhs_str = "rB["+utils::to_string(n)+"]";
                    else
                        rhs_str = "rB["+utils::to_string(n/simd_width_)+"].s"+utils::to_string(n%simd_width_);
                }
                stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
            }
        }

        if(!use_lhs_shared_ && !lhs->interpret_as_transposed())
            stream << lhs->name() << " += " << lhs->ld() << ";" << std::endl;

        if(!use_rhs_shared_ && rhs->interpret_as_transposed())
            stream << rhs->name() << " += " << rhs->ld() << ";" << std::endl;


        stream.dec_tab();
        stream << "}" << std::endl;
        if(use_lhs_shared_)
            stream << lhs->name() << " += " << KL_ << "*" << lhs->ld() << ";" << std::endl;
        if(use_rhs_shared_)
            stream << rhs->name() << " += " << KL_ << "*" << rhs->ld() << ";" << std::endl;

        stream.dec_tab();
        stream << "}" << std::endl;


        for(unsigned int m=0 ; m < ms_res ; ++m){
            for(unsigned int n=0 ; n < ns_res ; ++n){
                std::string i = "offset_x +" + utils::to_string((m/simd_width_)*(ls0_*simd_width_) + m%simd_width_);
                std::string j = "offset_y +" + utils::to_string((n/simd_width_)*(ls1_*simd_width_) + n%simd_width_);
                if(assigned->interpret_as_transposed())
                    std::swap(i,j);
                prod->access_name("rC["+utils::to_string(m)+"]["+utils::to_string(n)+"]");
                std::string str;
                tree_parsing::traverse(statements.front().first, statements.front().second, tree_parsing::expression_generation_traversal(std::make_pair(i, j), -1, str, mapping[0]), false);
                stream << str << ";" << std::endl;
            }
        }


    }

private:
    std::size_t ls0_;
    std::size_t ls1_;
    std::size_t KL_;

    std::size_t ML_;
    std::size_t NL_;

    std::size_t ms_;
    std::size_t ks_;
    std::size_t ns_;

    std::size_t DIM_XA_;
    std::size_t DIM_YA_;

    std::size_t DIM_XB_;
    std::size_t DIM_YB_;

    bool use_lhs_shared_;
    bool use_rhs_shared_;
};

}

}

#endif
