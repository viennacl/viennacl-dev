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

#include "viennacl/generator/templates/template_base.hpp"

#include "viennacl/generator/mapped_objects.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/tree_parsing/fetch.hpp"
#include "viennacl/generator/tree_parsing/elementwise_expression.hpp"
#include "viennacl/forwards.h"

#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{

    template<bool A_TRANS, bool B_TRANS>
    class matrix_product : public profile_base{

      std::size_t lmem_used(std::size_t scalartype_size) const {
        std::size_t lmem_used = 0;
        if(use_a_local_)
          lmem_used += KL_ * (ML_+1) * scalartype_size;
        if(use_b_local_)
          lmem_used += NL_ * (KL_ + 1) * scalartype_size;
        return lmem_used;
      }

      bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const{
        static const unsigned int alignment = 128;
        bool res = false;
        res |= alignment % ML_ > 0;
        res |= alignment % KL_ > 0;
        res |= alignment % NL_ > 0;
        res |= (ms_ % simd_width_) > 0;
        res |= (ns_ % simd_width_) > 0;
        if(use_a_local_)
          res |= (ML_ % (local_fetch0_*simd_width_)) > 0;
        if(use_b_local_)
          res |= (NL_ % (local_fetch0_*simd_width_)) > 0;
        if(use_a_local_ || use_b_local_){
          res |= (KL_ % local_fetch1_)> 0;
          res |= ((local_fetch0_*local_fetch1_) !=(ls0_*ls1_));
        }
        return res;
      }

    public:
      /** @brief The user constructor */
      matrix_product(unsigned int simd_width
                     , std::size_t ls0, std::size_t KL, std::size_t ls1
                     , unsigned int ms, unsigned int ks, unsigned int ns
                     , bool use_a_local, bool use_b_local
                     , std::size_t local_fetch0, std::size_t local_fetch1) : profile_base(simd_width,ls0, ls1,1){
        ls0_ = ls0;
        ls1_ = ls1;
        KL_=KL;
        ML_= ms*ls0;
        NL_=ns*ls1;
        ms_ = ms;
        ks_=ks;
        ns_=ns;
        use_a_local_ = use_a_local;
        use_b_local_ = use_b_local;
        local_fetch0_ = local_fetch0;
        local_fetch1_ = local_fetch1;
      }

      static std::string csv_format(){
        return "simd_width,local_size1,kl,local_size2,ms,ks,ns,use_a_local,use_b_local,local_fetch0,local_fetch1";
      }

      std::string csv_representation() const{
        std::ostringstream oss;
        oss << simd_width_ << "," << ls0_ << "," << KL_ << "," << ls1_ << "," << ms_ << "," << ks_ << ","
            << ns_ << "," << use_a_local_ << "," << use_b_local_ << "," << local_fetch0_ << "," << local_fetch1_;
        return oss.str();
      }

      virtual void init(std::pair<scheduler::statement, scheduler::statement_node> const & statement_pair, mapping_type & mapping) {
        scheduler::statement const & statement = statement_pair.first;
        scheduler::statement_node const & root_node = statement_pair.second;
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

      void configure_range_enqueue_arguments(std::size_t kernel_id, viennacl::ocl::kernel & k, unsigned int & n_arg)  const {
        //set M, N
        scheduler::statement_node const & first_node = statements_->front().second;
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
        scheduler::statement::container_type const & exprs = statements_->back().first.array();

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

      void add_kernel_arguments(std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "M");
        arguments_string += generate_value_kernel_argument("unsigned int", "N");
        arguments_string += generate_value_kernel_argument("unsigned int", "K");
      }

    private:

      void core(std::size_t /*kernel_id*/, utils::kernel_generation_stream& stream, std::vector<mapping_type> const & mapping) const {

        //////////////////
        /// INIT
        /// //////////////

        if(simd_width_>1){
          stream << A_->ld() << "/=" << simd_width_ << ";" << std::endl;
          stream << B_->ld() << "/=" << simd_width_ << ";" << std::endl;
        }

        std::string C_scalartype = C_->scalartype();
        std::string A_scalartype = use_a_local_?A_->scalartype():A_->simd_scalartype();
        std::string B_scalartype = use_b_local_?B_->scalartype():B_->simd_scalartype();

        //////////////////
        /// DECLARATIONS
        /// //////////////


        ///Result Values
        stream << C_scalartype << " " << "rC[" << ms_ << "][" << ns_ <<"]  = {(" << C_->scalartype() << ")0};" << std::endl;
        stream << A_scalartype << " " << "rA[" << ks_ << "][" << (use_a_local_?ms_:ms_/simd_width_) << "];" << std::endl;
        stream << B_scalartype << " " << "rB[" << ks_ << "][" << (use_b_local_?ns_:ns_/simd_width_) <<"];" << std::endl;
        stream << std::endl;


        if(use_a_local_)
          stream << "__local " << A_->scalartype() << " lA[" << KL_ * (ML_ + 1) << "];" << std::endl;
        if(use_b_local_)
          stream << "__local " << B_->scalartype() << " lB[" << KL_ * (NL_ + 1) << "];" << std::endl;
        stream << std::endl;


        stream << "uint gidx = get_group_id(0);" << std::endl;
        stream << "uint gidy = get_group_id(1);" << std::endl;
        stream << "uint idx = get_local_id(0);" << std::endl;
        stream << "uint idy = get_local_id(1);" << std::endl;
        if(use_a_local_ || use_b_local_){
          stream << std::endl;
          stream << "uint idt = " << ls0_ << "*idy + idx;" << std::endl;
          stream << "uint idxT = idt % " << local_fetch0_ << ";" << std::endl;
          stream << "uint idyT = idt / " << local_fetch0_ << ";" << std::endl;
        }
        stream << std::endl;


        if(use_a_local_)
          stream << A_->name() << " +=  gidx*" << ML_/simd_width_ << "+ idxT + idyT*" << A_->ld()  << ";" << std::endl;
        else
          stream << A_->name() << " += gidx*" << ML_/simd_width_ << "+ idx" << ";" << std::endl;

        if(use_b_local_)
          stream << B_->name() << " +=  gidy*" << NL_/simd_width_ << "+ idxT + idyT*" << B_->ld()  << ";" << std::endl;
        else
          stream << B_->name() << " +=  gidy*" << NL_/simd_width_ << "+ idy;" << std::endl;

        stream << std::endl;

        stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << KL_ << "){" << std::endl;
        stream.inc_tab();

        if(use_a_local_)
            stream << "__local " << A_->scalartype() << "* plA = lA + idyT*" << ML_+1 << "+" << simd_width_ << "*idxT;" << std::endl;

        if(use_b_local_)
            stream << "__local " << B_->scalartype() << "* plB = lB + idyT*" << NL_+1 << "+" << simd_width_ << "*idxT;" << std::endl;


        if(use_a_local_ || use_b_local_)
            stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
       
        ///Fetch LHS to Local Memory
        if(use_a_local_)
        {
          for(unsigned int k = 0 ; k < KL_ ; k += local_fetch1_){
            for(unsigned int m = 0 ; m < ML_ ; m += local_fetch0_*simd_width_){
              stream << "vstore" ;
              if(simd_width_>1)
                stream << simd_width_;
              stream << "(" <<  A_->name() << "[" << m/simd_width_ <<  "+"  << k << "*" << A_->ld() << "],0,plA+" << m << ");" << std::endl;
            }
            if((k+local_fetch1_)<KL_)
                stream << "plA += " << local_fetch1_*(ML_+1) << ";" << std::endl;
          }
        }

        ///Fetch RHS to Local Memory
        if(use_b_local_)
        {
          for(unsigned int k = 0 ; k < KL_ ; k += local_fetch1_){
            for(unsigned int n = 0 ; n < NL_ ; n += local_fetch0_*simd_width_){
              stream << "vstore" ;
              if(simd_width_>1)
                stream << simd_width_;
              stream << "(" <<  B_->name() << "[" << n/simd_width_ <<  "+"  << k << "*" << B_->ld() << "],0,plB+" << n << ");" << std::endl;
            }
            if((k+local_fetch1_)<KL_)
                stream << "plB += " << local_fetch1_*(NL_+1) << ";" << std::endl;
          }
        }

        if(use_a_local_ || use_b_local_)
          stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        stream << "uint offA = " << simd_width_ << "*idx;" << std::endl;
        stream << "uint offB = " << simd_width_ << "*idy;" << std::endl;

        //stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int k = 0 ; k < " << KL_ << "; k+=" << ks_ << "){" << std::endl;
        stream.inc_tab();

        ///Fetch LHS to registers
        for(unsigned int kk = 0 ; kk < ks_ ; ++kk){
          for(unsigned int mm = 0 ; mm < ms_/simd_width_ ; ++mm){
            if(use_a_local_)
              for(unsigned int ss = 0 ; ss < simd_width_ ; ++ss)
                  stream << "rA[" << kk << "][" << mm*simd_width_ + ss << "] = lA[offA + " << mm*ls0_*simd_width_ + ss << "];" << std::endl;
            else
              stream << "rA[" << kk << "][" << mm << "] = " << A_->name() << "[" << mm*ls0_ << "];" << std::endl;
          }


          ///Fetch RHS to registers
          for(unsigned int nn=0 ; nn < ns_/simd_width_ ; ++nn){
            if(use_b_local_)
              for(unsigned int ss = 0 ; ss < simd_width_ ; ++ss)
                  stream << "rB[" << kk << "][" << nn*simd_width_ + ss << "] = lB[offB + " << nn*ls1_*simd_width_ + ss << "];" << std::endl;
            else
              stream << "rB[" << kk << "][" << nn << "] = " << B_->name() << "[" << nn*ls1_ << "];" << std::endl;
          }

          ///Increment pointers
          if(!use_a_local_ && !A_->interpret_as_transposed())
            stream << A_->name() << " += " << A_->ld() << ";" << std::endl;
          else
            stream << "offA += " << ML_+1 << ";" << std::endl;

          if(!use_b_local_ && B_->interpret_as_transposed())
            stream << B_->name() << " += " << B_->ld() << ";" << std::endl;
          else
            stream << "offB += " << NL_+1 << ";" << std::endl;
        }


        for(unsigned int kk = 0 ; kk < ks_ ; ++kk){
            for(unsigned int nn=0 ; nn < ns_ ; ++nn){
                for(unsigned int mm=0 ; mm < ms_ ; ++mm){
                    std::string res_str, lhs_str, rhs_str;
                    res_str = "rC[" + utils::to_string(mm) + "][" + utils::to_string(nn) + "]";
                    if(!A_->interpret_as_transposed()){
                        if(use_a_local_ || simd_width_==1)
                            lhs_str = "rA[" + utils::to_string(kk) + "][" + utils::to_string(mm) + "]";
                        else
                            lhs_str = "rA[" + utils::to_string(kk) + "][" + utils::to_string(mm/simd_width_) + "].s" + utils::to_string(mm%simd_width_);
                    }
                    if(B_->interpret_as_transposed()){
                        if(use_b_local_ || simd_width_==1)
                            rhs_str = "rB[" + utils::to_string(kk) + "]["+utils::to_string(nn)+"]";
                        else
                            rhs_str = "rB[" + utils::to_string(kk) + "]["+utils::to_string(nn/simd_width_)+"].s"+utils::to_string(nn%simd_width_);
                    }
                    stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
                }
            }
        }




        stream.dec_tab();
        stream << "}" << std::endl;
        if(use_a_local_)
          stream << A_->name() << " += " << KL_ << "*" << A_->ld() << ";" << std::endl;
        if(use_b_local_)
          stream << B_->name() << " += " << KL_ << "*" << B_->ld() << ";" << std::endl;

        stream.dec_tab();
        stream << "}" << std::endl;


        stream << "uint offset_x = gidx*" << ML_ << "+ idx*" << simd_width_ << ";" << std::endl;
        stream << "uint offset_y = gidy*" << NL_ << "+ idy*" << simd_width_ << ";" << std::endl;
        for(unsigned int m=0 ; m < ms_ ; ++m){
          for(unsigned int n=0 ; n < ns_ ; ++n){
            std::string i = "offset_x +" + utils::to_string((m/simd_width_)*(ls0_*simd_width_) + m%simd_width_);
            std::string j = "offset_y +" + utils::to_string((n/simd_width_)*(ls1_*simd_width_) + n%simd_width_);
            if(C_->interpret_as_transposed())
              std::swap(i,j);
            prod_->access_name("rC["+utils::to_string(m)+"]["+utils::to_string(n)+"]");
            std::string str;
            tree_parsing::traverse(statements_->front().first, statements_->front().second, tree_parsing::expression_generation_traversal(std::make_pair(i, j), -1, str, mapping[0]), false);
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

      bool use_a_local_;
      bool use_b_local_;

      std::size_t local_fetch0_;
      std::size_t local_fetch1_;

      mapped_matrix_product * prod_;
      mapped_matrix * A_;
      mapped_matrix * B_;
      mapped_matrix * C_;
    };

  }

}

#endif
