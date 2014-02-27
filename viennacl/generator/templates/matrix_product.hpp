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


      std::size_t lmem_used(std::size_t scalartype_size) const {
        std::size_t lmem_used = 0;
        if(use_a_local_)
          lmem_used += KL_ * (ML_+1) * scalartype_size;
        if(use_b_local_)
          lmem_used += NL_ * (KL_ + 1) * scalartype_size;
        return lmem_used;
      }

      virtual void print(std::ostream & s) const{
        s << "{simd_width,local_size1,kl,local_size2,ms,ks,ns,use_a_local,use_b_local,local_fetch0,local_fetch1} = {"
          << simd_width_ << ","
          << ls0_ << ","
          << KL_ << ","
          << ls1_ << ","
          << ms_ << ","
          << ks_ << ","
          << ns_ << ","
          << use_a_local_ << ","
          << use_b_local_ << ","
          << local_fetch0_ << ","
          << local_fetch1_ << "}" ;
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

      static std::string csv_format() {
        return "simd_width, local_size1, kl, local_size2, ms, ks, ns, use_a_local, use_b_local, local_fetch0, local_fetch1";
      }

      std::string csv_representation() const{
        std::ostringstream oss;
        oss << simd_width_ << ","
            << ls0_ << ","
            << KL_ << ","
            << ls1_ << ","
            << ms_ << ","
            << ks_ << ","
            << ns_ << ","
            << use_a_local_ << ","
            << use_b_local_ << ","
            << local_fetch0_ << ","
            << local_fetch1_;
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

        transform_block(*lhs,use_a_local_,ml_lhs,cache_width_lhs,ms_lhs,ks_lhs);
        transform_block(*rhs,use_b_local_,cache_width_rhs,nl_rhs,ks_rhs,ns_rhs);

        std::string C_scalartype = assigned->scalartype();
        std::string A_scalartype = use_a_local_?lhs->scalartype():lhs->simd_scalartype();
        std::string B_scalartype = use_b_local_?rhs->scalartype():rhs->simd_scalartype();

        //////////////////
        /// DECLARATIONS
        /// //////////////


        ///Result Values
        stream << C_scalartype << " " << "rC[" << ms_ << "][" << ns_ <<"]  = {(" << assigned->scalartype() << ")0};" << std::endl;
        stream << A_scalartype << " " << "rA[" << ks_ << "][" << ms_lhs << "];" << std::endl;
        stream << B_scalartype << " " << "rB[" << ks_ << "][" << ns_rhs <<"];" << std::endl;
        stream << std::endl;


        if(use_a_local_)
          stream << "__local " << lhs->scalartype() << " lA[" << KL_ * (ML_ + 1) << "];" << std::endl;
        if(use_b_local_)
          stream << "__local " << rhs->scalartype() << " lB[" << KL_ * (NL_ + 1) << "];" << std::endl;
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
          stream << lhs->name() << " +=  gidx*" << ML_/simd_width_ << "+ idxT + idyT*" << lhs->ld()  << ";" << std::endl;
        else
          stream << lhs->name() << " += gidx*" << ML_/simd_width_ << "+ idx" << ";" << std::endl;

        if(use_b_local_)
          stream << rhs->name() << " +=  gidy*" << NL_/simd_width_ << "+ idxT + idyT*" << rhs->ld()  << ";" << std::endl;
        else
          stream << rhs->name() << " +=  gidy*" << NL_/simd_width_ << "+ idy;" << std::endl;

        stream << std::endl;

        stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << KL_ << "){" << std::endl;
        stream.inc_tab();

        if(use_a_local_)
            stream << "__local " << lhs->scalartype() << "* plA = lA + idyT*" << ML_+1 << "+" << simd_width_ << "*idxT;" << std::endl;

        if(use_b_local_)
            stream << "__local " << rhs->scalartype() << "* plB = lB + idyT*" << NL_+1 << "+" << simd_width_ << "*idxT;" << std::endl;


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
              stream << "(" <<  lhs->name() << "[" << m/simd_width_ <<  "+"  << k << "*" << lhs->ld() << "]," << m << ",plA);" << std::endl;
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
              stream << "(" <<  rhs->name() << "[" << n/simd_width_ <<  "+"  << k << "*" << rhs->ld() << "]," << n << ",plB);" << std::endl;
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
              stream << "rA[" << kk << "][" << mm << "] = " << lhs->name() << "[" << mm*ls0_ << "];" << std::endl;
          }


          ///Fetch RHS to registers
          for(unsigned int nn=0 ; nn < ns_/simd_width_ ; ++nn){
            if(use_b_local_)
              for(unsigned int ss = 0 ; ss < simd_width_ ; ++ss)
                  stream << "rB[" << kk << "][" << nn*simd_width_ + ss << "] = lB[offB + " << nn*ls1_*simd_width_ + ss << "];" << std::endl;
            else
              stream << "rB[" << kk << "][" << nn << "] = " << rhs->name() << "[" << nn*ls1_ << "];" << std::endl;
          }

          ///Increment pointers
          if(!use_a_local_ && !lhs->interpret_as_transposed())
            stream << lhs->name() << " += " << lhs->ld() << ";" << std::endl;
          else
            stream << "offA += " << ML_+1 << ";" << std::endl;

          if(!use_b_local_ && rhs->interpret_as_transposed())
            stream << rhs->name() << " += " << rhs->ld() << ";" << std::endl;
          else
            stream << "offB += " << NL_+1 << ";" << std::endl;
        }


        for(unsigned int kk = 0 ; kk < ks_ ; ++kk){
          for(unsigned int mm=0 ; mm < ms_ ; ++mm){
            for(unsigned int nn=0 ; nn < ns_ ; ++nn){
              std::string res_str, lhs_str, rhs_str;
              res_str = "rC[" + utils::to_string(mm) + "][" + utils::to_string(nn) + "]";
              if(!lhs->interpret_as_transposed()){
                if(use_a_local_ || simd_width_==1)
                  lhs_str = "rA[" + utils::to_string(kk) + "][" + utils::to_string(mm) + "]";
                else
                  lhs_str = "rA[" + utils::to_string(kk) + "][" + utils::to_string(mm/simd_width_) + "].s" + utils::to_string(mm%simd_width_);
              }
              if(rhs->interpret_as_transposed()){
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
          stream << lhs->name() << " += " << KL_ << "*" << lhs->ld() << ";" << std::endl;
        if(use_b_local_)
          stream << rhs->name() << " += " << KL_ << "*" << rhs->ld() << ";" << std::endl;

        stream.dec_tab();
        stream << "}" << std::endl;


        stream << "uint offset_x = gidx*" << ML_ << "+ idx*" << simd_width_ << ";" << std::endl;
        stream << "uint offset_y = gidy*" << NL_ << "+ idy*" << simd_width_ << ";" << std::endl;

//        stream << "#pragma unroll"<< std::endl;
//        stream << "for(unsigned int m = 0 ; m < " << ms_ << " ; ++m){" << std::endl;
//        stream.inc_tab();
//        stream << "#pragma unroll"<< std::endl;
//        stream << "for(unsigned int n = 0 ; n < " << ns_ << " ; ++n){" << std::endl;
//        stream.inc_tab();
//        prod->access_name("rC[m][n]");
//        std::string i = "offset_x + m*" + utils::to_string(ls0_);
//        std::string j = "offset_y + n*" + utils::to_string(ls1_);
//        std::string str;
//        tree_parsing::traverse(statements.front().first, statements.front().second, tree_parsing::expression_generation_traversal(std::make_pair(i, j), -1, str, mapping[0]), false);
//        stream << str << ";" << std::endl;
//        stream.dec_tab();
//        stream << "}" << std::endl;
//        stream.dec_tab();
//        stream << "}" << std::endl;


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

      bool use_a_local_;
      bool use_b_local_;

      std::size_t local_fetch0_;
      std::size_t local_fetch1_;
    };

  }

}

#endif
