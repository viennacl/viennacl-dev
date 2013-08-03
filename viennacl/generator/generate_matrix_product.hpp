#ifndef VIENNACL_GENERATOR_GENERATE_MATRIX_PRODUCT_HPP
#define VIENNACL_GENERATOR_GENERATE_MATRIX_PRODUCT_HPP

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


/** @file viennacl/generator/templates/matrix_product.hpp
 *
 * Kernel template for the vector reduction operation
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/generator/generate_template_base.hpp"
#include "viennacl/generator/mapped_types.hpp"
#include "viennacl/generator/utils.hpp"


#include "viennacl/tools/tools.hpp"

namespace viennacl{

  namespace generator{

    class matrix_product : public template_base{
        typedef template_base base_type;
      public:
        typedef base_type::statements_type statements_type;

        class profile : public template_base::profile{
            friend class matrix_product;
            std::size_t lmem_used(std::size_t scalartype_size) const {
              std::size_t lmem_used = 0;
              if(use_lhs_shared_)
                lmem_used += (ml_ + 1) * (kl_ + 1) * scalartype_size;
              if(use_rhs_shared_)
                lmem_used += (kl_ + 1) * (nl_ + 1) * scalartype_size;
              return lmem_used;
            }

            virtual std::ostream & print(std::ostream & s) const{
                s << "{vector_type, ms, ks, ns, ml, kl, nl, use_lhs_shared, use_rhs_shared, unroll} = {"
                  << vectorization_
                  << ms_ << ", "
                  << ks_ << ", "
                  << ns_ << ", "
                  << ml_ << ", "
                  << kl_ << ", "
                  << nl_ << ", "
                  << use_lhs_shared_ << ", " << use_rhs_shared_ << ", " << unroll_ << "}" ;
            }

          public:
            /** @brief The user constructor */
            profile(unsigned int vectorization, unsigned int ml, unsigned int kl, unsigned int nl
                    , unsigned int ms, unsigned int ks, unsigned int ns
                    , bool use_lhs_shared, bool use_rhs_shared
                    , unsigned int unroll) : template_base::profile(vectorization,1){

              ml_= ml; kl_=kl ; nl_=nl;
              ms_ = ms; ks_=ks; ns_=ns;
              use_lhs_shared_ = use_lhs_shared;
              use_rhs_shared_ = use_rhs_shared;
              unroll_ = unroll;
            }

            unsigned int ml() const { return ml_; }
            unsigned int kl() const { return kl_; }
            unsigned int nl() const { return nl_; }

            unsigned int ms() const { return ms_; }
            unsigned int ks() const { return ks_; }
            unsigned int ns() const { return ns_; }

            bool use_lhs_shared() const { return use_lhs_shared_; }
            bool use_rhs_shared() const { return use_rhs_shared_; }

            unsigned int unroll() const { return unroll_; }

            void set_local_sizes(std::size_t & size1, std::size_t & size2, std::size_t /*kernel_id*/) const{
              size1 = ml_/ms_;
              size2 = nl_/ns_;
            }

            void configure_range_enqueue_arguments(std::size_t kernel_id, statements_type  const & statements, viennacl::ocl::kernel & k, unsigned int & n_arg)  const {
              //set M, N
              scheduler::statement_node const & first_node = statements.front().second;
              unsigned int M = utils::call_on_matrix(first_node.lhs, utils::size1_fun());
              unsigned int N = utils::call_on_matrix(first_node.lhs, utils::size2_fun());

              //set ND range
              configure_local_sizes(k, kernel_id);
              k.global_work_size(0, M/ms_);
              k.global_work_size(1, N/ns_);

              //set arguments
              //M,N
              k.arg(n_arg++, cl_uint(M));
              k.arg(n_arg++, cl_uint(N));

              //K
              for(statements_type::const_iterator it = statements.begin() ; it != statements.end() ; ++it){
                scheduler::statement::container_type exprs = it->first.array();
                for(scheduler::statement::container_type::iterator iit = exprs.begin() ; iit != exprs.end() ; ++iit){
                  if(iit->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE){
                    scheduler::statement_node const * current_node = &(*iit);
                    //The LHS of the prod is a matrix
                    if(current_node->lhs.type_family==scheduler::MATRIX_ROW_TYPE_FAMILY
                       ||current_node->lhs.type_family==scheduler::MATRIX_COL_TYPE_FAMILY)
                    {
                      k.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::size2_fun())));
                    }
                    else{
                      //The LHS of the prod is a matrix expression
                      current_node = &exprs[current_node->lhs.node_index];
                      if(current_node->lhs.type_family==scheduler::MATRIX_ROW_TYPE_FAMILY
                         ||current_node->lhs.type_family==scheduler::MATRIX_COL_TYPE_FAMILY)
                      {
                        k.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::size2_fun())));
                      }
                      else if(current_node->rhs.type_family==scheduler::MATRIX_ROW_TYPE_FAMILY
                              ||current_node->rhs.type_family==scheduler::MATRIX_COL_TYPE_FAMILY)
                      {
                        k.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::size2_fun())));
                      }
                      else{
                        assert(false && bool("unexpected expression tree"));
                      }
                    }
                    return;
                  }
                }
              }
            }
            static std::string size1() { return "M";  }
            static std::string size2() { return "K"; }
            static std::string size3() { return "N"; }

            void kernel_arguments(statements_type  const & /*statements*/, std::string & arguments_string) const{
              arguments_string += detail::generate_value_kernel_argument("unsigned int", "M");
              arguments_string += detail::generate_value_kernel_argument("unsigned int", "N");
              arguments_string += detail::generate_value_kernel_argument("unsigned int", "K");
            }

          private:
            unsigned int ml_;
            unsigned int kl_;
            unsigned int nl_;

            unsigned int ms_;
            unsigned int ks_;
            unsigned int ns_;

            bool use_lhs_shared_;
            bool use_rhs_shared_;

            unsigned int unroll_;
        };

        void transform_block(detail::mapped_matrix const & mat_infos, bool is_transposed, bool store_shared
                             , unsigned int & large_block_1, unsigned int & large_block_2
                             , unsigned int & small_block_1, unsigned int & small_block_2) const {
          if(mat_infos.is_row_major()){
            if(is_transposed)
              large_block_1 /= profile_.vectorization_;
            else
              large_block_2/=profile_.vectorization_;
            if(!store_shared){
              if(is_transposed)
                small_block_1/=profile_.vectorization_;
              else
                small_block_2/=profile_.vectorization_;
            }
          }
          else{
            if(is_transposed)
              large_block_2 /= profile_.vectorization_;
            else
              large_block_1/=profile_.vectorization_;
            if(!store_shared){
              if(is_transposed)
                small_block_2/=profile_.vectorization_;
              else
                small_block_1/=profile_.vectorization_;
            }
          }

        }



        void declare_rhs_global_ptr(detail::mapped_matrix const & mat, utils::kernel_generation_stream & stream, unsigned int ks_rhs,unsigned int ns_rhs,
                                    unsigned int nl_rhs, std::string const & offset_n,
                                    bool is_transposed) const {
          if(mat.is_row_major())
            for(unsigned int k = 0 ; k < ks_rhs ; ++k){
              stream << "__global " << aligned_scalartype_ << " * " << "ptr_rhs_" << k << " = " << mat.name() << " + " ;
              if(is_transposed)
                stream<< mat.offset(std::make_pair(utils::to_string(k) + " + " + offset_n + " +  get_group_id(1)*" + utils::to_string(nl_rhs),"0"));
              else
                stream << mat.offset(std::make_pair(utils::to_string(k),offset_n + " +  get_group_id(1)*" + utils::to_string(nl_rhs)));
              stream << ";" << std::endl;
            }
          else
            for(unsigned int n = 0 ; n < ns_rhs ; ++n){
              stream << "__global " << aligned_scalartype_ << " * " << "ptr_rhs_" << n << " = " << mat.name() << " +  " ;
              if(is_transposed)
                stream << mat.offset(std::make_pair(offset_n + " +  get_group_id(1)*" + utils::to_string(nl_rhs), utils::to_string(n)));
              else
                stream << mat.offset(std::make_pair("0",offset_n + " +  get_group_id(1)*" + utils::to_string(nl_rhs) + " + " + utils::to_string(n)));
              stream << ";" << std::endl;
            }
        }

        void update_rhs_global_ptr(detail::mapped_matrix const & mat, utils::kernel_generation_stream & stream, unsigned int ks, unsigned int ns_rhs, unsigned int ks_rhs
                        ,std::string const & size1_rhs,
                        std::string const & size2_rhs
                        ,bool is_transposed) const {
          if(mat.is_row_major() && !is_transposed)
            for(unsigned int k=0 ; k<ks ; ++k)
              stream << "ptr_rhs_" << k << " += " << ks_rhs << "*" << size2_rhs << " - " << ns_rhs << ";" << std::endl;
          else if(is_transposed && !mat.is_row_major())
            for(unsigned int n=0 ; n<ns_rhs ; ++n)
              stream << "ptr_rhs_" << n << " += " << ns_rhs << "*" << size1_rhs << " - " << ks_rhs << ";" << std::endl;
        }



        void declare_lhs_global_ptr(detail::mapped_matrix const & mat, utils::kernel_generation_stream & stream,
                         unsigned int ms_lhs, unsigned int ks_lhs,
                         unsigned int ml_lhs, std::string const & offset_m
                         ,bool is_transposed) const {
          if(mat.is_row_major()){
            for(unsigned int m=0; m<ms_lhs; ++m){
              std::string ptr_name = "ptr_lhs_" + utils::to_string(m);
              stream << "__global " << aligned_scalartype_ << " * " << ptr_name << " = " << mat.name() << " + ";
              if(is_transposed)
                stream << mat.offset(std::make_pair(utils::to_string(m),"get_group_id(0)*" + utils::to_string(ml_lhs) + "+" + offset_m ));
              else
                stream << mat.offset(std::make_pair("get_group_id(0)*" + utils::to_string(ml_lhs) + "+" + offset_m + "+" + utils::to_string(m),"0"));
              stream << ";" << std::endl;
            }
          }
          else{
            for(unsigned int k=0; k<ks_lhs; ++k){
              std::string ptr_name = "ptr_lhs_" + utils::to_string(k);
              stream << "__global " << aligned_scalartype_<< " * " << ptr_name << " = " << mat.name() << " + " ;
              if(is_transposed)
                stream << mat.offset(std::make_pair("0", utils::to_string(k) + "+" + "get_group_id(0)*" + utils::to_string(ml_lhs) + "+" + offset_m ));
              else
                stream << mat.offset(std::make_pair( "get_group_id(0)*" + utils::to_string(ml_lhs) + "+" + offset_m, utils::to_string(k)));
              stream << ";" << std::endl;
            }
          }
        }

        void update_lhs_global_ptr(detail::mapped_matrix const & mat, utils::kernel_generation_stream & stream, unsigned int ks, unsigned int ms_lhs, unsigned int ks_lhs
                        ,std::string const & size1_lhs,
                        std::string const & size2_lhs
                        ,bool is_transposed) const {
          if(is_transposed && mat.is_row_major())
            for(unsigned int m=0 ; m<ms_lhs ; ++m)
              stream << "ptr_lhs_" << m << " += " << ks << "*" << size2_lhs << " - " <<  ks_lhs << ";" << std::endl;
          else if(!is_transposed && !mat.is_row_major())
            for(unsigned int k=0 ; k<ks_lhs ; ++k)
              stream << "ptr_lhs_" << k << " += " << ks_lhs << "*" << size1_lhs << " - " << ms_lhs << ";" << std::endl;
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

        void fetch_to_local_mem(utils::kernel_generation_stream & stream,
                                       std::string const & lmem_name,
                                       std::size_t lmem_size2,
                                       std::string const & offset,
                                       unsigned int bound1,
                                       unsigned int bound2,
                                       detail::mapped_matrix const & mat) const {
          stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
          stream << "for(unsigned int i = get_local_id(0)" << " ; i < " << bound1 << "; i+= get_local_size(0)){" << std::endl;
          stream.inc_tab();
          stream << "for(unsigned int j = get_local_id(1)" << " ; j < " << bound2 << "; j+= get_local_size(1)){" << std::endl;
          stream.inc_tab();
          if(mat.is_row_major()){
            stream << aligned_scalartype_ << " val = " << mat.name() +  "[" + offset + " + j  + " + mat.size2()  + "*i]" << ";" << std::endl;
            stream << "__local " << mat.scalartype() << "* ptr = " << lmem_name << " + i*" << lmem_size2 << "+j*" << profile_.vectorization_<<";" <<std::endl;
            for(unsigned int a = 0 ; a < profile_.vectorization_ ; ++a){
              if(profile_.vectorization_>1)
                stream << "*ptr++ =  val.s" << a << ";" << std::endl;
              else
                stream << "*ptr++ =  val;" << std::endl;
            }
          }
          else{
            stream << aligned_scalartype_ << " val = " << mat.name() + "[" + offset + "+ j*" + mat.size1() + " + i]" << ";" << std::endl;
            stream << "__local " << mat.scalartype() << "* ptr = " << lmem_name << " + i*" << profile_.vectorization_ * lmem_size2 << "+ j;" <<std::endl;
            for(unsigned int a = 0 ; a < profile_.vectorization_ ; ++a){
              if(profile_.vectorization_>1)
                stream << "*ptr =  val.s" << a << ";" << std::endl;
              else
                stream << "*ptr =  val;" << std::endl;
              stream << "ptr += " << lmem_size2 << ";" << std::endl;
            }
          }

          stream.dec_tab();
          stream << "}" << std::endl;
          stream.dec_tab();
          stream << "}" << std::endl;
          stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

        }


      public:
        matrix_product(template_base::statements_type const & s, profile const & p) : template_base(s, profile_), profile_(p){

        }

        void core(std::size_t /*idx*/, utils::kernel_generation_stream& stream) const{

          bool use_lhs_shared = profile_.use_lhs_shared_;
          bool use_rhs_shared = profile_.use_rhs_shared_;
          unsigned int kl = profile_.kl_;
          unsigned int ks = profile_.ks_;
          unsigned int ml = profile_.ml_;
          unsigned int ms = profile_.ms_;
          unsigned int nl = profile_.nl_;
          unsigned int ns = profile_.ns_;
          unsigned int unroll = profile_.unroll_;

          detail::mapped_matrix const * assigned = static_cast<detail::mapped_matrix const *>(mapping_.at(0).at(std::make_pair(&statements_.front().second,detail::LHS_NODE_TYPE)).get());
          detail::mapped_matrix const * lhs = NULL;
          detail::mapped_matrix const * rhs = NULL;
          bool is_lhs_transposed = false;
          bool is_rhs_transposed = false;


          for(statements_type::const_iterator it = statements_.begin() ; it != statements_.end() ; ++it){
            scheduler::statement::container_type const & exprs = it->first.array();
            std::size_t i = std::distance(statements_.begin(), it);
            for(scheduler::statement::container_type::const_iterator iit = exprs.begin() ; iit != exprs.end() ; ++iit){
              if(iit->op.type==scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE){
                if(iit->lhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY){
                  is_lhs_transposed = true;
                  lhs = (detail::mapped_matrix const *)mapping_.at(i).at(std::make_pair(&exprs[iit->lhs.node_index],detail::LHS_NODE_TYPE)).get();
                }
                else{
                  is_lhs_transposed = false;
                  lhs = (detail::mapped_matrix const *)mapping_.at(i).at(std::make_pair(&(*iit), detail::LHS_NODE_TYPE)).get();
                }

                if(iit->rhs.type_family == scheduler::COMPOSITE_OPERATION_FAMILY){
                  is_rhs_transposed = true;
                  rhs = (detail::mapped_matrix const *)mapping_.at(i).at(std::make_pair(&exprs[iit->rhs.node_index], detail::LHS_NODE_TYPE)).get();
                }
                else{
                  is_rhs_transposed = false;
                  rhs = (detail::mapped_matrix const *)mapping_.at(i).at(std::make_pair(&(*iit),detail::RHS_NODE_TYPE)).get();
                }

              }
            }
          }


          if(profile_.vectorization_>1){
            if(assigned->is_row_major())
              assigned->bind_sizes("M", "N/"+utils::to_string(profile_.vectorization_));
            else
              assigned->bind_sizes("M/"+utils::to_string(profile_.vectorization_), "N");

            if(lhs->is_row_major())
              lhs->bind_sizes("M", "K/"+utils::to_string(profile_.vectorization_));
            else
              lhs->bind_sizes("M/"+utils::to_string(profile_.vectorization_), "K");

            if(rhs->is_row_major())
              rhs->bind_sizes("K", "N/"+utils::to_string(profile_.vectorization_));
            else
              rhs->bind_sizes("K/"+utils::to_string(profile_.vectorization_), "N");

          }
          else{
            assigned->bind_sizes("M", "N");
            lhs->bind_sizes("M", "K");
            rhs->bind_sizes("K", "N");
          }



          aligned_scalartype_ = assigned->scalartype();
          if(profile_.vectorization_ > 1)
            aligned_scalartype_+=utils::to_string(profile_.vectorization_);



          bool is_lhs_rowmajor = lhs->is_row_major();
          bool is_rhs_rowmajor = rhs->is_row_major();
          bool is_result_rowmajor = assigned->is_row_major();

          std::string lhs_value_scalartype;
          if(use_lhs_shared)
            lhs_value_scalartype = lhs->scalartype();
          else
            lhs_value_scalartype = aligned_scalartype_;

          std::string rhs_value_scalartype;
          if(use_rhs_shared)
            rhs_value_scalartype = rhs->scalartype();
          else
            rhs_value_scalartype = aligned_scalartype_;

          unsigned int ml_res = ml, nl_res = nl, ms_res = ms, ns_res = ns;
          unsigned int ml_lhs = ml, kl_lhs = kl, ms_lhs = ms, ks_lhs = ks;
          unsigned int kl_rhs = kl, nl_rhs = nl, ks_rhs = ks, ns_rhs = ns;

          transform_block(*assigned,false,false,ml_res,nl_res,ms_res,ns_res);
          transform_block(*lhs,is_lhs_transposed,use_lhs_shared,ml_lhs,kl_lhs,ms_lhs,ks_lhs);
          transform_block(*rhs,is_rhs_transposed,use_rhs_shared,kl_rhs,nl_rhs,ks_rhs,ns_rhs);



          std::string size1_lhs = lhs->size1();
          std::string size2_lhs = lhs->size2();

          std::string size1_rhs = rhs->size1();
          std::string size2_rhs = rhs->size2();

          std::string size1_res = assigned->size1();
          std::string size2_res = assigned->size2();

          unsigned int lhs_size1 = ml, lhs_size2 = kl;
          unsigned int rhs_size1 = kl, rhs_size2 = nl;
          if(is_lhs_transposed) std::swap(lhs_size1, lhs_size2);
          if(is_rhs_transposed) std::swap(rhs_size1, rhs_size2);


          std::size_t local_lhs_size1 = lhs_size1;
          std::size_t local_lhs_size2 = lhs_size2 + 1;

          std::size_t local_rhs_size1 = rhs_size1;
          std::size_t local_rhs_size2 = rhs_size2 + 1;
          //Declaration of results registers
          //                        std::string res_table_name(first_prod->repr() + "_res");
          for(unsigned int m=0; m< ms_res; ++m)
            for(unsigned int n=0; n < ns_res ; ++n)
              stream << aligned_scalartype_ << " " << "res" << m << n << " = (" << aligned_scalartype_ << ")(0) ;" << std::endl;

          //Declaration of local memories
          if(use_lhs_shared)
            stream << "__local " << lhs->scalartype() << " lhs_buf[" << local_lhs_size1*local_lhs_size2 << "]" << ";" << std::endl;
          if(use_rhs_shared)
            stream << "__local " << rhs->scalartype() << " rhs_buf[" << local_rhs_size1*local_rhs_size2 << "]" << ";" << std::endl;

          //Declaration of helpers
          std::string offset_m = helper_variable(stream,false,"unsigned int", "offset_m", "get_local_id(0)*" + utils::to_string(ms_lhs));
          std::string offset_n = helper_variable(stream,false,"unsigned int", "offset_n", "get_local_id(1)*" + utils::to_string(ns_rhs));
          std::string block_num = helper_variable(stream,true,"unsigned int", "block_num", (is_lhs_transposed?size1_lhs:size2_lhs) + '/' + utils::to_string(kl_lhs));

          //Declaration of pointers and/or offsets to result, rhs, lhs.
          stream << "__global " << aligned_scalartype_ << "* res_ptr = " <<  assigned->name() << " + " << assigned->offset(std::make_pair("get_global_id(0)*" + utils::to_string(ms_res), "get_global_id(1)*" + utils::to_string(ns_res))) << ";" << std::endl;

          if(use_rhs_shared){
            if(is_rhs_transposed) stream << "unsigned int offsetRHS = " << rhs->offset(std::make_pair(" get_group_id(1)*" + utils::to_string(nl_rhs),"0")) << ";" << std::endl;
            else stream << "unsigned int offsetRHS = " << rhs->offset(std::make_pair("0", " get_group_id(1)*" + utils::to_string(nl_rhs))) << ";" << std::endl;
          }
          else{
            if(is_rhs_transposed)
              declare_rhs_global_ptr(*rhs,stream,ns_rhs,ks_rhs,nl_rhs,offset_n,is_rhs_transposed);
            else
              declare_rhs_global_ptr(*rhs,stream,ks_rhs,ns_rhs,nl_rhs,offset_n,is_rhs_transposed);
          }

          if(use_lhs_shared){
            if(is_lhs_transposed) stream << "unsigned int offsetLHS = " << lhs->offset(std::make_pair("0", "get_group_id(0)*" + utils::to_string(ml_lhs))) << ";" << std::endl;
            else stream << "unsigned int offsetLHS = " << lhs->offset(std::make_pair("get_group_id(0)*" + utils::to_string(ml_lhs), "0")) << ";" << std::endl;
          }
          else{
            if(is_lhs_transposed)
              declare_lhs_global_ptr(*lhs, stream,ks_lhs,ms_lhs,ml_lhs,offset_m, is_lhs_transposed);
            else
              declare_lhs_global_ptr(*lhs, stream,ms_lhs,ks_lhs,ml_lhs,offset_m, is_lhs_transposed);
          }



          //Main loop
          stream << "for(unsigned int bl=0 ; bl<" << block_num << " ; ++bl){" << std::endl;
          stream.inc_tab();

          //Fetches to local memory if necessary and declares pointers to local memory
          if(use_lhs_shared){
            if(is_lhs_transposed)
              fetch_to_local_mem(stream,"lhs_buf",local_lhs_size2,"offsetLHS",kl_lhs,ml_lhs,*lhs);
            else
              fetch_to_local_mem(stream,"lhs_buf",local_lhs_size2,"offsetLHS",ml_lhs,kl_lhs,*lhs);
            unsigned int upper_bound = is_lhs_transposed?ks_lhs:ms_lhs;
            for(unsigned int m=0; m<upper_bound; ++m){
              stream << "__local " << lhs_value_scalartype << "* ptr_lhs_" << m << " = lhs_buf + " ;
              if(is_lhs_transposed)
                stream << m*local_lhs_size2 << " + " << offset_m ;
              else
                stream << "(" << offset_m << "+" << m << ")" << "*" << local_lhs_size2 ;
              stream << ";" << std::endl;
            }
          }

          if(use_rhs_shared){
            if(is_rhs_transposed)
              fetch_to_local_mem(stream,"rhs_buf", local_rhs_size2, "offsetRHS",nl_rhs,kl_rhs,*rhs);
            else
              fetch_to_local_mem(stream,"rhs_buf", local_rhs_size2, "offsetRHS",kl_rhs,nl_rhs,*rhs);
            unsigned int upper_bound = is_rhs_transposed?ns_rhs:ks_rhs;
            for(unsigned int k=0; k<upper_bound; ++k){
              stream << "__local " << rhs_value_scalartype << "* ptr_rhs_" << k << " = rhs_buf + " ;
              if(is_rhs_transposed)
                stream << "(" << offset_n << "+" << k << ")*" << local_rhs_size2;
              else
                stream << k*local_rhs_size2 << " + " << offset_n;
              stream << ";" << std::endl;
            }
          }


          if(unroll > 1)
            stream << "#pragma unroll " << unroll << std::endl;
          stream << " for(unsigned int bs=0 ; bs < " << kl/ks  << " ; ++bs){" << std::endl;
          stream.inc_tab();


          unsigned int upperbound_1_rhs = is_rhs_transposed?ns_rhs:ks_rhs;
          unsigned int upperbound_2_rhs = is_rhs_transposed?ks_rhs:ns_rhs;

          for(unsigned int k = 0 ; k < upperbound_1_rhs ; ++k){
            for(unsigned int n=0 ; n < upperbound_2_rhs ; ++n){
              stream << rhs_value_scalartype << " val_rhs_" << k << "_" << n << " = " ;
              if(use_rhs_shared ) stream << "* ptr_rhs_" << k << "++";
              else{
                if(is_rhs_rowmajor)
                  stream << "* ptr_rhs_" << k;
                else
                  stream  << "* ptr_rhs_" << n;
              }
              stream << ";";
              if( !use_rhs_shared ){
                  if(is_rhs_rowmajor)stream << "++" << "ptr_rhs_" << k << ";" ;
                  else stream << "++" << "ptr_rhs_" << n << ";" ;
              }
              stream << std::endl;
            }
          }



          unsigned int upperbound_1_lhs = is_lhs_transposed?ms_lhs:ks_lhs;
          unsigned int upperbound_2_lhs = is_lhs_transposed?ks_lhs:ms_lhs;
          for(unsigned int k = 0 ; k < upperbound_1_lhs ; ++k){
            for(unsigned int m=0 ; m < upperbound_2_lhs ; ++m){
              stream << lhs_value_scalartype << " " << "val_lhs_" << m << "_" << k << " = ";
              if(use_lhs_shared) stream <<  "* ptr_lhs_" << m << "++" ;
              else if(is_lhs_rowmajor)
                stream << "* ptr_lhs_" << m;
              else
                stream << "* ptr_lhs_" << k;
              stream << ";";
              if( !use_lhs_shared ){
                  if(is_lhs_rowmajor) stream << "++" << "ptr_lhs_" << m << ";" ;
                  else stream << "++" << "ptr_lhs_" << k << ";" ;
              }
              stream << std::endl;
            }
          }



          for(unsigned int k = 0 ; k < ks ; ++k){
            for(unsigned int n=0 ; n < ns_res ; ++n){
              for(unsigned int m=0 ; m < ms_res ; ++m){
                for(unsigned int a = 0; a<profile_.vectorization_; ++a){

                  int ind_lhs_1 = m;
                  int ind_lhs_2 = k;
                  int ind_s_lhs=a;

                  int ind_rhs_1=k;
                  int ind_rhs_2=n;
                  int ind_s_rhs=a;

                  bool is_vectorized_lhs = false;
                  bool is_vectorized_rhs = false;

                  if(is_result_rowmajor){
                    if(is_lhs_transposed)
                      std::swap(ind_lhs_1,ind_lhs_2);

                    if(!use_lhs_shared){
                      if(is_lhs_rowmajor){
                        ind_s_lhs = ind_lhs_2%profile_.vectorization_;
                        ind_lhs_2 /= profile_.vectorization_;
                      }
                      else{
                        ind_s_lhs = ind_lhs_1%profile_.vectorization_;
                        ind_lhs_1 /= profile_.vectorization_;
                      }
                    }
                  }
                  else{
                    if(use_lhs_shared){
                      ind_lhs_1 = ind_lhs_1*profile_.vectorization_+a;
                    }
                    else{
                      if((is_lhs_rowmajor && !is_lhs_transposed)
                         ||(!is_lhs_rowmajor && is_lhs_transposed)){
                        ind_lhs_1 = ind_lhs_1*profile_.vectorization_+a;
                        ind_s_lhs = ind_lhs_2%profile_.vectorization_;
                        ind_lhs_2 /= profile_.vectorization_;

                      }
                    }
                    if(is_lhs_transposed) std::swap(ind_lhs_1,ind_lhs_2);
                  }

                  if(is_result_rowmajor){
                    if(use_rhs_shared){
                      ind_rhs_2 = ind_rhs_2*profile_.vectorization_+a;
                    }
                    else{
                      if((!is_rhs_rowmajor && !is_rhs_transposed)
                         ||(is_rhs_rowmajor && is_rhs_transposed)){
                        ind_rhs_2 = ind_rhs_2*profile_.vectorization_+a;
                        ind_s_rhs = ind_rhs_1%profile_.vectorization_;
                        ind_rhs_1 = ind_rhs_1/profile_.vectorization_;
                      }
                      else if( (is_rhs_rowmajor && !is_rhs_transposed) ){
                        is_vectorized_rhs=true;
                      }
                    }
                    if(is_rhs_transposed) std::swap(ind_rhs_1,ind_rhs_2);
                  }
                  else{
                    if(is_rhs_transposed) std::swap(ind_rhs_1,ind_rhs_2);
                    if(!use_rhs_shared){
                      if(is_rhs_rowmajor){
                        ind_s_rhs = ind_rhs_2%profile_.vectorization_;
                        ind_rhs_2/=profile_.vectorization_;
                      }
                      else{
                        ind_s_rhs = ind_rhs_1%profile_.vectorization_;
                        ind_rhs_1/=profile_.vectorization_;
                      }
                    }
                  }

                  bool is_vectorized = is_vectorized_lhs || is_vectorized_rhs;

                  std::ostringstream res_oss;
                  std::ostringstream lhs_oss;
                  std::ostringstream rhs_oss;

                  res_oss << "res" << m << n ;
                  if(!is_vectorized && profile_.vectorization_>1) res_oss << ".s" << a;

                  lhs_oss << "val_lhs_" << ind_lhs_1 << "_" << ind_lhs_2;
                  if(!is_vectorized_lhs && !use_lhs_shared && profile_.vectorization_>1) lhs_oss << ".s" << ind_s_lhs;


                  rhs_oss << "val_rhs_" << ind_rhs_1 << "_" << ind_rhs_2;
                  if(!is_vectorized_rhs && !use_rhs_shared && profile_.vectorization_>1) rhs_oss << ".s" << ind_s_rhs;


                  stream << res_oss.str() << "+=" << lhs_oss.str() << "*" << rhs_oss.str() << ";" << std::endl;



                  if(is_vectorized)
                    break;
                }
              }
            }
          }


          if(use_rhs_shared){
            for(unsigned int k=0 ; k<ks ; ++k)
              if(!is_rhs_transposed) stream << "ptr_rhs_" << k << " += " << ks_rhs*local_rhs_size2 - ns_rhs << ";" << std::endl;
          }
          else{
            if(is_rhs_transposed)
              update_rhs_global_ptr(*rhs, stream,ks,ks_rhs,ns_rhs,size1_rhs,size2_rhs, is_rhs_transposed);
            else
              update_rhs_global_ptr(*rhs,stream,ks,ns_rhs,ks_rhs,size1_rhs,size2_rhs, is_rhs_transposed);
          }



          if(use_lhs_shared){
            for(unsigned int m=0 ; m<ks_lhs ; ++m)
              if(is_lhs_transposed) stream << "ptr_lhs_" << m << " += " << ks*local_lhs_size2 - ms_lhs << ";" << std::endl;
          }
          else{
            if(is_lhs_transposed)
              update_lhs_global_ptr(*lhs, stream,ks,ks_lhs,ms_lhs,size1_lhs,size2_lhs, is_lhs_transposed);
            else
              update_lhs_global_ptr(*lhs, stream,ks,ms_lhs,ks_lhs,size1_lhs,size2_lhs, is_lhs_transposed);
          }



          stream.dec_tab();
          stream << "}" << std::endl;

          if(use_lhs_shared){
            if(is_lhs_transposed){
              if(is_lhs_rowmajor)
                stream << "offsetLHS += " << kl_lhs << "*" << size2_lhs << ";" << std::endl;
              else
                stream << "offsetLHS += " << kl_lhs  << ";" << std::endl;
            }
            else{
              if(is_lhs_rowmajor)
                stream << "offsetLHS += " << kl_lhs << ";" << std::endl;
              else
                stream << "offsetLHS += " << kl_lhs << "*" << size1_lhs << ";" << std::endl;
            }

          }

          if(use_rhs_shared){
            if(is_rhs_transposed){
              if(is_rhs_rowmajor)
                stream << "offsetRHS += " << kl_rhs << ";" << std::endl;
              else
                stream << "offsetRHS += " << kl_rhs << "*" << size1_rhs << ";" << std::endl;
            }
            else{
              if(is_rhs_rowmajor)
                stream << "offsetRHS += " << kl_rhs << "*" << size2_rhs << ";" << std::endl;
              else
                stream << "offsetRHS += " << kl_rhs << ";" << std::endl;
            }
          }

          stream.dec_tab();
          stream << "}" << std::endl;

          if(assigned->is_row_major()){
            for(unsigned int m=0 ; m < ms_res ; ++m){
              for(unsigned int n=0 ; n < ns_res ; ++n){
                stream << "*res_ptr++=" << "res" << m << n << ";" << std::endl;
              }
              if(m<ms_res-1)  stream << "res_ptr+=" << size2_res << " - " << ns_res << ";" << std::endl;
            }
          }
          else{
            for(unsigned int n=0 ; n < ns_res ; ++n){
              for(unsigned int m=0 ; m < ms_res ; ++m){
                stream << "*res_ptr++=" << "res" << m << n << ";" << std::endl;
              }
              if(n<ns_res-1) stream << "res_ptr+=" << size1_res << " - " << ms_res << ";" << std::endl;
            }
          }





        }

      private:
        profile profile_;

        mutable std::string aligned_scalartype_;
    };

  }

}

#endif
