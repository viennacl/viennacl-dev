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

            virtual void print(std::ostream & s) const{
                s << "{vector_type, ms, ks, ns, ml, kl, nl, use_lhs_shared, use_rhs_shared, unroll} = {"
                  << vectorization_ << ","
                  << ms_ << ", "
                  << ks_ << ", "
                  << ns_ << ", "
                  << ml_ << ", "
                  << kl_ << ", "
                  << nl_ << ", "
                  << use_lhs_shared_ << ", " << use_rhs_shared_ << ", " << unroll_ << "}" ;
            }

            bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const{
                return ml_ < ms_
                        || kl_ < ks_
                        || nl_ < ns_
                        || (ms_ % vectorization_) > 0
                        || (ks_ % vectorization_) > 0
                        || (ns_ % vectorization_) > 0;
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
                        if(current_node->op.type==scheduler::OPERATION_UNARY_TRANS_TYPE)
                          k.arg(n_arg++, cl_uint(utils::call_on_matrix(current_node->lhs, utils::size1_fun())));
                        else
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

        enum access_flow{
          REGULAR,
          STRIDED
        };

        void transform_block(detail::mapped_matrix const & mat_infos, bool store_shared
                             , unsigned int & large_block_1, unsigned int & large_block_2
                             , unsigned int & small_block_1, unsigned int & small_block_2
                             , access_flow flow) const {
          if(flow==REGULAR){
            large_block_2/=profile_.vectorization_;
            if(!store_shared)
              small_block_2/=profile_.vectorization_;
          }
          else{
            large_block_1/=profile_.vectorization_;
            if(!store_shared)
              small_block_1/=profile_.vectorization_;
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

        void fetch_to_local_mem(utils::kernel_generation_stream & stream,
                                       std::string const & lmem_name,
                                       std::size_t lmem_size2,
                                       std::string const & global_ptr,
                                       unsigned int bound1,
                                       unsigned int bound2,
                                       detail::mapped_matrix const & mat,
                                       access_flow flow) const {
          stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
          stream << "for(unsigned int i = get_local_id(0)" << " ; i < " << bound1 << "; i+= get_local_size(0)){" << std::endl;
          stream.inc_tab();
          stream << "for(unsigned int j = get_local_id(1)" << " ; j < " << bound2 << "; j+= get_local_size(1)){" << std::endl;
          stream.inc_tab();
          if(flow==REGULAR){
           stream << aligned_scalartype_ << " val = *(" << global_ptr + " + j  + " + mat.size2()  + "*i" << ");" << std::endl;
           stream << "__local " << mat.scalartype() << "* ptr = " << lmem_name << " + i*" << lmem_size2 << "+j*" << profile_.vectorization_<<";" <<std::endl;
           for(unsigned int a = 0 ; a < profile_.vectorization_ ; ++a){
             if(profile_.vectorization_>1)
               stream << "*ptr++ =  val.s" << a << ";" << std::endl;
             else
               stream << "*ptr++ =  val;" << std::endl;
           }
          }
          else{
           stream << aligned_scalartype_ << " val = *(" << global_ptr + "+ j*" + mat.size1() + " + i" << ");" << std::endl;
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

          //////////////////
          /// INIT
          /// //////////////

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

//          std::string assigned_mat_size1 = "M", assigned_mat_size2 = "N";
//          std::string lhs_mat_size1 = "M", lhs_mat_size2 = "K";
//          std::string rhs_mat_size1 = "K", rhs_mat_size2 = "N";


//          if(profile_.vectorization_>1){
//            std::string StrV = utils::to_string(profile_.vectorization_) ;
//            std::string Mv = "M/"+StrV;
//            std::string Nv = "N/"+StrV;
//            std::string Kv = "K/"+StrV;

//            if(assigned->is_row_major())
//              assigned_mat_size2 = Nv;
//            else
//              assigned_mat_size1 = Mv;

//            if(lhs->is_row_major())
//              lhs_mat_size2 = Kv;
//            else
//              lhs_mat_size1 = Mv;

//            if(rhs->is_row_major())
//              rhs_mat_size2 = Nv;
//            else
//              rhs_mat_size1 = Kv;
//          }

//          assigned->bind_sizes(assigned_mat_size1, assigned_mat_size2);
//          lhs->bind_sizes(lhs_mat_size1, lhs_mat_size2);
//          rhs->bind_sizes(rhs_mat_size1, rhs_mat_size2);


          if(profile_.vectorization_>1){
            std::string StrV = "/"+utils::to_string(profile_.vectorization_) ;

            if(assigned->is_row_major())
              assigned->bind_sizes("M", "N"+StrV);
            else
              assigned->bind_sizes("M"+StrV, "N");

            if(lhs->is_row_major())
              if(is_lhs_transposed)
                 lhs->bind_sizes("M"+StrV, "K");
              else
                 lhs->bind_sizes("M", "K"+StrV);
            else
              if(is_lhs_transposed)
                lhs->bind_sizes("M", "K"+StrV);
              else
                lhs->bind_sizes("M"+StrV, "K");


            if(rhs->is_row_major())
              if(is_rhs_transposed)
                 rhs->bind_sizes("K"+StrV, "N");
              else
                 rhs->bind_sizes("K", "N"+StrV);
            else
              if(is_rhs_transposed)
                rhs->bind_sizes("K", "N"+StrV);
              else
                rhs->bind_sizes("K"+StrV, "N");


          }
          else{
            assigned->bind_sizes("M", "N");
            lhs->bind_sizes("M", "K");
            rhs->bind_sizes("K", "N");
          }



          aligned_scalartype_ = assigned->scalartype();
          if(profile_.vectorization_ > 1)
            aligned_scalartype_+=utils::to_string(profile_.vectorization_);


          access_flow result_access_flow;
          if(assigned->is_row_major())
            result_access_flow = REGULAR;
          else
            result_access_flow = STRIDED;

          access_flow lhs_access_flow;
          if((lhs->is_row_major() && !is_lhs_transposed)
             ||(!lhs->is_row_major() && is_lhs_transposed))
            lhs_access_flow = REGULAR;
          else
            lhs_access_flow = STRIDED;

          access_flow rhs_access_flow;
          if((rhs->is_row_major() && !is_rhs_transposed)
             ||(!rhs->is_row_major() && is_rhs_transposed))
            rhs_access_flow = REGULAR;
          else
            rhs_access_flow = STRIDED;


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

          transform_block(*assigned,false,ml_res,nl_res,ms_res,ns_res,result_access_flow);
          transform_block(*lhs,use_lhs_shared,ml_lhs,kl_lhs,ms_lhs,ks_lhs,lhs_access_flow);
          transform_block(*rhs,use_rhs_shared,kl_rhs,nl_rhs,ks_rhs,ns_rhs,rhs_access_flow);

          //////////////////
          /// DECLARATIONS
          /// //////////////


          std::size_t local_lhs_size1 = ml ;
          std::size_t local_lhs_size2 = kl + 1;

          std::size_t local_rhs_size1 = kl;
          std::size_t local_rhs_size2 = nl + 1;

          ///Result Values
          for(unsigned int m=0; m< ms_res; ++m)
            for(unsigned int n=0; n < ns_res ; ++n)
              stream << aligned_scalartype_ << " " << "res" << m << n << " = (" << aligned_scalartype_ << ")(0) ;" << std::endl;

          ///Local memory
          if(use_lhs_shared)
            stream << "__local " << lhs->scalartype() << " lhs_buf[" << local_lhs_size1*local_lhs_size2 << "]" << ";" << std::endl;
          if(use_rhs_shared)
            stream << "__local " << rhs->scalartype() << " rhs_buf[" << local_rhs_size1*local_rhs_size2 << "]" << ";" << std::endl;

          ///Pointer to result
          stream << "__global " << aligned_scalartype_ << "* res_ptr = " <<  assigned->name() << " + " << assigned->offset(std::make_pair("get_global_id(0)*" + utils::to_string(ms_res), "get_global_id(1)*" + utils::to_string(ns_res))) << ";" << std::endl;


          ///LHS - Local Memory Offset
          if(use_lhs_shared){
            std::string i = "get_group_id(0)*" + utils::to_string(ml_lhs);
            stream << "__global " << aligned_scalartype_ << "* global_lhs_ptr = " << lhs->name() << " + ";
            if(lhs_access_flow==REGULAR)
              stream << "(" << i << ")" << "*" << lhs->size2();
            else
              stream << i;
            stream << ";" << std::endl;
          }

          ///LHS - Global Memory pointer
          else{
            if(lhs_access_flow==REGULAR)
              for(unsigned int m=0; m<ms_lhs; ++m)
                stream << "__global " << aligned_scalartype_ << " * " << "lhs_ptr_" << m << " = " << lhs->name() << " + "
                       << lhs->size2() << "* ("
                       << "get_group_id(0)*" << ml_lhs << "+" << "get_local_id(0)*" << ms_lhs << "+" << m
                       << " );" << std::endl;
            else
              for(unsigned int k=0; k<ks_lhs; ++k)
                stream << "__global " << aligned_scalartype_<< " * " << "lhs_ptr_" << k << " = " << lhs->name() << " + "
                       << "(" << lhs->size1() << ")*" << k
                       << "+ " << "get_group_id(0)*" << ml_lhs << "+" << "get_local_id(0)*" << ms_lhs << ";" << std::endl;
          }

          ///RHS - Local Memory Offset
          if(use_rhs_shared){
            std::string j = "get_group_id(1)*" + utils::to_string(nl_rhs);
            stream << "unsigned int offsetRHS = " ;
            if(rhs_access_flow==REGULAR)
              stream << j;
            else
              stream << "(" << j << ")" << "*" << rhs->size1();
            stream << ";" << std::endl;
          }

          ///RHS - Global Memory Pointer
          else{
            if(rhs_access_flow==REGULAR)
              for(unsigned int k = 0 ; k < ks_rhs ; ++k)
                stream << "__global " << aligned_scalartype_ << " * " << "rhs_ptr_" << k << " = " << rhs->name() << " + "
                       << "(" << k << ")" << "*" << rhs->size2()
                       << " + " << "get_local_id(1)*" << ns_rhs << " + get_group_id(1)*" << nl_rhs
                       << ";" << std::endl;
            else
              for(unsigned int n = 0 ; n < ns_rhs ; ++n)
                stream << "__global " << aligned_scalartype_ << " * " << "rhs_ptr_" << n << " = " << rhs->name() << " +  "
                       << "(" << "get_local_id(1)*" << ns_rhs << " + get_group_id(1)*" << nl_rhs << " + " << n << ")" << "*" << rhs->size1()
                       << ";" << std::endl;
          }


          ///Large Work-group Wise loop
          std::string block_num = helper_variable(stream,false,"unsigned int", "block_num", "K/" + utils::to_string(kl));
          stream << "for(unsigned int bl=0 ; bl<" << block_num << " ; ++bl){" << std::endl;
          stream.inc_tab();

          ///Update LHS Local Memory and pointers (if necessary)
          if(use_lhs_shared){
            fetch_to_local_mem(stream,"lhs_buf",local_lhs_size2,"global_lhs_ptr",ml_lhs,kl_lhs,*lhs,lhs_access_flow);
            for(unsigned int m=0; m<ms_lhs; ++m)
              stream << "__local " << lhs_value_scalartype << "* lhs_ptr_" << m << " = lhs_buf + "
                     << "(" << "get_local_id(0)*" << ms_lhs << "+" << m << ")" << "*" << local_lhs_size2
                     << ";" << std::endl;
          }

          ///Update RHS Local Memory and pointers (if necessary)
          if(use_rhs_shared){
            fetch_to_local_mem(stream,"rhs_buf", local_rhs_size2, "offsetRHS",kl_rhs,nl_rhs,*rhs,rhs_access_flow);
            for(unsigned int k=0; k<ks_rhs; ++k)
              stream << "__local " << rhs_value_scalartype << "* rhs_ptr_" << k << " = rhs_buf + "
                     << k*local_rhs_size2 << " + " << "get_local_id(1)*" << ns_rhs
                     << ";" << std::endl;
          }


          ///Small work-item wise loop
          if(unroll > 1)
            stream << "#pragma unroll " << unroll << std::endl;
          stream << " for(unsigned int bs=0 ; bs < " << kl/ks  << " ; ++bs){" << std::endl;
          stream.inc_tab();


          for(unsigned int k = 0 ; k < ks_rhs ; ++k){
            for(unsigned int n=0 ; n < ns_rhs ; ++n){
              stream << rhs_value_scalartype << " val_rhs_" << k << "_" << n << " = " ;
              if(use_rhs_shared )
                  stream << "* rhs_ptr_" << k << "++";
              else{
                if(rhs_access_flow==REGULAR)
                  stream << "* rhs_ptr_" << k << "++";
                else
                   stream  << "* rhs_ptr_" << n << "++";
              }
              stream << ";";
              stream << std::endl;
            }
          }


          for(unsigned int k = 0 ; k < ks_lhs ; ++k){
            for(unsigned int m=0 ; m < ms_lhs ; ++m){
              stream << lhs_value_scalartype << " " << "val_lhs_" << m << "_" << k << " = ";
              if(use_lhs_shared)
                stream <<  "* lhs_ptr_" << m << "++" ;
              else if(lhs_access_flow==REGULAR)
                stream << "* lhs_ptr_" << m << "++";
              else
                stream << "* lhs_ptr_" << k << "++";
              stream << ";";
              stream << std::endl;
            }
          }



          for(unsigned int k = 0 ; k < ks ; ++k){
            for(unsigned int n=0 ; n < ns_res ; ++n){
              for(unsigned int m=0 ; m < ms_res ; ++m){
                for(unsigned int a = 0; a<profile_.vectorization_; ++a){

                  int ind_lhs_1 = m;
                  int ind_lhs_2 = k;
                  int ind_s_lhs = a;

                  int ind_rhs_1=k;
                  int ind_rhs_2=n;
                  int ind_s_rhs=a;

                  if(result_access_flow==REGULAR){
                    if(!use_lhs_shared){
                      if(lhs_access_flow==REGULAR){
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
                      if(lhs_access_flow==REGULAR){
                        ind_lhs_1 = ind_lhs_1*profile_.vectorization_+a;
                        ind_s_lhs = ind_lhs_2%profile_.vectorization_;
                        ind_lhs_2 /= profile_.vectorization_;
                      }
                    }
                  }

                  if(result_access_flow==REGULAR){
                    if(use_rhs_shared){
                      ind_rhs_2 = ind_rhs_2*profile_.vectorization_+a;
                    }
                    else{
                      if(rhs_access_flow==STRIDED){
                        ind_rhs_2 = ind_rhs_2*profile_.vectorization_+a;
                        ind_s_rhs = ind_rhs_1%profile_.vectorization_;
                        ind_rhs_1 = ind_rhs_1/profile_.vectorization_;
                      }
                      else{
                      }
                    }
                  }
                  else{
                    if(!use_rhs_shared){
                      if(rhs_access_flow==REGULAR){
                        ind_s_rhs = ind_rhs_2%profile_.vectorization_;
                        ind_rhs_2/=profile_.vectorization_;
                      }
                      else{
                        ind_s_rhs = ind_rhs_1%profile_.vectorization_;
                        ind_rhs_1/=profile_.vectorization_;
                      }
                    }
                  }

                  std::ostringstream res_oss;
                  std::ostringstream lhs_oss;
                  std::ostringstream rhs_oss;

                  res_oss << "res" << m << n ;
                  if(profile_.vectorization_>1) res_oss << ".s" << a;

                  lhs_oss << "val_lhs_" << ind_lhs_1 << "_" << ind_lhs_2;
                  if(!use_lhs_shared && profile_.vectorization_>1) lhs_oss << ".s" << ind_s_lhs;


                  rhs_oss << "val_rhs_" << ind_rhs_1 << "_" << ind_rhs_2;
                  if(!use_rhs_shared && profile_.vectorization_>1) rhs_oss << ".s" << ind_s_rhs;


                  stream << res_oss.str() << "+=" << lhs_oss.str() << "*" << rhs_oss.str() << ";" << std::endl;
                }
              }
            }
          }


          if(use_rhs_shared){
            for(unsigned int k=0 ; k<ks ; ++k)
              stream << "rhs_ptr_" << k << " += " << ks_rhs*local_rhs_size2 - ns_rhs << ";" << std::endl;
          }
          else{
            if(rhs_access_flow==REGULAR)
              for(unsigned int k=0 ; k<ks ; ++k)
                stream << "rhs_ptr_" << k << " += " << ks_rhs << "*" << rhs->size2() << " - " << ns_rhs << ";" << std::endl;
          }

          if(!use_lhs_shared){
            if(lhs_access_flow==STRIDED)
              for(unsigned int k=0 ; k<ks_lhs ; ++k)
                stream << "lhs_ptr_" << k << " += " << ks_lhs << "*" << lhs->size1() << " - " << ms_lhs << ";" << std::endl;
          }



          stream.dec_tab();
          stream << "}" << std::endl;

          if(use_lhs_shared){
              if(lhs_access_flow==REGULAR)
                stream << "global_lhs_ptr += " << kl_lhs << ";" << std::endl;
              else
                stream << "global_lhs_ptr += " << kl_lhs << "*" << lhs->size1() << ";" << std::endl;
          }

          if(use_rhs_shared){
            if(rhs_access_flow==REGULAR)
                stream << "offsetRHS += " << kl_rhs << "*" << rhs->size2() << ";" << std::endl;
            else
                stream << "offsetRHS += " << kl_rhs << ";" << std::endl;
          }

          stream.dec_tab();
          stream << "}" << std::endl;

          if(result_access_flow==REGULAR){
            for(unsigned int m=0 ; m < ms_res ; ++m){
              for(unsigned int n=0 ; n < ns_res ; ++n){
                stream << "*res_ptr++=" << "res" << m << n << ";" << std::endl;
              }
              if(m<ms_res-1)  stream << "res_ptr+=" << assigned->size2() << " - " << ns_res << ";" << std::endl;
            }
          }
          else{
            for(unsigned int n=0 ; n < ns_res ; ++n){
              for(unsigned int m=0 ; m < ms_res ; ++m){
                stream << "*res_ptr++=" << "res" << m << n << ";" << std::endl;
              }
              if(n<ns_res-1) stream << "res_ptr+=" << assigned->size1() << " - " << ms_res << ";" << std::endl;
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
