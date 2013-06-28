#ifndef VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_VECTOR_REDUCTION_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_TEMPLATES_VECTOR_REDUCTION_HPP

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


/** @file viennacl/generator/templates/vector_reduction.hpp
 *
 * Kernel template for the vector reduction operation
*/

#include "viennacl/generator/templates/generator_base.hpp"
#include "viennacl/generator/templates/profile_base.hpp"

#include "viennacl/generator/symbolic_types.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{


        /** @brief Profile template for a vector reduction kernel
         *
         *  Implementation based on blocking.
         *  Each work group processes a block of M rows, by iterating horizontally over M*K blocks
         *  Uses persistents threads defined by NUM_GROUPS_0.
         */
        class vector_reduction_profile : public profile_base{
          public:

            /** @brief The user constructor */
            vector_reduction_profile(unsigned int m, unsigned int k, unsigned int num_groups_0) : profile_base(1), m_(m), k_(k), num_groups_0_(num_groups_0){ }

            /** @brief Returns M */
            unsigned int m() const { return m_; }

            /** @brief Returns K */
            unsigned int k() const { return k_; }

            /** @brief Returns NUM_GROUPS_0 */
            unsigned int num_groups_0() const { return num_groups_0_; }


            /** @brief Return the group sizes used by this kernel */
            std::pair<size_t,size_t> local_work_size() const{
              return std::make_pair(m_,k_);
            }

            /** @brief Configure the NDRange of a given kernel for this profile */
            void config_nd_range(viennacl::ocl::kernel & k, symbolic_expression_tree_base* p) const {
              k.local_work_size(0,m_);
              k.local_work_size(1,k_);
              k.global_work_size(0,m_*num_groups_0_);
              k.global_work_size(1,k_);
            }

            /** @brief Returns the representation string of this profile */
            std::string repr() const{
              std::ostringstream oss;
              oss << "V" << vectorization_;
              oss << "M" << m_;
              oss << "K" << k_;
              oss << "NG0" << num_groups_0_;
              return oss.str();
            }

            /** @brief returns whether or not the profile leads to undefined behavior on particular device
             *  @param dev the given device*/
            bool is_invalid(viennacl::ocl::device const & dev, size_t scalartype_size) const {
              return profile_base::invalid_base(dev,m_*(k_+1)*scalartype_size)
                  || vectorization_ > m_
                  || vectorization_ > k_;
            }


          private:
            unsigned int m_;
            unsigned int k_;
            unsigned int num_groups_0_;
        };


        class vector_reduction_generator : public generator_base{
          private:
            void generate_body_impl(unsigned int i, utils::kernel_generation_stream& kss){
              vector_reduction_profile const * casted_prof = static_cast<vector_reduction_profile const *>(prof_);

              std::list<symbolic_matrix_base *>  matrices;
              std::list<vector_reduction_base *>  prods;
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::const_iterator it=expressions_.begin() ; it!=expressions_.end() ; ++it){
                extract_as(*it, matrices, utils::is_type<symbolic_matrix_base>());
                extract_as(*it, prods, utils::is_type<vector_reduction_base>());
              }
              symbolic_matrix_base* first_matrix = *matrices.begin();
              vector_reduction_base * first_prod = *prods.begin();
              std::string scalartype = first_matrix->scalartype();
              std::string internal_size1 = first_matrix->internal_size1();
              std::string internal_size2 = first_matrix->internal_size2();

              unsigned int m = casted_prof->m();
              unsigned int k = casted_prof->k();

              bool is_lhs_transposed = is_transposed(&first_prod->lhs());
              //            bool is_lhs_row_major = first_matrix->is_rowmajor();
              std::map<vector_reduction_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > > reductions;
              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin(); it!=expressions_.end() ; ++it){
                unsigned int id = std::distance(expressions_.begin(),it);
                symbolic_vector_base* assigned = dynamic_cast<symbolic_vector_base*>(&(*it)->lhs());
                symbolic_local_memory<2> lmem("block_"+utils::to_string(id),m,k+1,scalartype);
                std::list<vector_reduction_base *>  prods;
                extract_as(*it, prods, utils::is_type<vector_reduction_base>());
                assert(prods.size()==1 && "More than one product involved in the expression");
                reductions.insert(std::make_pair(*prods.begin(),std::make_pair("reduction_"+utils::to_string(id),std::make_pair(lmem,assigned))));
              }
              kss << "unsigned int lid0 = get_local_id(0);" << std::endl;
              kss << "unsigned int lid1 = get_local_id(1);" << std::endl;
              for(std::map<vector_reduction_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > >::iterator it = reductions.begin() ; it != reductions.end() ; ++it){
                kss << it->second.second.first.declare() << ";" << std::endl;
              }
              if(is_lhs_transposed)
                kss << "for(unsigned int r = get_global_id(0) ; r < " << internal_size2 << " ; r += get_global_size(0)){" << std::endl;
              else
                kss << "for(unsigned int r = get_global_id(0) ; r < " << internal_size1 << " ; r += get_global_size(0)){" << std::endl;
              kss.inc_tab();

              for(std::map<vector_reduction_base*, std::pair<std::string,std::pair<symbolic_local_memory<2>, symbolic_vector_base*> > >::iterator it = reductions.begin() ; it != reductions.end() ; ++it){
                vector_reduction_base* prod = it->first;
                binary_operator const & op_reduce = prod->op_reduce();
                std::string const & sum_name = it->second.first;
                symbolic_local_memory<2> const & lmem = it->second.second.first;
                symbolic_vector_base * assigned = it->second.second.second;
                kss << scalartype << " " << sum_name << " = 0;" << std::endl;
                if(is_lhs_transposed)
                  kss << "for(unsigned int c = get_local_id(1) ; c < " << internal_size1 << " ; c += get_local_size(1)){" << std::endl;
                else
                  kss << "for(unsigned int c = get_local_id(1) ; c < " << internal_size2 << " ; c += get_local_size(1)){" << std::endl;
                kss.inc_tab();
                prod->lhs().access_index(0,"r","c");
                prod->rhs().access_index(0,"c","0");
                prod->fetch(0,kss);
                kss << sum_name << " = " << op_reduce.generate(sum_name,prod->symbolic_binary_expression_tree_infos_base::generate(0)) << ";" << std::endl;
                kss.dec_tab();
                kss << "}" << std::endl;
                kss << lmem.access("lid0", "lid1")<< " = " << sum_name << ";" << std::endl;
                for(unsigned int stride = k/2 ; stride>0 ; stride /=2){
                  kss << "barrier(CLK_LOCAL_MEM_FENCE); ";
                  kss <<  "if(lid1 < " << utils::to_string(stride) << ")" << lmem.access("lid0", "lid1") <<  " = " <<   op_reduce.generate(lmem.access("lid0", "lid1"),lmem.access("lid0", "lid1+" + utils::to_string(stride))) << ";" << std::endl;
                }
                it->first->access_name(lmem.access("lid0","0"));
                assigned->access_index(0,"r","0");
                kss << "if(lid1==0)" << expressions_.front()->generate(0) << ";" << std::endl;
              }


              kss.dec_tab();
              kss << "}" << std::endl;

              for(std::list<tools::shared_ptr<symbolic_binary_expression_tree_infos_base> >::iterator it = expressions_.begin() ; it != expressions_.end() ; ++it){
                (*it)->clear_private_value(0);
              }
            }

        };

      }

  }

}

#endif
