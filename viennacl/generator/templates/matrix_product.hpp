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
				lmem_used += (ml_ + 1) * (cache_width_ + 1) * scalartype_size;
				if(use_rhs_shared_)
				lmem_used += (cache_width_ + 1) * (nl_ + 1) * scalartype_size;
				return lmem_used;
			}

			virtual void print(std::ostream & s) const{
				s << "{vector_type, local_size1, cache_width, local_size2, ms, ks, ns, use_lhs_shared, use_rhs_shared} = {"
				<< simd_width_ << ","
				<< local_size1_ << ", "
				<< cache_width_ << ", "
				<< local_size2_ << ", "
				<< ms_ << ", "
				<< ks_ << ", "
				<< ns_ << ", "
				<< use_lhs_shared_ << ", " << use_rhs_shared_ << "}" ;
			}


			bool invalid_impl(viennacl::ocl::device const & /*dev*/, size_t /*scalartype_size*/) const{
				static const unsigned int alignment = 128;
				return alignment % ml_ > 0
				|| alignment % nl_ > 0
				|| (ms_ % simd_width_) > 0
				|| (ns_ % simd_width_) > 0;
			}

		public:
			/** @brief The user constructor */
			matrix_product(unsigned int vectorization
			, std::size_t local_size1, std::size_t cache_width, std::size_t local_size2
			, unsigned int ms, unsigned int ks, unsigned int ns
			, bool use_lhs_shared, bool use_rhs_shared) : profile_base(vectorization,local_size1, local_size2,1){
				local_size1_ = local_size1;
				local_size2_ = local_size2;
				cache_width_=cache_width;
				ml_= ms*local_size1;
				nl_=ns*local_size2;
				ms_ = ms;
				ks_=ks;
				ns_=ns;
				use_lhs_shared_ = use_lhs_shared;
				use_rhs_shared_ = use_rhs_shared;
			}

			static std::string csv_format() {
				return "Vec,LSize1,CacheWidth,LSize2,mS,kS,nS,NumGroups";
			}

			std::string csv_representation() const{
				std::ostringstream oss;
				oss << simd_width_
				<< "," << local_size1_
				<< "," << cache_width_
				<< "," << local_size2_
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

			void fetch_element_to_local_mem(utils::kernel_generation_stream & stream,
			std::string const & lmem_name,
			std::size_t lmem_size2,
			std::string const & global_ptr,
			mapped_matrix const & mat,
			std::string const & i,
			std::string const & j) const {

				if(mat.interpret_as_transposed()){
					stream << "val = *(" << global_ptr << " + " << j << " + " << mat.ld()  << "*" << i << ");" << std::endl;
					for(unsigned int a = 0 ; a < simd_width_ ; ++a)
					if(simd_width_>1)
					stream << lmem_name << "[" << i << "*" << lmem_size2 << " + " << j << "*" << simd_width_<<" + " << a << "] = val.s" << a << ";" <<std::endl;
					else
					stream << lmem_name << "[" << i << "*" << lmem_size2 << " + " << j << "*" << simd_width_ << "] = val" << ";" <<std::endl;
				}
				else{
					stream << "val = *(" << global_ptr << "+ " << j << "*" << mat.ld() << " + " << i << ");" << std::endl;
					for(unsigned int a = 0 ; a < simd_width_ ; ++a)
					if(simd_width_>1)
					stream << lmem_name << "[" << i << "*" << simd_width_*lmem_size2 << " + " << j << " + " << a*lmem_size2 << "] = val.s" << a << ";" <<std::endl;
					else
					stream << lmem_name << "[" << i << "*" << simd_width_*lmem_size2 << " + " << j << "] = val" << ";" <<std::endl;
				}
			}
			void fetch_to_local_mem(utils::kernel_generation_stream & stream,
									std::string const & lmem_name,
									std::size_t lmem_size2,
									std::string const & global_ptr,
									unsigned int bound1,
									unsigned int bound2,
									mapped_matrix const & mat) const 
			{
				stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
				stream << "{" << std::endl;
				stream << mat.simd_scalartype() << " val;" << std::endl;
				//Can unroll
				if(bound2%local_size2_==0 && bound1%local_size1_==0){
					for(unsigned int j = 0 ; j < bound2 ; j+=local_size2_){
						for(unsigned int i = 0 ; i < bound1 ; i+=local_size1_){
							std::string indi = "(get_local_id(0) + " + utils::to_string(i)+")";
							std::string indj = "(get_local_id(1) + " + utils::to_string(j)+")";
							fetch_element_to_local_mem(stream,lmem_name,lmem_size2,global_ptr,mat,indi,indj);
						}
					}
				}
				else{
					stream << "for(unsigned int j = get_local_id(1)" << " ; j < " << bound2 << "; j+= " << local_size2_ << "){" << std::endl;
					stream.inc_tab();
					stream << "for(unsigned int i = get_local_id(0)" << " ; i < " << bound1 << "; i+= " << local_size1_ << "){" << std::endl;
					stream.inc_tab();
					fetch_element_to_local_mem(stream,lmem_name,lmem_size2,global_ptr,mat,"i","j");
					stream.dec_tab();
					stream << "}" << std::endl;
					stream.dec_tab();
					stream << "}" << std::endl;

				}
				stream << "}" << std::endl;
				stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

			}

			void core(std::size_t /*kernel_id*/, utils::kernel_generation_stream& stream, expression_descriptor descriptor, statements_type const & statements, std::vector<mapping_type> const & mapping) const {

				if(use_lhs_shared_ || use_rhs_shared_)
					throw std::string("Not implemented");
				
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
				unsigned int ml_lhs = ml_, cache_width_lhs = cache_width_, ms_lhs = ms_, ks_lhs = ks_;
				unsigned int cache_width_rhs = cache_width_, nl_rhs = nl_, ks_rhs = ks_, ns_rhs = ns_;

				transform_block(*lhs,use_lhs_shared_,ml_lhs,cache_width_lhs,ms_lhs,ks_lhs);
				transform_block(*rhs,use_rhs_shared_,cache_width_rhs,nl_rhs,ks_rhs,ns_rhs);

				//////////////////
				/// DECLARATIONS
				/// //////////////


				std::size_t local_lhs_size1 = ml_ ;
				std::size_t local_lhs_size2 = cache_width_ + 1;

				std::size_t local_rhs_size1 = cache_width_;
				std::size_t local_rhs_size2 = nl_ + 1;

				

				///Result Values
				stream << assigned->scalartype() << " " << "res[" << ms_ << "][" << ns_ <<"]  = {(" << assigned->scalartype() << ")0};" << std::endl;
				stream << lhs->simd_scalartype() << " " << "val_lhs[" << ms_lhs << "];" << std::endl;
				stream << rhs->simd_scalartype() << " " << "val_rhs[" << ns_rhs <<"];" << std::endl;


				{
				std::string offset_i;
				std::string offset_j;
				
				if(strided){
					offset_i = "get_group_id(0)*"+utils::to_string(ml_) + "+ get_local_id(0)*"+utils::to_string(simd_width_);
					offset_j = "get_group_id(1)*"+utils::to_string(nl_) + "+ get_local_id(1)*"+utils::to_string(simd_width_);
				}
				stream << "uint2 offset = (uint2)(" << offset_i << "," << offset_j << ");" << std::endl;
				}
				
				if(lhs->interpret_as_transposed()){
					for(unsigned int m=0; m<ms_lhs; ++m)
					stream << "__global " << lhs->simd_scalartype() << "* " << "lhs_ptr_" << m << " = " << lhs->name() << " + "
					<< lhs->ld() << "* ("
					<< "get_group_id(0)*" << ml_lhs << "+" << "get_local_id(0)*" << ms_lhs << "+" << m
					<< " );" << std::endl;
				}
				else{
					stream << "__global " << lhs->simd_scalartype() << "* " << "lhs_ptr" << " = " 
						   << lhs->name() << " + offset.x/" << simd_width_  << ";" << std::endl;
				}
					
				if(rhs->interpret_as_transposed())
					stream << "__global " << rhs->simd_scalartype() << "* " << "rhs_ptr"
								<< " = " << rhs->name() << " + offset.y/" << simd_width_  << ";" << std::endl;
				else
					for(unsigned int n = 0 ; n < ns_rhs ; ++n)
							stream << "__global " << rhs->simd_scalartype() << "* " << "rhs_ptr_" << n << " = " << rhs->name() << " +  "
							<< "(" << "get_local_id(1)*" << ns_rhs << " + get_group_id(1)*" << nl_rhs << " + " << n << ")" << "*" << rhs->ld()
							<< ";" << std::endl;
				


				if(cache_width_>1)
				{
					stream << "for(unsigned int block_k=0 ; block_k< K ; block_k+=" << cache_width_ << "){" << std::endl;
					stream.inc_tab();
					stream << "#pragma unroll " << cache_width_ << std::endl;
					stream << "for(unsigned int k = 0 ; k < " << cache_width_ << "; ++k){" << std::endl;
					stream.inc_tab();
				}
				else
				{
					stream << "for(unsigned int k=0 ; k< K ; ++k){" << std::endl;
					stream.inc_tab();
				}
				///Fetch LHS to registers
				for(unsigned int m=0 ; m < ms_lhs ; ++m){
					if(lhs->interpret_as_transposed())
						stream << "val_lhs[" << m << "] = * lhs_ptr_" << m << "++;" << std::endl;
					else if(strided)
						stream << "val_lhs[" << m << "] = lhs_ptr[" << m*local_size_1_ << "];" << std::endl;
					else
						stream << "val_lhs[" << m << "] = lhs_ptr[" << m << "];" << std::endl;
				}

				///Fetch RHS to registers
				for(unsigned int n=0 ; n < ns_rhs ; ++n){
					if(!rhs->interpret_as_transposed())
						stream  << "val_rhs[" << n << "] = * rhs_ptr_" << n << "++;" << std::endl;
					else if(strided)
						stream << "val_rhs[" << n << "] = rhs_ptr[" << n*local_size_2_ << "];" << std::endl;
					else
						stream << "val_rhs[" << n << "] = rhs_ptr[" << n << "];" << std::endl;
				}
					
				for(unsigned int m=0 ; m < ms_ ; ++m){
					for(unsigned int n=0 ; n < ns_ ; ++n){
						std::string res_str, lhs_str, rhs_str;
						res_str = "res[" + utils::to_string(m) + "][" + utils::to_string(n) + "]";
						if(!lhs->interpret_as_transposed()){
							lhs_str = "val_lhs["+utils::to_string(m/simd_width_)+"]";
							if(simd_width_>1) lhs_str += ".s"+utils::to_string(m%simd_width_);
						}
						if(rhs->interpret_as_transposed()){
							rhs_str = "val_rhs["+utils::to_string(n/simd_width_)+"]";
							if(simd_width_>1) rhs_str += ".s"+utils::to_string(n%simd_width_);
						}
						stream << res_str << "=" << "mad(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
					}
				}

				if(!lhs->interpret_as_transposed())
					stream << "lhs_ptr += " << lhs->ld() << ";" << std::endl;

				if(rhs->interpret_as_transposed())
					stream << "rhs_ptr += " << rhs->ld() << ";" << std::endl;

				if(cache_width_>1)
				{
					stream.dec_tab();
					stream << "}" << std::endl;
					stream.dec_tab();
					stream << "}" << std::endl;
				}
				else
				{
					stream.dec_tab();
					stream << "}" << std::endl;
				}

				for(unsigned int m=0 ; m < ms_res ; ++m){
					for(unsigned int n=0 ; n < ns_res ; ++n){
						std::string i,j;
						if(strided){
							i = "offset.x +" + utils::to_string((m/simd_width_)*(local_size_1_*simd_width_) + m%simd_width_);
							j = "offset.y +" + utils::to_string((n/simd_width_)*(local_size_2_*simd_width_) + n%simd_width_);
						}
						else{
							i = "get_global_id(0)*" + utils::to_string(ms_res) + "+" + utils::to_string(m);
							j = "get_global_id(1)*" + utils::to_string(ns_res) + "+" + utils::to_string(n);
						}
						if(assigned->interpret_as_transposed())
						std::swap(i,j);
						prod->access_name("res["+utils::to_string(m)+"]["+utils::to_string(n)+"]");
						std::string str;
						tree_parsing::traverse(statements.front().first, statements.front().second, tree_parsing::expression_generation_traversal(std::make_pair(i, j), -1, str, mapping[0]), false);
						stream << str << ";" << std::endl;
					}
				}


			}

		private:
			std::size_t local_size1_;
			std::size_t local_size2_;
			std::size_t cache_width_;

			std::size_t ml_;
			std::size_t nl_;

			std::size_t ms_;
			std::size_t ks_;
			std::size_t ns_;

			bool use_lhs_shared_;
			bool use_rhs_shared_;
		};

	}

}

#endif
