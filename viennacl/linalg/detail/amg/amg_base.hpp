#ifndef VIENNACL_LINALG_DETAIL_AMG_AMG_BASE_HPP_
#define VIENNACL_LINALG_DETAIL_AMG_AMG_BASE_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
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

/** @file amg_base.hpp
    @brief Helper classes and functions for the AMG preconditioner. Experimental.

    AMG code contributed by Markus Wagner
*/

#include <cmath>
#include <set>
#include <list>
#include <stdexcept>
#include <algorithm>

#include <map>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

#include "viennacl/context.hpp"

#define VIENNACL_AMG_COARSE_RS      1
#define VIENNACL_AMG_COARSE_ONEPASS 2
#define VIENNACL_AMG_COARSE_RS0     3
#define VIENNACL_AMG_COARSE_RS3     4
#define VIENNACL_AMG_COARSE_AG      5
#define VIENNACL_AMG_COARSE_AG_MIS2 6

#define VIENNACL_AMG_INTERPOL_DIRECT  1
#define VIENNACL_AMG_INTERPOL_CLASSIC 2
#define VIENNACL_AMG_INTERPOL_AG      3
#define VIENNACL_AMG_INTERPOL_SA      4

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace amg
{

/** @brief A tag for algebraic multigrid (AMG). Used to transport information from the user to the implementation.
*/
class amg_tag
{
public:
  /** @brief The constructor.
  * @param coarse    Coarsening Routine (Default: VIENNACL_AMG_COARSE_CLASSIC)
  * @param interpol  Interpolation routine (Default: VIENNACL_AMG_INTERPOL_DIRECT)
  * @param threshold    Strength of dependence threshold for the coarsening process (Default: 0.25)
  * @param interpolweight  Interpolation parameter for SA interpolation and truncation parameter for direct+classical interpolation
  * @param jacobiweight  Weight of the weighted Jacobi smoother iteration step (Default: 1 = Regular Jacobi smoother)
  * @param presmooth    Number of presmoothing operations on every level (Default: 1)
  * @param postsmooth   Number of postsmoothing operations on every level (Default: 1)
  * @param coarselevels  Number of coarse levels that are constructed
  *      (Default: 0 = Optimize coarse levels for direct solver such that coarsest level has a maximum of COARSE_LIMIT points)
  *      (Note: Coarsening stops when number of coarse points = 0 and overwrites the parameter with actual number of coarse levels)
  */
  amg_tag(unsigned int coarse = 1,
          unsigned int interpol = 1,
          double threshold = 0.1,
          double interpolweight = 0.2,
          double jacobiweight = 1,
          unsigned int presmooth = 1,
          unsigned int postsmooth = 1,
          unsigned int coarselevels = 0)
  : coarse_(coarse), interpol_(interpol),
    threshold_(threshold), interpolweight_(interpolweight), jacobiweight_(jacobiweight),
    presmooth_(presmooth), postsmooth_(postsmooth), coarselevels_(coarselevels),
    coarse_info_(20), use_coarse_info_(false), save_coarse_info_(false) {}

  amg_tag & operator=(amg_tag const & other)
  {
    coarse_ = other.coarse_;
    interpol_ = other.interpol_;
    threshold_ = other.threshold_;
    interpolweight_ = other.interpolweight_;
    jacobiweight_ = other.jacobiweight_;
    presmooth_ = other.presmooth_;
    postsmooth_ = other.postsmooth_;
    coarselevels_ = other.coarselevels_;
    setup_ctx_ = other.setup_ctx_;
    target_ctx_ = other.target_ctx_;

    for (std::size_t i=0; i < coarselevels_; ++i)
    {
      if (other.coarse_info_[i].size() > 0)
      {
        std::vector<char> tmp(other.coarse_info_[i].size());
        viennacl::copy(other.coarse_info_[i], tmp);
        coarse_info_[i].resize(tmp.size(), false);
        viennacl::copy(tmp, coarse_info_[i]);
      }
    }
    use_coarse_info_ = other.use_coarse_info_;
    save_coarse_info_ = other.save_coarse_info_;

    return *this;
  }

  // Getter-/Setter-Functions
  void set_coarse(unsigned int coarse) { coarse_ = coarse; }
  unsigned int get_coarse() const { return coarse_; }

  void set_interpol(unsigned int interpol) { interpol_ = interpol; }
  unsigned int get_interpol() const { return interpol_; }

  void set_threshold(double threshold) { if (threshold > 0 && threshold <= 1) threshold_ = threshold; }
  double get_threshold() const { return threshold_; }

  void set_as(double jacobiweight) { if (jacobiweight > 0 && jacobiweight <= 2) jacobiweight_ = jacobiweight; }
  double get_interpolweight() const { return interpolweight_; }

  void set_interpolweight(double interpolweight) { if (interpolweight > 0 && interpolweight <= 2) interpolweight_ = interpolweight; }
  double get_jacobiweight() const { return jacobiweight_; }

  void set_presmooth(unsigned int presmooth) { presmooth_ = presmooth; }
  unsigned int get_presmooth() const { return presmooth_; }

  void set_postsmooth(unsigned int postsmooth) { postsmooth_ = postsmooth; }
  unsigned int get_postsmooth() const { return postsmooth_; }

  void set_coarselevels(unsigned int coarselevels)  { coarselevels_ = coarselevels; }
  unsigned int get_coarselevels() const { return coarselevels_; }

  void set_setup_context(viennacl::context ctx)  { setup_ctx_ = ctx; }
  viennacl::context const & get_setup_context() const { return setup_ctx_; }

  void set_target_context(viennacl::context ctx)  { target_ctx_ = ctx; }
  viennacl::context const & get_target_context() const { return target_ctx_; }

  void set_coarse_information(std::size_t level, viennacl::vector<char> const & c_info) { coarse_info_.at(level) = c_info; }
  void set_coarse_information(std::size_t level,      std::vector<char> const & c_info) { viennacl::copy(c_info, coarse_info_.at(level)); }
  viennacl::vector<char> & get_coarse_information(std::size_t level) { return coarse_info_.at(level); }

  void use_coarse_information(bool b) { use_coarse_info_ = b; }
  bool use_coarse_information() const { return use_coarse_info_; }

  void save_coarse_information(bool b) { save_coarse_info_ = b; }
  bool save_coarse_information() const { return save_coarse_info_; }

private:
  unsigned int coarse_, interpol_;
  double threshold_, interpolweight_, jacobiweight_;
  unsigned int presmooth_, postsmooth_, coarselevels_;
  viennacl::context setup_ctx_, target_ctx_;
  std::vector<viennacl::vector<char> > coarse_info_;
  bool use_coarse_info_, save_coarse_info_;
};



struct amg_level_context
{
  void resize(vcl_size_t num_points, vcl_size_t max_nnz)
  {
    influence_jumper_.resize(num_points + 1, false);
    influence_ids_.resize(max_nnz, false);
    influence_values_.resize(num_points, false);
    point_types_.resize(num_points, false);
    coarse_id_.resize(num_points, false);
  }

  void switch_context(viennacl::context ctx)
  {
    influence_jumper_.switch_memory_context(ctx);
    influence_ids_.switch_memory_context(ctx);
    influence_values_.switch_memory_context(ctx);
    point_types_.switch_memory_context(ctx);
    coarse_id_.switch_memory_context(ctx);
  }

  enum
  {
    POINT_TYPE_UNDECIDED = 0,
    POINT_TYPE_COARSE,
    POINT_TYPE_FINE
  } amg_point_types;

  viennacl::vector<unsigned int> influence_jumper_; // similar to row_buffer for CSR matrices
  viennacl::vector<unsigned int> influence_ids_;    // IDs of influencing points
  viennacl::vector<unsigned int> influence_values_; // Influence measure for each point
  viennacl::vector<unsigned int> point_types_;      // 0: undecided, 1: coarse point, 2: fine point. Using char here because type for enum might be a larger type
  viennacl::vector<unsigned int> coarse_id_;        // coarse ID used on the next level. Only valid for coarse points. Fine points may (ab)use their entry for something else.
  unsigned int num_coarse_;
};




} //namespace amg
}
}
}

#endif
