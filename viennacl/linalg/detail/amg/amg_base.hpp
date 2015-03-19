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

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <set>
#include <list>
#include <algorithm>

#include <map>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

#include "amg_debug.hpp"

#define VIENNACL_AMG_COARSE_RS      1
#define VIENNACL_AMG_COARSE_ONEPASS 2
#define VIENNACL_AMG_COARSE_RS0     3
#define VIENNACL_AMG_COARSE_RS3     4
#define VIENNACL_AMG_COARSE_AG      5

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
          double threshold = 0.25,
          double interpolweight = 0.2,
          double jacobiweight = 1,
          unsigned int presmooth = 1,
          unsigned int postsmooth = 1,
          unsigned int coarselevels = 0)
  : coarse_(coarse), interpol_(interpol),
    threshold_(threshold), interpolweight_(interpolweight), jacobiweight_(jacobiweight),
    presmooth_(presmooth), postsmooth_(postsmooth), coarselevels_(coarselevels) {}

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

private:
  unsigned int coarse_, interpol_;
  double threshold_, interpolweight_, jacobiweight_;
  unsigned int presmooth_, postsmooth_, coarselevels_;
};



class amg_pointlist
{
public:
  typedef const unsigned int * influence_const_iterator;

  amg_pointlist(std::size_t num_points = 0, std::size_t max_influences = 0)
    : influences_(num_points * max_influences, static_cast<unsigned int>(num_points)),
      influence_count_(num_points),
      influence_extra_(num_points),
      max_influences_(max_influences),
      point_types_(num_points),
      coarse_id_(num_points),
      num_coarse_(0) {}

  void resize(std::size_t num_points, std::size_t max_influences)
  {
    influences_.resize(num_points * max_influences, static_cast<unsigned int>(num_points));
    influence_count_.resize(num_points);
    influence_extra_.resize(num_points);
    max_influences_ = max_influences;
    point_types_.resize(num_points);
    coarse_id_.resize(num_points);
  }

  void add_influence(unsigned int point, unsigned int influenced_point)
  {
    if (influence_count_.at(point) < max_influences_) // found empty slot in list of influences, so add influence
    {
      influences_.at(point * max_influences_ + influence_count_.at(point)) = influenced_point;
      ++influence_count_.at(point);
      //std::cout << "New influence count for point " << point << ": " << influence_count_.at(point) << std::endl;
    }
    else
    {
      // this is only reached if the array of influences is too short
      std::stringstream ss;
      ss << "Fatal: Influence array not large enough for point " << point << " when inserting influence from " << influenced_point << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  influence_const_iterator influence_cbegin(unsigned int point) const { return &(influences_.at(point * max_influences_     )); }
  influence_const_iterator influence_cend  (unsigned int point) const { return &(influences_.at(point * max_influences_  + influence_count_.at(point))); }

  void augment_influence(unsigned int point) { influence_extra_.at(point) += 1; }
  std::size_t influence_count(unsigned int point) const { return influence_count_.at(point) + influence_extra_.at(point); }

  /** @brief Returns the number of points managed by the pointlist */
  std::size_t size() const { return influence_count_.size(); }

  bool is_undecided(unsigned int point) const { return point_types_.at(point) == POINT_TYPE_UNDECIDED; }
  bool is_coarse   (unsigned int point) const { return point_types_.at(point) == POINT_TYPE_COARSE; }
  bool is_fine     (unsigned int point) const { return point_types_.at(point) == POINT_TYPE_FINE; }

  void set_coarse(unsigned int point) { point_types_.at(point) = POINT_TYPE_COARSE; }
  void set_fine  (unsigned int point) { point_types_.at(point) = POINT_TYPE_FINE; }

  void enumerate_coarse_points()
  {
    num_coarse_ = 0;
    //std::cout << "Assigning coarse indices to array starting at " << &(coarse_id_.at(0)) << std::endl;
    for (std::size_t i=0; i<point_types_.size(); ++i)
    {
      if (is_coarse(i))
      {
        //std::cout << "Assigning coarse grid ID " << num_coarse_ << " to point " << i << std::endl;
        coarse_id_.at(i) = num_coarse_++;
      }
    }
  }

  std::size_t num_coarse_points() const { return num_coarse_; }
  unsigned int get_coarse_index(unsigned int point) const { return coarse_id_.at(point); }

  // Slow. Use for debugging only
  std::size_t num_fine_points() const
  {
    std::size_t num_fine = 0;
    for (std::size_t i=0; i<point_types_.size(); ++i)
    {
      if (is_fine(i))
        num_fine++;
    }

    return num_fine;
  }

private:

  std::vector<unsigned int> influences_;
  std::vector<std::size_t> influence_count_; // number of influences for each point
  std::vector<std::size_t> influence_extra_; // extra influence weight during coarsening procedure
  std::size_t max_influences_;

  enum
  {
    POINT_TYPE_UNDECIDED = 0,
    POINT_TYPE_COARSE,
    POINT_TYPE_FINE
  } amg_point_types;
  std::vector<char> point_types_;  // Note: Using char here because type for enum might be a larger type

  std::vector<unsigned int> coarse_id_;
  unsigned int num_coarse_;
};





/** @brief Sparse matrix product. Calculates RES = A*B.
  * @param A    Left Matrix
  * @param B    Right Matrix
  * @param RES    Result Matrix
  */
template<typename SparseMatrixT>
void amg_mat_prod (SparseMatrixT & A, SparseMatrixT & B, SparseMatrixT & RES)
{
  typedef typename SparseMatrixT::value_type ScalarType;
  typedef typename SparseMatrixT::iterator1 InternalRowIterator;
  typedef typename SparseMatrixT::iterator2 InternalColIterator;

  RES = SparseMatrixT(static_cast<unsigned int>(A.size1()), static_cast<unsigned int>(B.size2()));
  RES.clear();

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x<static_cast<long>(A.size1()); ++x)
  {
    InternalRowIterator row_iter = A.begin1();
    row_iter += vcl_size_t(x);
    for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
    {
      unsigned int y = static_cast<unsigned int>(col_iter.index2());
      InternalRowIterator row_iter2 = B.begin1();
      row_iter2 += vcl_size_t(y);

      for (InternalColIterator col_iter2 = row_iter2.begin(); col_iter2 != row_iter2.end(); ++col_iter2)
      {
        unsigned int z = static_cast<unsigned int>(col_iter2.index2());
        ScalarType prod = *col_iter * *col_iter2;
        RES.add(static_cast<unsigned int>(x),static_cast<unsigned int>(z),prod);
      }
    }
  }
}


/** @brief Computes B = trans(A).
  *
  * To be replaced by native functionality in ViennaCL.
  */
template<typename NumericT>
void amg_transpose(compressed_matrix<NumericT> const & A,
                   compressed_matrix<NumericT> & B)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  //std::cout << "Size of A: " << A.size1() << " times " << A.size2() << std::endl;
  //std::cout << "Matrix to transpose: " << std::endl;

  std::vector< std::map<unsigned int, NumericT> > B_temp(A.size2());

  for (std::size_t row = 0; row < A.size1(); ++row)
  {
    //std::cout << "Row " << row << ": ";
    unsigned int row_start = A_row_buffer[row];
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      //std::cout << A_col_buffer[nnz_index] << ": " << A_elements[nnz_index] << "; ";
      B_temp.at(A_col_buffer[nnz_index])[row] = A_elements[nnz_index];
    }
    //std::cout << std::endl;
  }

  /*std::cout << "Transposed matrix: " << std::endl;
  for (std::size_t i=0; i<B_temp.size(); ++i)
  {
    std::cout << "Row " << i << ": ";
    for (typename std::map<unsigned int, NumericT>::const_iterator it = B_temp.at(i).begin(); it != B_temp.at(i).end(); ++it)
      std::cout << it->first << ": " << it->second << "; ";
    std::cout << std::endl;
  }*/
  viennacl::copy(B_temp, B);


}


/** @brief Sparse Galerkin product: Calculates A_coarse = trans(P)*A_fine*P
  * @param A    Operator matrix on fine grid (quadratic)
  * @param P    Prolongation/Interpolation matrix
  * @param C_coarse    Result Matrix (Galerkin operator)
  */
template<typename NumericT>
void amg_galerkin_prod(compressed_matrix<NumericT> const & A_fine,
                       compressed_matrix<NumericT> const & P,
                       compressed_matrix<NumericT> & R, //P^T
                       compressed_matrix<NumericT> & A_coarse)
{

  compressed_matrix<NumericT> A_fine_times_P;

  // transpose P in memory (no known way of efficiently multiplying P^T * B for CSR-matrices P and B):
  amg_transpose(P, R);

  // compute Galerkin product using a temporary for the result of A_fine * P
  A_fine_times_P = viennacl::linalg::prod(A_fine, P);
  A_coarse = viennacl::linalg::prod(R, A_fine_times_P);

  /*std::vector< std::map<unsigned int, NumericT> > B(A_coarse.size1());
  viennacl::copy(A_coarse, B);
  std::cout << "Galerkin matrix: " << std::endl;
  for (std::size_t i=0; i<B.size(); ++i)
  {
    std::cout << "Row " << i << ": ";
    for (typename std::map<unsigned int, NumericT>::const_iterator it = B.at(i).begin(); it != B.at(i).end(); ++it)
      std::cout << it->first << ": " << it->second << "; ";
    std::cout << std::endl;
  }*/

}



} //namespace amg
}
}
}

#endif
