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

#define VIENNACL_AMG_COARSE_RS 1
#define VIENNACL_AMG_COARSE_ONEPASS 2
#define VIENNACL_AMG_COARSE_RS0 3
#define VIENNACL_AMG_COARSE_RS3 4
#define VIENNACL_AMG_COARSE_AG 5
#define VIENNACL_AMG_INTERPOL_DIRECT 1
#define VIENNACL_AMG_INTERPOL_CLASSIC 2
#define VIENNACL_AMG_INTERPOL_AG 3
#define VIENNACL_AMG_INTERPOL_SA 4

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

/** @brief A class for a scalar that can be written to the sparse matrix or sparse vector datatypes.
*  @brief Values are only written to those datatypes if non-zero to optimize memory usage and performance.
*  @brief Needed for the []- and ()-operators.
*/
template<typename InternalT, typename IteratorT, typename NumericT>
class amg_nonzero_scalar
{
private:
  InternalT *m_;
  IteratorT iter_;
  unsigned int i_,j_;
  NumericT s_;

  template <typename T>
  bool is_zero(T value) const { return value <= 0 && value >= 0; }

  template <typename T>
  bool is_zero(T * value) const { return value == NULL; }

public:
  amg_nonzero_scalar();

  /** @brief The constructor.
  *  @param m    Pointer to the sparse vector/matrix the scalar will be written to
  *  @param iter    Iterator pointing to the respective element in the vector/matrix if available
  *  @param i    Row index scalar will be written to
  *  @param j    Col index scalar will be written to
  *  @param s    Value of the scalar (usually used as dummy here as it will be set by the assignment operator)
  */
  amg_nonzero_scalar(InternalT *m,
                     IteratorT & iter,
                     unsigned int i,
                     unsigned int j,
                     NumericT s = 0): m_(m), iter_(iter), i_(i), j_(j), s_(s) {}

  /** @brief Assignment operator. Writes value into matrix at the given position.
  *  @param value  Value that will be written
  */
  NumericT operator=(const NumericT value)
  {
    s_ = value;
    // Only write if scalar is nonzero
    if (is_zero(s_)) return s_;
    // Write to m_ using iterator iter_ or indices (i_,j_)
    m_->addscalar(iter_,i_,j_,s_);
    return s_;
  }

  /** @brief Addition operator. Adds a constant.
  *  @param value  Value that will be written
  */
  NumericT operator+=(const NumericT value)
  {
    // If zero is added, then no change necessary
    if (is_zero(value))
      return s_;

    s_ += value;
    // Remove entry if resulting scalar is zero
    if (is_zero(s_))
    {
      m_->removescalar(iter_,i_);
      return s_;
    }
    //Write to m_ using iterator iter_ or indices (i_,j_)
    m_->addscalar(iter_,i_,j_,s_);
    return s_;
  }
  NumericT operator++(int)
  {
    s_++;
    if (is_zero(s_))
      m_->removescalar(iter_,i_);
    m_->addscalar (iter_,i_,j_,s_);
    return s_;
  }
  NumericT operator++()
  {
    s_++;
    if (is_zero(s_))
      m_->removescalar(iter_,i_);
    m_->addscalar(iter_,i_,j_,s_);
    return s_;
  }
  operator NumericT(void) { return s_; }
};

/** @brief Defines an iterator for the sparse vector type.
*/
template<typename InternalT>
class amg_sparsevector_iterator
{
private:
  typedef amg_sparsevector_iterator<InternalT> self_type;
  typedef typename InternalT::mapped_type      ScalarType;

  InternalT &                  internal_vec_;
  typename InternalT::iterator iter_;

public:

  /** @brief The constructor.
  *  @param vec    Internal sparse vector
  *  @param begin  Whether the iterator starts at the beginning or end of vec
  */
  amg_sparsevector_iterator(InternalT & vec, bool begin=true): internal_vec_(vec)
  {
    if (begin)
      iter_ = internal_vec_.begin();
    else
      iter_ = internal_vec_.end();
  }

  bool operator == (self_type other)
  {
    if (iter_ == other.iter_)
      return true;
    else
      return false;
  }
  bool operator != (self_type other)
  {
    if (iter_ != other.iter_)
      return true;
    else
      return false;
  }

  self_type const & operator ++ () const { iter_++; return *this; }
  self_type       & operator ++ ()       { iter_++; return *this; }
  self_type const & operator -- () const { iter_--; return *this; }
  self_type       & operator -- ()       { iter_--; return *this; }
  ScalarType const & operator * () const { return (*iter_).second; }
  ScalarType       & operator * ()       { return (*iter_).second; }
  unsigned int index() const { return (*iter_).first; }
  unsigned int index()       { return (*iter_).first; }
};

/** @brief A class for the sparse vector type.
*/
template<typename NumericT>
class amg_sparsevector
{
public:
  typedef NumericT value_type;

private:
  // A map is used internally which saves all non-zero elements with pairs of (index,value)
  typedef std::map<unsigned int, NumericT>   InternalType;
  typedef amg_sparsevector<NumericT>         self_type;
  typedef amg_nonzero_scalar<self_type,typename InternalType::iterator, NumericT> NonzeroScalarType;

  // Size is only a dummy variable. Not needed for internal map structure but for compatible vector interface.
  unsigned int size_;
  InternalType internal_vector_;

  template <typename T>
  bool is_zero(T value) const { return value <= 0 && value >= 0; }

  template <typename T>
  bool is_zero(T * value) const { return value == NULL; }

public:
  typedef amg_sparsevector_iterator<InternalType> iterator;
  typedef typename InternalType::const_iterator const_iterator;

public:
  /** @brief The constructor.
  *  @param size    Size of the vector
  */
  amg_sparsevector(unsigned int size = 0): size_(size) {}

  void resize(unsigned int size) { size_ = size; }
  unsigned int size() const { return size_;}

  // Returns number of non-zero entries in vector equal to the size of the underlying map.
  unsigned int internal_size() const { return static_cast<unsigned int>(internal_vector_.size()); }
  // Delete underlying map.
  void clear() { internal_vector_.clear();  }
  // Remove entry at position i.
  void remove(unsigned int i) { internal_vector_.erase(i); }

  // Add s to the entry at position i
  void add(unsigned int i, NumericT s)
  {
    typename InternalType::iterator iter = internal_vector_.find(i);
    // If there is no entry at position i, add new entry at that position
    if (iter == internal_vector_.end())
      addscalar(iter,i,i,s);
    else
    {
      s += (*iter).second;
      // If new value is zero, then erase the entry, otherwise write new value
      if (s < 0 || s > 0)
        (*iter).second = s;
      else
        internal_vector_.erase(iter);
    }
  }

  // Write to the map. Is called from non-zero scalar type.
  template<typename IteratorT>
  void addscalar(IteratorT & iter, unsigned int i, unsigned int /* j */, NumericT s)
  {
    // Don't write if value is zero
    if (is_zero(s))
      return;

    // If entry is already present, overwrite value, otherwise make new entry
    if (iter != internal_vector_.end())
      (*iter).second = s;
    else
      internal_vector_[i] = s;
  }

  // Remove value from the map. Is called from non-zero scalar type.
  template<typename IteratorT>
  void removescalar(IteratorT & iter, unsigned int /* i */) { internal_vector_.erase(iter); }

  // Bracket operator. Returns non-zero scalar type with actual values of the respective entry which calls addscalar/removescalar after value is altered.
  NonzeroScalarType operator [] (unsigned int i)
  {
    typename InternalType::iterator it = internal_vector_.find(i);
    // If value is already present then build non-zero scalar with actual value, otherwise 0.
    if (it != internal_vector_.end())
      return NonzeroScalarType(this,it,i,i,(*it).second);
    else
      return NonzeroScalarType(this,it,i,i,0);
  }

  // Use internal data structure directly for read-only access. No need to use non-zero scalar as no write access possible.
  NumericT operator [] (unsigned int i) const
  {
    const_iterator it = internal_vector_.find(i);

    if (it != internal_vector_.end())
      return (*it).second;
    else
      return 0;
  }

  // Iterator functions.
        iterator begin()       { return iterator(internal_vector_); }
  const_iterator begin() const { return internal_vector_.begin(); }
        iterator end()       { return iterator(internal_vector_, false); }
  const_iterator end() const { return internal_vector_.end(); }

  // checks whether value at index i is nonzero. More efficient than doing [] == 0.
  bool isnonzero(unsigned int i) const { return internal_vector_.find(i) != internal_vector_.end();  }

  // Copies data into a ublas vector type.
  operator boost::numeric::ublas::vector<NumericT>(void)
  {
    boost::numeric::ublas::vector<NumericT> vec (size_);
    for (iterator iter = begin(); iter != end(); ++iter)
      vec [iter.index()] = *iter;
    return vec;
  }
};

/** @brief A class for the sparse matrix type.
*  Uses vector of maps as data structure for higher performance and lower memory usage.
*  Uses similar interface as ublas::compressed_matrix.
*  Can deal with transposed of matrix internally: Creation, Storage, Iterators, etc.
*/
template<typename NumericT>
class amg_sparsematrix
{
public:
  typedef NumericT value_type;
private:
  typedef std::map<unsigned int,NumericT> RowType;
  typedef std::vector<RowType>            InternalType;
  typedef amg_sparsematrix<NumericT>      self_type;

  // Adapter is used for certain functionality, especially iterators.
  typedef typename viennacl::tools::sparse_matrix_adapter<NumericT>                AdapterType;
  typedef typename viennacl::tools::const_sparse_matrix_adapter<NumericT>     ConstAdapterType;

  // Non-zero scalar is used to write to the matrix.
  typedef amg_nonzero_scalar<self_type,typename RowType::iterator, NumericT> NonzeroScalarType;

  // Holds matrix coefficients.
  InternalType internal_mat_;
  // Holds matrix coefficient of transposed matrix if built.
  // Note: Only internal_mat is written using operators and methods while internal_mat_trans is built from internal_mat using do_trans().
  InternalType internal_mat_trans_;
  // Saves sizes.
  vcl_size_t s1_, s2_;

  // True if the transposed of the matrix is used (for calculations, iteration, etc.).
  bool transposed_mode_;
  // True if the transposed is already built (saved in internal_mat_trans) and also up to date (no changes to internal_mat).
  bool transposed_;

public:
  typedef typename AdapterType::iterator1                     iterator1;
  typedef typename AdapterType::iterator2                     iterator2;
  typedef typename ConstAdapterType::const_iterator1    const_iterator1;
  typedef typename ConstAdapterType::const_iterator2    const_iterator2;

  /** @brief Standard constructor. */
  amg_sparsematrix()
  {
    transposed_mode_ = false;
    transposed_ = false;
  }

  /** @brief Constructor. Builds matrix of size (i,j).
    * @param i  Size of first dimension
    * @param j  Size of second dimension
    */
  amg_sparsematrix(unsigned int i, unsigned int j)
  {
    AdapterType a(internal_mat_, i, j);
    a.resize(i,j,false);
    AdapterType a_trans(internal_mat_trans_, j, i);
    a_trans.resize(j,i,false);
    s1_ = i;
    s2_ = j;
    a.clear();
    a_trans.clear();
    transposed_mode_ = false;
    transposed_      = false;
  }

  /** @brief Constructor. Builds matrix via std::vector<std::map> by copying memory
  * (Only necessary feature of this other matrix type is to have const iterators)
  * @param mat  Vector of maps
  */
  amg_sparsematrix(std::vector<std::map<unsigned int, NumericT> > const & mat)
  {
    AdapterType a (internal_mat_, mat.size(), mat.size());
    AdapterType a_trans (internal_mat_trans_, mat.size(), mat.size());
    a.resize(mat.size(), mat.size());
    a_trans.resize(mat.size(), mat.size());

    internal_mat_ = mat;
    s1_ = s2_ = mat.size();

    transposed_mode_ = false;
    transposed_      = false;
  }

  /** @brief Constructor. Builds matrix via another matrix type.
    * (Only necessary feature of this other matrix type is to have const iterators)
    * @param mat  Matrix
    */
  template<typename MatrixT>
  amg_sparsematrix(MatrixT const & mat)
  {
    AdapterType a(internal_mat_, mat.size1(), mat.size2());
    AdapterType a_trans(internal_mat_trans_, mat.size2(), mat.size1());
    a.resize(mat.size1(), mat.size2());
    a_trans.resize(mat.size2(), mat.size1());
    s1_ = mat.size1();
    s2_ = mat.size2();
    a.clear();
    a_trans.clear();

    for (typename MatrixT::const_iterator1 row_iter = mat.begin1(); row_iter != mat.end1(); ++row_iter)
    {
      for (typename MatrixT::const_iterator2 col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
      {
        if (std::fabs(*col_iter) > 0)  // *col_iter != 0, but without floating point comparison warnings
        {
          unsigned int x = static_cast<unsigned int>(col_iter.index1());
          unsigned int y = static_cast<unsigned int>(col_iter.index2());
          a(x,y) = *col_iter;
          a_trans(y,x) = *col_iter;
        }
      }
    }
    transposed_mode_ = false;
    transposed_ = true;
  }

  // Build transposed of the current matrix.
  void do_trans()
  {
    // Do it only once if called in a parallel section
  #ifdef VIENNACL_WITH_OPENMP
    #pragma omp critical
  #endif
    {
      // Only build transposed if it is not built or not up to date
      if (!transposed_)
      {
        // Mode has to be set to standard mode temporarily
        bool save_mode = transposed_mode_;
        transposed_mode_ = false;

        for (iterator1 row_iter = begin1(); row_iter != end1(); ++row_iter)
          for (iterator2 col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
            internal_mat_trans_[col_iter.index2()][static_cast<unsigned int>(col_iter.index1())] = *col_iter;

        transposed_mode_ = save_mode;
        transposed_ = true;
      }
    }
  } //do_trans()

  // Set transposed mode (true=transposed, false=regular)
  void set_trans(bool mode)
  {
    transposed_mode_ = mode;
    if (mode)
      do_trans();
  }

  bool get_trans() const { return transposed_mode_; }

  // Checks whether coefficient (i,j) is non-zero. More efficient than using (i,j) == 0.
  bool isnonzero (unsigned int i, unsigned int j) const
  {
    if (!transposed_mode_)
    {
      if (internal_mat_[i].find(j) != internal_mat_[i].end())
        return true;
      else
        return false;
    }
    else
    {
      if (internal_mat_[j].find(i) != internal_mat_[j].end())
        return true;
      else
        return false;
    }
  } //isnonzero()

  // Add s to value at (i,j)
  void add(unsigned int i, unsigned int j, NumericT s)
  {
    // If zero is added then do nothing.
    if (s <= 0 && s >= 0)
      return;

    typename RowType::iterator col_iter = internal_mat_[i].find(j);
    // If there is no entry at position (i,j), then make new entry.
    if (col_iter == internal_mat_[i].end())
      addscalar(col_iter,i,j,s);
    else
    {
      s += (*col_iter).second;
      // Update value and erase entry if value is zero.
      if (s < 0 || s > 0)
        (*col_iter).second = s;
      else
        internal_mat_[i].erase(col_iter);
    }
    transposed_ = false;
  } //add()

  // Write to the internal data structure. Is called from non-zero scalar type.
  template<typename IteratorT>
  void addscalar(IteratorT & iter, unsigned int i, unsigned int j, NumericT s)
  {
    // Don't write if value is zero
    if (s >= 0 && s <= 0)
      return;

    if (iter != internal_mat_[i].end())
      (*iter).second = s;
    else
      internal_mat_[i][j] = s;

    transposed_ = false;
  }

  // Remove entry from internal data structure. Is called from non-zero scalar type.
  template<typename IteratorT>
  void removescalar(IteratorT & iter, unsigned int i)
  {
    internal_mat_[i].erase(iter);
    transposed_ = false;
  }

  // Return non-zero scalar at position (i,j). Value is written to the non-zero scalar and updated via addscalar()/removescalar().
  NonzeroScalarType operator()(unsigned int i, unsigned int j)
  {
    typename RowType::iterator iter;

    if (!transposed_mode_)
    {
      iter = internal_mat_[i].find(j);
      if (iter != internal_mat_[i].end())
        return NonzeroScalarType(this,iter,i,j,(*iter).second);
      else
        return NonzeroScalarType(this,iter,i,j,0);
    }
    else
    {
      iter = internal_mat_[j].find(i);
      if (iter != internal_mat_[j].end())
        return NonzeroScalarType(this,iter,j,i,(*iter).second);
      else
        return NonzeroScalarType(this,iter,j,i,0);
    }
  }

  // For read-only access return the actual value directly. Non-zero datatype not needed as no write access possible.
  NumericT operator()(unsigned int i, unsigned int j) const
  {
    typename RowType::const_iterator iter;

    if (!transposed_mode_)
    {
      iter = internal_mat_[i].find(j);
      if (iter != internal_mat_[i].end())
        return (*iter).second;
      else
        return 0;
    }
    else
    {
      iter = internal_mat_[j].find(i);
      if (iter != internal_mat_[j].end())
        return (*iter).second;
      else
        return 0;
    }
  }

  void resize(unsigned int i, unsigned int j, bool preserve = true)
  {
    AdapterType a (internal_mat_);
    a.resize(i,j,preserve);
    AdapterType a_trans (internal_mat_trans_);
    a_trans.resize(j,i,preserve);
    s1_ = i;
    s2_ = j;
  }

  void clear()
  {
    AdapterType a(internal_mat_, s1_, s2_);
    a.clear();
    AdapterType a_trans(internal_mat_trans_, s2_, s1_);
    a_trans.clear();
    transposed_ = true;
  }

  vcl_size_t size1()
  {
    if (!transposed_mode_)
      return s1_;
    else
      return s2_;
  }

  vcl_size_t size1() const
  {
    if (!transposed_mode_)
      return s1_;
    else
      return s2_;
  }


  vcl_size_t size2()
  {
    if (!transposed_mode_)
      return s2_;
    else
      return s1_;
  }

  vcl_size_t size2() const
  {
    if (!transposed_mode_)
      return s2_;
    else
      return s1_;
  }

  iterator1 begin1(bool trans = false)
  {
    if (!trans && !transposed_mode_)
    {
      AdapterType a(internal_mat_, s1_, s2_);
      return a.begin1();
    }
    else
    {
      do_trans();
      AdapterType a_trans(internal_mat_trans_, s2_, s1_);
      return a_trans.begin1();
    }
  }

  iterator1 end1(bool trans = false)
  {
    if (!trans && !transposed_mode_)
    {
      AdapterType a(internal_mat_, s1_, s2_);
      return a.end1();
    }
    else
    {
      //do_trans();
      AdapterType a_trans(internal_mat_trans_, s2_, s1_);
      return a_trans.end1();
    }
  }

  iterator2 begin2(bool trans = false)
  {
    if (!trans && !transposed_mode_)
    {
      AdapterType a(internal_mat_, s1_, s2_);
      return a.begin2();
    }
    else
    {
      do_trans();
      AdapterType a_trans(internal_mat_trans_, s2_, s1_);
      return a_trans.begin2();
    }
  }

  iterator2 end2(bool trans = false)
  {
    if (!trans && !transposed_mode_)
    {
      AdapterType a(internal_mat_, s1_, s2_);
      return a.end2();
    }
    else
    {
      //do_trans();
      AdapterType a_trans(internal_mat_trans_, s2_, s1_);
      return a_trans.end2();
    }
  }

  const_iterator1 begin1() const
  {
    // Const_iterator of transposed can only be used if transposed matrix is already built and up to date.
    assert((!transposed_mode_ || (transposed_mode_ && transposed_)) && bool("Error: Cannot build const_iterator when transposed has not been built yet!"));
    ConstAdapterType a_const(internal_mat_, s1_, s2_);
    return a_const.begin1();
  }

  const_iterator1 end1(bool trans = false) const
  {
    assert((!transposed_mode_ || (transposed_mode_ && transposed_)) && bool("Error: Cannot build const_iterator when transposed has not been built yet!"));
    ConstAdapterType a_const(internal_mat_, trans ? s2_ : s1_, trans ? s1_ : s2_);
    return a_const.end1();
  }

  const_iterator2 begin2(bool trans = false) const
  {
    assert((!transposed_mode_ || (transposed_mode_ && transposed_)) && bool("Error: Cannot build const_iterator when transposed has not been built yet!"));
    ConstAdapterType a_const(internal_mat_, trans ? s2_ : s1_, trans ? s1_ : s2_);
    return a_const.begin2();
  }

  const_iterator2 end2(bool trans = false) const
  {
    assert((!transposed_mode_ || (transposed_mode_ && transposed_)) && bool("Error: Cannot build const_iterator when transposed has not been built yet!"));
    ConstAdapterType a_const(internal_mat_, trans ? s2_ : s1_, trans ? s1_ : s2_);
    return a_const.end2();
  }

  // Returns pointer to the internal data structure. Improves performance of copy operation to GPU.
  std::vector<std::map<unsigned int, NumericT> > * get_internal_pointer()
  {
    if (!transposed_mode_)
      return &internal_mat_;

    if (!transposed_)
      do_trans();
    return &internal_mat_trans_;
  }

  operator boost::numeric::ublas::compressed_matrix<NumericT>(void)
  {
    boost::numeric::ublas::compressed_matrix<NumericT> mat;
    mat.resize(size1(), size2(), false);
    mat.clear();

    for (iterator1 row_iter = begin1(); row_iter != end1(); ++row_iter)
        for (iterator2 col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
          mat(col_iter.index1(), col_iter.index2()) = *col_iter;

    return mat;
  }

  operator boost::numeric::ublas::matrix<NumericT>(void)
  {
    boost::numeric::ublas::matrix<NumericT> mat;
    mat.resize(size1(), size2(), false);
    mat.clear();

    for (iterator1 row_iter = begin1(); row_iter != end1(); ++row_iter)
      for (iterator2 col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
        mat(col_iter.index1(), col_iter.index2()) = *col_iter;

    return mat;
  }
};

/** @brief A class for the AMG points.
*   Saves point index and influence measure
*  Holds information whether point is undecided, C or F point.
*  Holds lists of points that are influenced by or influencing this point
*/
class amg_point
{
private:
  typedef amg_sparsevector<amg_point*> ListType;

  unsigned int index_;
  unsigned int influence_;
  // Determines whether point is undecided.
  bool undecided_;
  // Determines wheter point is C point (true) or F point (false).
  bool cpoint_;
  unsigned int coarse_index_;
  // Index offset of parallel coarsening. In that case a point acts as if it had an index of index_-offset_ and treats other points as if they had an index of index+offset_
  unsigned int offset_;
  // Aggregate the point belongs to.
  unsigned int aggregate_;

  // Holds all points influencing this point.
  ListType influencing_points_;
  // Holds all points that are influenced by this point.
  ListType influenced_points_;

public:
  typedef ListType::iterator iterator;
  typedef ListType::const_iterator const_iterator;

  /** @brief The constructor.
  */
  amg_point (unsigned int index, unsigned int size): index_(index), influence_(0), undecided_(true), cpoint_(false), coarse_index_(0), offset_(0), aggregate_(0)
  {
    influencing_points_ = ListType(size);
    influenced_points_ = ListType(size);
  }

  void set_offset(unsigned int offset) { offset_ = offset; }
  unsigned int get_offset() { return offset_; }
  void set_index(unsigned int index) { index_ = index+offset_; }
  unsigned int get_index() const { return index_-offset_;  }
  unsigned int get_influence() const { return influence_;  }
  void set_aggregate(unsigned int aggregate) { aggregate_ = aggregate; }
  unsigned int get_aggregate () { return aggregate_; }

  bool is_cpoint() const { return cpoint_ && !undecided_;  }
  bool is_fpoint() const { return !cpoint_ && !undecided_; }
  bool is_undecided() const { return undecided_; }

  // Returns number of influencing points
  unsigned int number_influencing() const  { return influencing_points_.internal_size(); }
  // Returns true if *point is influencing this point
  bool is_influencing(amg_point* point) const { return influencing_points_.isnonzero(point->get_index()+offset_); }
  // Add *point to influencing points
  void add_influencing_point(amg_point* point) { influencing_points_[point->get_index()+offset_] = point;  }
  // Add *point to influenced points
  void add_influenced_point(amg_point* point) { influenced_points_[point->get_index()+offset_] = point; }

  // Clear influencing points
  void clear_influencing() { influencing_points_.clear(); }
  // Clear influenced points
  void clear_influenced() {influenced_points_.clear(); }


  unsigned int get_coarse_index() const { return coarse_index_; }
  void set_coarse_index(unsigned int index) { coarse_index_ = index; }

  // Calculates the initial influence measure equal to the number of influenced points.
  void calc_influence() { influence_ = influenced_points_.internal_size();  }

  // Add to influence measure.
  unsigned int add_influence(unsigned int add)
  {
    influence_ += add;
    return influence_;
  }
  // Make this point C point. Only call via amg_pointvector.
  void make_cpoint()
  {
    undecided_ = false;
    cpoint_ = true;
    influence_ = 0;
  }
  // Make this point F point. Only call via amg_pointvector.
  void make_fpoint()
  {
    undecided_ = false;
    cpoint_ = false;
    influence_ = 0;
  }
  // Switch point from F to C point. Only call via amg_pointvector.
  void switch_ftoc() { cpoint_ = true; }

  // Iterator handling for influencing and influenced points.
  iterator begin_influencing() { return influencing_points_.begin(); }
  iterator end_influencing() { return influencing_points_.end(); }
  const_iterator begin_influencing() const { return influencing_points_.begin(); }
  const_iterator end_influencing() const { return influencing_points_.end(); }
  iterator begin_influenced() { return influenced_points_.begin();  }
  iterator end_influenced() { return influenced_points_.end(); }
  const_iterator begin_influenced() const { return influenced_points_.begin(); }
  const_iterator end_influenced() const { return influenced_points_.end(); }
};

/** @brief Comparison class for the sorted set of points in amg_pointvector. Set is sorted by influence measure from lower to higher with the point-index as tie-breaker.
*/
struct classcomp
{
  // Function returns true if l comes before r in the ordering.
  bool operator() (amg_point* l, amg_point* r) const
  {
    // Map is sorted by influence number starting with the highest
    // If influence number is the same then lowest point index comes first
    return (l->get_influence() < r->get_influence() || (l->get_influence() == r->get_influence() && l->get_index() > r->get_index()));
  }
};

/** @brief A class for the AMG points.
*  Holds pointers of type amg_point in a vector that can be accessed using [point-index].
*  Additional list of pointers sorted by influence number and index to improve coarsening performance (see amg_coarse_classic_onepass() in amg_coarse.hpp)
*  Constructs indices for C points on the coarse level, needed for interpolation.
*/
class amg_pointvector
{
private:
  // Type for the sorted list
  typedef std::set<amg_point*,classcomp> ListType;
  // Type for the vector of pointers
  typedef std::vector<amg_point*>        VectorType;

  VectorType pointvector_;
  ListType pointlist_;
  unsigned int size_;
  unsigned int c_points_, f_points_;

public:
  typedef VectorType::iterator               iterator;
  typedef VectorType::const_iterator   const_iterator;

  /** @brief The constructor.
  *  @param size    Number of points
  */
  amg_pointvector(unsigned int size = 0): size_(size)
  {
    pointvector_ = VectorType(size);
    c_points_ = f_points_ = 0;
  }

  // Construct all the points dynamically and save pointers into vector.
  void init_points()
  {
    for (unsigned int i=0; i<size(); ++i)
      pointvector_[i] = new amg_point(i,size());
  }
  // Delete all the points.
  void delete_points()
  {
    for (unsigned int i=0; i<size(); ++i)
      delete pointvector_[i];
  }
  // Add point to the vector. Note: User has to make sure that point at point->get_index() does not exist yet, otherwise it will be overwritten!
  void add_point(amg_point *point)
  {
    pointvector_[point->get_index()] = point;
    if (point->is_cpoint()) c_points_++;
    else if (point->is_fpoint()) f_points_++;
  }

  // Update C and F count for point *point.
  // Necessary if C and F points were constructed outside this data structure (e.g. by parallel coarsening RS0 or RS3).
  void update_cf(amg_point *point)
  {
    if (point->is_cpoint()) c_points_++;
    else if (point->is_fpoint()) f_points_++;
  }
  // Clear the C and F point count.
  void clear_cf() { c_points_ = f_points_ = 0; }

  // Clear both point lists.
  void clear_influencelists()
  {
    for (iterator iter = pointvector_.begin(); iter != pointvector_.end(); ++iter)
    {
      (*iter)->clear_influencing();
      (*iter)->clear_influenced();
    }
  }

  amg_point* operator[](unsigned int i) const { return pointvector_[i]; }
  iterator begin() { return pointvector_.begin(); }
  iterator end() { return pointvector_.end(); }
  const_iterator begin() const { return pointvector_.begin(); }
  const_iterator end() const { return pointvector_.end(); }

  void resize(unsigned int size)
  {
    size_ = size;
    pointvector_ = VectorType(size);
  }
  unsigned int size() const { return size_; }

  // Returns number of C points
  unsigned int get_cpoints() const { return c_points_; }
  // Returns number of F points
  unsigned int get_fpoints() const { return f_points_; }

  // Does the initial sorting of points into the list. Sorting is automatically done by the std::set data type.
  void sort()
  {
    for (iterator iter = begin(); iter != end(); ++iter)
      pointlist_.insert(*iter);
  }
  // Returns the point with the highest influence measure
  amg_point* get_nextpoint()
  {
    // No points remaining? Return NULL such that coarsening will stop.
    if (pointlist_.size() == 0)
      return NULL;
    // If point with highest influence measure (end of the list) has measure of zero, then no further C points can be constructed. Return NULL.
    if ((*(--pointlist_.end()))->get_influence() == 0)
      return NULL;
    // Otherwise, return the point with highest influence measure located at the end of the list.
    else
      return *(--pointlist_.end());
  }
  // Add "add" to influence measure for point *point in the sorted list.
  void add_influence(amg_point* point, unsigned int add)
  {
    ListType::iterator iter = pointlist_.find(point);
    // If point is not in the list then stop.
    if (iter == pointlist_.end()) return;

    // Point has to be erased first as changing the value does not re-order the std::set
    pointlist_.erase(iter);
    point->add_influence(add);

    // Insert point back into the list. Using the iterator improves performance. The new position has to be at the same position or to the right of the old.
    pointlist_.insert(point);
  }
  // Make *point to C point and remove from sorted list
  void make_cpoint(amg_point* point)
  {
    pointlist_.erase(point);
    point->make_cpoint();
    c_points_++;
  }
  // Make *point to F point and remove from sorted list
  void make_fpoint(amg_point* point)
  {
    pointlist_.erase(point);
    point->make_fpoint();
    f_points_++;
  }
  // Swich *point from F to C point
  void switch_ftoc(amg_point* point)
  {
    point->switch_ftoc();
    c_points_++;
    f_points_--;
  }

  // Build vector of indices for C point on the coarse level.
  void build_index()
  {
    unsigned int count = 0;
    // Use simple counter for index creation.
    for (iterator iter = pointvector_.begin(); iter != pointvector_.end(); ++iter)
    {
      // Set index on coarse level using counter variable
      if ((*iter)->is_cpoint())
      {
        (*iter)->set_coarse_index(count);
        count++;
      }
    }
  }

  // Return information for debugging purposes
  template<typename MatrixT>
  void get_influence_matrix(MatrixT & mat) const
  {
    mat = MatrixT(size(),size());
    mat.clear();

    for (const_iterator row_iter = begin(); row_iter != end(); ++row_iter)
      for (amg_point::iterator col_iter = (*row_iter)->begin_influencing(); col_iter != (*row_iter)->end_influencing(); ++col_iter)
        mat((*row_iter)->get_index(),(*col_iter)->get_index()) = true;
  }
  template<typename VectorT>
  void get_influence(VectorT & vec) const
  {
    vec = VectorT(size_);
    vec.clear();

    for (const_iterator iter = begin(); iter != end(); ++iter)
      vec[(*iter)->get_index()] = (*iter)->get_influence();
  }
  template<typename VectorT>
  void get_sorting(VectorT & vec) const
  {
    vec = VectorT(pointlist_.size());
    vec.clear();
    unsigned int i=0;

    for (ListType::const_iterator iter = pointlist_.begin(); iter != pointlist_.end(); ++iter)
    {
      vec[i] = (*iter)->get_index();
      i++;
    }
  }
  template<typename VectorT>
  void get_C(VectorT & vec) const
  {
    vec = VectorT(size_);
    vec.clear();

    for (const_iterator iter = begin(); iter != end(); ++iter)
    {
      if ((*iter)->is_cpoint())
        vec[(*iter)->get_index()] = true;
    }
  }
  template<typename VectorT>
  void get_F(VectorT & vec) const
  {
    vec = VectorT(size_);
    vec.clear();

    for (const_iterator iter = begin(); iter != end(); ++iter)
    {
      if ((*iter)->is_fpoint())
        vec[(*iter)->get_index()] = true;
    }
  }
  template<typename MatrixT>
  void get_Aggregates(MatrixT & mat) const
  {
    mat = MatrixT(size_,size_);
    mat.clear();

    for (const_iterator iter = begin(); iter != end(); ++iter)
    {
      if (!(*iter)->is_undecided())
        mat((*iter)->get_aggregate(),(*iter)->get_index()) = true;
    }
  }
};

/** @brief A class for the matrix slicing for parallel coarsening schemes (RS0/RS3).
  * @brief Holds information on a per-processor basis and offers functionality to slice and join the data structures.
  */
template<typename InternalT1, typename InternalT2>
class amg_slicing
{
  typedef typename InternalT1::value_type    SparseMatrixType;
  typedef typename InternalT2::value_type    PointVectorType;

public:
  // Data structures on a per-processor basis.
  boost::numeric::ublas::vector<InternalT1> A_slice_;
  boost::numeric::ublas::vector<InternalT2> pointvector_slice_;
  // Holds the offsets showing the indices for which a new slice begins.
  boost::numeric::ublas::vector<boost::numeric::ublas::vector<unsigned int> > offset_;

  unsigned int threads_;
  unsigned int levels_;

  void init(unsigned int levels, unsigned int threads = 0)
  {
    // Either use the number of threads chosen by the user or the maximum number of threads available on the processor.
    if (threads == 0)
  #ifdef VIENNACL_WITH_OPENMP
      threads_ = omp_get_num_procs();
  #else
    threads_ = 1;
  #endif
    else
      threads_ = threads;

    levels_ = levels;

    A_slice_.resize(threads_);
    pointvector_slice_.resize(threads_);
    // Offset has threads_+1 entries to also hold the total size
    offset_.resize(threads_+1);

    for (unsigned int i=0; i<threads_; ++i)
    {
      A_slice_[i].resize(levels_);
      pointvector_slice_[i].resize(levels_);
      // Offset needs one more level for the build-up of the next offset
      offset_[i].resize(levels_+1);
    }
    offset_[threads_].resize(levels_+1);
  } //init()

  // Slice matrix A into as many parts as threads are used.
  void slice(unsigned int level, InternalT1 const & A, InternalT2 const & pointvector)
  {
    // On the finest level, build a new slicing first.
    if (level == 0)
      slice_new(level, A);

    // On coarser levels use the same slicing as on the finest level (Points stay together on the same thread on all levels).
    // This is necessary as due to interpolation and galerkin product there only exist connections between points on the same thread on coarser levels.
    // Note: Offset is determined in amg_coarse_rs0() after fine level was built.
    slice_build(level, A, pointvector);
  }

  // Join point data structure into Pointvector
  void join(unsigned int level, InternalT2 & pointvector) const
  {
    typedef typename InternalT2::value_type PointVectorType;

    // Reset index offset of all points and update overall C and F point count
    pointvector[level].clear_cf();
    for (typename PointVectorType::iterator iter = pointvector[level].begin(); iter != pointvector[level].end(); ++iter)
    {
      (*iter)->set_offset(0);
      pointvector[level].update_cf(*iter);
    }
  }

private:
  /** @brief Slices mat into this->threads parts of (almost) equal size
  * @param level    Level for which slicing is requested
  * @param A     System matrix on all levels
  */
  void slice_new(unsigned int level, InternalT1 const & A)
  {
    // Determine index offset of all the slices (index of A[level] when the respective slice starts).
  #ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
  #endif
    for (long i2=0; i2<=static_cast<long>(threads_); ++i2)
    {
      std::size_t i = static_cast<std::size_t>(i2);

      // Offset of first piece is zero. Pieces 1,...,threads-1 have equal size while the last one might be greater.
      if (i == 0) offset_[i][level] = 0;
      else if (i == threads_) offset_[i][level] = static_cast<unsigned int>(A[level].size1());
      else offset_[i][level] = static_cast<unsigned int>(i*(A[level].size1()/threads_));
    }
  }

  /** @brief Slices mat into pieces determined by this->Offset
  * @param level    Level to which Slices are saved
  * @param A     System matrix on all levels
  * @param Pointvector  Vector of points on all levels
  */
  void slice_build(unsigned int level, InternalT1 const & A, InternalT2 const & pointvector)
  {
    typedef typename SparseMatrixType::const_iterator1 ConstRowIterator;
    typedef typename SparseMatrixType::const_iterator2 ConstColIterator;

  #ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
  #endif
    for (long i2=0; i2<static_cast<long>(threads_); ++i2)
    {
      std::size_t i = static_cast<std::size_t>(i2);

      amg_point *point;

      // Allocate space for the matrix slice and the pointvector.
      A_slice_[i][level] = SparseMatrixType(offset_[i+1][level]-offset_[i][level], offset_[i+1][level]-offset_[i][level]);
      pointvector_slice_[i][level] = PointVectorType(offset_[i+1][level]-offset_[i][level]);

      // Iterate over the part that belongs to thread i (from Offset[i][level] to Offset[i+1][level]).
      ConstRowIterator row_iter = A[level].begin1();
      row_iter += offset_[i][level];
      unsigned int x = static_cast<unsigned int>(row_iter.index1());

      while (x < offset_[i+1][level] && row_iter != A[level].end1())
      {
        // Set offset for point index and save point for the respective thread
        point = pointvector[level][x];
        point->set_offset(offset_[i][level]);
        pointvector_slice_[i][level].add_point(point);

        ConstColIterator col_iter = row_iter.begin();
        unsigned int y = static_cast<unsigned int>(col_iter.index2());

        // Save all coefficients from the matrix slice
        while (y < offset_[i+1][level] && col_iter != row_iter.end())
        {
          if (y >= offset_[i][level])
            A_slice_[i][level](x-offset_[i][level], y-offset_[i][level]) = *col_iter;

          ++col_iter;
          y = static_cast<unsigned int>(col_iter.index2());
        }

        ++row_iter;
        x = static_cast<unsigned int>(row_iter.index1());
      }
    }
  }
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

/** @brief Sparse Galerkin product: Calculates RES = trans(P)*A*P
  * @param A    Operator matrix (quadratic)
  * @param P    Prolongation/Interpolation matrix
  * @param RES    Result Matrix (Galerkin operator)
  */
template<typename SparseMatrixT>
void amg_galerkin_prod (SparseMatrixT & A, SparseMatrixT & P, SparseMatrixT & RES)
{
  typedef typename SparseMatrixT::value_type   ScalarType;
  typedef typename SparseMatrixT::iterator1    InternalRowIterator;
  typedef typename SparseMatrixT::iterator2    InternalColIterator;

  RES = SparseMatrixT(static_cast<unsigned int>(P.size2()), static_cast<unsigned int>(P.size2()));
  RES.clear();

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long x=0; x<static_cast<long>(P.size2()); ++x)
  {
    amg_sparsevector<ScalarType> row(static_cast<unsigned int>(A.size2()));
    InternalRowIterator row_iter = P.begin1(true);
    row_iter += vcl_size_t(x);

    for (InternalColIterator col_iter = row_iter.begin(); col_iter != row_iter.end(); ++col_iter)
    {
      long y1 = static_cast<long>(col_iter.index2());
      InternalRowIterator row_iter2 = A.begin1();
      row_iter2 += vcl_size_t(y1);

      for (InternalColIterator col_iter2 = row_iter2.begin(); col_iter2 != row_iter2.end(); ++col_iter2)
      {
        long y2 = static_cast<long>(col_iter2.index2());
        row.add (static_cast<unsigned int>(y2), *col_iter * *col_iter2);
      }
    }
    for (typename amg_sparsevector<ScalarType>::iterator iter = row.begin(); iter != row.end(); ++iter)
    {
      long y2 = iter.index();
      InternalRowIterator row_iter3 = P.begin1();
      row_iter3 += vcl_size_t(y2);

      for (InternalColIterator col_iter3 = row_iter3.begin(); col_iter3 != row_iter3.end(); ++col_iter3)
      {
        long z = static_cast<long>(col_iter3.index2());
        RES.add (static_cast<unsigned int>(x), static_cast<unsigned int>(z), *col_iter3 * *iter);
      }
    }
  }

  #ifdef VIENNACL_AMG_DEBUG
  std::cout << "Galerkin Operator: " << std::endl;
  printmatrix (RES);
  #endif
}

/** @brief Test triple-matrix product by comparing it to ublas functions. Very slow for large matrices!
  * @param A    Operator matrix (quadratic)
  * @param P    Prolongation/Interpolation matrix
  * @param A_i1    Result Matrix
  */
template<typename SparseMatrixT>
void test_triplematprod(SparseMatrixT & A, SparseMatrixT & P, SparseMatrixT  & A_i1)
{
  typedef typename SparseMatrixT::value_type ScalarType;

  boost::numeric::ublas::compressed_matrix<ScalarType> A_temp (A.size1(), A.size2());
  A_temp = A;
  boost::numeric::ublas::compressed_matrix<ScalarType> P_temp (P.size1(), P.size2());
  P_temp = P;
  P.set_trans(true);
  boost::numeric::ublas::compressed_matrix<ScalarType> R_temp (P.size1(), P.size2());
  R_temp = P;
  P.set_trans(false);

  boost::numeric::ublas::compressed_matrix<ScalarType> RA (R_temp.size1(),A_temp.size2());
  RA = boost::numeric::ublas::prod(R_temp,A_temp);
  boost::numeric::ublas::compressed_matrix<ScalarType> RAP (RA.size1(),P_temp.size2());
  RAP = boost::numeric::ublas::prod(RA,P_temp);

  for (unsigned int x=0; x<RAP.size1(); ++x)
  {
    for (unsigned int y=0; y<RAP.size2(); ++y)
    {
      if (std::fabs(static_cast<ScalarType>(RAP(x,y)) - static_cast<ScalarType>(A_i1(x,y))) > 0.0001)
        std::cout << x << " " << y << " " << RAP(x,y) << " " << A_i1(x,y) << std::endl;
    }
  }
}

/** @brief Test if interpolation matrix makes sense. Only vanilla test though! Only checks if basic requirements are met!
  * @param A    Operator matrix (quadratic)
  * @param P    Prolongation/Interpolation matrix
  * @param Pointvector  Vector of points
  */
template<typename SparseMatrixT, typename PointVectorT>
void test_interpolation(SparseMatrixT & A, SparseMatrixT & P, PointVectorT & Pointvector)
{
  for (unsigned int i=0; i<P.size1(); ++i)
  {
    if (Pointvector.is_cpoint(i))
    {
      bool set = false;
      for (unsigned int j=0; j<P.size2(); ++j)
      {
        if (P.isnonzero(i,j))
        {
          if (P(i,j) != 1)
            std::cout << "Error 1 in row " << i << std::endl;
          if (P(i,j) == 1 && set)
            std::cout << "Error 2 in row " << i << std::endl;
          if (P(i,j) == 1 && !set)
            set = true;
        }
      }
    }

    if (Pointvector.is_fpoint(i))
      for (unsigned int j=0; j<P.size2(); ++j)
      {
        if (P.isnonzero(i,j) && j> Pointvector.get_cpoints()-1)
          std::cout << "Error 3 in row " << i << std::endl;
        if (P.isnonzero(i,j))
        {
          bool set = false;
          for (unsigned int k=0; k<P.size1(); ++k)
          {
            if (P.isnonzero(k,j))
            {
              if (Pointvector.is_cpoint(k) && P(k,j) == 1 && A.isnonzero(i,k))
                set = true;
            }
          }
          if (!set)
            std::cout << "Error 4 in row " << i << std::endl;
        }
      }
  }
}


} //namespace amg
}
}
}

#endif
