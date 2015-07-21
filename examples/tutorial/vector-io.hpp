#ifndef VECTOR_IO_HPP_
#define VECTOR_IO_HPP_

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

#include <string>
#include <iostream>
#include <fstream>

#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"


template<typename MatrixType, typename ScalarType>
void insert(MatrixType & matrix, long row, long col, ScalarType value)
{
  matrix(row, col) = value;
}

#ifdef VIENNACL_HAVE_EIGEN
template<typename ScalarType, int option>
void insert(Eigen::SparseMatrix<ScalarType, option> & matrix, long row, long col, double value)
{
  matrix.fill(row, col) = value;
}
#endif

template<typename MatrixType>
class my_inserter
{
  public:
    my_inserter(MatrixType & mat) : mat_(mat) {}

    void apply(long row, long col, double value)
    {
      insert(mat_, row, col, value);
    }

  private:
    MatrixType & mat_;
};

#ifdef VIENNACL_HAVE_MTL4
#include <boost/numeric/mtl/matrix/inserter.hpp>
/*template<typename ScalarType>
void insert(mtl::compressed2D<ScalarType> & matrix, long row, long col, ScalarType value)
{
  typedef mtl::compressed2D<ScalarType>   MatrixType;
  mtl::matrix::inserter<MatrixType>      ins(matrix);

  typename mtl::Collection<MatrixType>::value_type val(value);
  ins(row, col) << val;
  //matrix.fill(row, col) = val;
}*/

template<typename ScalarType>
void resize_vector(mtl::dense_vector<ScalarType> & vec, unsigned int size)
{
  vec.change_dim(size);
}

template<typename ScalarType>
class my_inserter<mtl::compressed2D<ScalarType> >
{
    typedef mtl::compressed2D<ScalarType>    MatrixType;
  public:
    my_inserter(MatrixType & mat) : mat_(mat), ins_(mat) {}

    void apply(long row, long col, ScalarType value)
    {
      typename mtl::Collection<MatrixType>::value_type val(value);
      ins_(row, col) << val;
    }

  private:
    MatrixType & mat_;
    mtl::matrix::inserter<MatrixType> ins_;
};
#endif

template<typename VectorType>
void resize_vector(VectorType & vec, unsigned int size)
{
  vec.resize(size);
}

template<typename VectorType>
bool readVectorFromFile(const std::string & filename,
                        VectorType & vec)
{
  typedef typename viennacl::result_of::value_type<VectorType>::type    ScalarType;

  std::ifstream file(filename.c_str());

  if (!file) return false;

  unsigned int size;
  file >> size;

  resize_vector(vec, size);

  for (unsigned int i = 0; i < size; ++i)
  {
    ScalarType element;
    file >> element;
    vec[i] = element;
  }

  return true;
}


template<class MatrixType>
bool readMatrixFromFile(const std::string & filename, MatrixType & matrix)
{
  typedef typename viennacl::result_of::value_type<MatrixType>::type    ScalarType;

  std::cout << "Reading matrix..." << std::endl;

  std::ifstream file(filename.c_str());

  if (!file) return false;

  std::string id;
  file >> id;
  if (id != "Matrix") return false;


  unsigned int num_rows, num_columns;
  file >> num_rows >> num_columns;
  if (num_rows != num_columns) return false;

  viennacl::traits::resize(matrix, num_rows, num_rows);

  my_inserter<MatrixType> ins(matrix);
  for (unsigned int row = 0; row < num_rows; ++row)
  {
    int num_entries;
    file >> num_entries;
    for (int j = 0; j < num_entries; ++j)
    {
      unsigned int column;
      ScalarType element;
      file >> column >> element;

      ins.apply(row, column, element);
      //insert(matrix, row, column, element);
      //note: the obvious 'matrix(row, column) = element;' does not work with Eigen, hence another level of indirection
    }
    //std::cout << "reading of row finished" << std::endl;
  }

  return true;
}


#endif
