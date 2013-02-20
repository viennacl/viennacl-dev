#ifndef VECTOR_IO_HPP_
#define VECTOR_IO_HPP_

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

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <iostream>
#include <fstream>

template <class TYPE>
bool readVectorFromFile(const std::string & filename, boost::numeric::ublas::vector<TYPE> & vec)
{
	std::ifstream file(filename.c_str());

	if (!file) return false;

	unsigned int size;
	file >> size;
	vec.resize(size);

	for (unsigned int i = 0; i < size; ++i)
	{
		TYPE element;
		file >> element;
		vec[i] = element;
	}

	return true;
}

template<class TYPE>
bool readVectorFromBinaryFile(const std::string & filename, boost::numeric::ublas::vector<TYPE> & vec)
{
	std::ifstream file(filename.c_str(), std::ios_base::binary);
	if (!file) return false;

	unsigned int size;
	file.read((char*)&size, sizeof(unsigned int));
	vec.resize(size);
	file.read((char*)&vec[0], sizeof(TYPE)*size);

	return true;
}

template<class TYPE>
bool saveVectorToBinaryFile(const std::string & filename, const boost::numeric::ublas::vector<TYPE> & vec)
{
	std::ofstream file(filename.c_str(), std::ios_base::binary);
	if (!file) return false;

	unsigned int size = vec.size();
	file.write((char*)&size, sizeof(unsigned int));
	file.write((char*)&vec[0], sizeof(TYPE)*size);

	return true;
}

template <class TYPE>
bool readMatrixFromFile(const std::string & filename, boost::numeric::ublas::compressed_matrix<TYPE> & matrix)
{
  std::cout << "Reading ublas matrix" << std::endl;
  
  std::ifstream file(filename.c_str());

  if (!file) return false;

  std::string id;
  file >> id;
  if (id != "Matrix") return false;

  unsigned int num_rows, num_columns;
  file >> num_rows >> num_columns;
  if (num_rows != num_columns) return false;
  
  matrix.resize(num_rows, num_rows, false);

  for (unsigned int row = 0; row < num_rows; ++row)
  {
    int num_entries;
    file >> num_entries;
    for (int j = 0; j < num_entries; ++j)
    {
      unsigned int column;
      TYPE element;
      file >> column >> element;

      //matrix.insert_element(row, column, element);
      matrix(row, column) = element;
    }
  }

  return true;
}




#endif
