/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

/** \example bandwidth-reduction.cpp
*
*  This tutorial shows how the bandwidth of the nonzero pattern of a sparse matrix can be reduced by renumbering the unknowns (i.e. rows and columns).
*  Such a reordering can significantly improve cache reuse for algorithms such as sparse matrix-vector products and may also reduce the iteration required in iterative solvers.
*
*  There are two parts: The first part defines a couple of helper function. Feel free to directly go to the second part, which shows the actual tutorial code.
**/


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/misc/bandwidth_reduction.hpp"


/**
*   <h2>Part 1: Helper functions</h2>
*
*  We start by defining a couple of helper functions for sparse matrices represented by C++ STL containers, i.e. std::vector<std::map<T,U> >
*
* <h3>Reorder Matrix</h3>
*
*
* Reorders a matrix according to a previously generated node number permutation vector r
**/
inline std::vector< std::map<int, double> > reorder_matrix(std::vector< std::map<int, double> > const & matrix, std::vector<int> const & r)
{
    std::vector< std::map<int, double> > matrix2(r.size());

    for (std::size_t i = 0; i < r.size(); i++)
      for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++)
        matrix2[static_cast<std::size_t>(r[i])][r[static_cast<std::size_t>(it->first)]] = it->second;

    return matrix2;
}

/** <h3>Bandwidth Calculation</h3>
*
* Calculates the bandwidth of a matrix
**/
inline int calc_bw(std::vector< std::map<int, double> > const & matrix)
{
    int bw = 0;

    for (std::size_t i = 0; i < matrix.size(); i++)
    {
      int min_index = static_cast<int>(matrix.size());
      int max_index = 0;
      for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++)
      {
        if (it->first > max_index)
          max_index = it->first;
        if (it->first < min_index)
          min_index = it->first;
      }

      if (max_index > min_index) //row isn't empty
        bw = std::max(bw, max_index - min_index);
    }

    return bw;
}

/** <h3> Bandwidth Calculation </h3>
*
* Calculate the bandwidth of a reordered matrix
**/
template<typename IndexT>
int calc_reordered_bw(std::vector< std::map<int, double> > const & matrix,  std::vector<IndexT> const & r)
{
    int bw = 0;

    for (std::size_t i = 0; i < r.size(); i++)
    {
      int min_index = static_cast<int>(matrix.size());
      int max_index = 0;
      for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++)
      {
        std::size_t col_idx = static_cast<std::size_t>(it->first);
        if (r[col_idx] > max_index)
          max_index = r[col_idx];
        if (r[col_idx] < min_index)
          min_index = r[col_idx];
      }
      if (max_index > min_index)
        bw = std::max(bw, max_index - min_index);
    }

    return bw;
}

/** <h3>Generate Random Permutation</h3>
*
* Generates a random permutation by Knuth shuffle algorithm
* reference: http://en.wikipedia.org/wiki/Knuth_shuffle
*  (URL taken on July 2nd, 2011)
**/
inline std::vector<int> generate_random_reordering(std::size_t n)
{
    std::vector<int> r(n);
    int tmp;

    for (std::size_t i = 0; i < n; i++)
        r[i] = static_cast<int>(i);

    for (std::size_t i = 0; i < n - 1; i++)
    {
        std::size_t j = i + static_cast<std::size_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * static_cast<double>(n - 1 - i));
        if (j != i)
        {
            tmp = r[i];
            r[i] = r[j];
            r[j] = tmp;
        }
    }

    return r;
}

/** <h3>Matrix Generation for 3D Meshes</h3>
*
* This function generates an incidence matrix from of an underlying three-dimensional mesh
*  l:  x dimension
*  m:  y dimension
*  n:  z dimension
*  tri: true for tetrahedral mesh, false for cubic mesh
*  return value: matrix of size l * m * n
**/
inline std::vector< std::map<int, double> > gen_3d_mesh_matrix(std::size_t l, std::size_t m, std::size_t n, bool tri)
{
    std::vector< std::map<int, double> > matrix;
    std::size_t s = l * m * n;
    std::size_t ind;
    std::size_t ind1;
    std::size_t ind2;

    matrix.resize(s);
    for (std::size_t i = 0; i < l; i++)
    {
        for (std::size_t j = 0; j < m; j++)
        {
            for (std::size_t k = 0; k < n; k++)
            {
                ind = i + l * j + l * m * k;

                matrix[ind][static_cast<int>(ind)] = 1.0;

                if (i > 0)
                {
                    ind2 = ind - 1;
                    matrix[ind][static_cast<int>(ind2)] = 1.0;
                    matrix[ind2][static_cast<int>(ind)] = 1.0;
                }
                if (j > 0)
                {
                    ind2 = ind - l;
                    matrix[ind][static_cast<int>(ind2)] = 1.0;
                    matrix[ind2][static_cast<int>(ind)] = 1.0;
                }
                if (k > 0)
                {
                    ind2 = ind - l * m;
                    matrix[ind][static_cast<int>(ind2)] = 1.0;
                    matrix[ind2][static_cast<int>(ind)] = 1.0;
                }

                if (tri)
                {
                    if (i < l - 1 && j < m - 1)
                    {
                        if ((i + j + k) % 2 == 0)
                        {
                            ind1 = ind;
                            ind2 = ind + 1 + l;
                        }
                        else
                        {
                            ind1 = ind + 1;
                            ind2 = ind + l;
                        }
                        matrix[ind1][static_cast<int>(ind2)] = 1.0;
                        matrix[ind2][static_cast<int>(ind1)] = 1.0;
                    }
                    if (i < l - 1 && k < n - 1)
                    {
                        if ((i + j + k) % 2 == 0)
                        {
                            ind1 = ind;
                            ind2 = ind + 1 + l * m;
                        }
                        else
                        {
                            ind1 = ind + 1;
                            ind2 = ind + l * m;
                        }
                        matrix[ind1][static_cast<int>(ind2)] = 1.0;
                        matrix[ind2][static_cast<int>(ind1)] = 1.0;
                    }
                    if (j < m - 1 && k < n - 1)
                    {
                        if ((i + j + k) % 2 == 0)
                        {
                            ind1 = ind;
                            ind2 = ind + l + l * m;
                        }
                        else
                        {
                            ind1 = ind + l;
                            ind2 = ind + l * m;
                        }
                        matrix[ind1][static_cast<int>(ind2)] = 1.0;
                        matrix[ind2][static_cast<int>(ind1)] = 1.0;
                    }
                }
            }
        }
    }

    return matrix;
}


/**
*    <h2>Part 2: Tutorial code</h2>
*
*   After all the helper functions have been defined, we are now ready to see how reordering routines are called.
**/

int main(int, char **)
{
  srand(42);
  std::cout << "-- Generating matrix --" << std::endl;
  std::size_t dof_per_dim = 64;   //number of grid points per coordinate direction
  std::size_t n = dof_per_dim * dof_per_dim * dof_per_dim; //total number of unknowns
  std::vector< std::map<int, double> > matrix = gen_3d_mesh_matrix(dof_per_dim, dof_per_dim, dof_per_dim, false);  //If last parameter is 'true', a tetrahedral grid instead of a hexahedral grid is used.

  /**
  * Shuffle the generated matrix
  **/
  std::vector<int> r = generate_random_reordering(n);
  std::vector< std::map<int, double> > matrix2 = reorder_matrix(matrix, r);


  /**
  * Print some statistics about the generated matrix:
  **/
  std::cout << " * Unknowns: " << n << std::endl;
  std::cout << " * Initial bandwidth: " << calc_bw(matrix) << std::endl;
  std::cout << " * Randomly reordered bandwidth: " << calc_bw(matrix2) << std::endl;

  /**
  * Reorder using Cuthill-McKee algorithm and print new bandwidth:
  **/
  std::cout << "-- Cuthill-McKee algorithm --" << std::endl;
  r = viennacl::reorder(matrix2, viennacl::cuthill_mckee_tag());
  r = viennacl::reorder(matrix2, viennacl::cuthill_mckee_tag());
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;

  /**
  * Reorder using advanced Cuthill-McKee algorithm and print new bandwidth:
  **/
  std::cout << "-- Advanced Cuthill-McKee algorithm --" << std::endl;
  double a = 0.0;
  std::size_t gmax = 1;
  r = viennacl::reorder(matrix2, viennacl::advanced_cuthill_mckee_tag(a, gmax));
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;

  /**
  * Reorder using Gibbs-Poole-Stockmeyer algorithm and print new bandwidth:
  **/
  std::cout << "-- Gibbs-Poole-Stockmeyer algorithm --" << std::endl;
  r = viennacl::reorder(matrix2, viennacl::gibbs_poole_stockmeyer_tag());
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;

  /**
  *  That's it.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}
