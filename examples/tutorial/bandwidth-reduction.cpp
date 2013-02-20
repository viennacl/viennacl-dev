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

/*
*   Tutorial: Matrix bandwidth reduction algorithms
*/


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/misc/bandwidth_reduction.hpp"


//
// Part 1: Helper functions
//

// Reorders a matrix according to a previously generated node
// number permutation vector r
std::vector< std::map<int, double> > reorder_matrix(std::vector< std::map<int, double> > const & matrix, std::vector<int> const & r)
{
    std::vector< std::map<int, double> > matrix2(r.size());
    std::vector<std::size_t> r2(r.size());
    
    for (std::size_t i = 0; i < r.size(); i++)
        r2[r[i]] = i;

    for (std::size_t i = 0; i < r.size(); i++)
        for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++)
            matrix2[i][r2[it->first]] = it->second;
    
    return matrix2;
}

// Calculates the bandwidth of a matrix
int calc_bw(std::vector< std::map<int, double> > const & matrix)
{
    int bw = 0;
    
    for (std::size_t i = 0; i < matrix.size(); i++)
        for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++)
            bw = std::max(bw, std::abs(static_cast<int>(i - it->first)));
    
    return bw;
}


// Calculate the bandwidth of a reordered matrix
int calc_reordered_bw(std::vector< std::map<int, double> > const & matrix,  std::vector<int> const & r)
{
    std::vector<int> r2(r.size());
    int bw = 0;
    
    for (std::size_t i = 0; i < r.size(); i++)
        r2[r[i]] = i;

    for (std::size_t i = 0; i < r.size(); i++)
        for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++)
            bw = std::max(bw, std::abs(static_cast<int>(i - r2[it->first])));
    
    return bw;
}


// Generates a random permutation by Knuth shuffle algorithm
// reference: http://en.wikipedia.org/wiki/Knuth_shuffle 
//  (URL taken on July 2nd, 2011)
std::vector<int> generate_random_reordering(int n)
{
    std::vector<int> r(n);
    int tmp;
    int j;
    
    for (int i = 0; i < n; i++)
        r[i] = i;
    
    for (int i = 0; i < n - 1; i++)
    {
        j = i + static_cast<std::size_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (n - 1 - i));
        if (j != i)
        {
            tmp = r[i];
            r[i] = r[j];
            r[j] = tmp;
        }
    }
    
    return r;
}


// function for the generation of a three-dimensional mesh incidence matrix
//  l:  x dimension
//  m:  y dimension
//  n:  z dimension
//  tri: true for tetrahedral mesh, false for cubic mesh
//  return value: matrix of size l * m * n
std::vector< std::map<int, double> > gen_3d_mesh_matrix(int l, int m, int n, bool tri)
{
    std::vector< std::map<int, double> > matrix;
    int s;
    int ind;
    int ind1;
    int ind2;
    
    s = l * m * n;
    matrix.resize(s);
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < n; k++)
            {
                ind = i + l * j + l * m * k;
                
                matrix[ind][ind] = 1.0;
                
                if (i > 0)
                {
                    ind2 = ind - 1;
                    matrix[ind][ind2] = 1.0;
                    matrix[ind2][ind] = 1.0;
                }
                if (j > 0)
                {
                    ind2 = ind - l;
                    matrix[ind][ind2] = 1.0;
                    matrix[ind2][ind] = 1.0;
                }
                if (k > 0)
                {
                    ind2 = ind - l * m;
                    matrix[ind][ind2] = 1.0;
                    matrix[ind2][ind] = 1.0;
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
                        matrix[ind1][ind2] = 1.0;
                        matrix[ind2][ind1] = 1.0;
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
                        matrix[ind1][ind2] = 1.0;
                        matrix[ind2][ind1] = 1.0;
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
                        matrix[ind1][ind2] = 1.0;
                        matrix[ind2][ind1] = 1.0;
                    }
                }
            }
        }
    }
    
    return matrix;
}


//
// Part 2: Tutorial code
//



int main(int, char **)
{
  srand(42);
  std::cout << "-- Generating matrix --" << std::endl;
  std::size_t dof_per_dim = 64;   //number of grid points per coordinate direction
  std::size_t n = dof_per_dim * dof_per_dim * dof_per_dim; //total number of unknowns
  std::vector< std::map<int, double> > matrix = gen_3d_mesh_matrix(dof_per_dim, dof_per_dim, dof_per_dim, false);  //If last parameter is 'true', a tetrahedral grid instead of a hexahedral grid is used.
  
  //
  // Shuffle the generated matrix
  //
  std::vector<int> r = generate_random_reordering(n);
  std::vector< std::map<int, double> > matrix2 = reorder_matrix(matrix, r);
  
  
  //
  // Print some statistics:
  //
  std::cout << " * Unknowns: " << n << std::endl;
  std::cout << " * Initial bandwidth: " << calc_bw(matrix) << std::endl;
  std::cout << " * Randomly reordered bandwidth: " << calc_bw(matrix2) << std::endl;

  //
  // Reorder using Cuthill-McKee algorithm
  //
  std::cout << "-- Cuthill-McKee algorithm --" << std::endl;
  r = viennacl::reorder(matrix2, viennacl::cuthill_mckee_tag());
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;
  
  //
  // Reorder using advanced Cuthill-McKee algorithm
  //
  std::cout << "-- Advanced Cuthill-McKee algorithm --" << std::endl;
  double a = 0.0;
  std::size_t gmax = 1;
  r = viennacl::reorder(matrix2, viennacl::advanced_cuthill_mckee_tag(a, gmax));
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;
  
  //
  // Reorder using Gibbs-Poole-Stockmeyer algorithm
  //
  std::cout << "-- Gibbs-Poole-Stockmeyer algorithm --" << std::endl;
  r = viennacl::reorder(matrix2, viennacl::gibbs_poole_stockmeyer_tag());
  std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;
    
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
    
  return EXIT_SUCCESS;
}
