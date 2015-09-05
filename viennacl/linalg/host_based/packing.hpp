#ifndef VIENNACL_LINALG_HOST_BASED_PACKING_HPP_
#define VIENNACL_LINALG_HOST_BASED_PACKING_HPP_

#include "viennacl/forwards.h"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/common.hpp"


/* SWITCHED OFFSETS FOR TRANSPOSED MATRICES: 
 * if a matrix in (mat-mat) prod() is transposed (e.g. prod(trans(A),B)), matrix A is left unaltered in any way (no memory movements!). 
 * Only A_trans is set to true and therefore A's sizes index the original, i.e. non-transposed, matrix A.
 * On the other hand, if A is transposed the offsets address blocks of that matrix, as if it were moved in memory accordingly.
 * Hence, we have to 'switch' the offsets for the transposed case.
 */

namespace viennacl
{
  template<typename NumericT>
  void pack_matrix_in_row_major(std::vector<NumericT> & buffer, vcl_size_t offset_i, vcl_size_t offset_j, vcl_size_t blocksize, 
                                NumericT const * data, 
                                vcl_size_t size1, vcl_size_t size2, 
                                vcl_size_t internal_size1, vcl_size_t internal_size2, 
                                vcl_size_t inc1, vcl_size_t inc2,
                                vcl_size_t start1,  vcl_size_t start2,
                                bool trans, bool row_major)
  {
    if (row_major)
    {
      if (!trans)
      {
        for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, size1); ++i)
        {
          for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, size2); ++j)
          {
            buffer[ (i-offset_i)*blocksize + (j-offset_j) ] = data[ (i*inc1+start1)*internal_size2 + (j*inc2+start2) ];
            //std::cout << "A: pack idx" << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i*size2 + j << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t i = offset_j; i < std::min(offset_j + blocksize, size1); ++i)
        {
          for (vcl_size_t j = offset_i; j < std::min(offset_i + blocksize, size2); ++j)
          {
            buffer[ (i-offset_j) + (j-offset_i)*blocksize ] = data[ (i*inc1+start1)*internal_size2 + (j*inc2+start2) ];
            //std::cout << "A: pack idx " << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i + j*size1  << std::endl;//DEBUG
          } 
        }
      }
    }
    else
    {
      if (!trans)
      {
        for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, size2); ++j)
        {
          for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, size1); ++i)
          {

            buffer[ (i-offset_i)*blocksize + (j-offset_j) ] = data[ (i*inc1+start1) + (j*inc2+start2)*internal_size1 ];
            //std::cout << "packed idx A " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t j = offset_i; j < std::min(offset_i + blocksize, size2); ++j)
        {
          for (vcl_size_t i = offset_j; i < std::min(offset_j + blocksize, size1); ++i)
          {
            buffer[ (i-offset_j) + (j-offset_i)*blocksize ] = data[ (i*inc1+start1) + (j*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
  }

  template<typename NumericT>
  void pack_matrix_in_col_major(std::vector<NumericT> & buffer, vcl_size_t offset_i, vcl_size_t offset_j, vcl_size_t blocksize, 
                                NumericT const * data, 
                                vcl_size_t size1, vcl_size_t size2, 
                                vcl_size_t internal_size1, vcl_size_t internal_size2, 
                                vcl_size_t inc1, vcl_size_t inc2,
                                vcl_size_t start1,  vcl_size_t start2,
                                bool trans, bool row_major)
  {
    if (row_major)
    {
      if (!trans)
      {
        for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, size1); ++i)
        {
          for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, size2); ++j)
          {
            buffer[ (i-offset_i) + (j-offset_j)*blocksize ] = data[ (i*inc1+start1)*internal_size2 + (j*inc2+start2) ];
            //std::cout << "A: pack idx" << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i*size2 + j << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t i = offset_j; i < std::min(offset_j + blocksize, size1); ++i)
        {
          for (vcl_size_t j = offset_i; j < std::min(offset_i + blocksize, size2); ++j)
          {
            buffer[ (i-offset_j)*blocksize + (j-offset_i) ] = data[ (i*inc1+start1)*internal_size2 + (j*inc2+start2) ];
            //std::cout << "A: pack idx " << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i + j*size1  << std::endl;//DEBUG
          } 
        }
      }
    }
    else
    {
      if (!trans)
      {
        for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, size2); ++j)
        {
          for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, size1); ++i)
          {

            buffer[ (i-offset_i) + (j-offset_j)*blocksize ] = data[ (i*inc1+start1) + (j*inc2+start2)*internal_size1 ];
            //std::cout << "packed idx A " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t j = offset_i; j < std::min(offset_i + blocksize, size2); ++j)
        {
          for (vcl_size_t i = offset_j; i < std::min(offset_j + blocksize, size1); ++i)
          {
            buffer[ (i-offset_j)*blocksize + (j-offset_i) ] = data[ (i*inc1+start1) + (j*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
  }
}//viennacl
#endif
