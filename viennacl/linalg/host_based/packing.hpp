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

  /* packs matrix A */
  template<typename NumericT>
  void pack_matrix_A(std::vector<NumericT> & buffer, vcl_size_t offset_i, vcl_size_t offset_j, 
                     vcl_size_t mc, vcl_size_t kc, vcl_size_t mr,
                     NumericT const * data, 
                     vcl_size_t size1, vcl_size_t size2, 
                     vcl_size_t internal_size1, vcl_size_t internal_size2, 
                     vcl_size_t inc1, vcl_size_t inc2,
                     vcl_size_t start1,  vcl_size_t start2,
                     bool trans, bool row_major)
  {
    //vcl_size_t num_A_slivers      = (size1-offset_i) < mc ? (size1-offset_i)/mr : mc/mr;
    vcl_size_t num_remaining_rows = (size1-offset_i) < mc ? (size1-offset_i)%mr :     0;
    /* indices used for zig-zag pattern inside sliver */
    //vcl_size_t l; 
    //vcl_size_t k;

    if (row_major)
    {
      if (!trans)
      {
        for (vcl_size_t i = 0; i < std::min(mc, size1-offset_i); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
          {
            buffer[ (i/mr)*kc*mr + (i%mr) + j*mr ] = data[ ((i+offset_i)*inc1+start1)*internal_size2 + ((j+offset_j)*inc2+start2) ];
            //std::cout << "A: pack idx" << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i*size2 + j << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t i = 0; i < std::min(mc, size1-offset_i); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
          {
            buffer[ i*mr + (j/mr)*kc*mr + (j%mr) ] = data[ ((i+offset_j)*inc1+start1)*internal_size2 + ((j+offset_i)*inc2+start2) ];
            //std::cout << "A: pack idx " << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i + j*size1  << std::endl;//DEBUG
          } 
        }
      }
    }
    else
    {
      if (!trans)
      {
        for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
        {
          for (vcl_size_t i = 0; i < std::min(mc, size1-offset_i); ++i)
          {
            buffer[ (i/mr)*kc*mr + (i%mr) + j*mr ] = data[ ((i+offset_i)*inc1+start1) + ((j+offset_j)*inc2+start2)*internal_size1 ];
            //std::cout << "packed idx A " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t j = 0; j < std::min(kc, size2-offset_i); ++j)
        {
          for (vcl_size_t i = 0; i < std::min(mc, size1-offset_j); ++i)
          {
            buffer[ i*mr + (j/mr)*kc*mr + (j%mr) ] = data[ ((i+offset_j)*inc1+start1) + ((j+offset_i)*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
    /* padd zeros if needed */
    /* if rows are "remaining", 'size1-offset_i' is smaller then mc and 
     * hence the correct starting point for i (j similar). */
    if (num_remaining_rows > 0)
    {
      for (vcl_size_t j = 0; j < kc; ++j)
      {
        for (vcl_size_t i = 0; i < num_remaining_rows; ++i)
        {
          if (!trans)
            buffer[ (i+size1-offset_i) + (j+size2-offset_j)*mr ] = NumericT(0);
          else //switched offsets
            buffer[ (i+size1-offset_j) + (j+size2-offset_i)*mr ] = NumericT(0);
        }
      }
    }          
  }

  /* packs matrix B */
  template<typename NumericT>
  void pack_matrix_B(std::vector<NumericT> & buffer, vcl_size_t offset_i, vcl_size_t offset_j, 
                     vcl_size_t kc, vcl_size_t nc, vcl_size_t nr,
                     NumericT const * data, 
                     vcl_size_t size1, vcl_size_t size2, 
                     vcl_size_t internal_size1, vcl_size_t internal_size2, 
                     vcl_size_t inc1, vcl_size_t inc2,
                     vcl_size_t start1,  vcl_size_t start2,
                     bool trans, bool row_major)
  {
    //vcl_size_t num_B_slivers      = (size2-offset_j) < nc ? (size2-offset_j)/nr : nc/nr;
    vcl_size_t num_remaining_cols = (size2-offset_j) < nc ? (size2-offset_j)%nr :     0;

    if (row_major)
    {
      if (!trans)
      {
        for (vcl_size_t i = 0; i < std::min(kc, size1-offset_i); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(nc, size2-offset_j); ++j)
          {
            buffer[ i*nr + (j/nr)*kc*nr + (j%nr) ] = data[ ((i+offset_i)*inc1+start1)*internal_size2 + ((j+offset_j)*inc2+start2) ];
            //std::cout << "A: pack idx" << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i*size2 + j << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t i = 0; i < std::min(kc, size1-offset_j); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(nc, size2-offset_i); ++j)
          {
            buffer[ (i/nr)*kc*nr + (i%nr) + j*nr ] = data[ ((i+offset_j)*inc1+start1)*internal_size2 + ((j+offset_i)*inc2+start2) ];
            //std::cout << "A: pack idx " << (i*inc1+start1)*internal_size2 + j*inc2+start2 << " --> " << i + j*size1  << std::endl;//DEBUG
          } 
        }
      }
    }
    else
    {
      if (!trans)
      {
        for (vcl_size_t j = 0; j < std::min(nc, size2-offset_j); ++j)
        {
          for (vcl_size_t i = 0; i < std::min(kc, size1-offset_i); ++i)
          {
            buffer[ i*nr + (j/nr)*kc*nr + (j%nr) ] = data[ ((i+offset_i)*inc1+start1) + ((j+offset_j)*inc2+start2)*internal_size1 ];
            //std::cout << "packed idx A " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
      else //attention: switched offsets intended! see top comment...
      {
        for (vcl_size_t j = 0; j < std::min(nc, size2-offset_i); ++j)
        {
          for (vcl_size_t i = 0; i < std::min(kc, size1-offset_j); ++i)
          {
            buffer[ (i/nr)*kc*nr + (i%nr) + j*nr ] = data[ ((i+offset_j)*inc1+start1) + ((j+offset_i)*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
    /* padd zeros if needed */
    /* if rows are "remaining", 'size1-offset_i' is smaller then mc and 
     * hence the correct starting point for i (j similar). */
    if (num_remaining_cols > 0)
    {
      for (vcl_size_t i = 0; i < kc; ++i)
      {
        for (vcl_size_t j = 0; j < num_remaining_cols; ++j)
        {
          if (!trans)
            buffer[ (i+size1-offset_i)*nr + (j+size2-offset_j) ] = NumericT(0);
          else //switched offsets
            buffer[ (i+size1-offset_j)*nr + (j+size2-offset_i) ] = NumericT(0);
        }
      }
    }          
  }


  /* OLD FUNCTIONS! of any use? */
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
