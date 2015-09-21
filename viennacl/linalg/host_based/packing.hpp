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
 * On the other hand, if A is transposed, the offsets address blocks of that matrix, as if it were moved in memory accordingly.
 * Hence, we have to 'switch' the offsets for the transposed case.
 */

namespace viennacl
{
  
  template<typename NumericT>
  void debug_print_package(NumericT *package, std::string name, vcl_size_t size1, vcl_size_t size2)
    {
      std::cout << std::endl << "printing package " << name << " : ";
      for (vcl_size_t i=0; i<(size1*size2); ++i)
      {
        std::cout << package[i] << ", ";
      }
      std::cout << std::endl;
    }
  
  template<typename NumericT>
  void pack_matrix_A(NumericT *buffer, vcl_size_t offset_i, vcl_size_t offset_j, 
                     vcl_size_t mc, vcl_size_t kc, vcl_size_t mr,
                     NumericT const * data, 
                     vcl_size_t size1, vcl_size_t size2, 
                     vcl_size_t internal_size1, vcl_size_t internal_size2, 
                     vcl_size_t inc1, vcl_size_t inc2,
                     vcl_size_t start1,  vcl_size_t start2,
                     bool trans, bool row_major)
  {
    /* see top comment */
    if(trans)
    {
      std::swap(offset_i, offset_j);
      std::swap(mc,kc);
    }
    
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
      else
      {
        for (vcl_size_t i = 0; i < std::min(mc, size1-offset_i); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
          {
            //std::cout << "new block " << size1 << " " << offset_i << " "<< std::min(kc, size2-offset_j) << " " << j << " " << ((i+offset_i)*inc1+start1)*internal_size2 + ((j+offset_j)*inc2+start2) << std::endl;//DEBUG
            /* Note that in '(j/nr)*mc*nr', mc is used instead of kc due to the swap at the beginning of this function! */
            buffer[ i*mr + (j/mr)*mc*mr + (j%mr) ] = data[ ((i+offset_i)*inc1+start1)*internal_size2 + ((j+offset_j)*inc2+start2) ];
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
     else
      {
        for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
        {                                                        
          for (vcl_size_t i = 0; i < std::min(mc, size1-offset_i); ++i)
          {
            /* Note that in '(j/nr)*mc*nr', mc is used instead of kc due to the swap at the beginning of this function! */
            buffer[ i*mr + (j/mr)*mc*mr + (j%mr) ] = data[ ((i+offset_i)*inc1+start1) + ((j+offset_j)*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
    /* padd zeros if needed */
    /* switch sizes when transposed, since they address the unaltered
     * matrix (not transposed). For padding, however, we address entries
     * as if they were transposed. Same thing holds for offsets:
     * as we transposed them at the beginning of this function,
     * they address the unaltered matrix, hence we re-swap them.*/
    if (trans)
    {
      std::swap(size1, size2);
      std::swap(offset_i, offset_j);
      std::swap(mc,kc);
    }
    
    vcl_size_t num_remaining_rows = (size1-offset_i) < mc ? (mc-size1+offset_i)%mr : 0;
    
    if (num_remaining_rows > 0)
    {
      for (vcl_size_t j = 0; j < std::min(kc, size2-offset_j); ++j)
      {
        for (vcl_size_t i = 0; i < num_remaining_rows; ++i)
        {
          //unsigned int index = ((i+size1-offset_i)/mr)*kc*mr + ((i+size1-offset_i)%mr) + j*mr;//DEBUG
          //std::cout << "index padding A is: " << index << std::endl;//DEBUG
          buffer[ ((i+size1-offset_i)/mr)*kc*mr + ((i+size1-offset_i)%mr) + j*mr ] = NumericT(0);
        }
      }
    }
    
    //    debug_print_package(buffer, "A", mc, kc);//DEBUG
  }//pack_matrix_A

  template<typename NumericT>
  void pack_matrix_B(NumericT *buffer, vcl_size_t offset_i, vcl_size_t offset_j, 
                     vcl_size_t kc, vcl_size_t nc, vcl_size_t nr,
                     NumericT const * data, 
                     vcl_size_t size1, vcl_size_t size2, 
                     vcl_size_t internal_size1, vcl_size_t internal_size2, 
                     vcl_size_t inc1, vcl_size_t inc2,
                     vcl_size_t start1,  vcl_size_t start2,
                     bool trans, bool row_major)
  {
    if (trans)
    {
      std::swap(offset_i, offset_j);
      std::swap(kc, nc);
    }
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
      else
      {
        for (vcl_size_t i = 0; i < std::min(kc, size1-offset_i); ++i)
        {
          for (vcl_size_t j = 0; j < std::min(nc, size2-offset_j); ++j)
          {
            /* Note that in '(i/nr)*nc*nr', nc is used instead of kc due to the swap at the beginning of this function! */
            buffer[ (i/nr)*nc*nr + (i%nr) + j*nr ] = data[ ((i+offset_i)*inc1+start1)*internal_size2 + ((j+offset_j)*inc2+start2) ];
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
      else
      {
        for (vcl_size_t j = 0; j < std::min(nc, size2-offset_j); ++j)
        {
          for (vcl_size_t i = 0; i < std::min(kc, size1-offset_i); ++i)
          {
            /* Note that in '(i/nr)*nc*nr', nc is used instead of kc due to the swap at the beginning of this function! */
            buffer[ (i/nr)*nc*nr + (i%nr) + j*nr ] = data[ ((i+offset_i)*inc1+start1) + ((j+offset_j)*inc2+start2)*internal_size1 ];
            //std::cout << "packed A idx " << i*inc1+start1 + (j*inc2+start2)*internal_size1 << std::endl;//DEBUG
          }
        }
      }
    }
    /* padd zeros if needed */
    /* switch sizes when transposed, since they address the unaltered
     * matrix (not transposed). For padding, however, we address entries
     * as if they were transposed. Same thing holds for offsets:
     * since we transposed them at the beginning of this function,
     * they address the unaltered matrix, hence we re-swap them.*/
    if (trans)
    {
      std::swap(size1, size2);
      std::swap(offset_i, offset_j);
      std::swap(kc,nc);
    }
    
    vcl_size_t num_remaining_cols = (size2-offset_j) < nc ? (nc-size2+offset_j)%nr : 0;
    
    if (num_remaining_cols > 0)
    {
      for (vcl_size_t i = 0; i < std::min(kc, size1-offset_i); ++i)
      {
        for (vcl_size_t j = 0; j < num_remaining_cols; ++j)
        {
          buffer[ i*nr + ((j+size2-offset_j)/nr)*kc*nr + ((j+size2-offset_j)%nr) ] = NumericT(0);
        }
      }
    }
    //    debug_print_package(buffer, "B", kc, nc);//DEBUG
  }//pack_matrix_B
}//viennacl
#endif
