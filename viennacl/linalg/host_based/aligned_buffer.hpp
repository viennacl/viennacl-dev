#ifndef VIENNACL_LINALG_HOST_BASED_ALIGNED_BUFFER_HPP_
#define VIENNACL_LINALG_HOST_BASED_ALIGNED_BUFFER_HPP_


namespace viennacl
{
  /**
   * @brief allocates aligned memory, e.g. for AVX 32-byte aligned
   */
  template<typename NumericT>
  inline NumericT *get_aligned_buffer(vcl_size_t size, bool fill_zeros)
  {
#ifdef VIENNACL_WITH_AVX
    
    void *ptr;
    int error = posix_memalign(&ptr, 32, size*sizeof(NumericT));
    
    if (error)
    {
      //dunno how to handle, throw exception? which one?
    }
    if (fill_zeros)
    {
      for (vcl_size_t i=0; i<size; ++i)
      {
        ((NumericT *)ptr)[i] = NumericT(0);
      }
    }
        
    return (NumericT *)ptr;
    
#else
    NumericT *ptr = (NumericT *)malloc(size*sizeof(NumericT));
    
    if (ptr == NULL)
    {
      //dunno how to handle, throw exception? which one?
    }
    if (fill_zeros)
    {
      for (vcl_size_t i=0; i<size; ++i)
      {
        ptr[i] = NumericT(0);
      }
    }
    return (NumericT *)ptr;
#endif    
  }

  /**
   * @brief frees a previously allocated and aligned buffer
   */
  template<typename NumericT>
  inline void free_aligned_buffer(NumericT *ptr)
  {
    free((void *)ptr);
  }
}//viennacl
#endif
