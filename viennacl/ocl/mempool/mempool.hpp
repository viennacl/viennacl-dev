// Abstract memory pool implementation
//
// Copyright (C) 2009-17 Andreas Kloeckner
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


#ifndef VIENNACL_MEMPOOL_MEMPOOL_HPP_
#define VIENNACL_MEMPOOL_MEMPOOL_HPP_


#include <cassert>
#include <vector>
#include <map>
#include <memory>
#include <ostream>
#include <iostream>

#include "viennacl/ocl/error.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace viennacl
{
namespace mempool
{

  // {{{ Allocator
  
  class cl_allocator_base
  {
    protected:
      tools::shared_ptr<viennacl::ocl::context> m_context;
      cl_mem_flags m_flags;

    public:
      // NOTE: pyopencl::context -> cl_context
      // Q: Should I make this viennacl::context
      cl_allocator_base(tools::shared_ptr<viennacl::ocl::context> const &ctx,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : m_context(ctx), m_flags(flags)
      {
        if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))
        {
          std::cerr << "[Allocator]: cannot specify USE_HOST_PTR or "
            "COPY_HOST_PTR flags" << std::endl;
          throw viennacl::ocl::invalid_value();
        }
      }

      cl_allocator_base(cl_allocator_base const &src)
      : m_context(src.m_context), m_flags(src.m_flags)
      { }

      virtual ~cl_allocator_base()
      { }

      typedef cl_mem cl_mem;
      typedef size_t size_t;

      virtual cl_allocator_base *copy() const = 0;
      virtual bool is_deferred() const = 0;
      virtual cl_mem allocate(size_t s) = 0;

      void free(cl_mem p)
      {
        cl_int err = clReleaseMemObject(p);
        VIENNACL_ERR_CHECK(err);
      }
      // NOTE: removed the function "try_release_blocks()"
  };

  class cl_deferred_allocator : public cl_allocator_base
  {
    private:
      typedef cl_allocator_base super;

    public:
      cl_deferred_allocator(tools::shared_ptr<viennacl::ocl::context> const &ctx,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : super(ctx, flags)
      { }

      cl_allocator_base *copy() const
      {
        return new cl_deferred_allocator(*this);
      }

      bool is_deferred() const
      { return true; }

      cl_mem allocate(size_t s)
      {
        return m_context->create_memory_without_smart_handle(m_flags, s, NULL);
      }
  };

  class cl_immediate_allocator : public cl_allocator_base
  {
    private:
      typedef cl_allocator_base super;
      tools::shared_ptr<viennacl::ocl::command_queue> const & m_queue;

    public:
      // NOTE: Changed the declaration as viennacl comman=d queue does nt store
      // the context
      cl_immediate_allocator(tools::shared_ptr<viennacl::ocl::context> const &ctx,
          tools::shared_ptr<viennacl::ocl::command_queue> const &queue,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : super(tools::shared_ptr<viennacl::ocl::context>(ctx), flags),
        m_queue(queue)
      { }

      cl_immediate_allocator(cl_immediate_allocator const &src)
        : super(src), m_queue(src.m_queue)
      { }

      cl_allocator_base *copy() const
      {
        return new cl_immediate_allocator(*this);
      }

      bool is_deferred() const
      { return false; }

      cl_mem allocate(size_t s)
      {
        cl_mem ptr =
          m_context->create_memory_without_smart_handle(m_flags, s, NULL);

        // Make sure the buffer gets allocated right here and right now.
        // This looks (and is) expensive. But immediate allocators
        // have their main use in memory pools, whose basic assumption
        // is that allocation is too expensive anyway--but they rely
        // on exact 'out-of-memory' information.
        unsigned zero = 0;
        cl_int err = clEnqueueWriteBuffer(
              m_queue->handle().get(),
              ptr,
              /* is blocking */ CL_FALSE,
              0, std::min(s, sizeof(zero)), &zero,
              0, NULL, NULL
              );
        VIENNACL_ERR_CHECK(err);

        // No need to wait for completion here. clWaitForEvents (e.g.)
        // cannot return mem object allocation failures. This implies that
        // the buffer is faulted onto the device on enqueue.

        return ptr;
      }
  };

  inline
  cl_mem allocator_call(cl_allocator_base &alloc, size_t size)
  {
    cl_mem mem;
    int try_count = 0;
    while (try_count < 2)
    {
      try
      {
        mem = alloc.allocate(size);
        break;
      }
      catch (viennacl::ocl::mem_object_allocation_failure &e)
      {
        if (++try_count == 2)
          throw;
      }

      //NOTE: There was a try_release blocks over here
      // which I got rid off. Is that fine?

      // alloc.try_release_blocks();
    }

    try
    {
      // Note: PyOpenCL retains this buffer, however in ViennaCL, there
      // doesn't seem to be any option to not retain it.
      return mem;
    }
    catch (...)
    {
      cl_int err = clReleaseMemObject(mem);
      VIENNACL_ERR_CHECK(err);
      throw;
    }
  }
  // }}}

  template <class T>
  inline T signed_left_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x >> -shift_amount;
    else
      return x << shift_amount;
  }

  template <class T>
  inline T signed_right_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x << -shift_amount;
    else
      return x >> shift_amount;
  }




  template<class Allocator>
  class memory_pool : mempool::noncopyable
  {
    private:
      std::map<uint32_t, std::vector<cl_mem>> m_container;

      std::unique_ptr<Allocator> m_allocator;

      // A held block is one that's been released by the application, but that
      // we are keeping around to dish out again.
      unsigned m_held_blocks;

      // An active block is one that is in use by the application.
      unsigned m_active_blocks;

      bool m_stop_holding;
      int m_trace;

    public:
      memory_pool(Allocator const &alloc=Allocator())
        : m_allocator(alloc.copy()),
        m_held_blocks(0), m_active_blocks(0), m_stop_holding(false),
        m_trace(false)
      {
        if (m_allocator->is_deferred())
        {
          throw std::runtime_error("Memory pools expect non-deferred "
              "semantics from their allocators. You passed a deferred "
              "allocator, i.e. an allocator whose allocations can turn out to "
              "be unavailable long after allocation.");
        }
      }

      virtual ~memory_pool()
      { free_held(); }

      static const unsigned mantissa_bits = 2;
      static const unsigned mantissa_mask = (1 << mantissa_bits) - 1;

      static uint32_t bin_number(size_t size)
      {
        signed l = bitlog2(size);
        size_t shifted = signed_right_shift(size, l-signed(mantissa_bits));
        if (size && (shifted & (1 << mantissa_bits)) == 0)
          throw std::runtime_error("memory_pool::bin_number: bitlog2 fault");
        size_t chopped = shifted & mantissa_mask;
        return l << mantissa_bits | chopped;
      }

      void set_trace(bool flag)
      {
        if (flag)
          ++m_trace;
        else
          --m_trace;
      }

      static size_t alloc_size(uint32_t bin)
      {
        uint32_t exponent = bin >> mantissa_bits;
        uint32_t mantissa = bin & mantissa_mask;

        size_t ones = signed_left_shift(1,
            signed(exponent)-signed(mantissa_bits)
            );
        if (ones) ones -= 1;

        size_t head = signed_left_shift(
           (1<<mantissa_bits) | mantissa,
            signed(exponent)-signed(mantissa_bits));
        if (ones & head)
          throw std::runtime_error("memory_pool::alloc_size: bit-counting fault");
        return head | ones;
      }

    protected:
      std::vector<cl_mem> &get_bin(uint32_t bin_nr)
      {
        typename std::map<uint32_t, std::vector<cl_mem>>::iterator it = m_container.find(bin_nr);
        if (it == m_container.end())
        {
          auto it_and_inserted = m_container.insert(std::make_pair(bin_nr, std::vector<cl_mem>()));
          assert(it_and_inserted.second);
          return it_and_inserted.first->second;
        }
        else
          return it->second;
      }

      void inc_held_blocks()
      {
        if (m_held_blocks == 0)
          start_holding_blocks();
        ++m_held_blocks;
      }

      void dec_held_blocks()
      {
        --m_held_blocks;
        if (m_held_blocks == 0)
          stop_holding_blocks();
      }

      virtual void start_holding_blocks()
      { }

      virtual void stop_holding_blocks()
      { }

    public:
      cl_mem allocate(size_t size)
      {
        uint32_t bin_nr = bin_number(size);
        std::vector<cl_mem> &bin = get_bin(bin_nr);

        if (bin.size())
        {
          if (m_trace)
            std::cout
              << "[pool] allocation of size " << size << " served from bin " << bin_nr
              << " which contained " << bin.size() << " entries" << std::endl;
          return pop_block_from_bin(bin, size);
        }

        size_t alloc_sz = alloc_size(bin_nr);

        assert(bin_number(alloc_sz) == bin_nr);

        if (m_trace)
          std::cout << "[pool] allocation of size " << size << " required new memory" << std::endl;

        try { return get_from_allocator(alloc_sz); }
        catch (mempool::error &e)
        {
          if (!e.is_out_of_memory())
            throw;
        }

        if (m_trace)
          std::cout << "[pool] allocation triggered OOM, running GC" << std::endl;

        m_allocator->try_release_blocks();
        if (bin.size())
          return pop_block_from_bin(bin, size);

        if (m_trace)
          std::cout << "[pool] allocation still OOM after GC" << std::endl;

        while (try_to_free_memory())
        {
          try { return get_from_allocator(alloc_sz); }
          catch (mempool::error &e)
          {
            if (!e.is_out_of_memory())
              throw;
          }
        }

        std::cerr << "memory_pool::allocate "
            "failed to free memory for allocation" << std::endl;
        throw viennacl::ocl::mem_object_allocation_failure();

      }

      void free(cl_mem p, size_t size)
      {
        --m_active_blocks;
        uint32_t bin_nr = bin_number(size);

        if (!m_stop_holding)
        {
          inc_held_blocks();
          get_bin(bin_nr).push_back(p);

          if (m_trace)
            std::cout << "[pool] block of size " << size << " returned to bin "
              << bin_nr << " which now contains " << get_bin(bin_nr).size()
              << " entries" << std::endl;
        }
        else
          m_allocator->free(p);
      }

      void free_held()
      {
        for (std::map<uint32_t, std::vector<cl_mem>>::value_type &bin_pair: m_container)
        {
          std::vector<cl_mem> &bin = bin_pair.second;

          while (bin.size())
          {
            m_allocator->free(bin.back());
            bin.pop_back();

            dec_held_blocks();
          }
        }

        assert(m_held_blocks == 0);
      }

      void stop_holding()
      {
        m_stop_holding = true;
        free_held();
      }

      unsigned active_blocks()
      { return m_active_blocks; }

      unsigned held_blocks()
      { return m_held_blocks; }

      bool try_to_free_memory()
      {
        // free largest stuff first
        for (std::map<uint32_t, std::vector<cl_mem>>::value_type &bin_pair: reverse(m_container))
        {
          std::vector<cl_mem> &bin = bin_pair.second;

          if (bin.size())
          {
            m_allocator->free(bin.back());
            bin.pop_back();

            dec_held_blocks();

            return true;
          }
        }

        return false;
      }

    private:
      cl_mem get_from_allocator(size_t alloc_sz)
      {
        cl_mem result = m_allocator->allocate(alloc_sz);
        ++m_active_blocks;

        return result;
      }

      cl_mem pop_block_from_bin(std::vector<cl_mem> &bin, size_t size)
      {
        cl_mem result = bin.back();
        bin.pop_back();

        dec_held_blocks();
        ++m_active_blocks;

        return result;
      }
  };


  template <class Pool>
  class pooled_allocation : public mempool::noncopyable
  {
    public:
      typedef Pool pool_type;
      typedef typename Pool::cl_mem cl_mem;
      typedef typename Pool::size_t size_t;

    private:
      tools::shared_ptr<pool_type> m_pool;

      cl_mem m_ptr;
      size_t m_size;
      bool m_valid;

    public:
      pooled_allocation(tools::shared_ptr<pool_type> p, size_t size)
        : m_pool(p), m_ptr(p->allocate(size)), m_size(size), m_valid(true)
      { }

      ~pooled_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          m_pool->free(m_ptr, m_size);
          m_valid = false;
        }
        else
          throw mempool::error(
              "pooled_device_allocation::free",
              CL_INVALID_VALUE
              );
      }

      cl_mem ptr() const
      { return m_ptr; }

      size_t size() const
      { return m_size; }
  };
}
}

#endif
