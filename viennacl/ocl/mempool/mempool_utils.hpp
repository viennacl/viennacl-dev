// Various odds and ends
//
// Copyright (C) 2009 Andreas Kloeckner
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


#ifndef VIENNACL_MEMPOOL_UTILS_HPP
#define VIENNACL_MEMPOOL_UTILS_HPP

#include <stdexcept>
#include <CL/cl.h>



namespace viennacl
{
namespace mempool
{

  // {{{ error
  class error : public std::runtime_error
  {
    private:
      std::string m_routine;
      cl_int m_code;

      // This is here because clLinkProgram returns a program
      // object *just* so that there is somewhere for it to
      // stuff the linker logs. :/
      bool m_program_initialized;
      cl_program m_program;

    public:
      error(const char *routine, cl_int c, const char *msg="")
        : std::runtime_error(msg), m_routine(routine), m_code(c),
        m_program_initialized(false), m_program(nullptr)
      { }

      error(const char *routine, cl_program prg, cl_int c,
          const char *msg="")
        : std::runtime_error(msg), m_routine(routine), m_code(c),
        m_program_initialized(true), m_program(prg)
      { }

      virtual ~error()
      {
        if (m_program_initialized)
          clReleaseProgram(m_program);
      }

      const std::string &routine() const
      {
        return m_routine;
      }

      cl_int code() const
      {
        return m_code;
      }

      bool is_out_of_memory() const
      {
        return (code() == CL_MEM_OBJECT_ALLOCATION_FAILURE
            || code() == CL_OUT_OF_RESOURCES
            || code() == CL_OUT_OF_HOST_MEMORY);
      }
  };

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

  // https://stackoverflow.com/a/44175911
  class noncopyable {
  public:
    noncopyable() = default;
    ~noncopyable() = default;

  private:
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
  };

}
}

#endif

// vim:foldmethod=marker
