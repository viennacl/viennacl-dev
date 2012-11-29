#ifndef VIENNACL_TOOLS_SHARED_PTR_HPP
#define VIENNACL_TOOLS_SHARED_PTR_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file tools/shared_ptr.hpp
    @brief Implementation of a shared pointer class (cf. std::shared_ptr, boost::shared_ptr). Will be used until C++11 is widely available.
    
    Contributed by Philippe Tillet.
*/

namespace viennacl
{
  namespace tools
  {

    namespace detail
    {

      class count
      {
        public:
          count(unsigned int val) : val_(val){ }
          void dec(){ --val_; }
          void inc(){ ++val_; }
          bool is_null(){ return val_ == 0; }
          unsigned int val(){ return val_; }
        private:
          unsigned int val_;
      };

    }

    /** @brief A shared pointer class similar to boost::shared_ptr. Reimplemented in order to avoid a Boost-dependency. Will be replaced by std::shared_ptr as soon as C++11 is widely available. */
    template<class T>
    class shared_ptr
    {
        struct aux
        {
          detail::count count;

          aux() :count(1) {}
          virtual void destroy()=0;
          virtual ~aux() {}
        };

        template<class U, class Deleter>
        struct auximpl: public aux
        {
          U* p;
          Deleter d;

          auximpl(U* pu, Deleter x) :p(pu), d(x) {}
          virtual void destroy() { d(p); }
        };

        template<class U>
        struct default_deleter
        {
          void operator()(U* p) const { delete p; }
        };

        aux* pa;
        T* pt;

        void inc() { if(pa) pa->count.inc(); }

        void dec()
        {
          if(pa)
          {
            pa->count.dec();
            
            if(pa->count.is_null())
            {
                pa->destroy();
                delete pa;
                pa = NULL;
            }
          }
        }

      public:

        shared_ptr() :pa(NULL), pt(NULL) {}

        template<class U, class Deleter>
        shared_ptr(U* pu, Deleter d) : pa(new auximpl<U, Deleter>(pu, d)), pt(pu) {}

        template<class U>
        explicit shared_ptr(U* pu) : pa(new auximpl<U, default_deleter<U> >(pu, default_deleter<U>())), pt(pu) {}

        shared_ptr(const shared_ptr& s) :pa(s.pa), pt(s.pt) { inc(); }

        template<class U>
        shared_ptr(const shared_ptr<U>& s) :pa(s.pa), pt(s.pt) { inc(); }

        ~shared_ptr() { dec(); }

        shared_ptr& operator=(const shared_ptr& s)
        {
            if(this!=&s)
            {
                dec();
                pa = s.pa;
                pt = s.pt;
                inc();
            }
            return *this;
        }

        T* get() const {  return pt; }

        T& operator*() const { return *pt; }
    };

  }

}

#endif // VIENNACL_UTILS_SHARED_PTR_HPP
