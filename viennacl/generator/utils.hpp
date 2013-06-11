#ifndef VIENNACL_GENERATOR_UTILS_HPP
#define VIENNACL_GENERATOR_UTILS_HPP

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


/** @file viennacl/generator/utils.hpp
    @brief Some utils for the generator. These are rather general purpose and should end up being merged in viennacl/tools/
*/

#include <sstream>
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/generator/forwards.h"
#include <vector>
#include <list>

namespace viennacl{

  namespace generator{

    namespace utils{

      template<class T>
      struct deref_eq : std::binary_function<T,T,bool>{
          bool operator()(T a, T b) const {
            return *a == *b;
          }
      };

      template <class T>
      inline std::string to_string ( T const t )
      {
        std::stringstream ss;
        ss << t;
        return ss.str();
      }

      //Are same type
      template<class T, class U>
      struct are_same_type
      {
          enum { value = 0 };
      };

      template<class T>
      struct are_same_type<T,T>
      {
          enum { value = 1 };
      };

      template<class Base,class Target>
      struct Base2Target { Target* operator ()( Base* value ) const { return dynamic_cast<Target*>(value); } };

      template<class T>
      struct SharedPtr2Raw {
          T* operator ()(viennacl::tools::shared_ptr<T> const & value ) const { return value.get(); }
      };


      template<class Base,class Target>
      struct UnsafeBase2Target { Target* operator ()( Base* value ) const { return static_cast<Target*>(value); } };

      inline void replace_all_occurences(std::string& str, const std::string& oldstr, const std::string& newstr)
      {
        size_t pos = 0;
        while((pos = str.find(oldstr, pos)) != std::string::npos)
        {
          str.replace(pos, oldstr.length(), newstr);
          pos += newstr.length();
        }
      }

      template<class T>
      struct print_type;

      template<>
      struct print_type<char>
      {
          static const char * value() { return "char"; }
      };

      template<>
      struct print_type<unsigned char>
      {
          static const char * value() { return "unsigned char"; }
      };


      template<>
      struct print_type<int>
      {
          static const char * value() { return "int"; }
      };

      template<>
      struct print_type<unsigned int>
      {
          static const char * value() { return "unsigned int"; }
      };

      template<>
      struct print_type<short>
      {
          static const char * value() { return "short"; }
      };

      template<>
      struct print_type<unsigned short>
      {
          static const char * value() { return "unsigned short"; }
      };

      template<>
      struct print_type<long>
      {
          static const char * value() { return "long"; }
      };

      template<>
      struct print_type<unsigned long>
      {
          static const char * value() { return "unsigned long"; }
      };

      template<>
      struct print_type<float>
      {
          static const char * value() { return "float"; }
      };

      template<>
      struct print_type<double>
      {
          static const char * value() { return "double"; }
      };


      class kernel_generation_stream : public std::ostream{
        private:
          class kgenstream : public std::stringbuf{
            public:
              kgenstream(std::ostream& final_destination
                         ,unsigned int const & tab_count) : final_destination_(final_destination)
              ,tab_count_(tab_count){ }
              ~kgenstream() {  pubsync(); }
              int sync() {
                for(unsigned int i=0 ; i<tab_count_;++i)
                  final_destination_ << '\t';
                final_destination_ << str();
                str("");
                return !final_destination_;
              }
            private:
              std::ostream& final_destination_;
              unsigned int const & tab_count_;
          };

        public:
          kernel_generation_stream(std::ostream& final_destination) : std::ostream(new kgenstream(final_destination,tab_count_))
          , tab_count_(0){ }
          ~kernel_generation_stream(){ delete rdbuf(); }
          std::string str(){
            return static_cast<std::stringbuf*>(rdbuf())->str();
          }

          void inc_tab(){ ++tab_count_; }
          void dec_tab(){ --tab_count_; }

        private:
          unsigned int tab_count_;
      };

      template<class U, class T>
      void unique_push_back(U & v, T t){
        if(std::find_if(v.begin(), v.end(), std::bind1st(deref_eq<T>(),t))==v.end())
          v.push_back(t);
      }

      template<class T, class U>
      typename std::vector<std::pair<T*, U> >::iterator unique_insert(std::vector<std::pair<T*, U> > & v, std::pair<T*, U> p){
        typename std::vector<std::pair<T*, U> >::iterator res = v.begin();
        while(res != v.end()){
          if(*res->first == *p.first) return res;
          ++res;
        }
        return v.insert(res,p);
      }

      template<class T>
      struct is_type{
          template<class U>
          bool operator()(U* p) const{
            return dynamic_cast<T *>(p);
          }
      };



    }
  }
}
#endif
