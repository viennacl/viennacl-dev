#ifndef VIENNACL_GENERATOR_AUTOTUNE_HPP
#define VIENNACL_GENERATOR_AUTOTUNE_HPP


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


/** @file viennacl/generator/autotune.hpp
 *
 * User interface for the autotuning procedure
*/

#include <ctime>
#include <iomanip>
#include <cmath>
#include <iterator>

#include "viennacl/generator/forwards.h"
#include "viennacl/generator/code_generation.hpp"
#include "viennacl/generator/builtin_database.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/infos.hpp"

namespace viennacl{

  namespace generator{

    namespace autotune{

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

      class Timer
      {
        public:

          Timer()
          {
            QueryPerformanceFrequency(&freq);
          }

          void start()
          {
            QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
          }

          double get() const
          {
            LARGE_INTEGER  end_time;
            QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
            return (static_cast<double>(end_time.QuadPart) - static_cast<double>(start_time.QuadPart)) / static_cast<double>(freq.QuadPart);
          }


        private:
          LARGE_INTEGER freq;
          LARGE_INTEGER start_time;
      };

#else

#include <sys/time.h>

      class Timer
      {
        public:

          Timer() : ts(0)
          {}

          void start()
          {
            struct timeval tval;
            gettimeofday(&tval, NULL);
            ts = tval.tv_sec * 1000000 + tval.tv_usec;
          }

          double get() const
          {
            struct timeval tval;
            gettimeofday(&tval, NULL);
            double end_time = tval.tv_sec * 1000000 + tval.tv_usec;

            return static_cast<double>(end_time-ts) / 1000000.0;
          }

        private:
          double ts;
      };


#endif


      /** @brief class for a tuning parameter */
      class tuning_param{
        public:

          /** @brief The constructor
           *
           *  @param min minimal value
           *  @param max maximal value
           *  @param policy for increasing the tuning parameter
           */
          tuning_param(std::vector<int> const & values) : values_(values){
              reset();
          }

          /** @brief Returns true if the parameter has reached its maximum value */
          bool is_max() const {
            return current_ ==  (values_.size()-1);
          }

          /** @brief Increments the parameter */
          bool inc(){
            ++current_ ;
            if(current_ < values_.size() )
              return false;
            reset();
            return true;
          }

          /** @brief Returns the current value of the parameter */
          int current() const{
              return values_[current_];
          }

          /** @brief Resets the parameter to its minimum value */
          void reset() {
              current_ = 0;
          }

        private:
          std::vector<int> values_;
          unsigned int current_;
      };

      /** @brief Tuning configuration
       *
       *  ConfigT must have a profile_t typedef
       *  ConfigT must implement is_invalid that returns whether or not a given parameter is invalid
       *  ConfigT must implement create_profile that creates a profile_t given a set of parameters
       *
       *  Parameters are stored in a std::map<std::string, viennacl::generator::autotune::tuning_param>
       */
      template<class ConfigT>
      class tuning_config{
        private:
          /** @brief Storage type of the parameters */
          typedef std::map<std::string, viennacl::generator::autotune::tuning_param> params_t;

        public:
          /** @brief Accessor for profile_t */
          typedef typename ConfigT::profile_t profile_t;

          /** @brief Add a tuning parameter to the config */
          void add_tuning_param(std::string const & name, std::vector<int> const & values){
            params_.insert(std::make_pair(name,values));
          }

          /** @brief Returns true if the tuning config has still not explored all its possibilities */
          bool has_next() const{
            bool res = false;
            for(typename params_t::const_iterator it = params_.begin() ; it != params_.end() ; ++it)
              res = res || !it->second.is_max();
            return res;
          }

          /** @brief Update the parameters of the config */
          void update(){
            for(typename params_t::iterator it = params_.begin() ; it != params_.end() ; ++it)
              if(it->second.inc()==false)
                break;
          }

          /** @brief Returns true if the compilation/execution of the underlying profile has an undefined behavior */
          bool is_invalid(viennacl::ocl::device const & dev) const{
            return ConfigT::is_invalid(dev,params_);
          }

          /** @brief Returns the current profile */
          typename ConfigT::profile_t get_current(){
            return ConfigT::create_profile(params_);
          }

          /** @brief Reset the config */
          void reset(){
            for(params_t::iterator it = params_.begin() ; it != params_.end() ; ++it){
              it->second.reset();
            }
          }

        private:
          params_t params_;
      };

      /** @brief Add the timing value for a given profile and an operation */
      template<class OpT, class ProfileT>
      void benchmark_impl(std::map<double, ProfileT> & timings, viennacl::ocl::device const & dev, OpT const & operation, code_generation::profile_id const & id, ProfileT const & prof){

        Timer t;

        unsigned int n_runs = 10;

        //Skips if use too much local memory.
        viennacl::generator::custom_operation op;
        op.add(operation);
        op.force_profile(id,prof);
        viennacl::ocl::program & pgm = op.program();
        viennacl::ocl::kernel & k = pgm.get_kernel("_k0");


        //Anticipates kernel failure
        size_t max_workgroup_size = viennacl::ocl::info<CL_KERNEL_WORK_GROUP_SIZE>(k,dev);
        std::pair<size_t,size_t> work_group_sizes = prof.local_work_size();
        if(work_group_sizes.first*work_group_sizes.second > max_workgroup_size)  return;

        //Doesn't execute because it would likelily be a waste of time
        size_t prefered_workgroup_size_multiple = viennacl::ocl::info<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(k,dev);
        if( (prof.local_work_size().first*prof.local_work_size().second) % prefered_workgroup_size_multiple > 0) return;

        op.execute(true);
        viennacl::backend::finish();

        double exec_time = 0;
        t.start();
        for(unsigned int n=0; n<n_runs ; ++n){
          op.execute();
        }
        viennacl::backend::finish();
        exec_time = t.get()/(float)n_runs;
        timings.insert(std::make_pair(exec_time, ProfileT(prof)));
      }

      /** @brief Fills a timing map for a given operation and a benchmark configuration
       *
       * @tparam OpT type of the operation
       * @tparam ConfigT type of the benchmark configuration
       * @param timings the timings to fill
       * @param op the given operation
       * @param the given config */
      template<class OpT, class ConfigT>
      void benchmark(std::map<double, typename ConfigT::profile_t> & timings, OpT const & op, code_generation::profile_id const & id, ConfigT & config){
        viennacl::ocl::device const & dev = viennacl::ocl::current_device();

        unsigned int n=0, n_conf = 0;
        while(config.has_next()){
          config.update();
          if(config.is_invalid(dev)) continue;
          ++n_conf;
        }

        config.reset();
        while(config.has_next()){
          config.update();
          if(config.is_invalid(dev)) continue;
          ++n;
          std::cout << '\r' << "Test " << n << "/" << n_conf << " [" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << (double)n*100/n_conf << "%" << "]" << std::flush;
          benchmark_impl(timings,dev,op,id,config.get_current());
        }

        std::cout << std::endl;
      }

      /** @brief Fills a timing map for a given operation and a list of profiles */
      template<class OpT, class ProfT>
      void benchmark(std::map<double, ProfT> & timings, OpT const & op, code_generation::profile_id const & id, std::list<ProfT> const & profiles, size_t scalartype_size){
        viennacl::ocl::device const & dev = viennacl::ocl::current_device();

        unsigned int n=0;
        unsigned int n_conf = 0;

        for(typename std::list<ProfT>::const_iterator it = profiles.begin(); it!=profiles.end(); ++it){
          if(it->is_invalid(dev,scalartype_size)) continue;
          ++n_conf;
        }

        for(typename std::list<ProfT>::const_iterator it = profiles.begin(); it!=profiles.end(); ++it){
          if(it->is_invalid(dev,scalartype_size)) continue;
          std::cout << '\r' << "Test " << n << "/" << n_conf << " [" << std::setprecision(2) << std::setfill (' ') << std::setw(6) << std::fixed  << (double)n*100/n_conf << "%" << "]" << std::flush;
          benchmark_impl<OpT>(timings,dev,op,id,*it);
        }

        std::cout << std::endl;
      }



    }

  }

}
#endif // AUTOTUNE_HPP
