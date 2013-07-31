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

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/infos.hpp"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/generate.hpp"
#include "viennacl/generator/builtin_database.hpp"

#include "viennacl/tools/timer.hpp"

namespace viennacl{

  namespace generator{

    namespace autotune{


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
      template<class ProfileT>
      void benchmark_impl(std::map<double, ProfileT> & timings, viennacl::ocl::device const & dev, viennacl::scheduler::statement const & operation, ProfileT const & prof){

        tools::Timer t;

        unsigned int n_runs = 10;

        //Skips if use too much local memory.
        std::list<viennacl::ocl::kernel *> kernels;
        viennacl::generator::code_generator gen;
        gen.add(operation);
        gen.force_profile(prof);
        viennacl::generator::get_configured_program(gen, kernels, true);

        viennacl::ocl::kernel & k = *kernels.front();
        //Anticipates kernel failure
        size_t max_workgroup_size = viennacl::ocl::info<CL_KERNEL_WORK_GROUP_SIZE>(k,dev);
        size_t size1, size2;
        prof.set_local_sizes(size1, size2, 0);
        if(size1*size2 > max_workgroup_size)  return;

        //Doesn't execute because it would likelily be a waste of time
        size_t prefered_workgroup_size_multiple = viennacl::ocl::info<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(k,dev);
        if( (size1*size2) % prefered_workgroup_size_multiple > 0) return;

        viennacl::generator::enqueue(gen);
        viennacl::backend::finish();

        double exec_time = 0;
        t.start();
        for(unsigned int n=0; n<n_runs ; ++n){
          generator::enqueue(gen);
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
      template<class ConfigT>
      void benchmark(std::map<double, typename ConfigT::profile_t> & timings, scheduler::statement const & op, ConfigT & config, size_t /*scalartype_size*/){
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
          benchmark_impl(timings,dev,op,config.get_current());
        }

        std::cout << std::endl;
      }

      /** @brief Fills a timing map for a given operation and a list of profiles */
      template< class ProfT>
      void benchmark(std::map<double, ProfT> & timings, scheduler::statement const & op, std::list<ProfT> const & profiles, size_t scalartype_size){
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
          benchmark_impl(timings,dev,op,*it);
        }

        std::cout << std::endl;
      }



    }

  }

}
#endif // AUTOTUNE_HPP
