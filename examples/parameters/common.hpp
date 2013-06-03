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

////////////////////// some functions that aid testing to follow /////////////////////////////////

#include "benchmark-utils.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/io/kernel_parameters.hpp"

#ifndef BENCHMARK_RUNS
 #define BENCHMARK_RUNS          10
#endif


void set_kernel_params(std::string program_name,
                       std::string kernel_name,
                       unsigned int work_groups, //total no. of threads
                       unsigned int loc)  //threads per work group
{
  //get kernel from pool and set work sizes:
  viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(program_name, kernel_name);
  k.global_work_size(0, work_groups * loc);
  k.local_work_size(0, loc);
}

bool validate_result(std::string program_name,
                     std::string kernel_name,
                     unsigned int work_groups,
                     unsigned int local_workers)
{
  viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(program_name, kernel_name);
  bool ret = (k.global_work_size() == work_groups * local_workers)
           && (k.local_work_size() == local_workers);
  if (!ret)
  {
    std::cout << "Failed: " << k.global_work_size() << " vs. " << work_groups * local_workers << " and " << k.local_work_size() << " vs. " << local_workers << std::endl;
  }
  return ret;
}



template <typename T, typename TestData>
double execute(T functor, TestData & data)
{
  Timer t;
  functor(data); //one startup calculation
  viennacl::ocl::get_queue().finish();
  t.start();
  for (int i=0; i<BENCHMARK_RUNS; ++i)
    functor(data);
  viennacl::ocl::get_queue().finish();
  return t.get();
}


template <typename TimingType, typename F, typename TestConfig, typename TestData>
void record_full_timings(TimingType & timings,
                         F functor,
                         TestConfig & config,
                         TestData & data)
{
  typedef typename TestData::value_type  ScalarType;
  
  double result = 0;
  functor(data); //startup run (ensures kernel compilation)
  for (unsigned int work_groups = config.min_work_groups(); work_groups <= config.max_work_groups(); work_groups *= 2)           //iterate over number of work groups (compute units)
  {
    for (unsigned int local_workers = config.min_local_size(); local_workers <= config.max_local_size(); local_workers *= 2)   //iterate over local thread number
    {
      //set parameter:
      set_kernel_params(config.program_name(), config.kernel_name(), work_groups, local_workers);
      
      //std::cout << "Benchmarking kernel " << config.kernel_name() << std::endl;
      result = execute(functor, data);
      
      //check for valid result: (kernels have an automatic fallback to smaller values included)
      if (!validate_result(config.program_name(), config.kernel_name(), work_groups, local_workers))
      {
      std::cout << "Kernel start failed for kernel " << config.kernel_name() << " [" << work_groups << " groups, " << local_workers << " per group]" << std::endl;
        break;
      }
      else
        timings[result] = std::make_pair(work_groups * local_workers, local_workers);
    }
  }
}

template <typename TimingType, typename F, typename TestConfig, typename TestData>
void record_restricted_timings(TimingType & timings,
                               F functor,
                               TestConfig & config,
                               TestData & data)
{
  typedef typename TestData::value_type  ScalarType;
  
  double result = 0;
  functor(data); //startup run (ensures kernel compilation)
  for (unsigned int local_workers = config.min_local_size(); local_workers <= config.max_local_size(); local_workers *= 2)   //iterate over local thread number, up to 512
  {
    //set parameter:
    set_kernel_params(config.program_name(), config.kernel_name(), 1, local_workers);
    
    result = execute(functor, data);
    
    //check for valid result: (kernels have an automatic fallback to smaller values included)
    if (!validate_result(config.program_name(), config.kernel_name(), 1, local_workers))
    {
      std::cout << "Kernel start failed for kernel " << config.kernel_name() << " [1 group, " << local_workers << " per group]" << std::endl;
      //break;
    }
    else
      timings[result] = std::make_pair(local_workers, local_workers);
  }
}

template <typename TimingType>
void print_best(TimingType const & timings, std::string kernel_name)
{
  //give some feedback to stdout:
  std::cout << "Best parameter set for " << kernel_name << ": [" << timings.begin()->second.first << " global workers, " << timings.begin()->second.second << " local workers] " << timings.begin()->first << std::endl;
  
}

template <typename TimingType>
void print_default(TimingType const & timings, std::string /*kernel_name*/)
{
  bool found = false;
  std::cout << "Default parameter set: [16384 global workers, 128 local workers] ";
  for (typename TimingType::const_iterator it = timings.begin(); it != timings.end(); ++it)
  {
    if (it->second.first == 128*128 && it->second.second == 128)
    {
      std::cout << it->first << std::endl;
      found = true;
    }
  }
  if (!found)
    std::cout << "n.a." << std::endl;
}

template <typename TimingType>
void print_default_restricted(TimingType const & timings, std::string /*kernel_name*/)
{
  bool found = false;
  std::cout << "Default parameter set: [128 global workers, 128 local workers] ";
  for (typename TimingType::const_iterator it = timings.begin(); it != timings.end(); ++it)
  {
    if (it->second.first == 128 && it->second.second == 128)
    {
      std::cout << it->first << std::endl;
      found = true;
    }
  }
  if (!found)
    std::cout << "n.a." << std::endl;
}


class test_config
{
  public:
    test_config() {}
    test_config(std::string const & prog_name) : prog_(prog_name) {}
    test_config(std::string const & prog_name, std::string const & kernel_name) : prog_(prog_name), kernel_(kernel_name) {}
    
    std::string const & program_name() const { return prog_; }
    void program_name(std::string const & name) { prog_ = name; }
    std::string const & kernel_name() const { return kernel_; }
    void kernel_name(std::string const & name) { kernel_ = name; }
    
    unsigned int min_work_groups() const { return min_work_groups_; }
    void min_work_groups(unsigned int i) { min_work_groups_ = i; }
    unsigned int max_work_groups() const { return max_work_groups_; }
    void max_work_groups(unsigned int i) { max_work_groups_ = i; }
    
    
    unsigned int min_local_size() const { return min_local_size_; }
    void min_local_size(unsigned int i) { min_local_size_ = i; }
    unsigned int max_local_size() const { return max_local_size_; }
    void max_local_size(unsigned int i) { max_local_size_ = i; }
    
  private:
    std::string prog_;
    std::string kernel_;
    unsigned int min_work_groups_;
    unsigned int max_work_groups_;
    unsigned int min_local_size_;
    unsigned int max_local_size_;
};

template <typename TimingType>
void record_kernel_parameters(viennacl::io::parameter_database& paras, std::string kernel, TimingType& timings)
{
   paras.add_kernel();  
   paras.add_data_node(viennacl::io::tag::name, kernel);   
   paras.add_parameter();  
   paras.add_data_node(viennacl::io::tag::name, viennacl::io::val::globsize);
   paras.add_data_node(viennacl::io::tag::value, timings.begin()->second.first);         
   paras.add_parameter();     
   paras.add_data_node(viennacl::io::tag::name, viennacl::io::val::locsize);
   paras.add_data_node(viennacl::io::tag::value, timings.begin()->second.second);            
}




template <typename TimingType, typename F, typename TestConfig, typename TestData>
void optimize_full(viennacl::io::parameter_database & paras,
                   TimingType & timings,
                   F functor,
                   TestConfig & config,
                   TestData & data)
{
  record_full_timings(timings, functor, config, data);
  record_kernel_parameters(paras, config.kernel_name(), timings);
#ifdef ENABLE_VIENNAPROFILER
  write_viennaprofiler(timings, config.program_name(), config.kernel_name());
#endif
  print_best(timings, config.kernel_name());
  print_default(timings, config.kernel_name());
}

template <typename TimingType, typename F, typename TestConfig, typename TestData>
void optimize_restricted(viennacl::io::parameter_database & paras,
                         TimingType & timings,
                         F functor,
                         TestConfig & config,
                         TestData & data)
{
  record_restricted_timings(timings, functor, config, data);
  record_kernel_parameters(paras, config.kernel_name(), timings);
#ifdef ENABLE_VIENNAPROFILER
  write_viennaprofiler(timings, config.program_name(), config.kernel_name());
#endif
  print_best(timings, config.kernel_name());
  print_default_restricted(timings, config.kernel_name());
}
