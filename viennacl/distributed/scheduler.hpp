#ifndef VIENNACL_DISTRIBUTED_SCHEDULER_HPP_
#define VIENNACL_DISTRIBUTED_SCHEDULER_HPP_

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file scheduler.hpp
    @brief Implementation of the scheduler classes
*/

#include "deque"

#include "CL/cl.h"


#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/distributed/task.hpp"
#include "viennacl/distributed/utils.hpp"

#include "viennacl/distributed/timer.hpp"

#include "viennacl/tools/shared_ptr.hpp"

#include <thread>
#include <pthread.h>
#include <chrono>
#include <functional>

#include <mutex>
#include <thread>

namespace viennacl{

namespace distributed{


class scheduler{
private:
    typedef std::vector<viennacl::tools::shared_ptr<task> > pending_tasks_t;
    typedef std::multimap<task*, task*> active_dependancies_t;
    typedef std::map<cl_device_id, viennacl::tools::shared_ptr<std::thread> > context_map_t;

private:

    static void solve_dependancies(task* ptr){
        active_dependancies_t::iterator it = active_dependancies_.begin();
        while(it!=active_dependancies_.end()){
            if(it->second==ptr) active_dependancies_.erase(it++);
            else ++it;
        }
    }

    static viennacl::tools::shared_ptr<task> pop_task(){
        pending_tasks_t::iterator it = pending_tasks_.begin();
        for( ; it!=pending_tasks_.end() ; ++it){
            if(active_dependancies_.find(it->get())==active_dependancies_.end()){
                viennacl::tools::shared_ptr<task> res(*it);
                pending_tasks_.erase(it);
                return res;
            }
        }
        return viennacl::tools::shared_ptr<task>();
    }

   static  void print_dep(){
        for(active_dependancies_t::iterator it = active_dependancies_.begin() ; it != active_dependancies_.end() ; ++it){
            std::cout << it->first->info() << " DEPENDS ON " << it->second->info() << std::endl;
        }
    }


    static void update(long id){
        viennacl::ocl::backend<>::switch_context(id);
//        viennacl::ocl::backend<>::current_context().add_queue(viennacl::ocl::current_device());
//        viennacl::ocl::switch_context(ctxt_id);
        viennacl::ocl::context & context = viennacl::ocl::current_context();
        viennacl::ocl::device const & device = context.devices()[0];
        while(!pending_tasks_.empty()){
            mutex_.lock();
            viennacl::tools::shared_ptr<task> tsk(pop_task());
            mutex_.unlock();
            if(tsk.get()==NULL){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else{
                double begin = timeline_.get();
#ifdef VIENNACL_DEBUG_SCHEDULER
                std::cout << "[" << begin << "] : % Start :" << tsk->info() << " on " << device.name() << std::endl;
#endif
                viennacl::ocl::event * evt = tsk->run();
                while(evt->status() > 0){
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                mutex_.lock();
                double end = timeline_.get();
#ifdef VIENNACL_DEBUG_SCHEDULER
                std::cout << "[" << end << "] : #Finished : " << tsk->info() <<  " on device " << device.name() << " : \n"
                          << "      Kernel duration : " << viennacl::ocl::time_of_execution_us(*evt)/1000  << "ms\n"
                          << "      Total duration : " << (end-begin)*1000 << "ms" << std::endl;
#endif
                solve_dependancies(tsk.get()); //Removes current task
                mutex_.unlock();
            }
        }
    }


public:

    static void add_device(viennacl::ocl::device const & device){
#ifdef VIENNACL_DEBUG_SCHEDULER
        std::cout << "Adding " << device.name() << " to the scheduler " << std::endl;
#endif
//        long id = context_map_.size();
//        std::vector<cl_device_id> devs; devs.push_back(device.id());
//        viennacl::ocl::setup_context(id,devs);
//        viennacl::ocl::switch_context(id);
//        viennacl::ocl::current_context().add_queue(device);
        context_map_.insert(std::make_pair(device.id(),viennacl::tools::shared_ptr<std::thread>()));

    }

    static std::vector<cl_device_id> devices(){
        std::vector<cl_device_id> res;
        for(context_map_t::const_iterator it = context_map_.begin() ; it != context_map_.end() ; ++it){
            res.push_back(it->first);
        }
        return res;
    }

    static void add_all_available_devices(cl_device_type dtype = CL_DEVICE_TYPE_DEFAULT){
        cl_uint num_platforms = viennacl::ocl::num_platforms();
        for(cl_uint i = 0 ; i < num_platforms ; ++i){
            viennacl::ocl::platform pf(i);
            std::vector<viennacl::ocl::device> devices = pf.devices(dtype);
            for(std::vector<viennacl::ocl::device>::iterator it = devices.begin() ; it != devices.end() ; ++it){
                viennacl::distributed::scheduler::add_device(*it);
            }
        }
    }

    static unsigned int n_devices(){
        return context_map_.size();
    }

    template<class F, class A>
    static task* create_task(std::function<F> fun, A const & args){
        pending_tasks_.push_back(viennacl::tools::shared_ptr<task2<F,A> >(new task2<F,A>(fun,args)));
        return pending_tasks_.back().get();
    }


    //Add an edge between i and j (j depends on i)
    static void connect(task* i, task* j){
        active_dependancies_.insert(std::make_pair(j,i));
    }


    static void init(){
        timeline_.start();
        for(context_map_t::iterator it = context_map_.begin() ; it != context_map_.end() ; ++it){
            unsigned int id = 1 + std::distance(context_map_.begin(),it);
            std::vector<cl_device_id> devs; devs.push_back(it->first);
            viennacl::ocl::setup_context(id,devs);
            std::thread* t = new std::thread(std::bind(&scheduler::update,id));
            it->second.reset(t);
        }
    }

    static void finish(){
        for(context_map_t::const_iterator it = context_map_.begin() ; it != context_map_.end() ; ++it){
            it->second->join();
        }
    }

    friend void on_task_end_callback(cl_event , cl_int, void*);

private:
    static context_map_t context_map_;
    static pending_tasks_t pending_tasks_;
    static std::map<cl_mem, unsigned int> handle_in_use_;
    static active_dependancies_t active_dependancies_;
    static std::mutex mutex_;
    static viennacl::distributed::timer timeline_;
};

scheduler::context_map_t scheduler::context_map_;
scheduler::pending_tasks_t scheduler::pending_tasks_;
std::map<cl_mem, unsigned int> scheduler::handle_in_use_;
scheduler::active_dependancies_t scheduler::active_dependancies_;
std::mutex scheduler::mutex_;
viennacl::distributed::timer scheduler::timeline_;

}

}

#endif // SCHEDULER_HPP
