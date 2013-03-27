#ifndef VIENNACL_DISTRIBUTED_FISSION_HPP_
#define VIENNACL_DISTRIBUTED_FISSION_HPP_

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

/** @file fission.hpp
    @brief Implementation of the fission policy for an opencl device
*/

#include "viennacl/ocl/device.hpp"
#include <memory>

#include "CL/cl.h"

#include <map>

namespace viennacl{

namespace distributed{

#ifdef CL_VERSION_1_2

class fission_base{
public:
    virtual std::vector<viennacl::ocl::device>  sub_devices(viennacl::ocl::device const & in_device) = 0;
};


template<cl_device_partition_property PROP>
class fission;

class fission_policy{
private:
    typedef std::map<cl_device_id, viennacl::tools::shared_ptr<fission_base> > map_t;
public:
    template<cl_device_partition_property PROP>
    void add_device_fission(viennacl::ocl::device const & d, fission<PROP> f){
        map_.insert(std::make_pair(d.id(), new fission<PROP>(f)));
    }

    std::vector<viennacl::ocl::device> sub_devices(viennacl::ocl::device const & d){
        map_t::iterator it = map_.find(d.id());
        if(it==map_.end()){
            return  std::vector<viennacl::ocl::device>();
        }
        return it->second->sub_devices(d);
    }

private:
    map_t map_;
};


template<>
class fission<CL_DEVICE_PARTITION_EQUALLY> : public fission_base{
public:

    /** @brief constructor
      * @param n : The number of core per sub-device
      */
    fission(unsigned int n){
        properties_[0] = CL_DEVICE_PARTITION_EQUALLY;
        properties_[1] = n;
        properties_[2] = 0;
    }

    std::vector<viennacl::ocl::device>  sub_devices(viennacl::ocl::device const & in_device)
    {
        cl_device_id  out_devices_id[16];
        cl_uint num_devices_ret = 0;
        cl_int err = clCreateSubDevices(in_device.id(),
                                       properties_,
                                       16,
                                       out_devices_id,
                                       &num_devices_ret);
        VIENNACL_ERR_CHECK(err);
        std::vector<viennacl::ocl::device> res;
        res.reserve(num_devices_ret);
        for(unsigned int i = 0 ; i < num_devices_ret ; ++i){
            res.push_back(viennacl::ocl::device(out_devices_id[i]));
        }
        return res;
    }

private:
    cl_device_partition_property properties_[3];
};

template<>
class fission<CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN> : public fission_base{
public:

    /** @brief constructor
      * @param affinity_domain : The affinity domain
      */
    fission(cl_device_affinity_domain affinity_domain){
        properties_[0] = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
        properties_[1] = affinity_domain;
        properties_[2] = 0;
    }

    std::vector<viennacl::ocl::device>  sub_devices(viennacl::ocl::device const & in_device)
    {
        cl_device_id  out_devices_id[16];
        cl_uint num_devices_ret = 0;
        cl_int err = clCreateSubDevices(in_device.id(),
                                       properties_,
                                       16,
                                       out_devices_id,
                                       &num_devices_ret);
        VIENNACL_ERR_CHECK(err);
        std::vector<viennacl::ocl::device> res;
        res.reserve(num_devices_ret);
        for(unsigned int i = 0 ; i < num_devices_ret ; ++i){
            res.push_back(viennacl::ocl::device(out_devices_id[i]));
        }
        return res;
    }
private:
    cl_device_partition_property properties_[3];
};

template<>
class fission<CL_DEVICE_PARTITION_BY_COUNTS> : public fission_base{
public:

    /** @brief constructor */
    fission(){
        properties_.push_back(CL_DEVICE_PARTITION_BY_COUNTS);
    }

    /** @brief Adds a sub-device to the list
      * @param n : Number of cores the user wants on this subdevice
      */
    void add(unsigned int n){
        properties_.push_back(n);
    }

    std::vector<viennacl::ocl::device>  sub_devices(viennacl::ocl::device const & in_device)
    {
        cl_device_id out_devices_id[16];
        cl_uint num_devices_ret = 0;
        properties_.push_back(CL_DEVICE_PARTITION_BY_COUNTS_LIST_END);
        properties_.push_back(0);
        if(properties_.size()>3){
            cl_int err = clCreateSubDevices(in_device.id(),
                                           properties_.data(),
                                           16,
                                           out_devices_id,
                                           &num_devices_ret);
            VIENNACL_ERR_CHECK(err);
        }
        std::vector<viennacl::ocl::device> res;
        res.reserve(num_devices_ret);
        for(unsigned int i = 0 ; i < num_devices_ret ; ++i){
            res.push_back(viennacl::ocl::device(out_devices_id[i]));
        }
        return res;
    }


private:
    std::vector<cl_device_partition_property> properties_;
};

#endif

}

}

#endif // FISSION_HPP
