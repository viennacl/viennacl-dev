#ifndef VIENNACL_GENERATOR_CODE_GENERATION_BUILTIN_DATABASE_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_BUILTIN_DATABASE_HPP

#include "CL/cl.h"
#include <string>

#include <map>
#include "viennacl/generator/code_generation/templates.hpp"
#include "viennacl/tools/shared_ptr.hpp"


namespace viennacl{

namespace generator{

namespace code_generation{

typedef std::map< std::pair<cl_uint, cl_device_type>, std::map<std::string,viennacl::tools::shared_ptr<optimization_profile> > > builtin_database_t;

static  builtin_database_t make_database(){
    builtin_database_t res;



    //AMD GPUs
    {
        std::map<std::string,viennacl::tools::shared_ptr<optimization_profile> > tmp;

        //BLAS 1
            tmp.insert(std::make_pair("p_mf_1_0eqmf_1_0_p", viennacl::tools::shared_ptr<optimization_profile> ( new vector_saxpy::profile(2,1,128))));
            tmp.insert(std::make_pair("p_vfeqvf_p", viennacl::tools::shared_ptr<optimization_profile> ( new vector_saxpy::profile(2,1,128))));

        //BLAS 3
        //Row * Row and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));

            //Double
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));

        //Row * Col and analogs
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));

        //Col * Row and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));

            //Double
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));

        // Col * Col and analogs
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));


            res.insert(std::make_pair(std::make_pair(4098,CL_DEVICE_TYPE_GPU),tmp));
    }

    //NVidia GPUs
    {
            std::map<std::string, viennacl::tools::shared_ptr<optimization_profile> > tmp;

            tmp.insert(std::make_pair("p_mf_1_0eqmf_1_0_p", viennacl::tools::shared_ptr<optimization_profile> ( new vector_saxpy::profile(2,2,128))));


        //Row * Row and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,32))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));

            //Double
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,64,128,2,2,8,1,0,1,32))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));


        //Row * Col and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));

        //Col * Row and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));

            //double
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));

        // Col * Col and analogs

            //Float
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_0_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_1_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));


        res.insert(std::make_pair(std::make_pair(4318,CL_DEVICE_TYPE_GPU),tmp));
    }

    //Intel CPUs
    {
        std::map<std::string, viennacl::tools::shared_ptr<optimization_profile> > tmp;

        tmp.insert(std::make_pair("p_mf_1_0eqmf_1_0_p", viennacl::tools::shared_ptr<optimization_profile> ( new vector_saxpy::profile(4,16,64))));

        //Row * Row and analogs

        //float
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));

        //double
        tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,64,8,4,64,0,0,2,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_1_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_0_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));

        //Row * Col and analogs

        //Col * Row and analogs

        //float
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_0_0prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_1_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("p_mf_0_0eqp_mf_1_1prodmf_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));

        //double
        tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_1_0_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_0_0prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_1_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("p_md_0_0eqp_md_1_1prodmd_0_1_p_p", viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        res.insert(std::make_pair(std::make_pair(32902,CL_DEVICE_TYPE_CPU),tmp));
        res.insert(std::make_pair(std::make_pair(4098,CL_DEVICE_TYPE_CPU),tmp));
    }
    return res;
}


static builtin_database_t builtin_dabase = make_database();

}

}

}


#endif

