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
            tmp.insert(std::make_pair("assign(vecf,vecf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(2,1,128))));
            tmp.insert(std::make_pair("assign(pscalf,prod(vecf,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(4,128,32))));
            tmp.insert(std::make_pair("assign(matf,matf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(2,1,128))));

            tmp.insert(std::make_pair("assign(vecd,vecd)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(1,2,256))));
            tmp.insert(std::make_pair("assign(pscald,prod(vecd,vecd))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(8,64,64))));
            tmp.insert(std::make_pair("assign(matd,matd)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(1,2,256))));

        //BLAS2
            tmp.insert(std::make_pair("assign(vecd,prod(matdC,vecd))", viennacl::tools::shared_ptr<optimization_profile>(new gemv::profile(32,8,256))));
            tmp.insert(std::make_pair("assign(vecf,prod(matfC,vecf))", viennacl::tools::shared_ptr<optimization_profile>(new gemv::profile(32,8,256))));


        //BLAS 3
        //Row * Row and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,256,8,8,4,1,0,4,1))));

            //Double
            tmp.insert(std::make_pair("assign(matdR,prod(matdR,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdR,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(matdR,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdR,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,4,4,4,1,0,4,1))));

        //Row * Col and analogs
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,32,64,4,8,4,1,0,4,1))));

        //Col * Row and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,256,256,4,4,8,0,0,4,1))));

            //Double
            tmp.insert(std::make_pair("assign(matdR,prod(matdC,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdC,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdR),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(matdC,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdC,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,128,4,2,4,0,0,2,1))));

        // Col * Col and analogs
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,8,4,8,0,0,4,1))));


            res.insert(std::make_pair(std::make_pair(4098,CL_DEVICE_TYPE_GPU),tmp));
    }

    //NVidia GPUs
    {
            std::map<std::string, viennacl::tools::shared_ptr<optimization_profile> > tmp;

        //BLAS1
            tmp.insert(std::make_pair("assign(vecf,vecf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(4,1,64))));
            tmp.insert(std::make_pair("assign(pscalf,prod(vecf,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(4,64,64))));
            tmp.insert(std::make_pair("assign(matf,matf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(2,2,128))));

            tmp.insert(std::make_pair("assign(vecd,vecd)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(2,1,64))));
            tmp.insert(std::make_pair("assign(pscald,prod(vecd,vecd))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(2,128,64))));
            tmp.insert(std::make_pair("assign(matd,matd)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(2,1,64))));

        //BLAS2
            tmp.insert(std::make_pair("assign(pvecd,prod(matdC,vecd))", viennacl::tools::shared_ptr<optimization_profile> ( new gemv::profile(16,32,1024))));
            tmp.insert(std::make_pair("assign(pvecf,prod(matfC,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new gemv::profile(64,4,128))));



        //Row * Row and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,32))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,matfR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),matfR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),matfR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,trans(matfC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,trans(matfC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),trans(matfC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),trans(matfC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,128,128,8,4,2,1,0,1,1))));

            //Double
            tmp.insert(std::make_pair("assign(matdR,prod(matdR,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(16,64,128,2,2,8,1,0,1,32))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdR,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(matdR,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdR,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,64,128,2,2,8,1,0,1,1))));


        //Row * Col and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfR,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfR,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,2,4,8,1,1,2,1))));

        //Col * Row and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,128,256,8,4,2,1,0,1,1))));

            //double
            tmp.insert(std::make_pair("assign(matdR,prod(matdC,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdC,matdR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(trans(matdR),matdR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(matdC,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdC,prod(matdC,trans(matdC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));
            tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(256,128,32,2,2,8,0,1,2,1))));

        // Col * Col and analogs

            //Float
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,matfC))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),matfC))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(matfC,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(matfC,trans(matfR)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));
            tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),trans(matfR)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(32,32,128,4,4,4,1,1,4,1))));


        res.insert(std::make_pair(std::make_pair(4318,CL_DEVICE_TYPE_GPU),tmp));
    }

    //Intel CPUs
    {
        std::map<std::string, viennacl::tools::shared_ptr<optimization_profile> > tmp;

     //BLAS1
        tmp.insert(std::make_pair("assign(vecf,vecf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(1,1,256))));
        tmp.insert(std::make_pair("assign(pscalf,prod(vecf,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(2,32,64))));
        tmp.insert(std::make_pair("assign(matf,matf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(4,16,64))));

        tmp.insert(std::make_pair("assign(vecf,vecf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(4,16,512))));
        tmp.insert(std::make_pair("assign(pscalf,prod(vecf,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new inner_product::profile(3,32,32))));
        tmp.insert(std::make_pair("assign(matf,matf)", viennacl::tools::shared_ptr<optimization_profile> ( new saxpy::profile(4,16,64))));


     //BLAS2
        tmp.insert(std::make_pair("assign(pvecd,prod(matdC,vecd))", viennacl::tools::shared_ptr<optimization_profile> ( new gemv::profile(16,8,32))));
        tmp.insert(std::make_pair("assign(pvecf,prod(matfC,vecf))", viennacl::tools::shared_ptr<optimization_profile> ( new gemv::profile(32,4,32))));

     //BLAS3
        //Row * Row and analogs

        //float
        tmp.insert(std::make_pair("assign(matfR,prod(matfR,matfR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(matfR,matfR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),matfR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),matfR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(matfR,trans(matfC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(matfR,trans(matfC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(trans(matfC),trans(matfC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(trans(matfC),trans(matfC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(64,64,128,4,4,128,0,0,4,1))));

        //double
        tmp.insert(std::make_pair("assign(matdR,prod(matdR,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,64,8,4,64,0,0,2,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(matdR,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(matdR,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(matdR,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(trans(matdC),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(trans(matdC),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,128,8,4,128,0,0,4,1))));

        //Row * Col and analogs

        //Col * Row and analogs

        //float
        tmp.insert(std::make_pair("assign(matfR,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(matfC,matfR))"                 , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(trans(matfR),matfR))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfC,prod(matfC,trans(matfC)))"          , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matfR,prod(trans(matfR),trans(matfC)))"   , viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,64,32,16,4,32,0,0,1,1))));

        //double
        tmp.insert(std::make_pair("assign(matdR,prod(matdC,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(matdC,matdR))"                 ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(trans(matdR),matdR))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(matdC,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdC,prod(matdC,trans(matdC)))"          ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));
        tmp.insert(std::make_pair("assign(matdR,prod(trans(matdR),trans(matdC)))"   ,  viennacl::tools::shared_ptr<optimization_profile> ( new gemm::profile(128,128,32,8,4,16,0,0,1,1))));


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

