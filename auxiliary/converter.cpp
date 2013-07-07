/*
* Converts OpenCL sources to header file string constants
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <map>

// uncomment the following if you are using Boost versions prior to 1.44
//#define USE_OLD_BOOST_FILESYSTEM_VERSION

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>

namespace fs = boost::filesystem;

void writeSourceFile(std::ofstream & out_file, std::string const & filename, const char * dirname, const char * alignment)
{
    std::string fullpath(dirname);
    fullpath += "/";
    fullpath += alignment;
    fullpath += "/";
    fullpath += filename;
    std::ifstream in_file(fullpath.c_str());
    std::string tmp;

    if (in_file.is_open())
    {
        //write variable declaration:
        out_file << "const char * const " << dirname << "_" << alignment << "_" << filename.substr(0, filename.size()-3) << " = " << std::endl;

        //write source string:
        while (getline(in_file, tmp, '\n'))
        {
            if (tmp.size() > 0)
            {

                //Replaces " by \" in the source file.
                size_t start_pos = 0;
                while((start_pos = tmp.find("\"", start_pos)) != std::string::npos) {
                         tmp.replace(start_pos, 1,"\\\"");
                         start_pos += 2;
                }

//              std::replace( tmp.begin(), tmp.end(), '"', '\"');

                //out_file << "\"" << tmp.replace(tmp.end()-1, tmp.end(), "\\n\"") << std::endl;
                if ( *(tmp.end()-1) == '\r')  //Windows line delimiter, \r\n
                    out_file << "\"" << tmp.replace(tmp.end()-1, tmp.end(), "\\n\"") << std::endl;
                else //Unix line delimiter \n
                    out_file << "\"" << tmp.append("\\n\"") << std::endl;
            }
        }
        out_file << "; //" << dirname << "_" << alignment << "_" << filename.substr(0, filename.size()-3)  << std::endl << std::endl;

    }
    else
        std::cerr << "Failed to open file " << filename << std::endl;
}

void createSourceFile(const char * dirname)
{
    std::multimap<std::string, std::string> sorted_paths;

    //Step 1: Open source file
    std::string header_name(dirname);
    std::ofstream source_file(("@PROJECT_BINARY_DIR@/viennacl/linalg/kernels/" + header_name + "_source.h").c_str());

    //Step 2: Write source header file preamble
    std::string dirname_uppercase(dirname);
    std::transform(dirname_uppercase.begin(), dirname_uppercase.end(), dirname_uppercase.begin(), toupper);
    source_file << "#ifndef VIENNACL_LINALG_KERNELS_" << dirname_uppercase << "_SOURCE_HPP_" << std::endl;
    source_file << "#define VIENNACL_LINALG_KERNELS_" << dirname_uppercase << "_SOURCE_HPP_" << std::endl;
    source_file << "//Automatically generated file from auxiliary-directory, do not edit manually!" << std::endl;
    source_file << "/** @file " << header_name << "_source.h" << std::endl;
    source_file << " *  @brief OpenCL kernel source file, generated automatically from scripts in auxiliary/. */" << std::endl;
    source_file << "namespace viennacl" << std::endl;
    source_file << "{" << std::endl;
    source_file << " namespace linalg" << std::endl;
    source_file << " {" << std::endl;
    source_file << "  namespace kernels" << std::endl;
    source_file << "  {" << std::endl;

    //Step 3: Write all OpenCL kernel sources into header file
    fs::path filepath = fs::system_complete( fs::path( dirname ) );
    if ( fs::is_directory( filepath ) )
    {
        //std::cout << "\n In directory " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
                std::cout << "\nGenerating kernels from directory " << alignment_itr->path().string() << std::endl;

                //write and register single precision sources:
                for ( fs::directory_iterator cl_itr( alignment_itr->path() );
                      cl_itr != end_iter;
                      ++cl_itr )
                {
#ifdef USE_OLD_BOOST_FILESYSTEM_VERSION
                    std::string fname = cl_itr->path().filename();
                    std::string alignment = alignment_itr->path().filename();
#else
                    std::string fname = cl_itr->path().filename().string();
                    std::string alignment = alignment_itr->path().filename().string();
#endif

                    size_t pos = fname.find(".cl");
                    if ( pos == std::string::npos )
                      continue;

                    if (fname.substr(fname.size()-3, 3) == ".cl")
                        sorted_paths.insert(std::make_pair(fname,alignment));
                        //std::cout << alignment_itr->path().filename() << "/" << fname << std::endl;
                } //for
            } //if is_directory
        } //for alignment_iterator
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    for(std::multimap<std::string,std::string>::const_iterator it = sorted_paths.begin() ; it != sorted_paths.end() ; ++it){
      writeSourceFile(source_file, it->first, dirname, it->second.c_str());
    }
    //Final Step: Write file tail:
    source_file << "  }  //namespace kernels" << std::endl;
    source_file << " }  //namespace linalg" << std::endl;
    source_file << "}  //namespace viennacl" << std::endl;
    source_file << "#endif" << std::endl;
    source_file << std::endl;
    source_file.close();
}


unsigned int getBestKernel(const char * dirname, std::string & kernel_name, unsigned int alignment)
{
    unsigned int search_alignment = alignment;
    //std::cout << "Searching for best match for " << kernel_name << " with alignment " << alignment << std::endl;

    while (search_alignment > 1)
    {
        std::ostringstream oss;
        oss << dirname << "/align" << search_alignment;
        //std::cout << "Searching " << oss.str() << std::endl;

        //try to find kernel in directory:
        fs::path filepath = fs::system_complete( fs::path( oss.str() ) );
        if ( fs::is_directory( filepath ) ) //directory exists?
        {
            fs::directory_iterator end_iter;
            for ( fs::directory_iterator cl_itr( filepath );
                  cl_itr != end_iter;
                  ++cl_itr )
            {
#ifdef USE_OLD_BOOST_FILESYSTEM_VERSION
                std::string fname = cl_itr->path().filename();
#else
                std::string fname = cl_itr->path().filename().string();
#endif
                if (fname == kernel_name)
                {
                  //std::cout << "Found matching kernel for " << kernel_name << " with alignment " << alignment << " at alignment " << search_alignment << std::endl;
                    return search_alignment;
                }
            }
        }

        search_alignment /= 2;
    }

    //std::cout << "Found alignment 1 only..." << std::endl;
    //nothing found: return alignment 1:
    return 1;
}


void writeKernelInit(std::ostream & kernel_file, const char * dirname, std::string & subfolder, bool is_float)
{
    //extract alignment information from subfolder string:
    std::istringstream stream(subfolder.substr(5, subfolder.size()-5));
    unsigned int alignment = 0;
    stream >> alignment;
    if (alignment == 0)
        std::cerr << "ERROR: Could not extract alignment from " << subfolder << std::endl;

    kernel_file << "   template <>" << std::endl;
    kernel_file << "   struct " << dirname;
    if (is_float)
        kernel_file << "<float, ";
    else
        kernel_file << "<double, ";
    kernel_file << alignment << ">" << std::endl;
    kernel_file << "   {" << std::endl;

    kernel_file << "    static std::string program_name()" << std::endl;
    kernel_file << "    {" << std::endl;
    kernel_file << "      return \"";
    if (is_float)
        kernel_file << "f";
    else
        kernel_file << "d";
    kernel_file << "_" << dirname << "_" << alignment << "\";" << std::endl;
    kernel_file << "    }" << std::endl;

    kernel_file << "    static void init(viennacl::ocl::context & ctx)" << std::endl;
    kernel_file << "    {" << std::endl;
    if (is_float)
      kernel_file << "      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);" << std::endl;
    else
      kernel_file << "      viennacl::ocl::DOUBLE_PRECISION_CHECKER<double>::apply(ctx);" << std::endl;
    kernel_file << "      static std::map<cl_context, bool> init_done;" << std::endl;
    kernel_file << "      if (!init_done[ctx.handle().get()])" << std::endl;
    kernel_file << "      {" << std::endl;
    kernel_file << "        std::string source;" << std::endl;
    kernel_file << "        source.reserve(8192);" << std::endl; //to avoid some early reallocations
    if (!is_float)
      kernel_file << "        std::string fp64_ext = ctx.current_device().double_support_extension();" << std::endl;

    //iterate over all kernels in align1-folder:
    std::string current_dir(dirname);
    current_dir += "/align1";
    fs::path filepath = fs::system_complete( fs::path( current_dir ) );

    fs::directory_iterator end_iter;
    //write and register single precision sources:
    for ( fs::directory_iterator cl_itr( filepath );
          cl_itr != end_iter;
          ++cl_itr )
    {
#ifdef USE_OLD_BOOST_FILESYSTEM_VERSION
        std::string fname = cl_itr->path().filename();
#else
        std::string fname = cl_itr->path().filename().string();
#endif
        size_t pos = fname.find(".cl");
        if ( pos == std::string::npos )
          continue;

        if (fname.substr(fname.size()-3, 3) == ".cl")
        {
            //add kernel source to program string:
            std::string kernel_name_ending = fname.size() > 8 ? fname.substr(fname.size()-7, 4) : " ";
            if (kernel_name_ending == "_amd")
              kernel_file << "        if (ctx.current_device().local_mem_size() > 20000)" << std::endl << "  ";  //fast AMD kernels require more than 20 kB of local memory

            kernel_file << "        source.append(";
            if (!is_float)
                kernel_file << "viennacl::tools::make_double_kernel(";
            kernel_file << dirname << "_align" << getBestKernel(dirname, fname, alignment) << "_" << fname.substr(0, fname.size()-3);
            if (!is_float)
                kernel_file << ", fp64_ext)";
            kernel_file << ");" << std::endl;
        }
    } //for

    kernel_file << "        std::string prog_name = program_name();" << std::endl;
    kernel_file << "        #ifdef VIENNACL_BUILD_INFO" << std::endl;
    kernel_file << "        std::cout << \"Creating program \" << prog_name << std::endl;" << std::endl;
    kernel_file << "        #endif" << std::endl;
    kernel_file << "        ctx.add_program(source, prog_name);" << std::endl;

    kernel_file << "        init_done[ctx.handle().get()] = true;" << std::endl;
    kernel_file << "       } //if" << std::endl;
    kernel_file << "     } //init" << std::endl;
    kernel_file << "    }; // struct" << std::endl << std::endl;
}




void createKernelFile(const char * dirname)
{
    //Step 1: Open kernel file
    std::string header_name(dirname);
    std::ofstream kernel_file(("@PROJECT_BINARY_DIR@/viennacl/linalg/kernels/" + header_name + "_kernels.h").c_str());

    //Step 2: Write kernel header file preamble
    std::string dirname_uppercase(dirname);
    std::transform(dirname_uppercase.begin(), dirname_uppercase.end(), dirname_uppercase.begin(), toupper);
    kernel_file << "#ifndef VIENNACL_" << dirname_uppercase << "_KERNELS_HPP_" << std::endl;
    kernel_file << "#define VIENNACL_" << dirname_uppercase << "_KERNELS_HPP_" << std::endl;
    kernel_file << "#include \"viennacl/tools/tools.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/ocl/kernel.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/ocl/platform.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/ocl/utils.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/linalg/kernels/" << dirname << "_source.h\"" << std::endl;
    kernel_file << std::endl;
    kernel_file << "//Automatically generated file from aux-directory, do not edit manually!" << std::endl;
    kernel_file << "/** @file " << header_name << "_kernels.h" << std::endl;
    kernel_file << " *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */" << std::endl;
    kernel_file << "namespace viennacl" << std::endl;
    kernel_file << "{" << std::endl;
    kernel_file << " namespace linalg" << std::endl;
    kernel_file << " {" << std::endl;
    kernel_file << "  namespace kernels" << std::endl;
    kernel_file << "  {" << std::endl;

    //Step 3: Write class information:
    kernel_file << "   template<class TYPE, unsigned int alignment>" << std::endl;
    kernel_file << "   struct " << dirname << ";" << std::endl << std::endl;

    //Step 4: Write single precision kernels
    std::string dir(dirname);
    kernel_file << std::endl << "    /////////////// single precision kernels //////////////// " << std::endl;
    fs::path filepath = fs::system_complete( fs::path( dir ) );
    if ( fs::is_directory( filepath ) )
    {
        //std::cout << "\nIn directory: " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
#ifdef USE_OLD_BOOST_FILESYSTEM_VERSION
                std::string subfolder = alignment_itr->path().filename();
#else
                std::string subfolder = alignment_itr->path().filename().string();
#endif
                if( subfolder.find("align") == std::string::npos )
                  continue;
                writeKernelInit(kernel_file, dirname, subfolder, true);
            } //if is_directory
        } //for alignment_iterator
        kernel_file << std::endl;
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    //Step 5: Write double precision kernels
    kernel_file << std::endl << "    /////////////// double precision kernels //////////////// " << std::endl;
    filepath = fs::system_complete( fs::path( dir ) );
    if ( fs::is_directory( filepath ) )
    {
        //std::cout << "\nIn directory: " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
#ifdef USE_OLD_BOOST_FILESYSTEM_VERSION
                std::string subfolder = alignment_itr->path().filename();
#else
                std::string subfolder = alignment_itr->path().filename().string();
#endif
                if( subfolder.find("align") == std::string::npos )
                  continue;
                writeKernelInit(kernel_file, dirname, subfolder, false);
            } //if is_directory
        } //for alignment_iterator
        kernel_file << std::endl;
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    //Final Step: Write file tail:
    kernel_file << "  }  //namespace kernels" << std::endl;
    kernel_file << " }  //namespace linalg" << std::endl;
    kernel_file << "}  //namespace viennacl" << std::endl;
    kernel_file << "#endif" << std::endl;
    kernel_file << std::endl;
    kernel_file.close();
}

void createHeaders(const char * dirname)
{
    createKernelFile(dirname);
    createSourceFile(dirname);
}

int main(int , char **)
{
    createHeaders("compressed_matrix");
    createHeaders("coordinate_matrix");
    createHeaders("ell_matrix");
    createHeaders("hyb_matrix");
    createHeaders("matrix_row");
    createHeaders("matrix_row_element");
    createHeaders("matrix_col");
    createHeaders("matrix_col_element");
    createHeaders("matrix_prod_row_row_row");
    createHeaders("matrix_prod_row_row_col");
    createHeaders("matrix_prod_row_col_row");
    createHeaders("matrix_prod_row_col_col");
    createHeaders("matrix_prod_col_row_row");
    createHeaders("matrix_prod_col_row_col");
    createHeaders("matrix_prod_col_col_row");
    createHeaders("matrix_prod_col_col_col");
    createHeaders("matrix_solve_col_col");
    createHeaders("matrix_solve_col_row");
    createHeaders("matrix_solve_row_col");
    createHeaders("matrix_solve_row_row");
    createHeaders("scalar");
    createHeaders("fft");
    createHeaders("rand");
    createHeaders("svd");
    createHeaders("spai");
    createHeaders("nmf");
    createHeaders("ilu");
}

