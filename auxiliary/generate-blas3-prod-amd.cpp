#include <iostream>
#include <sstream>
#include <cstdlib>

template <class T>
inline std::string to_string ( T const t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}


template<unsigned int dim>
class local_memory;

template<>
class local_memory<1>{
public:

    local_memory(std::string const & name, unsigned int size, std::string const & scalartype): name_(name), size_(size), scalartype_(scalartype){ }

    std::string declare() const{
        return "__local " + scalartype_ + " " + name_ + '[' + to_string(size_) + ']';
    }

    unsigned int size() const{ return size_; }

    std::string const & name() const{
        return name_;
    }

    std::string access(std::string const & index) const{
        return name_ + '[' + index + ']';
    }

private:
    std::string name_;
    unsigned int size_;
    std::string const & scalartype_;
};

template<>
class local_memory<2>{
public:

    local_memory(std::string const & name
                 , unsigned int size1
                 , unsigned int size2
                 , std::string const & scalartype): size1_(size1), size2_(size2), impl_(name,size1*size2,scalartype){

    }

    std::string declare() const{
        return impl_.declare();
    }

    std::string const & name() const { return impl_.name(); }

    unsigned int size1() const{ return size1_; }

    unsigned int size2() const{ return size2_; }

    std::string offset(std::string const & i, std::string const & j) const{
        return '('+i+')' + '*' + to_string(size2_) + '(' + j + ')';
    }

private:
    unsigned int size1_;
    unsigned int size2_;
    local_memory<1> impl_;
};


struct matrix_descriptor{
  matrix_descriptor(unsigned int _alignment, bool /*_use_shared*/, bool _is_rowmajor, bool _is_transposed, std::string const & _name, std::string const & _scalartype){
    alignment = _alignment;
    is_rowmajor = _is_rowmajor;
    is_transposed = _is_transposed;
    name = _name;
    size1 = _name+"_size1";
    size2 = _name+"_size2";
    internal_size1 = _name+"_internal_size1";
    internal_size2 = _name+"_internal_size2";
    row_inc = _name+"_stride1";
    col_inc = _name+"_stride2";
    row_start = _name+"_offset1";
    col_start = _name+"_offset2";
    scalartype = _scalartype;
    aligned_scalartype = _scalartype; if(alignment>1) aligned_scalartype+=to_string(alignment);
  }

  std::string arguments_string() const{
      return " __global " + aligned_scalartype + "*"  + " " + name
                                                  + ", unsigned int " + row_start
                                                  + ", unsigned int " + col_start
                                                  + ", unsigned int " + row_inc
                                                  + ", unsigned int " + col_inc
                                                  + ", unsigned int " + size1
                                                  + ", unsigned int " + size2
                                                  + ", unsigned int " + internal_size1
                                                  + ", unsigned int " + internal_size2;
  }


  std::string offset(std::string const & offset_i, std::string const & offset_j){
                if(is_rowmajor){
                    return '(' + offset_i + ')' + '*' + internal_size2 + "+ (" + offset_j + ')';
                }
                return '(' + offset_i + ')' + "+ (" + offset_j + ')' + '*' + internal_size1;
    }

  unsigned int alignment;
  bool is_rowmajor;
  bool is_transposed;
  std::string name;
  std::string size1;
  std::string size2;
  std::string internal_size1;
  std::string internal_size2;
  std::string row_start;
  std::string row_inc;
  std::string col_start;
  std::string col_inc;
  std::string scalartype;
  std::string aligned_scalartype;
};

struct scalar_descriptor{
  scalar_descriptor(std::string const & _name, std::string const & _scalartype) : name(_name), scalartype(_scalartype){ }
  std::string name;
  std::string scalartype;
  std::string arguments_string() const{
    return scalartype + " " + name;
  }
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


 class blas3_generator{
static void transform_block(matrix_descriptor const & mat_infos, bool store_shared
                            , unsigned int & large_block_1, unsigned int & large_block_2
                            , unsigned int & small_block_1, unsigned int & small_block_2){
    unsigned int alignment = mat_infos.alignment;
    if(mat_infos.is_rowmajor){
        if(mat_infos.is_transposed) large_block_1 /= alignment;
        else large_block_2/=alignment;
        if(!store_shared){
            if(mat_infos.is_transposed) small_block_1/=alignment;
            else small_block_2/=alignment;
        }
    }
    else{
        if(mat_infos.is_transposed) large_block_2 /= alignment;
        else large_block_1/=alignment;
        if(!store_shared){
            if(mat_infos.is_transposed)  small_block_2/=alignment;
            else    small_block_1/=alignment;
        }
    }

}


void transform_size(matrix_descriptor const & mat_infos){
    if(mat_infos.is_rowmajor){
        kss << mat_infos.internal_size2 << "/=" << mat_infos.alignment << ";" << std::endl;
    }
    else{
        kss << mat_infos.internal_size1 << "/=" << mat_infos.alignment << ";" << std::endl;
    }
}

void declare_rhs_global_ptr(unsigned int ks_rhs,unsigned int ns_rhs,
                           unsigned int nl_rhs, std::string const & offset_n){
    if(rhs_descriptor.is_rowmajor)
            for(unsigned int k = 0 ; k < ks_rhs ; ++k){
        std::string ptr_name = rhs_descriptor.name + "_ptr_" + to_string(k);
                kss << "__global " << rhs_descriptor.aligned_scalartype << " * " << ptr_name << " = " << rhs_descriptor.name << " + " ;
                if(rhs_descriptor.is_transposed) kss<< rhs_descriptor.offset(to_string(k) + " + " + offset_n + " +  get_group_id(1)*" + to_string(nl_rhs),"0");
                else kss << rhs_descriptor.offset(to_string(k),offset_n + " +  get_group_id(1)*" + to_string(nl_rhs));
                kss << ";" << std::endl;
            }
       else
            for(unsigned int n = 0 ; n < ns_rhs ; ++n){
        std::string ptr_name = rhs_descriptor.name + "_ptr_" + to_string(n);
                kss << "__global " << rhs_descriptor.aligned_scalartype << " * " << ptr_name << " = " << rhs_descriptor.name << " +  " ;
                if(rhs_descriptor.is_transposed)  kss << rhs_descriptor.offset(offset_n + " +  get_group_id(1)*" + to_string(nl_rhs), to_string(n));
                else kss << rhs_descriptor.offset("0",offset_n + " +  get_group_id(1)*" + to_string(nl_rhs) + " + " + to_string(n));
                kss << ";" << std::endl;
            }
}

void declare_lhs_global_ptr(unsigned int ms_lhs, unsigned int ks_lhs,
                           unsigned int ml_lhs, std::string const & offset_m){
         if(lhs_descriptor.is_rowmajor){
            for(unsigned int m=0; m<ms_lhs; ++m){
        std::string ptr_name = lhs_descriptor.name + "_ptr_" + to_string(m);
                kss << "__global " << lhs_descriptor.aligned_scalartype << " * " << ptr_name << " = " << lhs_descriptor.name << " + ";
                if(lhs_descriptor.is_transposed) kss << lhs_descriptor.offset(to_string(m),"get_group_id(0)*" + to_string(ml_lhs) + "+" + offset_m );
                else kss << lhs_descriptor.offset("get_group_id(0)*" + to_string(ml_lhs) + "+" + offset_m + "+" + to_string(m),"0");
                kss << ";" << std::endl;
            }
        }
        else{
            for(unsigned int k=0; k<ks_lhs; ++k){
        std::string ptr_name = lhs_descriptor.name + "_ptr_" + to_string(k);
                kss << "__global " << lhs_descriptor.aligned_scalartype << " * " << ptr_name << " = " << lhs_descriptor.name << " + " ;
                if(lhs_descriptor.is_transposed) kss << lhs_descriptor.offset("0", to_string(k) + "+" + "get_group_id(0)*" + to_string(ml_lhs) + "+" + offset_m );
                else kss << lhs_descriptor.offset( "get_group_id(0)*" + to_string(ml_lhs) + "+" + offset_m, to_string(k));
                kss << ";" << std::endl;
            }
        }
}

void update_rhs_global_ptr(unsigned int ks, unsigned int ns_rhs, unsigned int ks_rhs
                          ,std::string const & internal_size1_rhs,
                          std::string const & internal_size2_rhs){
      if(rhs_descriptor.is_rowmajor && !rhs_descriptor.is_transposed)
            for(unsigned int k=0 ; k<ks ; ++k)
                kss << rhs_descriptor.name << "_ptr_" << k << " += " << ks_rhs << "*" << internal_size2_rhs << " - " << ns_rhs << ";" << std::endl;
        else if(rhs_descriptor.is_transposed && !rhs_descriptor.is_rowmajor)
            for(unsigned int n=0 ; n<ns_rhs ; ++n)
                kss << rhs_descriptor.name << "_ptr_" << n << " += " << ns_rhs << "*" << internal_size1_rhs << " - " << ks_rhs << ";" << std::endl;
}

void update_lhs_global_ptr(unsigned int ks, unsigned int ms_lhs, unsigned int ks_lhs
                          ,std::string const & internal_size1_lhs,
                          std::string const & internal_size2_lhs){
      if(lhs_descriptor.is_transposed && lhs_descriptor.is_rowmajor)
            for(unsigned int m=0 ; m<ms_lhs ; ++m)
                kss << lhs_descriptor.name << "_ptr_" << m << " += " << ks << "*" << internal_size2_lhs << " - " <<  ks_lhs << ";" << std::endl;
        else if(!lhs_descriptor.is_transposed && !lhs_descriptor.is_rowmajor)
            for(unsigned int k=0 ; k<ks_lhs ; ++k)
                kss << lhs_descriptor.name << "_ptr_" << k << " += " << ks_lhs << "*" << internal_size1_lhs << " - " << ms_lhs << ";" << std::endl;
}

static std::string helper_variable(kernel_generation_stream & kss
                                    , bool store_in_register
                                    , std::string const & type
                                    , std::string const & name
                                    , std::string const & expr){
    if(!store_in_register)
        return expr;
    kss << type << " " << name << " = " << expr << ";" << std::endl;
    return name;
}

void fetch_to_local_mem(local_memory<2> const & lmem,
                               std::string const & offset,
                               unsigned int bound1,
                               unsigned int bound2,
                               matrix_descriptor & matrix){
    unsigned int alignment = matrix.alignment;
    std::string aligned_scalartype = matrix.aligned_scalartype;
    std::string scalartype = matrix.scalartype;
    std::string internal_size2 = matrix.internal_size2;
    std::string internal_size1 = matrix.internal_size1;
    kss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
    kss << "for(unsigned int i = get_local_id(0)" << " ; i < " << bound1 << "; i+= get_local_size(0)){" << std::endl;
    kss.inc_tab();
    kss << "for(unsigned int j = get_local_id(1)" << " ; j < " << bound2 << "; j+= get_local_size(1)){" << std::endl;
    kss.inc_tab();
    if(matrix.is_rowmajor){
        kss << aligned_scalartype << " val = " << matrix.name +  "[" + offset + " + j  + " + internal_size2 + "*i]" << ";" << std::endl;
        kss << "__local " << scalartype << "* ptr = " << lmem.name() << " + i*" << lmem.size2() << "+j*" << alignment<<";" <<std::endl;
        for(unsigned int a = 0 ; a < alignment ; ++a){
            if(alignment>1)
                kss << "*ptr++ =  val.s" << a << ";" << std::endl;
            else
                kss << "*ptr++ =  val;" << std::endl;
        }
    }
    else{
        kss << aligned_scalartype << " val = " << matrix.name + "[" + offset + "+ j*" + internal_size1 + " + i]" << ";" << std::endl;
        kss << "__local " << scalartype << "* ptr = " << lmem.name() << " + i*" << alignment * lmem.size2() << "+ j;" <<std::endl;
        for(unsigned int a = 0 ; a < alignment ; ++a){
            if(alignment>1)
                kss << "*ptr =  val.s" << a << ";" << std::endl;
            else
                kss << "*ptr =  val;" << std::endl;
            kss << "ptr += " << lmem.size2() << ";" << std::endl;
        }
    }

    kss.dec_tab();
    kss << "}" << std::endl;
    kss.dec_tab();
    kss << "}" << std::endl;
    kss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

}


public:
    blas3_generator(kernel_generation_stream & _kss,
          std::string const & scalartype,

          unsigned int _alignment,

          unsigned int _ml,
          unsigned int _kl,
          unsigned int _nl,

          unsigned int _ms,
          unsigned int _ks,
          unsigned int _ns,

          bool _use_LHS_shared,
          bool _use_RHS_shared,

          bool _is_lhs_rowmajor,
          bool _is_lhs_transposed,

          bool _is_rhs_rowmajor,
          bool _is_rhs_transposed,

          bool _is_result_rowmajor) : kss(_kss), lhs_descriptor(_alignment, _use_LHS_shared, _is_lhs_rowmajor, _is_lhs_transposed, "lhs", scalartype)
                            , rhs_descriptor(_alignment, _use_RHS_shared, _is_rhs_rowmajor, _is_rhs_transposed, "rhs", scalartype)
                            , result_descriptor(_alignment, false, _is_result_rowmajor, false, "res", scalartype)
                            , alpha_descriptor("alpha",scalartype)
                            , beta_descriptor("beta",scalartype){

    alignment = _alignment;
    ml = _ml;
    kl = _kl;
    nl = _nl;

    ms = _ms;
    ks = _ks;
    ns = _ns;

    use_LHS_shared = _use_LHS_shared;
    use_RHS_shared = _use_RHS_shared;
  }

    void operator()() {

    kss << "__kernel void prod_" << (lhs_descriptor.is_transposed ? "T" : "A") << (rhs_descriptor.is_transposed ? "T" : "A")
                        << "_amd( "
                        << alpha_descriptor.arguments_string() << std::endl
                        << "," << lhs_descriptor.arguments_string() << std::endl
                        << "," << rhs_descriptor.arguments_string() << std::endl
                        << "," << beta_descriptor.arguments_string() << std::endl
                        << "," << result_descriptor.arguments_string() << ")" << std::endl;
    kss << "{" << std::endl;
    kss.inc_tab();
        std::string lhs_value_scalartype;
        if(use_LHS_shared) lhs_value_scalartype = lhs_descriptor.scalartype;
        else lhs_value_scalartype = lhs_descriptor.aligned_scalartype;

        std::string rhs_value_scalartype;
        if(use_RHS_shared) rhs_value_scalartype = rhs_descriptor.scalartype;
        else rhs_value_scalartype = rhs_descriptor.aligned_scalartype;

        unsigned int ml_res = ml, nl_res = nl, ms_res = ms, ns_res = ns;
        unsigned int ml_lhs = ml, kl_lhs = kl, ms_lhs = ms, ks_lhs = ks;
        unsigned int kl_rhs = kl, nl_rhs = nl, ks_rhs = ks, ns_rhs = ns;

        transform_block(result_descriptor,false,ml_res,nl_res,ms_res,ns_res);
        transform_block(lhs_descriptor,use_LHS_shared,ml_lhs,kl_lhs,ms_lhs,ks_lhs);
        transform_block(rhs_descriptor,use_RHS_shared,kl_rhs,nl_rhs,ks_rhs,ns_rhs);

        std::string internal_size1_lhs = lhs_descriptor.internal_size1;
        std::string internal_size2_lhs = lhs_descriptor.internal_size2;

        std::string internal_size1_rhs = rhs_descriptor.internal_size1;
        std::string internal_size2_rhs = rhs_descriptor.internal_size2;

        std::string internal_size1_res = result_descriptor.internal_size1;
        std::string internal_size2_res = result_descriptor.internal_size2;

        unsigned int lhs_size1 = ml, lhs_size2 = kl;
        unsigned int rhs_size1 = kl, rhs_size2 = nl;
        if(lhs_descriptor.is_transposed) std::swap(lhs_size1, lhs_size2);
        if(rhs_descriptor.is_transposed) std::swap(rhs_size1, rhs_size2);

        local_memory<2> lmem_lhs("local_lhs",lhs_size1,lhs_size2+1,lhs_descriptor.scalartype);
        local_memory<2> lmem_rhs("local_rhs",rhs_size1,rhs_size2+1,lhs_descriptor.scalartype);

        //Declaration of results registers
        for(unsigned int m=0; m< ms_res; ++m)
            for(unsigned int n=0; n < ns_res ; ++n)
                kss << result_descriptor.aligned_scalartype << " " << "res" << m << n  << " = (" << result_descriptor.aligned_scalartype << ")(0) ;" << std::endl;

        //Declaration of local memories
        if(use_LHS_shared) kss << lmem_lhs.declare() << ";" << std::endl;
        if(use_RHS_shared) kss << lmem_rhs.declare() << ";" << std::endl;

        //Declaration of helpers
        transform_size(lhs_descriptor);
        transform_size(rhs_descriptor);
        transform_size(result_descriptor);

        std::string offset_m = helper_variable(kss,false,"unsigned int", "offset_m", "get_local_id(0)*" + to_string(ms_lhs));
        std::string offset_n = helper_variable(kss,false,"unsigned int", "offset_n", "get_local_id(1)*" + to_string(ns_rhs));
        std::string block_num = helper_variable(kss,true,"unsigned int", "block_num", (lhs_descriptor.is_transposed?internal_size1_lhs:internal_size2_lhs) + '/' + to_string(kl_lhs));

        //Declaration of pointers and/or offsets to result, rhs, lhs.
        kss << "__global " << result_descriptor.aligned_scalartype << "* res_ptr = " <<  result_descriptor.name << " + " << result_descriptor.offset("get_global_id(0)*" + to_string(ms_res), "get_global_id(1)*" + to_string(ns_res)) << ";" << std::endl;

        if(use_RHS_shared){
            if(rhs_descriptor.is_transposed) kss << "unsigned int offsetRHS = " << rhs_descriptor.offset(" get_group_id(1)*" + to_string(nl_rhs),"0") << ";" << std::endl;
            else kss << "unsigned int offsetRHS = " << rhs_descriptor.offset("0", " get_group_id(1)*" + to_string(nl_rhs)) << ";" << std::endl;
        }
        else{
            if(rhs_descriptor.is_transposed)
                declare_rhs_global_ptr(ns_rhs,ks_rhs,nl_rhs,offset_n);
            else
                declare_rhs_global_ptr(ks_rhs,ns_rhs,nl_rhs,offset_n);
        }

        if(use_LHS_shared){
            if(lhs_descriptor.is_transposed) kss << "unsigned int offsetLHS = " << lhs_descriptor.offset("0", "get_group_id(0)*" + to_string(ml_lhs)) << ";" << std::endl;
            else kss << "unsigned int offsetLHS = " << lhs_descriptor.offset("get_group_id(0)*" + to_string(ml_lhs), "0") << ";" << std::endl;
        }
        else{
            if(lhs_descriptor.is_transposed)
                declare_lhs_global_ptr(ks_lhs,ms_lhs,ml_lhs,offset_m);
            else
                declare_lhs_global_ptr(ms_lhs,ks_lhs,ml_lhs,offset_m);
        }



        //Main loop
        kss << "for(unsigned int bl=0 ; bl<" << block_num << " ; ++bl){" << std::endl;
        kss.inc_tab();

        //Fetches to local memory if necessary and declares pointers to local memory
        if(use_LHS_shared){
            if(lhs_descriptor.is_transposed) fetch_to_local_mem(lmem_lhs,"offsetLHS",kl_lhs,ml_lhs,lhs_descriptor);
            else fetch_to_local_mem(lmem_lhs,"offsetLHS",ml_lhs,kl_lhs,lhs_descriptor);
            unsigned int upper_bound = lhs_descriptor.is_transposed?ks_lhs:ms_lhs;
            for(unsigned int m=0; m<upper_bound; ++m){
                 kss << "__local " << lhs_value_scalartype << "* ptr_lhs_" << m << " = local_lhs + " ;
                if(lhs_descriptor.is_transposed) kss << m*lmem_lhs.size2() << " + " << offset_m ;
                else kss << "(" << offset_m << "+" << m << ")" << "*" << lmem_lhs.size2() ;
                kss << ";" << std::endl;
            }
        }

        if(use_RHS_shared){
            if(rhs_descriptor.is_transposed) fetch_to_local_mem(lmem_rhs,"offsetRHS",nl_rhs,kl_rhs,rhs_descriptor);
            else fetch_to_local_mem(lmem_rhs,"offsetRHS",kl_rhs,nl_rhs,rhs_descriptor);
            unsigned int upper_bound = rhs_descriptor.is_transposed?ns_rhs:ks_rhs;
            for(unsigned int k=0; k<upper_bound; ++k){
                kss << "__local " << rhs_value_scalartype << "* ptr_rhs_" << k << " = local_rhs + " ;
                if(rhs_descriptor.is_transposed) kss << "(" << offset_n << "+" << k << ")*" << lmem_rhs.size2();
                else kss << k*lmem_rhs.size2() << " + " << offset_n;
                kss << ";" << std::endl;
            }
        }


        kss << " for(unsigned int bs=0 ; bs < " << kl/ks  << " ; ++bs){" << std::endl;
        kss.inc_tab();


        unsigned int upperbound_1_rhs = rhs_descriptor.is_transposed?ns_rhs:ks_rhs;
        unsigned int upperbound_2_rhs = rhs_descriptor.is_transposed?ks_rhs:ns_rhs;
        for(unsigned int k = 0 ; k < upperbound_1_rhs ; ++k){
            for(unsigned int n=0 ; n < upperbound_2_rhs ; ++n){

                kss << rhs_value_scalartype << " val_rhs_" << k << "_" << n << " = " ;
                if(use_RHS_shared ) kss << "* ptr_rhs_" << k << "++";
                else{
                    if(rhs_descriptor.is_rowmajor) kss << "*" << rhs_descriptor.name + "_ptr_" + to_string(k);
                    else kss  << "*" << rhs_descriptor.name + "_ptr_" + to_string(n);
                }
                kss << ";";
                if( !use_RHS_shared ){
                        if(rhs_descriptor.is_rowmajor)kss << "++" << rhs_descriptor.name << "_ptr_" << k << ";" ;
                        else kss << "++" << rhs_descriptor.name << "_ptr_" << n << ";" ;
                }
                kss << std::endl;
            }
        }



        unsigned int upperbound_1_lhs = lhs_descriptor.is_transposed?ms_lhs:ks_lhs;
        unsigned int upperbound_2_lhs = lhs_descriptor.is_transposed?ks_lhs:ms_lhs;
        for(unsigned int k = 0 ; k < upperbound_1_lhs ; ++k){
            for(unsigned int m=0 ; m < upperbound_2_lhs ; ++m){
                kss << lhs_value_scalartype << " " << "val_lhs_" << m << "_" << k << " = ";
                if(use_LHS_shared) kss <<  "* ptr_lhs_" << m << "++" ;
                else if(lhs_descriptor.is_rowmajor) kss << "*" << lhs_descriptor.name + "_ptr_" + to_string(m);
                else kss << "*" << lhs_descriptor.name + "_ptr_" + to_string(k);
                kss << ";";
                if( !use_LHS_shared ){
                        if(lhs_descriptor.is_rowmajor) kss << "++" << lhs_descriptor.name << "_ptr_" << m << ";" ;
                        else kss << "++" << lhs_descriptor.name << "_ptr_" << k << ";" ;
                }
                kss << std::endl;
            }
        }



        for(unsigned int k = 0 ; k < ks ; ++k){
            for(unsigned int n=0 ; n < ns_res ; ++n){
                for(unsigned int m=0 ; m < ms_res ; ++m){
                    for(unsigned int a = 0; a<alignment; ++a){

                        int ind_lhs_1 = m;
                        int ind_lhs_2 = k;
                        int ind_s_lhs=a;

                        int ind_rhs_1=k;
                        int ind_rhs_2=n;
                        int ind_s_rhs=a;

                        bool is_vectorized_lhs = false;
                        bool is_vectorized_rhs = false;

                        if(result_descriptor.is_rowmajor){
                            if(lhs_descriptor.is_transposed) std::swap(ind_lhs_1,ind_lhs_2);

                            if(!use_LHS_shared){
                                if(lhs_descriptor.is_rowmajor){
                                    ind_s_lhs = ind_lhs_2%alignment;
                                    ind_lhs_2 /= alignment;
                                }
                                else{
                                    ind_s_lhs = ind_lhs_1%alignment;
                                    ind_lhs_1 /= alignment;
                                }
                            }
                        }
                        else{
                            if(use_LHS_shared){
                                ind_lhs_1 = ind_lhs_1*alignment+a;
                            }
                            else{
                                if((lhs_descriptor.is_rowmajor && !lhs_descriptor.is_transposed)
                                        ||(!lhs_descriptor.is_rowmajor && lhs_descriptor.is_transposed)){
                                    ind_lhs_1 = ind_lhs_1*alignment+a;
                                    ind_s_lhs = ind_lhs_2%alignment;
                                    ind_lhs_2 /= alignment;

                                }
                            }
                            if(lhs_descriptor.is_transposed) std::swap(ind_lhs_1,ind_lhs_2);
                        }

                        if(result_descriptor.is_rowmajor){
                            if(use_RHS_shared){
                                ind_rhs_2 = ind_rhs_2*alignment+a;
                            }
                            else{
                                if((!rhs_descriptor.is_rowmajor && !rhs_descriptor.is_transposed)
                                    ||(rhs_descriptor.is_rowmajor && rhs_descriptor.is_transposed)){
                                    ind_rhs_2 = ind_rhs_2*alignment+a;
                                    ind_s_rhs = ind_rhs_1%alignment;
                                    ind_rhs_1 = ind_rhs_1/alignment;
                                }
                                else if( (rhs_descriptor.is_rowmajor && !rhs_descriptor.is_transposed) ){
                                    is_vectorized_rhs=true;
                                }
                            }
                            if(rhs_descriptor.is_transposed) std::swap(ind_rhs_1,ind_rhs_2);
                        }
                        else{
                            if(rhs_descriptor.is_transposed) std::swap(ind_rhs_1,ind_rhs_2);
                            if(!use_RHS_shared){
                                if(rhs_descriptor.is_rowmajor){
                                    ind_s_rhs = ind_rhs_2%alignment;
                                    ind_rhs_2/=alignment;
                                }
                                else{
                                    ind_s_rhs = ind_rhs_1%alignment;
                                    ind_rhs_1/=alignment;
                                }
                            }
                        }

                        bool is_vectorized = is_vectorized_lhs || is_vectorized_rhs;

                        std::ostringstream res_oss;
                        std::ostringstream lhs_oss;
                        std::ostringstream rhs_oss;

                        res_oss << "res" << m << n ;
                        if(!is_vectorized && alignment>1) res_oss << ".s" << a;

                        lhs_oss << "val_lhs_" << ind_lhs_1 << "_" << ind_lhs_2;
                        if(!is_vectorized_lhs && !use_LHS_shared && alignment>1) lhs_oss << ".s" << ind_s_lhs;


                        rhs_oss << "val_rhs_" << ind_rhs_1 << "_" << ind_rhs_2;
                        if(!is_vectorized_rhs && !use_RHS_shared && alignment>1) rhs_oss << ".s" << ind_s_rhs;


                        kss << res_oss.str() << "+=" <<  lhs_oss.str() << "*" << rhs_oss.str() << ";" << std::endl;


                        if(is_vectorized)
                            break;
                    }
                }
            }
        }


        if(use_RHS_shared){
            for(unsigned int k=0 ; k<ks ; ++k)
                if(!rhs_descriptor.is_transposed) kss << "ptr_rhs_" << k << " += " << ks_rhs*lmem_rhs.size2() - ns_rhs << ";" << std::endl;
        }
        else{
            if(rhs_descriptor.is_transposed)
                update_rhs_global_ptr(ks,ks_rhs,ns_rhs,internal_size1_rhs,internal_size2_rhs);
            else
                update_rhs_global_ptr(ks,ns_rhs,ks_rhs,internal_size1_rhs,internal_size2_rhs);
        }



        if(use_LHS_shared){
            for(unsigned int m=0 ; m<ks_lhs ; ++m)
                if(lhs_descriptor.is_transposed) kss << "ptr_lhs_" << m << " += " << ks*lmem_lhs.size2() - ms_lhs << ";" << std::endl;
        }
        else{
            if(lhs_descriptor.is_transposed)
                update_lhs_global_ptr(ks,ks_lhs,ms_lhs,internal_size1_lhs,internal_size2_lhs);
            else
                update_lhs_global_ptr(ks,ms_lhs,ks_lhs,internal_size1_lhs,internal_size2_lhs);
        }



        kss.dec_tab();
        kss << "}" << std::endl;

        if(use_LHS_shared){
            if(lhs_descriptor.is_transposed){
                if(lhs_descriptor.is_rowmajor)
                    kss << "offsetLHS += " << kl_lhs << "*" << internal_size2_lhs << ";" << std::endl;
                else
                    kss << "offsetLHS += " << kl_lhs  << ";" << std::endl;
            }
            else{
                if(lhs_descriptor.is_rowmajor)
                    kss << "offsetLHS += " << kl_lhs << ";" << std::endl;
                else
                    kss << "offsetLHS += " << kl_lhs << "*" << internal_size1_lhs << ";" << std::endl;
            }

        }

        if(use_RHS_shared){
            if(rhs_descriptor.is_transposed){
                if(rhs_descriptor.is_rowmajor)
                    kss << "offsetRHS += " << kl_rhs << ";" << std::endl;
                else
                    kss << "offsetRHS += " << kl_rhs << "*" << internal_size1_rhs << ";" << std::endl;
            }
            else{
                if(rhs_descriptor.is_rowmajor)
                    kss << "offsetRHS += " << kl_rhs << "*" << internal_size2_rhs << ";" << std::endl;
                else
                    kss << "offsetRHS += " << kl_rhs << ";" << std::endl;
            }
        }

        kss.dec_tab();
        kss << "}" << std::endl;

        if(result_descriptor.is_rowmajor){
            for(unsigned int m=0 ; m < ms_res ; ++m){
                for(unsigned int n=0 ; n < ns_res ; ++n){
                    kss << "*res_ptr = (" << beta_descriptor.name << " != 0) ? " << alpha_descriptor.name << "*" << "res" << m << n  << " + " << beta_descriptor.name << " * *res_ptr : "
                                                                                 << alpha_descriptor.name << "*" << "res" << m << n  << ";" << std::endl;
                    kss << "res_ptr++;" << std::endl;
                }
                if(m<ms_res-1)  kss << "res_ptr+=" << internal_size2_res << " - " << ns_res << ";" << std::endl;
            }
        }
        else{
            for(unsigned int n=0 ; n < ns_res ; ++n){
                for(unsigned int m=0 ; m < ms_res ; ++m){
                    kss << "*res_ptr = (" << beta_descriptor.name << " != 0) ? " << alpha_descriptor.name << "*" << "res" << m << n  << " + " << beta_descriptor.name << " * *res_ptr : "
                                                                                 << alpha_descriptor.name << "*" << "res" << m << n  << ";" << std::endl;
                    kss << "res_ptr++;" << std::endl;
                }
                if(n<ns_res-1) kss << "res_ptr+=" << internal_size1_res << " - " << ms_res << ";" << std::endl;
            }
        }
        kss.dec_tab();
        kss << "}" << std::endl;


    }

private:
    kernel_generation_stream & kss;

  unsigned int alignment;

  unsigned int ml;
  unsigned int kl;
  unsigned int nl;

  unsigned int ms;
  unsigned int ks;
  unsigned int ns;

  bool use_LHS_shared;
  bool use_RHS_shared;

  matrix_descriptor lhs_descriptor;
  matrix_descriptor rhs_descriptor;
  matrix_descriptor result_descriptor;
  scalar_descriptor alpha_descriptor;
  scalar_descriptor beta_descriptor;
};

int main(int args, char **argv){
  kernel_generation_stream kss(std::cout);

  std::string scalartype = "float";

  unsigned int alignment = 4;

  unsigned int ml = 32;
  unsigned int kl = 64;
  unsigned int nl = 128;

  unsigned int ms = 4;
  unsigned int ks = 8;
  unsigned int ns = 4;

        if (args != 6)
        {
          std::cout << "Wrong number of arguments (" << args << " instead of 6). Aborting..." << std::endl;
          return EXIT_FAILURE;
        }

  bool use_LHS_shared = true;
  bool use_RHS_shared = false;

  bool is_lhs_rowmajor   = (argv[1][0] == '1');
  bool is_lhs_transposed = (argv[4][0] == '1');

  bool is_rhs_rowmajor   = (argv[2][0] == '1');
  bool is_rhs_transposed = (argv[5][0] == '1');

  bool is_result_rowmajor = (argv[3][0] == '1');

  blas3_generator gen(kss, scalartype, alignment
          , ml, kl, nl
          , ms, ks, ns
          , use_LHS_shared, use_RHS_shared,
           is_lhs_rowmajor, is_lhs_transposed
           , is_rhs_rowmajor, is_rhs_transposed
           , is_result_rowmajor);

  gen();

}
