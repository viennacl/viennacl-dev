#ifndef VIENNACL_DISTRIBUTED_CPU_MATRIX
#define VIENNACL_DISTRIBUTED_CPU_MATRIX

#include <vector>

namespace viennacl{

namespace distributed{

template<typename SCALARTYPE, class F>
class cpu_matrix{
public:

    cpu_matrix(unsigned int size1, unsigned int size2){
        size1_ = size1;
        size2_ = size2;
        elements_.resize(size1*size2);
    }

    SCALARTYPE const & operator()(unsigned int i, unsigned int j) const{
        assert(i < size1_ && "Out of bound access");
        assert(j < size2_ && "Out of bound access");
        return elements_[F::mem_index(i, j, size1_, size2_)];
    }


    SCALARTYPE &  operator()(unsigned int i, unsigned int j){
        assert(i < size1_ && "Out of bound access");
        assert(j < size2_ && "Out of bound access");
        return elements_[F::mem_index(i, j, size1_, size2_)];
    }

    unsigned int size1() const { return size1_; }

    unsigned int size2() const { return size2_; }

private:
    std::vector<SCALARTYPE> elements_;
    unsigned int size1_;
    unsigned int size2_;
};

}

}
#endif
