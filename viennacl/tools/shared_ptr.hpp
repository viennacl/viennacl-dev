#ifndef VIENNACL_TOOLS_SHARED_PTR_HPP
#define VIENNACL_TOOLS_SHARED_PTR_HPP

namespace viennacl{

namespace tools{


namespace detail{

class count{
public:
    count(unsigned int val) : val_(val){ }
    void dec(){ --val_; }
    void inc(){ ++val_; }
    bool is_null(){ return val_ == 0; }
    unsigned int val(){ return val_; }
private:
    unsigned int val_;
};

}


template<class T>
class shared_ptr
{
    struct aux
    {
        detail::count count;

        aux() :count(1) {}
        virtual void destroy()=0;
        virtual ~aux() {}
    };

    template<class U, class Deleter>
    struct auximpl: public aux
    {
        U* p;
        Deleter d;

        auximpl(U* pu, Deleter x) :p(pu), d(x) {}
        virtual void destroy() { d(p); }
    };

    template<class U>
    struct default_deleter
    {
        void operator()(U* p) const { delete p; }
    };

    aux* pa;
    T* pt;

    void inc() { if(pa) pa->count.inc(); }

    void dec()
    {
        if(pa){
            pa->count.dec();
            if(pa->count.is_null()){
                pa->destroy();
                delete pa;
            }
        }
    }

public:

    shared_ptr() :pa(), pt() {}

    template<class U, class Deleter>
    shared_ptr(U* pu, Deleter d) : pa(new auximpl<U,Deleter>(pu,d)), pt(pu) {}

    template<class U>
    explicit shared_ptr(U* pu) : pa(new auximpl<U,default_deleter<U> >(pu,default_deleter<U>())), pt(pu) {}

    shared_ptr(const shared_ptr& s) :pa(s.pa), pt(s.pt) {
        inc();
    }

    template<class U>
    shared_ptr(const shared_ptr<U>& s) :pa(s.pa), pt(s.pt) { inc(); }

    ~shared_ptr() {
        dec();
    }

    shared_ptr& operator=(const shared_ptr& s)
    {
        if(this!=&s)
        {
            dec();
            pa = s.pa;
            pt = s.pt;
            inc();
        }
        return *this;
    }

    T* get() const {  return pt; }

    T& operator*() const { return *pt; }
};

}

}

#endif // VIENNACL_UTILS_SHARED_PTR_HPP
