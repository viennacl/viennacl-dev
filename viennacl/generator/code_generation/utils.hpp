#ifndef VIENNACL_GENERATOR_CODE_GENERATION_UTILS_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_UTILS_HPP

#include <list>
#include <set>
#include <algorithm>
#include <ostream>
#include <map>

namespace viennacl{

    namespace generator{



        namespace code_generation{



            namespace utils{


                template<class T>
                struct is_type{
                    bool operator()(infos_base* p) const{
                        return dynamic_cast<T *>(p);
                    }
                };


                template<class T>
                struct deref_t{ typedef deref_less type; };

                template<class T>
                struct deref_t<T*>{ typedef double_deref_less type; };

                template<class T>
                void remove_unsorted_duplicates(std::list<T> &the_list) {
                  std::set<typename std::list<T>::iterator, typename deref_t<T>::type> found;
                  for (typename std::list<T>::iterator x = the_list.begin(); x != the_list.end();) {
                    if (!found.insert(x).second) {
                      x = the_list.erase(x);
                    }
                    else {
                      ++x;
                    }
                  }
                }



                struct EXTRACT_IF{
                    typedef std::list<infos_base*> result_type_single;
                    typedef std::list<infos_base*> result_type_all;
                    static void do_on_new_res(result_type_single & new_res, result_type_single & res){
                        res.merge(new_res);
                    }
                    static void do_on_pred_true(infos_base* tree,result_type_single & res){
                        res.push_back(tree);
                    }
                    static void do_on_next_operation_merge(result_type_single & new_res, result_type_all & final_res){
                        final_res.merge(new_res);
                    }
                };


                template<class FILTER_T, class Pred>
                static typename FILTER_T::result_type_single filter(infos_base* const tree, Pred pred){
                    typedef typename FILTER_T::result_type_single res_t;
                    res_t res;
                    if(binary_tree_infos_base * p = dynamic_cast<binary_tree_infos_base *>(tree)){
                        res_t  reslhs(filter<FILTER_T,Pred>(&p->lhs(),pred));
                        res_t resrhs(filter<FILTER_T,Pred>(&p->rhs(),pred));
                        FILTER_T::do_on_new_res(reslhs,res);
                        FILTER_T::do_on_new_res(resrhs,res);
                    }
                    else if(unary_tree_infos_base * p = dynamic_cast<unary_tree_infos_base *>(tree)){
                        res_t  ressub(filter<FILTER_T,Pred>(&p->sub(),pred));
                        FILTER_T::do_on_new_res(ressub,res);
                    }
                    if(pred(tree)){
                        FILTER_T::do_on_pred_true(tree,res);
                    }
                    return res;
                }

                template<class FILTER_T, class Pred>
                static typename FILTER_T::result_type_all filter(std::list<infos_base*> const & trees, Pred pred){
                    typedef typename FILTER_T::result_type_single res_t_single;
                    typedef typename FILTER_T::result_type_all res_t_all;
                    res_t_all res;
                    for(std::list<infos_base*>::const_iterator it = trees.begin() ; it != trees.end() ; ++it){
                        res_t_single tmp(filter<FILTER_T,Pred>(*it,pred));

                        FILTER_T::do_on_next_operation_merge(tmp,res);
                    }

                    return res;
                }




                template<class T, class B>
                static std::list<T *> cast(std::list<B *> const & in){
                    std::list<T*> res;
                    for(typename std::list<B *>::const_iterator it = in.begin(); it!= in.end() ; ++it){
                        if(T* p = dynamic_cast<T*>(*it)){
                            res.push_back(p);
                        }
                    }
                    return res;
                }

                template<class T>
                static std::list<T *> extract_cast(std::list<infos_base*> const & trees){
                    return cast<T,infos_base>(filter<EXTRACT_IF>(trees,is_type<T>()));
                }


//                template<class T>
//                class cache_manager{
//                public:
//                    typedef std::set<T *, viennacl::generator::deref_less> expressions_read_t;
//                    typedef std::list<T *> expressions_write_t;
//                    cache_manager( expressions_read_t & expressions_read
//                                  ,expressions_write_t const & expressions_write
//                                  ,kernel_generation_stream & kss) : expressions_read_(expressions_read), expressions_write_(expressions_write)
//                                                                              ,kss_(kss){
//                    }

//                    void fetch(unsigned int i){
//                        for(typename expressions_read_t::iterator it = expressions_read_.begin() ; it != expressions_read_.end() ; ++it){
//                            T * p = *it;
//                            std::string val_name = p->name()+"_val_"+to_string(i);
//                            old_access_names_[i] = p->generate(i);
//                            kss_ << p->aligned_scalartype() << " " << val_name << " = " << old_access_names_[i] << ";" << std::endl;
//                            p->access_name(i,val_name);
//                        }
//                    }

//                    void writeback(unsigned int i){
//                        for(typename expressions_write_t::iterator it = expressions_write_.begin() ; it != expressions_write_.end() ; ++it){
//                            T * p = *it;
//                            kss_<< old_access_names_[i] << " = "  << p->generate(i) << ";" << std::endl;
//                        }
//                    }

//                private:
//                    std::map<unsigned int, std::string> old_access_names_;
//                    expressions_read_t & expressions_read_;
//                    expressions_write_t expressions_write_;
//                    kernel_generation_stream & kss_;

//                };

//                class loop_unroller{
//                public:
//                    loop_unroller(unsigned int n_unroll) : n_unroll_(n_unroll){

//                    }



//                private:
//                    unsigned int n_unroll_;

//                };

//                template<class ExprT, class CacheExpr>
//                static void unroll_loop(kernel_generation_stream & kss
//                                                       ,unsigned int n_unroll
//                                                   ,ExprT & expressions
//                                                   , CacheExpr & cache
//                                                   , std::string const & upper_bound){
//                        kss << "unsigned int i = get_global_id(0)" ;
//                        if(n_unroll>1) kss << "*" << n_unroll;
//                        kss << ";" << std::endl;

//                        kss << "if(i < " << upper_bound << "){" << std::endl;
//                        kss.inc_tab();
//                        cache.fetch_entries(0, "i");
//                        for(unsigned int j=1 ; j<n_unroll  ; ++j){
//                            cache.fetch_entries(j, "i + " + to_string(j));
//                        }


//                        for(typename ExprT::iterator it=expressions.begin() ; it!=expressions.end();++it){
//                            for(unsigned int j=0 ; j < n_unroll ; ++j){
//                                kss << (*it)->generate(j) << ";" << std::endl;
//                            }
//                        }

//                        cache.writeback_entries(0,"i");
//                        for(unsigned int j=1 ; j<n_unroll  ; ++j){
//                            cache.writeback_entries(j,"i + " + to_string(j));
//                        }
//                        kss.dec_tab();
//                        kss << "}" << std::endl;

//                }

            }

        }

    }

}
#endif // UTILS_HPP
