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

            }

        }

    }

}
#endif // UTILS_HPP
