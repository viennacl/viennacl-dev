/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#ifndef _COMMAND_LINE_UTILS_HPP_
#define _COMMAND_LINE_UTILS_HPP_

#include <tclap/CmdLine.h>

std::vector<unsigned int> get_values_in_commas(std::string const & s){
    std::vector<unsigned int> res;
    std::size_t old_comma_pos = 0, new_comma_pos;
    while((new_comma_pos = s.find(',',old_comma_pos))!= std::string::npos){
        res.push_back(atoi(s.substr(old_comma_pos,new_comma_pos).c_str()));
        old_comma_pos = new_comma_pos+1;
    }
    res.push_back(atoi(s.substr(old_comma_pos,s.length()).c_str()));
    return res;
}

class pow_2_interval_constraint : public TCLAP::Constraint<std::string>{
    static bool is_pow_of_two(const unsigned int x){ return ((x != 0) && !(x & (x - 1))); }
public:
    bool check(std::string const & s) const{
        std::vector<unsigned int> vals = get_values_in_commas(s);
        return vals.size()==2 && is_pow_of_two(vals[0]) && is_pow_of_two(vals[1]);
    }
    std::string shortID() const { return "min,max"; }
    std::string description() const { return "Must be a power of two"; }
};

class min_max_inc_constraint : public TCLAP::Constraint<std::string>{
public:
    bool check(std::string const & s) const{
        std::vector<unsigned int> vals = get_values_in_commas(s);
        return vals.size()==3;
    }
    std::string shortID() const { return "min,max,inc"; }
    std::string description() const { return "Must contain minimum value, maximum value and increment"; }
};

#endif
