
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

#ifndef _RANDOM_HPP_
#define _RANDOM_HPP_

#include <time.h>
#include <stdlib.h>

inline void init()
{
	static bool init = false;
	if (!init)
	{
		srand( (unsigned int)time(NULL) );
		init = true;
	}
}

template<class TYPE>
TYPE random();

template<>
double random<double>()
{
  init();
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

template<>
float random<float>()
{
  init();
  return static_cast<float>(random<double>());
}

#endif

