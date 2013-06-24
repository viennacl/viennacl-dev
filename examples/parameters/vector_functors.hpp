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



/////////////////// full optimization functors /////////////////////

// add kernels:
// struct vector_add
// {
//   template <typename TestData>
//   void operator()(TestData & data)
//   {
//     std::cerr << "add now!" << std::endl;
//     data.v3 = data.v1 + data.v2;
//   }
// };

template <typename TestData>
void vector_add(TestData & data)
{
  data.v3 = data.v1 + data.v2;
}

template <typename TestData>
void vector_inplace_add(TestData & data)
{
  data.v3 += data.v1;
}

template <typename TestData>
void vector_mul_add(TestData & data)
{
  data.v3 = data.v1 + data.s1 * data.v2;
}

template <typename TestData>
void vector_cpu_mul_add(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  data.v3 = data.v1 + NumericT(2.0) * data.v2;
}

template <typename TestData>
void vector_inplace_mul_add(TestData & data)
{
  data.v3 += data.s1 * data.v2;
}

template <typename TestData>
void vector_cpu_inplace_mul_add(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  data.v3 += NumericT(2.0) * data.v2;
}


template <typename TestData>
void vector_inplace_div_add(TestData & data)
{
  data.v3 += data.v2 / data.s1;
}


// sub kernels:
template <typename TestData>
void vector_sub(TestData & data)
{
  data.v3 = data.v1 - data.v2; //a plain vector subtraction
}

template <typename TestData>
void vector_inplace_sub(TestData & data)
{
  data.v3 -= data.v1; //a plain vector subtraction
}

template <typename TestData>
void vector_mul_sub(TestData & data)
{
  data.v3 = data.v1 - data.s1 * data.v2;
}

template <typename TestData>
void vector_inplace_mul_sub(TestData & data)
{
  data.v3 -= data.s1 * data.v2;
}

template <typename TestData>
void vector_inplace_div_sub(TestData & data)
{
  data.v3 -= data.v2 / data.s1;
}


// mult kernels:
template <typename TestData>
void vector_mult(TestData & data)
{
  data.v3 = data.s1 * data.v2;
}

template <typename TestData>
void vector_cpu_mult(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  data.v3 = NumericT(2.0) * data.v2;
}

template <typename TestData>
void vector_inplace_mult(TestData & data)
{
  data.v3 *= data.s1;
}

template <typename TestData>
void vector_cpu_inplace_mult(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  data.v3 *= NumericT(2.0);
}


// div kernels:
template <typename TestData>
void vector_divide(TestData & data)
{
  data.v3 = data.v2 / data.s1;
}

template <typename TestData>
void vector_inplace_divide(TestData & data)
{
  data.v3 /= data.s1;
}


// other kernels:
template <typename TestData>
void vector_inner_prod(TestData & data)
{
  data.s1 = viennacl::linalg::inner_prod(data.v1, data.v2);
}

template <typename TestData>
void vector_swap(TestData & data)
{
  swap(data.v2, data.v3);
}

template <typename TestData>
void vector_clear(TestData & data)
{
  data.v3.clear();
}

template <typename TestData>
void vector_plane_rotation(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  viennacl::linalg::plane_rotation(data.v1, data.v2, NumericT(1.0), NumericT(2.0)); //a plain vector addition
}

template <typename TestData>
void vector_norm_1(TestData & data)
{
  data.s1 = viennacl::linalg::norm_1(data.v3);
}

template <typename TestData>
void vector_norm_2(TestData & data)
{
  data.s1 = viennacl::linalg::norm_2(data.v3);
}

template <typename TestData>
void vector_norm_inf(TestData & data)
{
  data.s1 = viennacl::linalg::norm_inf(data.v3);
}

/////////////////// restricted optimization functors /////////////////////

template <typename TestData>
void vector_index_norm_inf(TestData & data)
{
  viennacl::linalg::index_norm_inf(data.v3);
}


