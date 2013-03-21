/*
Part of MWC64X by David Thomas, dt10@imperial.ac.uk
This is provided under BSD, full license is with the main package.
See http://www.doc.ic.ac.uk/~dt10/research
*/


// Pre: a<M, b<M
// Post: r=(a+b) mod M
ulong MWC_AddMod64(ulong a, ulong b, ulong M)
{
        ulong v=a+b;
        if( (v>=M) || (v<a) )
                v=v-M;
        return v;
}

// Pre: a<M,b<M
// Post: r=(a*b) mod M
// This could be done more efficently, but it is portable, and should
// be easy to understand. It can be replaced with any of the better
// modular multiplication algorithms (for example if you know you have
// double precision available or something).
ulong MWC_MulMod64(ulong a, ulong b, ulong M)
{
        ulong r=0;
        while(a!=0){
                if(a&1)
                        r=MWC_AddMod64(r,b,M);
                b=MWC_AddMod64(b,b,M);
                a=a>>1;
        }
        return r;
}


// Pre: a<M, e>=0
// Post: r=(a^b) mod M
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on
// most architectures
ulong MWC_PowMod64(ulong a, ulong e, ulong M)
{
        ulong sqr=a, acc=1;
        while(e!=0){
                if(e&1)
                        acc=MWC_MulMod64(acc,sqr,M);
                sqr=MWC_MulMod64(sqr,sqr,M);
                e=e>>1;
        }
        return acc;
}

uint2 MWC_SkipImpl_Mod64(uint2 curr, ulong A, ulong M, ulong distance)
{
        ulong m=MWC_PowMod64(A, distance, M);
        ulong x=curr.x*(ulong)A+curr.y;
        x=MWC_MulMod64(x, m, M);
        return (uint2)((uint)(x/A), (uint)(x%A));
}

uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap)
{
        // This is an arbitrary constant for starting LCG jumping from. I didn't
        // want to start from 1, as then you end up with the two or three first values
        // being a bit poor in ones - once you've decided that, one constant is as
        // good as any another. There is no deep mathematical reason for it, I just
        // generated a random number.
        enum{ MWC_BASEID = 4077358422479273989UL };

        ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap;
        ulong m=MWC_PowMod64(A, dist, M);

        ulong x=MWC_MulMod64(MWC_BASEID, m, M);
        return (uint2)((uint)(x/A), (uint)(x%A));
}

//! Represents the state of a particular generator

typedef struct{ uint x; uint c; } mwc64x_state_t;

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

void MWC64X_Step(mwc64x_state_t *s)
{
        uint X=s->x, C=s->c;

        uint Xn=MWC64X_A*X+C;
        uint carry=(uint)(Xn<C);				// The (Xn<C) will be zero or one for scalar
        uint Cn=mad_hi(MWC64X_A,X,carry);

        s->x=Xn;
        s->c=Cn;
}

void MWC64X_Skip(mwc64x_state_t *s, ulong distance)
{
        uint2 tmp=MWC_SkipImpl_Mod64((uint2)(s->x,s->c), MWC64X_A, MWC64X_M, distance);
        s->x=tmp.x;
        s->c=tmp.y;
}

void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
        uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
        s->x=tmp.x;
        s->c=tmp.y;
}

//! Return a 32-bit integer in the range [0..2^32)
uint MWC64X_NextUint(mwc64x_state_t *s)
{
        uint res=s->x ^ s->c;
        MWC64X_Step(s);
        return res;
}



//! Represents the state of a particular generator
typedef struct{ uint2 x; uint2 c; } mwc64xvec2_state_t;

enum{ MWC64XVEC2_A = 4294883355U };
enum{ MWC64XVEC2_M = 18446383549859758079UL };

void MWC64XVEC2_Step(mwc64xvec2_state_t *s)
{
        uint2 X=s->x, C=s->c;

        uint2 Xn=MWC64XVEC2_A*X+C;
        // Note that vector comparisons return -1 for true, so we have to do this negation
        // I would hope that the compiler would do something sensible if possible...
        uint2 carry=as_uint2(-(Xn<C));
        uint2 Cn=mad_hi((uint2)MWC64XVEC2_A,X,carry);

        s->x=Xn;
        s->c=Cn;
}

void MWC64XVEC2_Skip(mwc64xvec2_state_t *s, ulong distance)
{
        uint2 tmp0=MWC_SkipImpl_Mod64((uint2)(s->x.s0,s->c.s0), MWC64XVEC2_A, MWC64XVEC2_M, distance);
        uint2 tmp1=MWC_SkipImpl_Mod64((uint2)(s->x.s1,s->c.s1), MWC64XVEC2_A, MWC64XVEC2_M, distance);
        s->x=(uint2)(tmp0.x, tmp1.x);
        s->c=(uint2)(tmp0.y, tmp1.y);
}

void MWC64XVEC2_SeedStreams(mwc64xvec2_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
        uint2 tmp0=MWC_SeedImpl_Mod64(MWC64XVEC2_A, MWC64XVEC2_M, 2, 0, baseOffset, perStreamOffset);
        uint2 tmp1=MWC_SeedImpl_Mod64(MWC64XVEC2_A, MWC64XVEC2_M, 2, 1, baseOffset, perStreamOffset);
        s->x=(uint2)(tmp0.x, tmp1.x);
        s->c=(uint2)(tmp0.y, tmp1.y);
}

//! Return a 32-bit integer in the range [0..2^32)
uint2 MWC64XVEC2_NextUint2(mwc64xvec2_state_t *s)
{
        uint2 res=s->x ^ s->c;
        MWC64XVEC2_Step(s);
        return res;
}


//! Represents the state of a particular generator
typedef struct{ uint4 x; uint4 c; } mwc64xvec4_state_t;

enum{ MWC64XVEC4_A = 4294883355U };
enum{ MWC64XVEC4_M = 18446383549859758079UL };

void MWC64XVEC4_Step(mwc64xvec4_state_t *s)
{
        uint4 X=s->x, C=s->c;

        uint4 Xn=MWC64XVEC4_A*X+C;
        // Note that vector comparisons return -1 for true, so we have to do this odd negation
        // I would hope that the compiler would do something sensible if possible...
        uint4 carry=as_uint4(-(Xn<C));
        uint4 Cn=mad_hi((uint4)MWC64XVEC4_A,X,carry);

        s->x=Xn;
        s->c=Cn;
}

void MWC64XVEC4_Skip(mwc64xvec4_state_t *s, ulong distance)
{
        uint2 tmp0=MWC_SkipImpl_Mod64((uint2)(s->x.s0,s->c.s0), MWC64XVEC4_A, MWC64XVEC4_M, distance);
        uint2 tmp1=MWC_SkipImpl_Mod64((uint2)(s->x.s1,s->c.s1), MWC64XVEC4_A, MWC64XVEC4_M, distance);
        uint2 tmp2=MWC_SkipImpl_Mod64((uint2)(s->x.s2,s->c.s2), MWC64XVEC4_A, MWC64XVEC4_M, distance);
        uint2 tmp3=MWC_SkipImpl_Mod64((uint2)(s->x.s3,s->c.s3), MWC64XVEC4_A, MWC64XVEC4_M, distance);
        s->x=(uint4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
        s->c=(uint4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
}

void MWC64XVEC4_SeedStreams(mwc64xvec4_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
        uint2 tmp0=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 0, baseOffset, perStreamOffset);
        uint2 tmp1=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 1, baseOffset, perStreamOffset);
        uint2 tmp2=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 2, baseOffset, perStreamOffset);
        uint2 tmp3=MWC_SeedImpl_Mod64(MWC64XVEC4_A, MWC64XVEC4_M, 4, 3, baseOffset, perStreamOffset);
        s->x=(uint4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
        s->c=(uint4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
}

//! Return a 32-bit integer in the range [0..2^32)
uint4 MWC64XVEC4_NextUint4(mwc64xvec4_state_t *s)
{
        uint4 res=s->x ^ s->c;
        MWC64XVEC4_Step(s);
        return res;
}


//! Represents the state of a particular generator
typedef struct{ uint8 x; uint8 c; } mwc64xvec8_state_t;

enum{ MWC64XVEC8_A = 4294883355U };
enum{ MWC64XVEC8_M = 18446383549859758079UL };

void MWC64XVEC8_Step(mwc64xvec8_state_t *s)
{
        uint8 X=s->x, C=s->c;

        uint8 Xn=MWC64XVEC8_A*X+C;
        // Note that vector comparisons return -1 for true, so we have to do this odd negation
        // I would hope that the compiler would do something sensible if possible...
        uint8 carry=as_uint8(-(Xn<C));
        uint8 Cn=mad_hi((uint8)MWC64XVEC8_A,X,carry);

        s->x=Xn;
        s->c=Cn;
}

void MWC64XVEC8_Skip(mwc64xvec8_state_t *s, ulong distance)
{
        uint2 tmp0=MWC_SkipImpl_Mod64((uint2)(s->x.s0,s->c.s0), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp1=MWC_SkipImpl_Mod64((uint2)(s->x.s1,s->c.s1), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp2=MWC_SkipImpl_Mod64((uint2)(s->x.s2,s->c.s2), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp3=MWC_SkipImpl_Mod64((uint2)(s->x.s3,s->c.s3), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp4=MWC_SkipImpl_Mod64((uint2)(s->x.s4,s->c.s4), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp5=MWC_SkipImpl_Mod64((uint2)(s->x.s5,s->c.s5), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp6=MWC_SkipImpl_Mod64((uint2)(s->x.s6,s->c.s6), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        uint2 tmp7=MWC_SkipImpl_Mod64((uint2)(s->x.s7,s->c.s7), MWC64XVEC8_A, MWC64XVEC8_M, distance);
        s->x=(uint8)(tmp0.x, tmp1.x, tmp2.x, tmp3.x, tmp4.x, tmp5.x, tmp6.x, tmp7.x);
        s->c=(uint8)(tmp0.y, tmp1.y, tmp2.y, tmp3.y, tmp4.y, tmp5.y, tmp6.y, tmp7.y);
}

void MWC64XVEC8_SeedStreams(mwc64xvec8_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
        uint2 tmp0=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 0, baseOffset, perStreamOffset);
        uint2 tmp1=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 1, baseOffset, perStreamOffset);
        uint2 tmp2=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 2, baseOffset, perStreamOffset);
        uint2 tmp3=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 3, baseOffset, perStreamOffset);
        uint2 tmp4=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 4, baseOffset, perStreamOffset);
        uint2 tmp5=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 5, baseOffset, perStreamOffset);
        uint2 tmp6=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 6, baseOffset, perStreamOffset);
        uint2 tmp7=MWC_SeedImpl_Mod64(MWC64XVEC8_A, MWC64XVEC8_M, 8, 7, baseOffset, perStreamOffset);
        s->x=(uint8)(tmp0.x, tmp1.x, tmp2.x, tmp3.x, tmp4.x, tmp5.x, tmp6.x, tmp7.x);
        s->c=(uint8)(tmp0.y, tmp1.y, tmp2.y, tmp3.y, tmp4.y, tmp5.y, tmp6.y, tmp7.y);
}

//! Return a 32-bit integer in the range [0..2^32)
uint8 MWC64XVEC8_NextUint8(mwc64xvec8_state_t *s)
{
        uint8 res=s->x ^ s->c;
        MWC64XVEC8_Step(s);
        return res;
}

