#ifndef FLO_INCLUDED_TEST_COMMON_MACROS
#define FLO_INCLUDED_TEST_COMMON_MACROS

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define STRINGIFY_NX(A) #A
#define STRINGIFY(A) STRINGIFY_NX(A)

#define FLOAT_SOFT_EPSILON 0.005f
#define EXPECT_FLOAT_NEAR(A, B) EXPECT_NEAR(A, B, FLOAT_SOFT_EPSILON);

#define EXPECT_VEC3_EQ(A, B)       \
  EXPECT_FLOAT_EQ((A)[0], (B)[0]); \
  EXPECT_FLOAT_EQ((A)[1], (B)[1]); \
  EXPECT_FLOAT_EQ((A)[2], (B)[2]);

#define EXPECT_VEC3_NE(A, B) \
  EXPECT_NE((A)[0], (B)[0]); \
  EXPECT_NE((A)[1], (B)[1]); \
  EXPECT_NE((A)[2], (B)[2]);

#define EXPECT_VEC2_EQ(A, B)       \
  EXPECT_FLOAT_EQ((A)[0], (B)[0]); \
  EXPECT_FLOAT_EQ((A)[1], (B)[1]);

#define EXPECT_VEC2_NE(A, B) \
  EXPECT_NE((A)[0], (B)[0]); \
  EXPECT_NE((A)[1], (B)[1]);

#define EXPECT_VEC3_NEAR(A, B)       \
  EXPECT_FLOAT_NEAR((A)[0], (B)[0]); \
  EXPECT_FLOAT_NEAR((A)[1], (B)[1]); \
  EXPECT_FLOAT_NEAR((A)[2], (B)[2]);

#define EXPECT_VEC2_NEAR(A, B)       \
  EXPECT_FLOAT_NEAR((A)[0], (B)[0]); \
  EXPECT_FLOAT_NEAR((A)[1], (B)[1]);

#endif//FLO_INCLUDED_TEST_COMMON_MACROS
