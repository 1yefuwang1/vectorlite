#include "vector_space.h"

#include "gtest/gtest.h"

TEST(ParseVectorSpace, ShouldSupport_L2_InnerProduct_Cosine) {
  auto l2 = vectorlite::ParseSpaceType("l2");
  EXPECT_TRUE(l2);
  EXPECT_TRUE(*l2 == vectorlite::SpaceType::L2);

  auto ip = vectorlite::ParseSpaceType("ip");
  EXPECT_TRUE(ip);
  EXPECT_TRUE(*ip == vectorlite::SpaceType::InnerProduct);

  auto cosine = vectorlite::ParseSpaceType("cosine");
  EXPECT_TRUE(cosine);
  EXPECT_TRUE(*cosine == vectorlite::SpaceType::Cosine);
}

TEST(ParseVectorSpace, ShouldRetturnNullOptForInvalidSpaceType) {
  auto l2 = vectorlite::ParseSpaceType("aaa");
  EXPECT_FALSE(l2);
}

TEST(CreateVectorSpace, ShouldWorkWithValidInput) {
  auto l2 = vectorlite::CreateNamedVectorSpace(3, vectorlite::SpaceType::L2,
                                               "my_vector");
  EXPECT_TRUE(l2.ok());
  EXPECT_TRUE(l2->type == vectorlite::SpaceType::L2);
  EXPECT_TRUE(l2->normalize == false);
  EXPECT_TRUE(l2->space != nullptr);
  EXPECT_EQ(3, l2->dimension());

  auto ip = vectorlite::CreateNamedVectorSpace(
      4, vectorlite::SpaceType::InnerProduct, "my_vector");
  EXPECT_TRUE(ip.ok());
  EXPECT_TRUE(ip->type == vectorlite::SpaceType::InnerProduct);
  EXPECT_TRUE(ip->normalize == false);
  EXPECT_TRUE(ip->space != nullptr);
  EXPECT_EQ(4, ip->dimension());

  auto cosine = vectorlite::CreateNamedVectorSpace(
      5, vectorlite::SpaceType::Cosine, "my_vector");
  EXPECT_TRUE(cosine->type == vectorlite::SpaceType::Cosine);
  EXPECT_TRUE(cosine.ok());
  EXPECT_TRUE(cosine->normalize == true);
  EXPECT_TRUE(cosine->space != nullptr);
  EXPECT_EQ(5, cosine->dimension());
}

TEST(CreateVectorSpace, ShouldReturnErrorForDimOfZero) {
  auto l2 = vectorlite::CreateNamedVectorSpace(0, vectorlite::SpaceType::L2,
                                               "my_vector");
  EXPECT_FALSE(l2.ok());

  auto ip = vectorlite::CreateNamedVectorSpace(
      0, vectorlite::SpaceType::InnerProduct, "my_vector");
  EXPECT_FALSE(ip.ok());

  auto cosine = vectorlite::CreateNamedVectorSpace(
      0, vectorlite::SpaceType::Cosine, "my_vector");
  EXPECT_FALSE(cosine.ok());
}

TEST(VectorSpace_FromString, ShouldWorkWithValidInput) {
  auto space = vectorlite::NamedVectorSpace::FromString("my_vec(3, \"l2\")");
  EXPECT_TRUE(space.ok());
  EXPECT_TRUE(space->normalize == false);
  EXPECT_TRUE(space->space != nullptr);
  EXPECT_TRUE(space->type == vectorlite::SpaceType::L2);
  EXPECT_EQ(3, space->dimension());
  EXPECT_EQ("my_vec", space->vector_name);

  space = vectorlite::NamedVectorSpace::FromString("my_vec(10086, \"cosine\")");
  EXPECT_TRUE(space.ok());
  EXPECT_TRUE(space->normalize == true);
  EXPECT_TRUE(space->space != nullptr);
  EXPECT_TRUE(space->type == vectorlite::SpaceType::Cosine);
  EXPECT_EQ(10086, space->dimension());
  EXPECT_EQ("my_vec", space->vector_name);

  space = vectorlite::NamedVectorSpace::FromString("my_vec(42, \"ip\")");
  EXPECT_TRUE(space.ok());
  EXPECT_TRUE(space->normalize == false);
  EXPECT_TRUE(space->space != nullptr);
  EXPECT_TRUE(space->type == vectorlite::SpaceType::InnerProduct);
  EXPECT_EQ(42, space->dimension());
  EXPECT_EQ("my_vec", space->vector_name);
}
