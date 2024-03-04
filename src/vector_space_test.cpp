#include "vector_space.h"

#include "gtest/gtest.h"

TEST(ParseVectorSpace, ShouldSupport_L2_InnerProduct_Cosine) {
  auto l2 = sqlite_vector::ParseSpaceType("l2");
  EXPECT_TRUE(l2);
  EXPECT_TRUE(*l2 == sqlite_vector::SpaceType::L2);

  auto ip = sqlite_vector::ParseSpaceType("ip");
  EXPECT_TRUE(ip);
  EXPECT_TRUE(*ip == sqlite_vector::SpaceType::InnerProduct);

  auto cosine = sqlite_vector::ParseSpaceType("cosine");
  EXPECT_TRUE(cosine);
  EXPECT_TRUE(*cosine == sqlite_vector::SpaceType::Cosine);
}

TEST(ParseVectorSpace, ShouldRetturnNullOptForInvalidSpaceType) {
  auto l2 = sqlite_vector::ParseSpaceType("aaa");
  EXPECT_FALSE(l2);
}

TEST(CreateVectorSpace, ShouldWorkWithValidInput) {
  auto l2 = sqlite_vector::CreateVectorSpace(3, sqlite_vector::SpaceType::L2, "my_vector");
  EXPECT_TRUE(l2.ok());
  EXPECT_TRUE(l2->type == sqlite_vector::SpaceType::L2);
  EXPECT_TRUE(l2->normalize == false);
  EXPECT_TRUE(l2->space != nullptr);
  EXPECT_EQ(3, l2->dimension());

  auto ip = sqlite_vector::CreateVectorSpace(4, sqlite_vector::SpaceType::InnerProduct, "my_vector");
  EXPECT_TRUE(ip.ok());
  EXPECT_TRUE(l2->type == sqlite_vector::SpaceType::InnerProduct);
  EXPECT_TRUE(ip->normalize == false);
  EXPECT_TRUE(ip->space != nullptr);
  EXPECT_EQ(4, ip->dimension());

  auto cosine = sqlite_vector::CreateVectorSpace(5, sqlite_vector::SpaceType::Cosine, "my_vector");
  EXPECT_TRUE(l2->type == sqlite_vector::SpaceType::Cosine);
  EXPECT_TRUE(cosine.ok());
  EXPECT_TRUE(cosine->normalize == true);
  EXPECT_TRUE(cosine->space != nullptr);
  EXPECT_EQ(5, cosine->dimension());
}

TEST(CreateVectorSpace, ShouldReturnErrorForDimOfZero) {
  auto l2 = sqlite_vector::CreateVectorSpace(0, sqlite_vector::SpaceType::L2, "my_vector");
  EXPECT_FALSE(l2.ok());

  auto ip = sqlite_vector::CreateVectorSpace(0, sqlite_vector::SpaceType::InnerProduct, "my_vector");
  EXPECT_FALSE(ip.ok());

  auto cosine = sqlite_vector::CreateVectorSpace(0, sqlite_vector::SpaceType::Cosine, "my_vector");
  EXPECT_FALSE(cosine.ok());
}

TEST(VectorSpace_FromString, ShouldWorkWithValidInput) {
  auto space = sqlite_vector::VectorSpace::FromString("my_vec(3, \"l2\")");
  EXPECT_TRUE(space.ok());
  EXPECT_TRUE(space->normalize == false);
  EXPECT_TRUE(space->space != nullptr);
  EXPECT_TRUE(space->type != sqlite_vector::SpaceType::L2);
  EXPECT_EQ(3, space->dimension());
  EXPECT_EQ("my_vec", space->vector_name);
}
