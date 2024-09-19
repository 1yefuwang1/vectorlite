#include "vector_space.h"

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"

TEST(ParseDistanceType, ShouldSupport_L2_InnerProduct_Cosine) {
  auto l2 = vectorlite::ParseDistanceType("l2");
  ASSERT_TRUE(l2);
  EXPECT_TRUE(*l2 == vectorlite::DistanceType::L2);

  auto ip = vectorlite::ParseDistanceType("ip");
  ASSERT_TRUE(ip);
  EXPECT_TRUE(*ip == vectorlite::DistanceType::InnerProduct);

  auto cosine = vectorlite::ParseDistanceType("cosine");
  ASSERT_TRUE(cosine);
  EXPECT_TRUE(*cosine == vectorlite::DistanceType::Cosine);
}

TEST(ParseDistanceType, ShouldRetturnNullOptForInvalidSpaceType) {
  auto l2 = vectorlite::ParseDistanceType("aaa");
  ASSERT_FALSE(l2);
}

TEST(ParseVectorType, ShouldSupportFloat32) {
  auto float32 = vectorlite::ParseVectorType("float32");
  ASSERT_TRUE(float32);
  EXPECT_TRUE(*float32 == vectorlite::VectorType::Float32);
}

TEST(ParseVectorType, ShouldReturnNullOptForInvalidVectorType) {
  auto float16 = vectorlite::ParseVectorType("float16");
  EXPECT_FALSE(float16);

  auto uint8 = vectorlite::ParseVectorType("uint8");
  EXPECT_FALSE(uint8);
}

TEST(ParseVectorType, ShouldSupportBFloat16) {
  auto float16 = vectorlite::ParseVectorType("bfloat16");
  EXPECT_TRUE(float16);
}

TEST(CreateVectorSpace, ShouldWorkWithValidInput) {
  for (auto vector_type :
       {vectorlite::VectorType::Float32, vectorlite::VectorType::BFloat16}) {
    auto l2 = vectorlite::CreateNamedVectorSpace(
        3, vectorlite::DistanceType::L2, "my_vector", vector_type);
    ASSERT_TRUE(l2.ok());
    EXPECT_EQ(l2->distance_type, vectorlite::DistanceType::L2);
    EXPECT_EQ(l2->normalize, false);
    EXPECT_NE(l2->space, nullptr);
    EXPECT_EQ(l2->dimension(), 3);
    EXPECT_EQ(l2->vector_type, vector_type);

    auto ip = vectorlite::CreateNamedVectorSpace(
        4, vectorlite::DistanceType::InnerProduct, "my_vector", vector_type);
    ASSERT_TRUE(ip.ok());
    EXPECT_EQ(ip->distance_type, vectorlite::DistanceType::InnerProduct);
    EXPECT_EQ(ip->normalize, false);
    EXPECT_NE(ip->space, nullptr);
    EXPECT_EQ(ip->dimension(), 4);
    EXPECT_EQ(ip->vector_type, vector_type);

    auto cosine = vectorlite::CreateNamedVectorSpace(
        5, vectorlite::DistanceType::Cosine, "my_vector", vector_type);
    ASSERT_TRUE(cosine.ok());
    EXPECT_EQ(cosine->distance_type, vectorlite::DistanceType::Cosine);
    EXPECT_EQ(cosine->normalize, true);
    EXPECT_NE(cosine->space, nullptr);
    EXPECT_EQ(cosine->dimension(), 5);
    EXPECT_EQ(cosine->vector_type, vector_type);
  }
}

TEST(CreateNamedVectorSpace, ShouldReturnErrorForDimOfZero) {
  for (auto vector_type :
       {vectorlite::VectorType::Float32, vectorlite::VectorType::BFloat16}) {
    auto l2 = vectorlite::CreateNamedVectorSpace(
        0, vectorlite::DistanceType::L2, "my_vector", vector_type);
    EXPECT_FALSE(l2.ok());

    auto ip = vectorlite::CreateNamedVectorSpace(
        0, vectorlite::DistanceType::InnerProduct, "my_vector", vector_type);
    EXPECT_FALSE(ip.ok());

    auto cosine = vectorlite::CreateNamedVectorSpace(
        0, vectorlite::DistanceType::Cosine, "my_vector", vector_type);
    EXPECT_FALSE(cosine.ok());
  }
}

static std::string VectorTypeToString(vectorlite::VectorType type) {
  switch (type) {
    case vectorlite::VectorType::Float32:
      return "float32";
    case vectorlite::VectorType::BFloat16:
      return "bfloat16";
    default:
      return "unknown";
  }
}

TEST(NamedVectorSpace_FromString, ShouldWorkWithValidInput) {
  for (auto vector_type :
       {vectorlite::VectorType::Float32, vectorlite::VectorType::BFloat16}) {
    // If distance type is not specifed, it should default to L2
    std::string vector_type_str = VectorTypeToString(vector_type);
    auto space = vectorlite::NamedVectorSpace::FromString(
        absl::StrFormat("my_vec  %s[3]", vector_type_str));
    ASSERT_TRUE(space.ok());
    EXPECT_EQ(space->normalize, false);
    EXPECT_NE(space->space, nullptr);
    EXPECT_EQ(space->distance_type, vectorlite::DistanceType::L2);
    EXPECT_EQ(3, space->dimension());
    EXPECT_EQ("my_vec", space->vector_name);
    EXPECT_EQ(vector_type, space->vector_type);

    space = vectorlite::NamedVectorSpace::FromString(
        absl::StrFormat("my_vec  %s[3]   l2", vector_type_str));
    ASSERT_TRUE(space.ok());
    EXPECT_EQ(space->normalize, false);
    EXPECT_NE(space->space, nullptr);
    EXPECT_EQ(space->distance_type, vectorlite::DistanceType::L2);
    EXPECT_EQ(3, space->dimension());
    EXPECT_EQ("my_vec", space->vector_name);
    EXPECT_EQ(vector_type, space->vector_type);

    space = vectorlite::NamedVectorSpace::FromString(
        absl::StrFormat("my_vec  %s[10086] cosine", vector_type_str));
    ASSERT_TRUE(space.ok());
    EXPECT_EQ(space->normalize, true);
    EXPECT_NE(space->space, nullptr);
    EXPECT_EQ(space->distance_type, vectorlite::DistanceType::Cosine);
    EXPECT_EQ(10086, space->dimension());
    EXPECT_EQ("my_vec", space->vector_name);
    EXPECT_EQ(vector_type, space->vector_type);

    space = vectorlite::NamedVectorSpace::FromString(
        absl::StrFormat("my_vec %s[42]   ip", vector_type_str));
    ASSERT_TRUE(space.ok());
    EXPECT_EQ(space->normalize, false);
    EXPECT_NE(space->space, nullptr);
    EXPECT_EQ(space->distance_type, vectorlite::DistanceType::InnerProduct);
    EXPECT_EQ(42, space->dimension());
    EXPECT_EQ("my_vec", space->vector_name);
    EXPECT_EQ(vector_type, space->vector_type);
  }
}
