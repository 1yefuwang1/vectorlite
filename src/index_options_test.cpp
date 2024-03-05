#include "index_options.h"

#include "absl/strings/match.h"
#include "gtest/gtest.h"

TEST(ParseIndexOptions, ShouldWorkWithValidInput) {
  auto options = sqlite_vector::IndexOptions::FromString(
      "hnsw(max_elements=1000,M=32,ef_construction=400,random_seed=10000,allow_"
      "replace_deleted=true)");
  EXPECT_TRUE(options.ok());
  EXPECT_EQ(1000, options->max_elements);
  EXPECT_EQ(32, options->M);
  EXPECT_EQ(400, options->ef_construction);
  EXPECT_EQ(10000, options->random_seed);
  EXPECT_EQ(true, options->allow_replace_deleted);
}

TEST(ParseIndexOptions, ShouldWorkWithOnlyMaxElements) {
  auto options =
      sqlite_vector::IndexOptions::FromString("hnsw(max_elements=1000)");
  EXPECT_TRUE(options.ok());
  EXPECT_EQ(1000, options->max_elements);
  // Below are default values.
  EXPECT_EQ(16, options->M);
  EXPECT_EQ(200, options->ef_construction);
  EXPECT_EQ(100, options->random_seed);
  EXPECT_EQ(false, options->allow_replace_deleted);
}

TEST(ParseIndexOptions, ShouldFailWithoutMaxElements) {
  auto options = sqlite_vector::IndexOptions::FromString(
      "hnsw(M=16,ef_construction=200,random_seed=100,allow_replace_deleted="
      "false)");
  EXPECT_FALSE(options.ok());
  EXPECT_TRUE(absl::StrContains(options.status().message(), "max_elements is required"));
}

TEST(ParseIndexOptions, ShouldWorkWithAnyOrder) {
  auto options = sqlite_vector::IndexOptions::FromString(
      "hnsw(M=16,max_elements=1000,ef_construction=200,random_seed=100,allow_"
      "replace_deleted=false)");
  EXPECT_TRUE(options.ok());
  EXPECT_EQ(1000, options->max_elements);
  EXPECT_EQ(16, options->M);
  EXPECT_EQ(200, options->ef_construction);
  EXPECT_EQ(100, options->random_seed);
  EXPECT_EQ(false, options->allow_replace_deleted);
}

TEST(ParseIndexOptions, ShouldFailWithInvalidNumber) {
  auto options = sqlite_vector::IndexOptions::FromString(
      "hnsw(M=16,max_elements=aaa,ef_construction=200,random_seed=100,allow_"
      "replace_deleted=false)");
  EXPECT_FALSE(options.ok());

  options = sqlite_vector::IndexOptions::FromString(
      "hnsw(M=16,max_elements=1111111111111111111111111111,ef_construction=200,random_"
      "seed=100,allow_"
      "replace_deleted=false)");
  EXPECT_FALSE(options.ok());
  EXPECT_TRUE(absl::StrContains(options.status().message(), "Cannot parse max_elements"));
}

TEST(ParseIndexOptions, ShouldFailWithNonHNSWString) {
  auto options = sqlite_vector::IndexOptions::FromString(
      "xxxx(M=16,max_elements=1000,ef_construction=200,random_seed=100,allow_"
      "replace_deleted=false)");
  EXPECT_FALSE(options.ok());
}