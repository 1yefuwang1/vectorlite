#include "index_registry.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "hnswlib/hnswlib.h"
#include "index_options.h"
#include "vector_space.h"

namespace vectorlite {
namespace {

IndexHandle MakeTestHandle(
    std::string_view space_str = "emb float32[4]",
    std::string_view options_str = "hnsw(max_elements=100)") {
  auto space = NamedVectorSpace::FromString(space_str);
  EXPECT_TRUE(space.ok());
  auto options = IndexOptions::FromString(options_str);
  EXPECT_TRUE(options.ok());
  auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space->space.get(), options->max_elements, options->M,
      options->ef_construction, options->random_seed,
      options->allow_replace_deleted);
  return IndexHandle{std::move(*space), std::move(index),
                     options->allow_replace_deleted, std::string(space_str),
                     std::string(options_str)};
}

TEST(IndexRegistry, FindReturnsNullForMissingKey) {
  IndexRegistry registry;
  EXPECT_EQ(registry.Find({"main", "absent"}), nullptr);
}

TEST(IndexRegistry, InsertThenFindReturnsSameHandle) {
  IndexRegistry registry;
  IndexHandle* inserted = registry.Insert({"main", "t"}, MakeTestHandle());
  ASSERT_NE(inserted, nullptr);
  EXPECT_EQ(registry.Find({"main", "t"}), inserted);
}

TEST(IndexRegistry, InsertReplacesExistingEntry) {
  IndexRegistry registry;
  IndexHandle* first = registry.Insert({"main", "t"}, MakeTestHandle());
  IndexHandle* second = registry.Insert({"main", "t"}, MakeTestHandle());
  EXPECT_NE(first, second);
  EXPECT_EQ(registry.Find({"main", "t"}), second);
}

TEST(IndexRegistry, EraseRemovesEntry) {
  IndexRegistry registry;
  registry.Insert({"main", "t"}, MakeTestHandle());
  registry.Erase({"main", "t"});
  EXPECT_EQ(registry.Find({"main", "t"}), nullptr);
}

TEST(IndexRegistry, KeysWithDifferentSchemasAreDistinct) {
  IndexRegistry registry;
  IndexHandle* main_handle = registry.Insert({"main", "t"}, MakeTestHandle());
  IndexHandle* temp_handle = registry.Insert({"temp", "t"}, MakeTestHandle());
  EXPECT_NE(main_handle, temp_handle);
  EXPECT_EQ(registry.Find({"main", "t"}), main_handle);
  EXPECT_EQ(registry.Find({"temp", "t"}), temp_handle);
}

TEST(IndexRegistry, RenameMovesEntryAndKeepsHandleAddress) {
  IndexRegistry registry;
  IndexHandle* handle = registry.Insert({"main", "old"}, MakeTestHandle());
  registry.Rename({"main", "old"}, {"main", "new"});
  EXPECT_EQ(registry.Find({"main", "old"}), nullptr);
  // The handle address is preserved so live references stay valid.
  EXPECT_EQ(registry.Find({"main", "new"}), handle);
}

TEST(IndexRegistry, RenameReplacesExistingEntryAtNewKey) {
  IndexRegistry registry;
  IndexHandle* src = registry.Insert({"main", "old"}, MakeTestHandle());
  registry.Insert({"main", "new"}, MakeTestHandle());
  registry.Rename({"main", "old"}, {"main", "new"});
  EXPECT_EQ(registry.Find({"main", "old"}), nullptr);
  EXPECT_EQ(registry.Find({"main", "new"}), src);
}

TEST(IndexRegistry, RenameIsNoOpForMissingOrIdenticalKey) {
  IndexRegistry registry;
  registry.Rename({"main", "absent"}, {"main", "whatever"});
  EXPECT_EQ(registry.Find({"main", "whatever"}), nullptr);

  IndexHandle* handle = registry.Insert({"main", "t"}, MakeTestHandle());
  registry.Rename({"main", "t"}, {"main", "t"});
  EXPECT_EQ(registry.Find({"main", "t"}), handle);
}

}  // namespace
}  // namespace vectorlite
