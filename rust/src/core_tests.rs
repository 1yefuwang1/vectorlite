use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TestDirectory(std::path::PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vectorlite-core-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn index(vector_type: VectorType, metric: DistanceType) -> Index {
    Index::create(4, metric, vector_type, 10, 16, 100, 42, true).unwrap()
}

#[test]
fn distance_matches_known_values_without_mutating_inputs() {
    let first = [1., 2., 3.];
    let second = [4., 5., 6.];
    assert_eq!(distance(&first, &second, DistanceType::L2), Some(27.));
    assert_eq!(
        distance(&first, &second, DistanceType::InnerProduct),
        Some(-31.)
    );
    let cosine = distance(&first, &second, DistanceType::Cosine).unwrap();
    assert!((cosine - 0.025_368_2).abs() < 1e-6);
    assert_eq!(first, [1., 2., 3.]);
    assert_eq!(second, [4., 5., 6.]);
    assert_eq!(distance(&first, &second[..2], DistanceType::L2), None);
}

#[test]
fn distance_handles_empty_vectors() {
    assert_eq!(distance(&[], &[], DistanceType::L2), Some(0.));
    assert_eq!(distance(&[], &[], DistanceType::InnerProduct), Some(1.));
    assert_eq!(distance(&[], &[], DistanceType::Cosine), Some(1.));
}

#[test]
fn persistence_save_reports_missing_parent() {
    let dir = TestDirectory::new();
    let index = index(VectorType::Float32, DistanceType::L2);
    index.add(&[1., 2., 3., 4.], 1).unwrap();
    let path = dir.file("missing/index.bin");
    assert!(index.save(&path).is_err());
    assert!(!Path::new(&path).exists());
}

#[test]
fn persistence_rejects_same_width_type_and_preserves_existing_index() {
    let dir = TestDirectory::new();
    let src = index(VectorType::BFloat16, DistanceType::L2);
    src.add(&[1., 2., 3., 4.], 1).unwrap();
    let path = dir.file("index.bin");
    src.save(&path).unwrap();
    let dst = index(VectorType::Float16, DistanceType::L2);
    dst.add(&[4., 3., 2., 1.], 2).unwrap();
    assert!(dst.load(&path).is_err());
    assert_eq!(dst.get_vector(2), Some(vec![4., 3., 2., 1.]));
    assert!(!dst.contains(1));
}

#[test]
fn persistence_rejects_metric_changes() {
    let dir = TestDirectory::new();
    let src = index(VectorType::Float32, DistanceType::L2);
    src.add(&[1., 2., 3., 4.], 1).unwrap();
    let path = dir.file("index.bin");
    src.save(&path).unwrap();
    let dst = index(VectorType::Float32, DistanceType::Cosine);
    assert!(dst.load(&path).is_err());
    assert!(!dst.contains(1));
}

#[test]
fn persistence_roundtrips_all_types_and_overwrites_atomically() {
    for kind in [
        VectorType::Float32,
        VectorType::BFloat16,
        VectorType::Float16,
    ] {
        let dir = TestDirectory::new();
        let src = index(kind, DistanceType::L2);
        let path = dir.file("index.bin");
        src.save(&path).unwrap();
        let dst = index(kind, DistanceType::L2);
        dst.load(&path).unwrap();
        assert!(!dst.contains(1));
        src.add(&[1., 2., 3., 4.], 1).unwrap();
        src.add(&[4., 3., 2., 1.], 2).unwrap();
        src.save(&path).unwrap();
        dst.load(&path).unwrap();
        assert_eq!(dst.get_vector(1), Some(vec![1., 2., 3., 4.]));
        assert_eq!(
            dst.search(&[1., 2., 3., 4.], 100, Some(30), SearchFilter::None)
                .unwrap(),
            vec![SearchResult::new(0., 1), SearchResult::new(20., 2)]
        );
        assert_eq!(
            std::fs::read_dir(&dir.0).unwrap().count(),
            1,
            "temporary files must be removed"
        );
    }
}

#[test]
fn failed_atomic_replace_leaves_destination_untouched() {
    let dir = TestDirectory::new();
    let destination = dir.0.join("existing-directory");
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(destination.join("marker"), "keep").unwrap();
    let src = index(VectorType::Float32, DistanceType::L2);
    assert!(src.save(destination.to_str().unwrap()).is_err());
    assert_eq!(
        std::fs::read_to_string(destination.join("marker")).unwrap(),
        "keep"
    );
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[test]
fn persistence_loads_legacy_and_resaves_with_descriptor() {
    for kind in [
        VectorType::Float32,
        VectorType::BFloat16,
        VectorType::Float16,
    ] {
        for metric in [
            DistanceType::L2,
            DistanceType::InnerProduct,
            DistanceType::Cosine,
        ] {
            let dir = TestDirectory::new();
            let src = index(kind, metric);
            src.add(&[1., 2., 3., 4.], 1).unwrap();
            let expected = src.get_vector(1);
            let raw = dir.file("legacy.bin");
            src.index.borrow().save(&raw).unwrap();

            let dst = index(kind, metric);
            dst.add(&[4., 3., 2., 1.], 2).unwrap();
            dst.load(&raw).unwrap();
            assert_eq!(dst.get_vector(1), expected);
            assert!(!dst.contains(2));

            let upgraded = dir.file("versioned.bin");
            dst.save(&upgraded).unwrap();
            let bytes = std::fs::read(&upgraded).unwrap();
            assert_eq!(&bytes[..8], FILE_MAGIC);
            let restored = index(kind, metric);
            restored.load(&upgraded).unwrap();
            assert_eq!(restored.get_vector(1), expected);
        }
    }
}

#[test]
fn persistence_rejects_truncated_envelopes_without_replacement() {
    let dir = TestDirectory::new();
    let src = index(VectorType::Float32, DistanceType::L2);
    src.add(&[1., 2., 3., 4.], 1).unwrap();
    let path = dir.file("index.bin");
    src.save(&path).unwrap();
    let original_size = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(original_size - 1)
        .unwrap();
    assert!(src.load(&path).is_err());
    assert_eq!(src.get_vector(1), Some(vec![1., 2., 3., 4.]));
}

#[test]
fn raw_payload_save_also_reports_missing_parent() {
    let dir = TestDirectory::new();
    let src = index(VectorType::Float32, DistanceType::L2);
    assert!(src
        .index
        .borrow()
        .save(&dir.file("missing/raw.bin"))
        .is_err());
}

#[test]
fn non_normalizing_f32_encoding_borrows_the_input() {
    let values = [1., 2., 3., 4.];
    for metric in [DistanceType::L2, DistanceType::InnerProduct] {
        let src = index(VectorType::Float32, metric);
        let encoded = src.encode(&values).unwrap();
        assert_eq!(encoded.bytes().as_ptr(), values.as_ptr().cast::<u8>());
    }
    let src = index(VectorType::Float32, DistanceType::Cosine);
    src.add(&values, 1).unwrap();
    assert_eq!(
        values,
        [1., 2., 3., 4.],
        "normalization must not mutate borrowed input"
    );
}

#[test]
fn safe_index_methods_reject_dimension_mismatches() {
    for kind in [
        VectorType::Float32,
        VectorType::BFloat16,
        VectorType::Float16,
    ] {
        let src = index(kind, DistanceType::L2);
        assert!(src.add(&[1., 2.], 1).is_err());
        assert!(src.add(&[1., 2., 3., 4., 5.], 1).is_err());
        assert!(src.search(&[1.], 1, None, SearchFilter::None).is_err());
        assert!(!src.contains(1));
    }
}

#[test]
fn construction_rejects_invalid_layouts_before_native_allocation() {
    assert!(Index::create(
        0,
        DistanceType::L2,
        VectorType::Float32,
        10,
        16,
        100,
        1,
        true
    )
    .is_err());
    assert!(Index::create(
        usize::MAX,
        DistanceType::L2,
        VectorType::Float32,
        10,
        16,
        100,
        1,
        true
    )
    .is_err());
    for m in [0, 1, 10_001] {
        assert!(Index::create(
            4,
            DistanceType::L2,
            VectorType::Float32,
            10,
            m,
            100,
            1,
            true
        )
        .is_err());
    }
    assert!(Index::create(
        4,
        DistanceType::L2,
        VectorType::Float32,
        usize::MAX,
        16,
        100,
        1,
        true
    )
    .is_err());
    assert!(Index::create(4, DistanceType::L2, VectorType::Float32, 10, 16, 0, 1, true).is_err());
}

#[test]
fn native_index_retains_its_space_and_validates_buffers() {
    let space = Space::new(DistanceType::L2, VectorType::Float32, 4).unwrap();
    let hnsw = Hnsw::create(&space, 10, 16, 100, 1, true).unwrap();
    drop(space);
    let values = [1f32, 2., 3., 4.];
    let bytes = bytemuck::cast_slice(&values);
    hnsw.add_point(bytes, 9, true).unwrap();
    assert!(hnsw.add_point(&bytes[..4], 10, true).is_err());
    assert!(hnsw.search(&bytes[..4], 1, &SearchFilter::None).is_err());
    assert!(!hnsw.get_data(9, &mut [0; 32]));
    let mut output = [0f32; 4];
    assert!(hnsw.get_data(9, bytemuck::cast_slice_mut(&mut output)));
    assert_eq!(output, values);
    assert_eq!(
        hnsw.search(bytes, 10, &SearchFilter::Equals(9)).unwrap(),
        vec![SearchResult::new(0., 9)]
    );
    assert!(hnsw
        .search(bytes, 0, &SearchFilter::None)
        .unwrap()
        .is_empty());
    assert!(hnsw
        .search(bytes, 1, &SearchFilter::Equals(42))
        .unwrap()
        .is_empty());
}

#[test]
fn persistence_preserves_multilevel_graph_deletions_and_query_ef() {
    let dir = TestDirectory::new();
    let src = Index::create(
        4,
        DistanceType::L2,
        VectorType::Float32,
        200,
        8,
        100,
        42,
        true,
    )
    .unwrap();
    for i in 0..128 {
        src.add(&[i as f32, (i % 7) as f32, 0., 1.], i).unwrap();
    }
    src.mark_delete(3).unwrap();
    let query = [20., 6., 0., 1.];
    let expected = src
        .search(&query, 12, Some(60), SearchFilter::None)
        .unwrap();
    assert_eq!(
        src.index.borrow().get_ef(),
        10,
        "per-query ef must be restored"
    );
    let path = dir.file("graph.bin");
    src.save(&path).unwrap();
    let dst = Index::create(
        4,
        DistanceType::L2,
        VectorType::Float32,
        300,
        16,
        100,
        1,
        true,
    )
    .unwrap();
    dst.load(&path).unwrap();
    assert!(!dst.contains(3));
    assert_eq!(
        dst.search(&query, 12, Some(60), SearchFilter::None)
            .unwrap(),
        expected
    );
    dst.add(&[3., 3., 0., 1.], 3).unwrap();
    assert!(dst.contains(3));
}
