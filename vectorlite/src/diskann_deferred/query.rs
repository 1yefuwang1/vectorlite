//! Operation-local, read-only search of one published Format4 generation.
//! The metadata cursor anchors every read and all accepted projected bytes.
//! Native ID zero is ordinary data; the only frozen start point is explicit N.
use std::mem::size_of;

use crate::core::{SearchFilter, SearchResult};
use crate::diskann_core::{self, GraphConfig, GraphStore, NodeState, ResourceLimits, StoredVector};
use crate::diskann_store::codec::{
    blob_projection, bounded_blob, checked_id, checked_product, corrupt, nonnegative,
    normalize_input, reserve, too_big, with_decoded_vector,
};
use crate::diskann_store::{DiskAnnTable, Generation, QueryRows};
use crate::ffi;
use crate::index_error::IndexError;
use crate::ops;
use crate::sqlite::{Statement, Step};
use crate::vector_space::DistanceType;

type Result<T> = std::result::Result<T, IndexError>;

fn unsupported() -> IndexError {
    IndexError::with_code(
        ffi::SQLITE_READONLY as i32,
        "DiskANN published query adapter is read-only",
    )
}

// Explicitly finalize on both success and failure. A primary SQL/read error is
// never replaced by a later cleanup failure. No auxiliary statement escapes.
fn finish<T>(statement: Statement<'_>, result: Result<T>) -> Result<T> {
    let cleanup = statement.finish();
    match result {
        Err(error) => Err(error),
        Ok(value) => cleanup.map(|()| value),
    }
}

#[derive(Debug)]
struct Budget {
    visits: usize,
    max_visits: usize,
    core_bytes: usize,
}
impl Budget {
    fn admit(
        table: &DiskAnnTable,
        count: usize,
        filter: &SearchFilter<'_>,
        project: bool,
    ) -> Result<Self> {
        // Reserve SQL names/VMs, metadata, owner decode/payload buffers, both
        // result representations during final materialization and filter storage.
        // Core receives only the remainder, not a second full cache_bytes budget.
        let names = [
            &table.names().meta,
            &table.names().entries,
            &table.names().current,
            &table.names().generations,
            &table.names().graph_nodes,
        ]
        .iter()
        .try_fold(0usize, |sum, name| {
            sum.checked_add(name.len())
                .ok_or_else(|| too_big("DiskANN query identifier overflow"))
        })?;
        let filter_bytes = match filter {
            SearchFilter::In(ids) => checked_product(ids.capacity(), 64, "query filter")?,
            _ => 0,
        };
        let per_result = size_of::<SearchResult>()
            .checked_mul(2)
            .and_then(|n| {
                n.checked_add(if project {
                    table.vector_bytes() + size_of::<Vec<u8>>()
                } else {
                    0
                })
            })
            .ok_or_else(|| too_big("DiskANN query result overflow"))?;
        let owner = checked_product(table.vector_bytes(), 5, "query vector buffers")?
            .checked_add(checked_product(
                table.options().physical_degree(),
                24,
                "query adjacency",
            )?)
            .and_then(|n| {
                checked_product(names, 32, "query SQL names")
                    .ok()
                    .and_then(|s| n.checked_add(s))
            })
            .and_then(|n| n.checked_add(64 * 1024))
            .and_then(|n| n.checked_add(filter_bytes))
            .and_then(|n| {
                checked_product(count, per_result, "query results")
                    .ok()
                    .and_then(|s| n.checked_add(s))
            })
            .ok_or_else(|| too_big("DiskANN query workspace overflow"))?;
        let core_bytes = table
            .options()
            .cache_bytes
            .checked_sub(owner)
            .ok_or_else(|| too_big("DiskANN query workspace exceeds cache_bytes"))?;
        Ok(Self {
            visits: 0,
            max_visits: table.options().max_visits,
            core_bytes,
        })
    }
    fn visit(&mut self) -> Result<()> {
        if self.visits >= self.max_visits {
            return Err(too_big("DiskANN query exceeded max_visits"));
        }
        self.visits += 1;
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.max_visits - self.visits
    }
}

struct PublishedGraph<'a> {
    table: &'a DiskAnnTable,
    generation: Generation,
    revision: u64,
    budget: Budget,
    decode: Vec<f32>,
}

impl PublishedGraph<'_> {
    fn node_sql(&self, vector: bool) -> String {
        let names = self.table.names();
        let payload = if vector {
            format!(
                ",{}",
                blob_projection(
                    "CASE WHEN n.state=2 THEN n.frozen_vector ELSE e.vector END",
                    self.table.vector_bytes(),
                    true
                )
            )
        } else {
            String::new()
        };
        format!("SELECT n.entry_id,n.state,n.frozen_vector IS NULL,e.public_rowid,e.created_revision,e.retired_revision,c.entry_id{payload} FROM {} n LEFT JOIN {} e ON e.entry_id=n.entry_id LEFT JOIN {} c ON c.public_rowid=e.public_rowid WHERE n.generation_id=?1 AND n.native_local_id=?2", names.graph_nodes, names.entries, names.current)
    }
    fn bind_node(&self, statement: &mut Statement<'_>, id: u64) -> Result<()> {
        if self.generation.total_count == 0 || id > self.generation.total_count {
            return Err(corrupt("DiskANN query local ID exceeds generation"));
        }
        statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
        statement.bind_i64(2, checked_id(id)?)
    }
    // Read immutable identity and current membership before accepting top-k.
    // Missing/corrupt metadata is never interpreted as a deleted embedding.
    fn classify(&self, statement: &Statement<'_>, id: u64) -> Result<(NodeState, Option<u64>)> {
        let role = nonnegative(statement, 1, "query node role")?;
        if Some(id) == self.generation.root_local_id {
            if role != 2
                || !statement.column_is_null(0)?
                || nonnegative(statement, 2, "root vector NULL")? != 0
            {
                return Err(corrupt("DiskANN explicit frozen root role invalid"));
            }
            return Ok((NodeState::Frozen, None));
        }
        if role != 0
            || id >= self.generation.total_count
            || nonnegative(statement, 0, "query entry ID")? == 0
            || nonnegative(statement, 2, "data frozen vector NULL")? != 1
        {
            return Err(corrupt("DiskANN query node role/version invalid"));
        }
        let rowid = nonnegative(statement, 3, "query original rowid")?;
        let created = nonnegative(statement, 4, "query creation revision")?;
        if created == 0 || created > self.generation.source_revision {
            return Err(corrupt("DiskANN graph entry lies outside frozen snapshot"));
        }
        let retired = if statement.column_is_null(5)? {
            None
        } else {
            Some(nonnegative(statement, 5, "query retirement revision")?)
        };
        if retired.is_some_and(|r| r <= created || r > self.revision) {
            return Err(corrupt("DiskANN query immutable interval invalid"));
        }
        let current = if statement.column_is_null(6)? {
            None
        } else {
            Some(nonnegative(statement, 6, "query current entry ID")?)
        };
        if current == Some(0) {
            return Err(corrupt("DiskANN query current entry is zero"));
        }
        if current == Some(nonnegative(statement, 0, "query entry ID")?) {
            if retired.is_some() {
                return Err(corrupt("DiskANN current graph entry is retired"));
            }
            Ok((NodeState::Live, Some(rowid)))
        } else {
            Ok((NodeState::Deleted, None))
        }
    }
    fn identity(&self, id: u64) -> Result<(NodeState, Option<u64>)> {
        let mut statement = self.table.connection().prepare(&self.node_sql(false))?;
        let outcome = (|| {
            self.bind_node(&mut statement, id)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced query node/entry missing"));
            }
            let identity = self.classify(&statement, id)?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN query node identity not unique"));
            }
            Ok(identity)
        })();
        finish(statement, outcome)
    }
    fn eligible_sql(&self, point: bool, vector: bool) -> String {
        let names = self.table.names();
        let payload = if vector {
            format!(
                ",{}",
                blob_projection("e.vector", self.table.vector_bytes(), true)
            )
        } else {
            String::new()
        };
        // current joins the GRAPH version, never the latest canonical payload by
        // rowid alone. LEFT entries makes a broken current entry a corruption.
        format!("SELECT n.native_local_id,n.entry_id,e.public_rowid,e.created_revision,e.retired_revision,c.public_rowid,n.frozen_vector IS NULL{payload} FROM {} n JOIN {} c ON c.entry_id=n.entry_id LEFT JOIN {} e ON e.entry_id=n.entry_id WHERE n.generation_id=?1 AND n.state=0{} ORDER BY n.native_local_id", names.graph_nodes, names.current, names.entries, if point { " AND c.public_rowid=?2" } else { "" })
    }
    fn eligible_identity(&self, statement: &Statement<'_>) -> Result<(u64, u64)> {
        let id = nonnegative(statement, 0, "eligible local ID")?;
        let entry = nonnegative(statement, 1, "eligible entry ID")?;
        let rowid = nonnegative(statement, 2, "eligible original rowid")?;
        let created = nonnegative(statement, 3, "eligible creation revision")?;
        if id >= self.generation.total_count
            || entry == 0
            || created == 0
            || created > self.generation.source_revision
            || !statement.column_is_null(4)?
            || nonnegative(statement, 6, "eligible frozen vector NULL")? != 1
            || rowid != nonnegative(statement, 5, "eligible current rowid")?
        {
            return Err(corrupt(
                "DiskANN eligible graph/current/entry identity invalid",
            ));
        }
        Ok((id, rowid))
    }
    fn exact(
        &mut self,
        query: &[f32],
        k: usize,
        filter: &SearchFilter<'_>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        reserve(&mut results, k)?;
        match filter {
            SearchFilter::Equals(rowid) => self.exact_scan(query, k, Some(*rowid), &mut results)?,
            SearchFilter::In(ids) => {
                for &rowid in *ids {
                    self.exact_scan(query, k, Some(rowid), &mut results)?;
                }
            }
            SearchFilter::None => self.exact_scan(query, k, None, &mut results)?,
        }
        Ok(results)
    }
    fn exact_scan(
        &mut self,
        query: &[f32],
        k: usize,
        rowid: Option<u64>,
        results: &mut Vec<SearchResult>,
    ) -> Result<()> {
        if rowid.is_some_and(|r| r > i64::MAX as u64) {
            return Ok(());
        }
        let mut statement = self
            .table
            .connection()
            .prepare(&self.eligible_sql(rowid.is_some(), true))?;
        let outcome = (|| {
            statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
            if let Some(rowid) = rowid {
                statement.bind_i64(2, checked_id(rowid)?)?;
            }
            // Charge point misses too: a huge IN set cannot evade the visit bound.
            if rowid.is_some() {
                self.budget.visit()?;
            }
            let mut count = 0usize;
            while statement.step()? == Step::Row {
                if rowid.is_none() {
                    self.budget.visit()?;
                }
                count += 1;
                if count
                    > usize::try_from(self.generation.eligible_count)
                        .map_err(|_| too_big("DiskANN eligible population overflow"))?
                    || (rowid.is_some() && count > 1)
                {
                    return Err(corrupt("DiskANN eligible count/rowid uniqueness mismatch"));
                }
                let (_, label) = self.eligible_identity(&statement)?;
                check_blob(&statement, 7, 8, self.table.vector_bytes())?;
                let metric = self.table.space().distance_type;
                let mut distance = 0.0;
                statement.with_column_blob(8, |bytes| {
                    with_decoded_vector(
                        bytes,
                        self.table.space().dim,
                        &mut self.decode,
                        &mut |vector| {
                            let vector = vector
                                .validate(self.table.space().dim)
                                .map_err(|_| corrupt("DiskANN eligible vector invalid"))?;
                            distance = score(metric, query, vector)?;
                            Ok(())
                        },
                    )
                })?;
                insert_best(results, SearchResult::new(distance, label), k);
            }
            if rowid.is_none() && count as u64 != self.generation.eligible_count {
                return Err(corrupt("DiskANN published eligible count mismatch"));
            }
            Ok(())
        })();
        finish(statement, outcome)
    }
    fn project(&mut self, rowid: u64) -> Result<Vec<u8>> {
        self.budget.visit()?;
        let mut statement = self
            .table
            .connection()
            .prepare(&self.eligible_sql(true, true))?;
        let outcome = (|| {
            statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
            statement.bind_i64(2, checked_id(rowid)?)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN accepted immutable projection missing"));
            }
            self.eligible_identity(&statement)?;
            let bytes = bounded_blob(
                &statement,
                7,
                8,
                self.table.vector_bytes(),
                true,
                "accepted vector",
            )?;
            with_decoded_vector(
                &bytes,
                self.table.space().dim,
                &mut self.decode,
                &mut |_| Ok(()),
            )?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN accepted rowid not unique"));
            }
            Ok(bytes)
        })();
        finish(statement, outcome)
    }
}

fn check_blob(
    statement: &Statement<'_>,
    length: usize,
    payload: usize,
    expected: usize,
) -> Result<()> {
    if nonnegative(statement, length, "query vector length")? != expected as u64
        || statement.column_type(payload)? != ffi::SQLITE_BLOB as i32
    {
        return Err(corrupt(
            "DiskANN query vector has an invalid byte length/type",
        ));
    }
    Ok(())
}
fn score(metric: DistanceType, query: &[f32], vector: &[f32]) -> Result<f32> {
    // Dimension/finite proof precedes these unchecked native SIMD operations.
    let distance = match metric {
        DistanceType::L2 => ops::l2_sq_f32(query, vector),
        DistanceType::Cosine => ops::ip_dist_f32(query, vector),
        DistanceType::InnerProduct => {
            return Err(IndexError::new("DiskANN does not support inner product"))
        }
    };
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(corrupt("DiskANN non-finite query distance"))
    }
}
fn insert_best(results: &mut Vec<SearchResult>, result: SearchResult, k: usize) {
    let position = results
        .binary_search_by(|other| {
            other
                .distance
                .total_cmp(&result.distance)
                .then(other.rowid.cmp(&result.rowid))
        })
        .unwrap_or_else(|p| p);
    if position < k {
        if results.len() == k {
            results.pop();
        }
        results.insert(position, result);
    }
}

impl GraphStore for PublishedGraph<'_> {
    fn allocate(&mut self, _: u64, _: &[f32]) -> Result<u64> {
        Err(unsupported())
    }
    fn internal_id(&mut self, rowid: u64) -> Result<u64> {
        let mut statement = self
            .table
            .connection()
            .prepare(&self.eligible_sql(true, false))?;
        let outcome = (|| {
            statement.bind_i64(1, checked_id(self.generation.generation_id)?)?;
            statement.bind_i64(2, checked_id(rowid)?)?;
            if statement.step()? != Step::Row {
                return Err(IndexError::new("DiskANN published rowid absent"));
            }
            let (id, _) = self.eligible_identity(&statement)?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN published rowid not unique"));
            }
            Ok(id)
        })();
        finish(statement, outcome)
    }
    fn rowid(&mut self, id: u64) -> Result<Option<u64>> {
        // InlineFilterSearch requests rowid for every expansion, including
        // distance-cache hits. Together with with_vector fetch charges this
        // conservatively covers core visits in the SAME fallback budget.
        self.budget.visit()?;
        self.identity(id).map(|(_, rowid)| rowid)
    }
    fn state(&mut self, id: u64) -> Result<NodeState> {
        self.identity(id).map(|(state, _)| state)
    }
    fn vector(&mut self, id: u64) -> Result<Vec<f32>> {
        let dimension = self.table.space().dim;
        let mut output = Vec::new();
        reserve(&mut output, dimension)?;
        self.with_vector(id, &mut |vector| {
            output.extend_from_slice(
                vector
                    .validate(dimension)
                    .map_err(|_| corrupt("DiskANN query vector invalid"))?,
            );
            Ok(())
        })?;
        Ok(output)
    }
    fn remaining_vector_visits(&self) -> usize {
        self.budget.remaining()
    }
    fn with_vector(
        &mut self,
        id: u64,
        action: &mut dyn FnMut(StoredVector<'_>) -> Result<()>,
    ) -> Result<()> {
        self.budget.visit()?;
        let mut statement = self.table.connection().prepare(&self.node_sql(true))?;
        let outcome = (|| {
            self.bind_node(&mut statement, id)?;
            if statement.step()? != Step::Row {
                return Err(corrupt("DiskANN referenced vector node/entry missing"));
            }
            self.classify(&statement, id)?;
            check_blob(&statement, 7, 8, self.table.vector_bytes())?;
            // The borrowed checked proof lives only in this synchronous column
            // call; action is core pure scoring/copying, with no SQL or yield.
            statement.with_column_blob(8, |bytes| {
                with_decoded_vector(bytes, self.table.space().dim, &mut self.decode, action)
            })?;
            if statement.step()? != Step::Done {
                return Err(corrupt("DiskANN query vector node not unique"));
            }
            Ok(())
        })();
        finish(statement, outcome)
    }
    fn neighbors(&mut self, id: u64) -> Result<Vec<u64>> {
        self.budget.visit()?;
        let node = self.table.read_graph_node(&self.generation, id)?;
        // Range/order/duplicates/self are checked by the frozen accessor. Check
        // each bounded target identity too: a dangling edge is corruption, not EOF.
        for &neighbor in &node.neighbors {
            self.identity(neighbor)?;
        }
        Ok(node.neighbors)
    }
    #[cfg(test)]
    fn set_neighbors(&mut self, _: u64, _: &[u64]) -> Result<()> {
        Err(unsupported())
    }
    fn mark_delete(&mut self, _: u64) -> Result<()> {
        Err(unsupported())
    }
    fn start_points(&mut self) -> Result<Vec<u64>> {
        let root = self
            .generation
            .root_local_id
            .ok_or_else(|| corrupt("DiskANN nonempty generation has no frozen root"))?;
        if self.identity(root)?.0 != NodeState::Frozen {
            return Err(corrupt("DiskANN start point is not frozen"));
        }
        Ok(vec![root])
    }
}

pub(crate) fn knn(
    table: &DiskAnnTable,
    query: &[f32],
    k: usize,
    override_l: Option<usize>,
    filter: SearchFilter<'_>,
    project_vector: bool,
) -> Result<QueryRows> {
    if override_l.is_some_and(|l| l == 0 || l > table.options().max_visits) {
        return Err(IndexError::new(
            "DiskANN search list override must be positive and no greater than max_visits",
        ));
    }
    if query.len() != table.space().dim {
        return Err(IndexError::new("DiskANN vector dimension mismatch"));
    }
    let mut normalized = Vec::new();
    reserve(&mut normalized, query.len())?;
    normalized.extend_from_slice(query);
    normalize_input(
        &mut normalized,
        table.space().dim,
        table.space().distance_type,
    )?;
    let (metadata, anchor) = table.read_metadata()?;
    let Some(id) = metadata.published_generation else {
        Budget::admit(table, 0, &filter, project_vector)?;
        return Ok(QueryRows {
            results: Vec::new(),
            vectors: project_vector.then(Vec::new),
            anchor,
        });
    };
    let generation = table.read_generation(id)?;
    if generation.state != 1 || generation.source_revision > metadata.revision {
        return Err(corrupt(
            "DiskANN published generation identity/revision invalid",
        ));
    }
    if let Some(root) = generation.root_local_id {
        table.read_graph_node(&generation, root)?;
    }
    let eligible = usize::try_from(generation.eligible_count)
        .map_err(|_| too_big("DiskANN eligible population overflow"))?;
    let count = k.min(eligible).min(match &filter {
        SearchFilter::Equals(_) => 1,
        SearchFilter::In(ids) => ids.len(),
        SearchFilter::None => usize::MAX,
    });
    let budget = Budget::admit(table, count, &filter, project_vector)?;
    let mut graph = PublishedGraph {
        table,
        generation,
        revision: metadata.revision,
        budget,
        decode: Vec::new(),
    };
    if count == 0 {
        return Ok(QueryRows {
            results: Vec::new(),
            vectors: project_vector.then(Vec::new),
            anchor,
        });
    }
    let l = override_l
        .unwrap_or(table.options().search_list_size)
        .max(count);
    let filtered_small = match &filter {
        SearchFilter::Equals(_) => true,
        SearchFilter::In(ids) => ids.len() <= l,
        SearchFilter::None => false,
    };
    let mut results = if eligible <= l || count >= eligible || filtered_small {
        graph.exact(&normalized, count, &filter)?
    } else {
        let options = table.options();
        let config = GraphConfig {
            dimension: table.space().dim,
            distance: table.space().distance_type,
            max_degree: options.degree,
            construction_l: options.build_list_size,
            search_l: options.search_list_size,
            alpha: options.alpha,
            limits: ResourceLimits {
                max_visits: options.max_visits,
                max_cached_vectors: 2048,
                max_vector_bytes: graph.budget.core_bytes,
            },
        };
        let set = match &filter {
            SearchFilter::In(ids) => Some(*ids),
            _ => None,
        };
        let found = diskann_core::search(&mut graph, &config, &normalized, count, override_l, set)?;
        if found.len() < count {
            // Same owner budget; never reset visits or search raw current rows.
            graph.exact(&normalized, count, &filter)?
        } else {
            found
        }
    };
    results.sort_unstable_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then(a.rowid.cmp(&b.rowid))
    });
    let vectors = if project_vector {
        let mut vectors = Vec::new();
        reserve(&mut vectors, results.len())?;
        for result in &results {
            vectors.push(graph.project(result.rowid)?);
        }
        Some(vectors)
    } else {
        None
    };
    Ok(QueryRows {
        results,
        vectors,
        anchor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_topk_has_deterministic_public_label_ties() {
        let mut results = Vec::with_capacity(2);
        for rowid in [8, 0, i64::MAX as u64, 3] {
            insert_best(&mut results, SearchResult::new(1.0, rowid), 2);
        }
        assert_eq!(results.iter().map(|r| r.rowid).collect::<Vec<_>>(), [0, 3]);
        insert_best(&mut results, SearchResult::new(0.0, 7), 2);
        assert_eq!(results.iter().map(|r| r.rowid).collect::<Vec<_>>(), [7, 0]);
    }
    #[test]
    fn zero_k_does_not_insert() {
        let mut results = Vec::new();
        insert_best(&mut results, SearchResult::new(0.0, 0), 0);
        assert!(results.is_empty());
    }
}
