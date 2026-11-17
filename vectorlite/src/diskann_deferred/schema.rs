//! Format4 durable layouts. No triggers, foreign keys, or recovery on open.
use super::*;

pub(super) struct Layout {
    pub suffix: &'static str,
    pub definition: &'static str,
    // Columns are name/type/not-null/primary-key ordinal.
    pub columns: &'static [(&'static str, &'static str, u64, u64)],
    pub without_rowid: bool,
    pub unique: &'static [&'static [&'static str]],
}

pub(super) const LAYOUTS: &[Layout] = &[
 Layout { suffix:"diskann_meta", without_rowid:false, unique:&[],
 definition:"singleton INTEGER PRIMARY KEY CHECK(singleton=1),format_version INTEGER NOT NULL CHECK(format_version=4),instance_id BLOB NOT NULL CHECK(length(instance_id)=16),descriptor BLOB NOT NULL,revision INTEGER NOT NULL CHECK(revision>=0),next_entry_id INTEGER NOT NULL CHECK(next_entry_id>=1),next_job_id INTEGER NOT NULL CHECK(next_job_id>=1),next_generation_id INTEGER NOT NULL CHECK(next_generation_id>=1),canonical_count INTEGER NOT NULL CHECK(canonical_count>=0),retired_count INTEGER NOT NULL CHECK(retired_count>=0),published_generation INTEGER CHECK(published_generation>=1),candidate_generation INTEGER CHECK(candidate_generation>=1),active_job INTEGER CHECK(active_job>=1),next_fence INTEGER NOT NULL CHECK(next_fence>=1),cleanup_entry_id INTEGER NOT NULL CHECK(cleanup_entry_id>=0 AND cleanup_entry_id<next_entry_id)",
 columns:&[("singleton","INTEGER",0,1),("format_version","INTEGER",1,0),("instance_id","BLOB",1,0),("descriptor","BLOB",1,0),("revision","INTEGER",1,0),("next_entry_id","INTEGER",1,0),("next_job_id","INTEGER",1,0),("next_generation_id","INTEGER",1,0),("canonical_count","INTEGER",1,0),("retired_count","INTEGER",1,0),("published_generation","INTEGER",0,0),("candidate_generation","INTEGER",0,0),("active_job","INTEGER",0,0),("next_fence","INTEGER",1,0),("cleanup_entry_id","INTEGER",1,0)] },
 Layout {suffix:"diskann_entries",without_rowid:false,unique:&[],
 definition:"entry_id INTEGER PRIMARY KEY CHECK(entry_id>=1),public_rowid INTEGER NOT NULL CHECK(public_rowid>=0),created_revision INTEGER NOT NULL CHECK(created_revision>=1),retired_revision INTEGER CHECK(retired_revision>created_revision),vector BLOB NOT NULL CHECK(length(vector)={vector_bytes})",
 columns:&[("entry_id","INTEGER",0,1),("public_rowid","INTEGER",1,0),("created_revision","INTEGER",1,0),("retired_revision","INTEGER",0,0),("vector","BLOB",1,0)]},
 Layout {suffix:"diskann_current",without_rowid:false,unique:&[&["entry_id"]],
 definition:"public_rowid INTEGER PRIMARY KEY CHECK(public_rowid>=0),entry_id INTEGER NOT NULL UNIQUE CHECK(entry_id>=1)",
 columns:&[("public_rowid","INTEGER",0,1),("entry_id","INTEGER",1,0)]},
 Layout {suffix:"diskann_generations",without_rowid:false,unique:&[],
 definition:"generation_id INTEGER PRIMARY KEY CHECK(generation_id>=1),job_id INTEGER NOT NULL CHECK(job_id>=1),state INTEGER NOT NULL CHECK(state IN (0,1,2)),source_revision INTEGER NOT NULL CHECK(source_revision>=0),source_hash BLOB CHECK(source_hash IS NULL OR length(source_hash)=32),config_hash BLOB NOT NULL CHECK(length(config_hash)=32),total_count INTEGER NOT NULL CHECK(total_count>=0 AND total_count<=4294967294),eligible_count INTEGER NOT NULL CHECK(eligible_count>=0 AND eligible_count<=total_count),root_local_id INTEGER,CHECK(state!=1 OR source_hash IS NOT NULL),CHECK((total_count=0 AND root_local_id IS NULL) OR (total_count>0 AND root_local_id=total_count))",
 columns:&[("generation_id","INTEGER",0,1),("job_id","INTEGER",1,0),("state","INTEGER",1,0),("source_revision","INTEGER",1,0),("source_hash","BLOB",0,0),("config_hash","BLOB",1,0),("total_count","INTEGER",1,0),("eligible_count","INTEGER",1,0),("root_local_id","INTEGER",0,0)]},
 Layout {suffix:"diskann_graph_nodes",without_rowid:true,unique:&[&["generation_id","native_local_id"],&["generation_id","entry_id"]],
 definition:"generation_id INTEGER NOT NULL CHECK(generation_id>=1),native_local_id INTEGER NOT NULL CHECK(native_local_id>=0 AND native_local_id<=4294967294),entry_id INTEGER CHECK(entry_id>=1),state INTEGER NOT NULL CHECK(state IN (0,2)),frozen_vector BLOB,neighbors BLOB NOT NULL CHECK(length(neighbors)<={adjacency_bytes} AND length(neighbors)%8=0),PRIMARY KEY(generation_id,native_local_id),UNIQUE(generation_id,entry_id),CHECK((state=0 AND entry_id IS NOT NULL AND frozen_vector IS NULL) OR (state=2 AND entry_id IS NULL AND typeof(frozen_vector)='blob' AND length(frozen_vector)={vector_bytes}))",
 columns:&[("generation_id","INTEGER",1,1),("native_local_id","INTEGER",1,2),("entry_id","INTEGER",0,0),("state","INTEGER",1,0),("frozen_vector","BLOB",0,0),("neighbors","BLOB",1,0)]},
 Layout {suffix:"diskann_jobs",without_rowid:false,unique:&[],
 definition:"job_id INTEGER PRIMARY KEY CHECK(job_id>=1),generation_id INTEGER NOT NULL CHECK(generation_id>=1),parent_generation INTEGER CHECK(parent_generation>=1),cutoff_revision INTEGER NOT NULL CHECK(cutoff_revision>=0),fence INTEGER NOT NULL CHECK(fence>=1),owner_nonce BLOB CHECK(owner_nonce IS NULL OR length(owner_nonce)=16),phase INTEGER NOT NULL CHECK(phase BETWEEN 0 AND 6),expected_count INTEGER NOT NULL CHECK(expected_count>=0 AND expected_count<=4294967294),scan_entry_id INTEGER NOT NULL CHECK(scan_entry_id>=0),captured_count INTEGER NOT NULL CHECK(captured_count>=0 AND captured_count<=expected_count),import_cursor INTEGER NOT NULL CHECK(import_cursor>=0),validation_cursor INTEGER NOT NULL CHECK(validation_cursor>=0),cleanup_cursor INTEGER NOT NULL CHECK(cleanup_cursor>=0),source_hash BLOB CHECK(source_hash IS NULL OR length(source_hash)=32),config_hash BLOB NOT NULL CHECK(length(config_hash)=32),candidate_hash BLOB CHECK(candidate_hash IS NULL OR length(candidate_hash)=32),failure_code INTEGER,cutoff_entry_id INTEGER NOT NULL CHECK(cutoff_entry_id>=0),snapshot_chain BLOB NOT NULL CHECK(typeof(snapshot_chain)='blob' AND length(snapshot_chain)=32),CHECK(scan_entry_id<=cutoff_entry_id AND expected_count<=cutoff_entry_id)",
 columns:&[("job_id","INTEGER",0,1),("generation_id","INTEGER",1,0),("parent_generation","INTEGER",0,0),("cutoff_revision","INTEGER",1,0),("fence","INTEGER",1,0),("owner_nonce","BLOB",0,0),("phase","INTEGER",1,0),("expected_count","INTEGER",1,0),("scan_entry_id","INTEGER",1,0),("captured_count","INTEGER",1,0),("import_cursor","INTEGER",1,0),("validation_cursor","INTEGER",1,0),("cleanup_cursor","INTEGER",1,0),("source_hash","BLOB",0,0),("config_hash","BLOB",1,0),("candidate_hash","BLOB",0,0),("failure_code","INTEGER",0,0),("cutoff_entry_id","INTEGER",1,0),("snapshot_chain","BLOB",1,0)]},
 Layout {suffix:"diskann_snapshot_map",without_rowid:true,unique:&[&["job_id","ordinal"],&["job_id","entry_id"]],
 definition:"job_id INTEGER NOT NULL CHECK(job_id>=1),ordinal INTEGER NOT NULL CHECK(ordinal>=0 AND ordinal<=4294967293),entry_id INTEGER NOT NULL CHECK(entry_id>=1),public_rowid INTEGER NOT NULL CHECK(public_rowid>=0),PRIMARY KEY(job_id,ordinal),UNIQUE(job_id,entry_id)",
 columns:&[("job_id","INTEGER",1,1),("ordinal","INTEGER",1,2),("entry_id","INTEGER",1,0),("public_rowid","INTEGER",1,0)]},
 Layout {suffix:"diskann_txn",without_rowid:false,unique:&[],
 definition:"singleton INTEGER PRIMARY KEY CHECK(singleton IN (1,2)),value INTEGER NOT NULL CHECK(value=0)",
 columns:&[("singleton","INTEGER",0,1),("value","INTEGER",1,0)]},
];

impl DiskAnnTable {
    pub(super) fn create_schema(&self) -> Result<()> {
        for layout in LAYOUTS {
            let definition = layout
                .definition
                .replace("{vector_bytes}", &self.vector_bytes.to_string())
                .replace("{adjacency_bytes}", &self.adjacency_bytes.to_string());
            self.connection.execute(&format!(
                "CREATE TABLE {} ({}){}",
                self.names.qualified(layout.suffix)?,
                definition,
                if layout.without_rowid {
                    " WITHOUT ROWID"
                } else {
                    ""
                }
            ))?;
        }
        self.connection
            .execute(&format!("INSERT INTO {} VALUES(1,0),(2,0)", self.names.txn))?;
        if self.connection.changes() != 2 {
            return Err(corrupt("DiskANN guard initialization count"));
        }
        let bytes =
            serde_json::to_vec(&self.descriptor).map_err(|e| IndexError::new(e.to_string()))?;
        if bytes.len() > MAX_DESCRIPTOR_BYTES {
            return Err(too_big("DiskANN descriptor too large"));
        }
        let mut stmt = self.connection.prepare(&format!(
            "INSERT INTO {} VALUES(1,4,?1,?2,0,1,1,1,0,0,NULL,NULL,NULL,1,0)",
            self.names.meta
        ))?;
        stmt.bind_blob(1, &self.connection.random_instance_id())?;
        stmt.bind_blob(2, &bytes)?;
        done(stmt)?;
        self.expect_one_change("metadata initialization")
    }
    /// Per-layout catalog validation (D2a): the eight layouts keep their own
    /// checks and message order, and only the five pragma shapes are retained.
    pub(super) fn validate_schema(&self) -> Result<()> {
        for layout in LAYOUTS {
            let table = self.names.base(layout.suffix);
            let (kind, columns, without_rowid, strict) = self.pragma_table_row(
                &table,
                ["column count", "WITHOUT ROWID", "STRICT"],
                "DiskANN shadow table kind is incompatible",
            )?;
            if !matches!(kind.as_str(), "table" | "shadow")
                || columns != layout.columns.len() as u64
                || without_rowid != u64::from(layout.without_rowid)
                || strict != 0
            {
                return Err(corrupt("DiskANN shadow table kind is incompatible"));
            }
            self.validate_columns(&table, layout.columns, SLOT_CATALOG_COLUMNS)?;
            self.validate_indexes(&table, layout.unique)?;
            self.with_writer(
                SLOT_CATALOG_FOREIGN_KEYS,
                || String::from("SELECT 1 FROM pragma_foreign_key_list(?1,?2) LIMIT 1"),
                |fks| {
                    fks.bind_text(1, &table)?;
                    fks.bind_text(2, &self.names.schema)?;
                    if fks.step()? != Step::Done {
                        return Err(corrupt("DiskANN shadows cannot have foreign keys"));
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }
    pub(super) fn validate_indexes(&self, table: &str, expected: &[&[&str]]) -> Result<()> {
        self.with_writer(
            SLOT_CATALOG_INDEX_LIST,
            || {
                String::from(
                    "SELECT length(name),name,\"unique\",origin,partial FROM pragma_index_list(?1,?2)",
                )
            },
            |list| {
                list.bind_text(1, table)?;
                list.bind_text(2, &self.names.schema)?;
                let mut seen = vec![false; expected.len()];
                while list.step()? == Step::Row {
                    if nonnegative(list, 0, "index name length")? > (table.len() + 64) as u64
                        || nonnegative(list, 2, "unique flag")? != 1
                        || !matches!(list.column_text(3)?.as_str(), "u" | "pk")
                        || nonnegative(list, 4, "partial flag")? != 0
                    {
                        return Err(corrupt("DiskANN unexpected shadow index"));
                    }
                    let name = list.column_text(1)?;
                    let mut names = Vec::new();
                    self.with_writer(
                        SLOT_CATALOG_INDEX_INFO,
                        || {
                            String::from(
                                "SELECT seqno,length(name),name FROM pragma_index_info(?1,?2) ORDER BY seqno",
                            )
                        },
                        |cols| {
                            cols.bind_text(1, &name)?;
                            cols.bind_text(2, &self.names.schema)?;
                            while cols.step()? == Step::Row {
                                if names.len() >= 2
                                    || nonnegative(cols, 0, "index sequence")? != names.len() as u64
                                    || nonnegative(cols, 1, "index column length")? > 32
                                {
                                    return Err(corrupt("DiskANN incompatible unique index"));
                                }
                                names.push(cols.column_text(2)?);
                            }
                            Ok(())
                        },
                    )?;
                    let pos = expected
                        .iter()
                        .position(|e| {
                            e.len() == names.len() && e.iter().zip(&names).all(|(a, b)| a == b)
                        })
                        .ok_or_else(|| corrupt("DiskANN unique index does not match layout"))?;
                    if seen[pos] {
                        return Err(corrupt("DiskANN duplicate layout index"));
                    }
                    seen[pos] = true;
                }
                if seen.iter().any(|v| !*v) {
                    return Err(corrupt("DiskANN required unique index is missing"));
                }
                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts_and_unique_reverse_maps_are_fixed_and_bounded() {
        assert_eq!(LAYOUTS.len(), 8);
        for (index, layout) in LAYOUTS.iter().enumerate() {
            assert!(!LAYOUTS[..index].iter().any(|l| l.suffix == layout.suffix));
            assert!(layout.columns.len() <= 19);
            assert!(layout.unique.len() <= 2);
            for names in layout.unique {
                assert!(names.len() <= 2);
                for name in *names {
                    assert!(layout.columns.iter().any(|c| c.0 == *name));
                }
            }
            assert!(!layout.definition.contains("FOREIGN KEY"));
        }
        let graph = LAYOUTS
            .iter()
            .find(|l| l.suffix == "diskann_graph_nodes")
            .unwrap();
        assert!(graph.without_rowid);
        assert!(graph.unique.contains(&&["generation_id", "entry_id"][..]));
        assert!(graph.definition.contains("state=2 AND entry_id IS NULL"));
        assert!(graph
            .definition
            .contains("state=0 AND entry_id IS NOT NULL"));
    }
    #[test]
    fn immutable_original_labels_and_native_root_n_are_explicit() {
        let entries = LAYOUTS
            .iter()
            .find(|l| l.suffix == "diskann_entries")
            .unwrap();
        assert!(entries.definition.contains("public_rowid INTEGER NOT NULL"));
        assert!(entries
            .definition
            .contains("retired_revision>created_revision"));
        let generations = LAYOUTS
            .iter()
            .find(|l| l.suffix == "diskann_generations")
            .unwrap();
        assert!(generations
            .definition
            .contains("total_count=0 AND root_local_id IS NULL"));
        assert!(generations.definition.contains("root_local_id=total_count"));
        assert!(generations.definition.contains("total_count<=4294967294"));
    }
}

#[cfg(test)]
mod refinement_tests {
    use super::*;
    #[test]
    fn positive_fence_highwater_chain_and_cleanup_are_explicit_tail_fields() {
        let meta = LAYOUTS.iter().find(|l| l.suffix == "diskann_meta").unwrap();
        assert_eq!(
            &meta.columns[13..],
            &[
                ("next_fence", "INTEGER", 1, 0),
                ("cleanup_entry_id", "INTEGER", 1, 0)
            ]
        );
        assert!(meta.definition.contains("next_fence>=1"));
        assert!(meta.definition.contains("cleanup_entry_id<next_entry_id"));
        let jobs = LAYOUTS.iter().find(|l| l.suffix == "diskann_jobs").unwrap();
        assert_eq!(
            &jobs.columns[17..],
            &[
                ("cutoff_entry_id", "INTEGER", 1, 0),
                ("snapshot_chain", "BLOB", 1, 0)
            ]
        );
        assert!(jobs.definition.contains("scan_entry_id<=cutoff_entry_id"));
        assert!(jobs.definition.contains("expected_count<=cutoff_entry_id"));
        assert!(jobs
            .definition
            .contains("typeof(snapshot_chain)='blob' AND length(snapshot_chain)=32"));
    }
}
