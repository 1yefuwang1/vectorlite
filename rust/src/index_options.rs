//! Parsing of the HNSW index-options string, e.g.
//! `hnsw(max_elements=1000, M=16)`. Mirrors `index_options.cpp`.

#[derive(Clone, Debug)]
pub struct IndexOptions {
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub random_seed: usize,
    pub allow_replace_deleted: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            max_elements: 0,
            m: 16,
            ef_construction: 200,
            random_seed: 100,
            allow_replace_deleted: true,
        }
    }
}

fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Mirrors `absl::SimpleAtob`: accepts a small set of boolean spellings,
/// case-insensitively.
fn parse_bool(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "y" | "on" | "1" => Some(true),
        "false" | "f" | "no" | "n" | "off" | "0" => Some(false),
        _ => None,
    }
}

impl IndexOptions {
    pub fn parse(input: &str) -> Result<IndexOptions, String> {
        let trimmed = input.trim();
        let inner = trimmed
            .strip_prefix("hnsw(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| "Invalid index option. Only hnsw is supported".to_string())?;

        let mut options = IndexOptions::default();
        let mut has_max_elements = false;

        let body = inner.trim();
        if !body.is_empty() {
            for pair in body.split(',') {
                let pair = pair.trim();
                let (key, value) = pair.split_once('=').ok_or_else(|| {
                    format!(
                        "Invalid index option. Expected comma-separated key=value pairs, got: {inner}"
                    )
                })?;
                let key = key.trim();
                let value = value.trim();
                if !is_word(key) || !is_word(value) {
                    return Err(format!(
                        "Invalid index option. Expected comma-separated key=value pairs, got: {inner}"
                    ));
                }
                match key {
                    "max_elements" => {
                        options.max_elements = value
                            .parse()
                            .map_err(|_| format!("Cannot parse max_elements: {value}"))?;
                        has_max_elements = true;
                    }
                    "M" => {
                        options.m = value
                            .parse()
                            .map_err(|_| format!("Cannot parse M: {value}"))?;
                    }
                    "ef_construction" => {
                        options.ef_construction = value
                            .parse()
                            .map_err(|_| format!("Cannot parse ef_construction: {value}"))?;
                    }
                    "random_seed" => {
                        options.random_seed = value
                            .parse()
                            .map_err(|_| format!("Cannot parse random_seed: {value}"))?;
                    }
                    "allow_replace_deleted" => {
                        options.allow_replace_deleted = parse_bool(value).ok_or_else(|| {
                            format!("Cannot parse allow_replace_deleted: {value}")
                        })?;
                    }
                    other => {
                        return Err(format!("Invalid index option: {other}"));
                    }
                }
            }
        }

        if !has_max_elements {
            return Err("max_elements is required but not provided".to_string());
        }
        Ok(options)
    }
}
