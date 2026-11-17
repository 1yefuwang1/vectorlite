//! Parsing of the HNSW index-options string, e.g.
//! `hnsw(max_elements=1000, M=16)`.

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

/// Backend selection preserves the original HNSW parser and configuration.
#[derive(Clone, Debug)]
pub enum BackendOptions {
    Hnsw(IndexOptions),
    Diskann(DiskAnnOptions),
}

impl BackendOptions {
    pub fn parse(input: &str) -> Result<Self, String> {
        if input.trim().starts_with("diskann(") {
            DiskAnnOptions::parse(input).map(Self::Diskann)
        } else {
            IndexOptions::parse(input).map(Self::Hnsw)
        }
    }
}

/// DiskANN settings bound working memory independently of the stored row count.
#[derive(Clone, Debug, PartialEq)]
pub struct DiskAnnOptions {
    pub degree: usize,
    pub build_list_size: usize,
    pub search_list_size: usize,
    pub alpha: f32,
    pub cache_bytes: usize,
    pub max_visits: usize,
}

impl Default for DiskAnnOptions {
    fn default() -> Self {
        Self {
            degree: 32,
            build_list_size: 100,
            search_list_size: 64,
            alpha: 1.2,
            cache_bytes: 64 * 1024 * 1024,
            max_visits: 65_536,
        }
    }
}

impl DiskAnnOptions {
    pub fn parse(input: &str) -> Result<Self, String> {
        let inner = input
            .trim()
            .strip_prefix("diskann(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| {
                "Invalid DiskANN options; expected diskann(key=value, ...)".to_owned()
            })?;
        let mut options = Self::default();
        let mut seen = std::collections::HashSet::new();
        if !inner.trim().is_empty() {
            for pair in inner.split(',') {
                let (key, value) = pair
                    .trim()
                    .split_once('=')
                    .ok_or_else(|| "DiskANN options require key=value pairs".to_owned())?;
                let (key, value) = (key.trim(), value.trim());
                if !seen.insert(key) {
                    return Err(format!("Duplicate DiskANN option: {key}"));
                }
                let parse_size = || {
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("Cannot parse DiskANN {key}: {value}"))
                };
                match key {
                    "degree" => options.degree = parse_size()?,
                    "build_list_size" => options.build_list_size = parse_size()?,
                    "search_list_size" => options.search_list_size = parse_size()?,
                    "cache_bytes" => options.cache_bytes = parse_size()?,
                    "max_visits" => options.max_visits = parse_size()?,
                    "alpha" => {
                        options.alpha = value
                            .parse()
                            .map_err(|_| format!("Cannot parse DiskANN alpha: {value}"))?;
                    }
                    _ => return Err(format!("Invalid DiskANN option: {key}")),
                }
            }
        }
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(2..=10_000).contains(&self.degree) {
            return Err("DiskANN degree must be between 2 and 10000".into());
        }
        if self.build_list_size == 0
            || self.search_list_size == 0
            || self.max_visits == 0
            || self.max_visits > u32::MAX as usize
            || self.build_list_size > self.max_visits
            || self.search_list_size > self.max_visits
        {
            return Err("DiskANN list sizes must be positive and no greater than max_visits (at most u32::MAX)".into());
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err("DiskANN alpha must be finite and at least 1".into());
        }
        if self.cache_bytes < 4096 || self.cache_bytes > isize::MAX as usize {
            return Err("DiskANN cache_bytes must be between 4096 and isize::MAX".into());
        }
        Ok(())
    }

    pub fn physical_degree(&self) -> usize {
        (self.degree as f32 * 1.3) as usize
    }
}

fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Accepts common boolean spellings case-insensitively.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_options() {
        let o = IndexOptions::parse(
            "hnsw(max_elements=1000, M=32, ef_construction=100, random_seed=7, allow_replace_deleted=false)",
        )
        .unwrap();
        assert_eq!(o.max_elements, 1000);
        assert_eq!(o.m, 32);
        assert_eq!(o.ef_construction, 100);
        assert_eq!(o.random_seed, 7);
        assert!(!o.allow_replace_deleted);
    }

    #[test]
    fn applies_defaults_for_unspecified_options() {
        let o = IndexOptions::parse("hnsw(max_elements=10)").unwrap();
        assert_eq!(o.max_elements, 10);
        assert_eq!(o.m, 16);
        assert_eq!(o.ef_construction, 200);
        assert_eq!(o.random_seed, 100);
        assert!(o.allow_replace_deleted);
    }

    #[test]
    fn requires_max_elements() {
        assert!(IndexOptions::parse("hnsw(M=16)").is_err());
        assert!(IndexOptions::parse("hnsw()").is_err());
    }

    #[test]
    fn rejects_non_hnsw_and_malformed() {
        assert!(IndexOptions::parse("flat(max_elements=1)").is_err());
        assert!(IndexOptions::parse("hnsw(max_elements)").is_err());
        assert!(IndexOptions::parse("hnsw(max_elements=1, bogus=2)").is_err());
        assert!(IndexOptions::parse("hnsw(max_elements=notanumber)").is_err());
    }

    #[test]
    fn accepts_arbitrary_option_order() {
        let options = IndexOptions::parse(
            "hnsw(random_seed=7, M=8, allow_replace_deleted=false, max_elements=100, ef_construction=50)",
        )
        .unwrap();
        assert_eq!(options.max_elements, 100);
        assert_eq!(options.m, 8);
        assert_eq!(options.ef_construction, 50);
        assert_eq!(options.random_seed, 7);
        assert!(!options.allow_replace_deleted);
    }

    #[test]
    fn rejects_integer_overflow_and_missing_separators() {
        assert!(IndexOptions::parse("hnsw(max_elements=99999999999999999999999999999)").is_err());
        assert!(IndexOptions::parse("hnsw(max_elements=10 M=16)").is_err());
        assert!(IndexOptions::parse("hnsw(max_elements=10,)").is_err());
    }

    #[test]
    fn bool_spellings() {
        assert_eq!(parse_bool("Yes"), Some(true));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("OFF"), Some(false));
        assert_eq!(parse_bool("maybe"), None);
    }
}
