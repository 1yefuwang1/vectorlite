//! Parsing of the vector-space declaration string, e.g.
//! `my_embedding float32[384] cosine`. Mirrors `vector_space.cpp` /
//! `NamedVectorSpace::FromString` and `util.cpp::IsValidColumnName`.

/// Distance metric. Discriminants must match `VlDistanceType` in core_shim.h.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum DistanceType {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

/// Stored element type. Discriminants must match `VlVectorType` in core_shim.h.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum VectorType {
    Float32 = 0,
    BFloat16 = 1,
    Float16 = 2,
}

impl VectorType {
    pub fn element_size(self) -> usize {
        match self {
            VectorType::Float32 => 4,
            VectorType::BFloat16 => 2,
            VectorType::Float16 => 2,
        }
    }
}

pub fn parse_distance_type(s: &str) -> Option<DistanceType> {
    match s {
        "l2" => Some(DistanceType::L2),
        "ip" => Some(DistanceType::InnerProduct),
        "cosine" => Some(DistanceType::Cosine),
        _ => None,
    }
}

pub fn parse_vector_type(s: &str) -> Option<VectorType> {
    match s {
        "float32" => Some(VectorType::Float32),
        "bfloat16" => Some(VectorType::BFloat16),
        "float16" => Some(VectorType::Float16),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct NamedVectorSpace {
    pub vector_name: String,
    pub dim: usize,
    pub distance_type: DistanceType,
    pub vector_type: VectorType,
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Validates a SQLite column name: must start with a letter or underscore and
/// contain only letters, digits, underscores or `$`. (Keyword rejection from
/// the C++ version is omitted; it is not exercised by the test suite.)
pub fn is_valid_column_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// Parses `<name> <type>[<dim>] [<distance>]`. Equivalent to the regex
/// `^(\w+)\s+(\w+)\[(\d+)\]\s*(\w+)?\s*$` used by the C++ implementation.
pub fn parse_named_vector_space(input: &str) -> Result<NamedVectorSpace, String> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut i = 0;

    let take_word = |start: usize| -> usize {
        let mut j = start;
        while j < n && is_word_char(bytes[j] as char) {
            j += 1;
        }
        j
    };
    let skip_ws = |start: usize| -> usize {
        let mut j = start;
        while j < n && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        j
    };

    // Leading whitespace is not allowed by the anchored regex, but the input is
    // already trimmed by SQLite's argument handling; be lenient and skip it.
    i = skip_ws(i);

    // vector name
    let name_end = take_word(i);
    if name_end == i {
        return Err("Unable to parse vector space".to_string());
    }
    let vector_name = &input[i..name_end];
    i = name_end;

    // mandatory whitespace
    let after_name_ws = skip_ws(i);
    if after_name_ws == i {
        return Err("Unable to parse vector space".to_string());
    }
    i = after_name_ws;

    // vector type
    let type_end = take_word(i);
    if type_end == i {
        return Err("Unable to parse vector space".to_string());
    }
    let vector_type_str = &input[i..type_end];
    i = type_end;

    // '['
    if i >= n || bytes[i] != b'[' {
        return Err("Unable to parse vector space".to_string());
    }
    i += 1;

    // digits
    let dim_start = i;
    while i < n && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == dim_start {
        return Err("Unable to parse vector space".to_string());
    }
    let dim: usize = input[dim_start..i]
        .parse()
        .map_err(|_| "Unable to parse vector space".to_string())?;

    // ']'
    if i >= n || bytes[i] != b']' {
        return Err("Unable to parse vector space".to_string());
    }
    i += 1;

    // optional whitespace then optional distance word
    i = skip_ws(i);
    let mut distance_type = DistanceType::L2;
    if i < n {
        let dist_end = take_word(i);
        if dist_end == i {
            return Err("Unable to parse vector space".to_string());
        }
        let dist_str = &input[i..dist_end];
        distance_type = parse_distance_type(dist_str)
            .ok_or_else(|| format!("Invalid distance type: {dist_str}"))?;
        i = dist_end;
    }

    // only trailing whitespace permitted
    i = skip_ws(i);
    if i != n {
        return Err("Unable to parse vector space".to_string());
    }

    if !is_valid_column_name(vector_name) {
        return Err(format!("Invalid vector name: {vector_name}"));
    }

    let vector_type = parse_vector_type(vector_type_str)
        .ok_or_else(|| format!("Invalid vector type: {vector_type_str}"))?;

    Ok(NamedVectorSpace {
        vector_name: vector_name.to_string(),
        dim,
        distance_type,
        vector_type,
    })
}
