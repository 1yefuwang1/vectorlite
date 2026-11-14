//! Callback-local evidence for one exact stored-vector slice.
//!
//! Only `ValidatedVector::new` can create the proof. A provider may instead
//! lend an unvalidated slice; core checks it without letting a validation error
//! short-circuit the provider's statement cleanup. Neither variant owns data.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VectorValidationError {
    Dimension,
    NonFinite,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedVector<'a> {
    values: &'a [f32],
}

impl<'a> ValidatedVector<'a> {
    pub(crate) fn new(values: &'a [f32], dimension: usize) -> Result<Self, VectorValidationError> {
        if values.len() != dimension {
            return Err(VectorValidationError::Dimension);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(VectorValidationError::NonFinite);
        }
        Ok(Self { values })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum StoredVector<'a> {
    Unvalidated(&'a [f32]),
    Validated(ValidatedVector<'a>),
}

impl<'a> StoredVector<'a> {
    /// A finite proof can have been checked for a different provider dimension.
    /// Always match the current GraphConfig before any distance kernel or copy.
    pub(crate) fn validate(self, dimension: usize) -> Result<&'a [f32], VectorValidationError> {
        let validated = match self {
            Self::Unvalidated(values) => ValidatedVector::new(values, dimension)?,
            Self::Validated(validated) => {
                if validated.values.len() != dimension {
                    return Err(VectorValidationError::Dimension);
                }
                validated
            }
        };
        Ok(validated.values)
    }

    /// Access for the decoder's already-checked copy path, not a validation step.
    pub(crate) fn as_slice(self) -> &'a [f32] {
        match self {
            Self::Unvalidated(values) => values,
            Self::Validated(validated) => validated.values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_checks_exact_dimension_and_all_nonfinite_values() {
        let values = [1.25, -2.5, 0.0];
        let proof = ValidatedVector::new(&values, values.len()).unwrap();
        let view = StoredVector::Validated(proof);
        assert_eq!(view.validate(3).unwrap(), values);
        assert_eq!(view.as_slice().as_ptr(), values.as_ptr());
        assert_eq!(view.validate(2), Err(VectorValidationError::Dimension));
        assert!(matches!(
            ValidatedVector::new(&values, 2),
            Err(VectorValidationError::Dimension)
        ));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let values = [0.0, value];
            assert!(matches!(
                ValidatedVector::new(&values, 2),
                Err(VectorValidationError::NonFinite)
            ));
            assert_eq!(
                StoredVector::Unvalidated(&values).validate(2),
                Err(VectorValidationError::NonFinite)
            );
        }
        assert_eq!(
            StoredVector::Unvalidated(&values).validate(4),
            Err(VectorValidationError::Dimension)
        );
    }
}
