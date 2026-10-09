use core::fmt;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::Deserializer;

use crate::merkle::{Digest, MAX_CONTEXT_FIELD_BYTES, MAX_LEAF_BYTES};
use crate::spot_check::MAX_CHALLENGE_OPENINGS;

const MAX_PROOF_SIBLINGS: usize = 20;

pub(crate) fn context_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringVisitor;

    impl Visitor<'_> for StringVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "a nonempty UTF-8 string of at most {MAX_CONTEXT_FIELD_BYTES} bytes"
            )
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            validate_string(value)
                .map(|()| value.to_owned())
                .map_err(E::custom)
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            validate_string(&value).map(|_| value).map_err(E::custom)
        }
    }

    deserializer.deserialize_string(StringVisitor)
}

pub(crate) fn leaf_bytes<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_byte_buf(BytesVisitor)
}

pub(crate) fn proof_siblings<'de, D>(deserializer: D) -> Result<Vec<Digest>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(DigestVecVisitor)
}

pub(crate) fn challenge_indices<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(IndexVecVisitor)
}

fn validate_string(value: &str) -> Result<(), &'static str> {
    if value.is_empty() {
        Err("context identifier must not be empty")
    } else if value.len() > MAX_CONTEXT_FIELD_BYTES {
        Err("context identifier exceeds the byte limit")
    } else {
        Ok(())
    }
}

struct BytesVisitor;

impl<'de> Visitor<'de> for BytesVisitor {
    type Value = Vec<u8>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX_LEAF_BYTES} bytes")
    }

    fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.len() > MAX_LEAF_BYTES {
            return Err(E::custom("Merkle leaf exceeds the byte limit"));
        }
        Ok(value.to_vec())
    }

    fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.len() > MAX_LEAF_BYTES {
            return Err(E::custom("Merkle leaf exceeds the byte limit"));
        }
        Ok(value)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let capacity = sequence.size_hint().unwrap_or(0).min(MAX_LEAF_BYTES);
        let mut bytes = Vec::with_capacity(capacity);
        while let Some(byte) = sequence.next_element()? {
            if bytes.len() == MAX_LEAF_BYTES {
                return Err(A::Error::custom("Merkle leaf exceeds the byte limit"));
            }
            bytes.push(byte);
        }
        Ok(bytes)
    }
}

struct DigestVecVisitor;

impl<'de> Visitor<'de> for DigestVecVisitor {
    type Value = Vec<Digest>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX_PROOF_SIBLINGS} proof siblings")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let capacity = sequence.size_hint().unwrap_or(0).min(MAX_PROOF_SIBLINGS);
        let mut siblings = Vec::with_capacity(capacity);
        while let Some(sibling) = sequence.next_element()? {
            if siblings.len() == MAX_PROOF_SIBLINGS {
                return Err(A::Error::custom("Merkle proof has too many siblings"));
            }
            siblings.push(sibling);
        }
        Ok(siblings)
    }
}

struct IndexVecVisitor;

impl<'de> Visitor<'de> for IndexVecVisitor {
    type Value = Vec<u64>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "at most {MAX_CHALLENGE_OPENINGS} challenge indices"
        )
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let capacity = sequence
            .size_hint()
            .unwrap_or(0)
            .min(MAX_CHALLENGE_OPENINGS);
        let mut indices = Vec::with_capacity(capacity);
        while let Some(index) = sequence.next_element()? {
            if indices.len() == MAX_CHALLENGE_OPENINGS {
                return Err(A::Error::custom("challenge contains too many indices"));
            }
            indices.push(index);
        }
        Ok(indices)
    }
}
