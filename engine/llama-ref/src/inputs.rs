//! Reading token-id files. A file is JSON in one of these shapes:
//!
//! * `[1, 2, 3]`, a bare array;
//! * `{"ids": [...]}`, as the wikitext fixtures;
//! * `{"prompts": [{"ids": [...]}, ...]}`, as the GSM8K fixtures;
//! * `{"agents": [{"ids": [...]}, ...]}`, as `results/prompts/<pair>/<preset>.json`.

use crate::{Error, Result};
use serde_json::Value;
use std::path::Path;

fn ids_of(v: &Value) -> Result<Vec<i32>> {
    let a = v.as_array().ok_or_else(|| Error("ids is not an array".into()))?;
    a.iter()
        .map(|x| x.as_i64().filter(|n| i32::try_from(*n).is_ok()).map(|n| n as i32).ok_or_else(|| Error(format!("bad token id {x}"))))
        .collect()
}

/// Every id list in the file, in order (one for the `ids` / bare array shapes).
pub fn parse_id_lists(v: &Value) -> Result<Vec<Vec<i32>>> {
    if v.is_array() {
        return Ok(vec![ids_of(v)?]);
    }
    if let Some(ids) = v.get("ids") {
        return Ok(vec![ids_of(ids)?]);
    }
    for key in ["prompts", "agents"] {
        if let Some(list) = v.get(key).and_then(Value::as_array) {
            return list.iter().map(|p| ids_of(p.get("ids").ok_or_else(|| Error(format!("{key} entry without ids")))?)).collect();
        }
    }
    Err(Error("no ids / prompts / agents in the file".into()))
}

pub fn read_id_lists(path: &Path) -> Result<Vec<Vec<i32>>> {
    let s = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let v: Value = serde_json::from_str(&s).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    parse_id_lists(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let p = |s: &str| parse_id_lists(&serde_json::from_str(s).unwrap()).unwrap();
        assert_eq!(p("[1,2]"), vec![vec![1, 2]]);
        assert_eq!(p(r#"{"ids":[3],"x":1}"#), vec![vec![3]]);
        assert_eq!(p(r#"{"prompts":[{"ids":[1]},{"ids":[2,3]}]}"#), vec![vec![1], vec![2, 3]]);
        assert_eq!(p(r#"{"agents":[{"ids":[9],"text":"x"}]}"#), vec![vec![9]]);
        assert!(parse_id_lists(&serde_json::from_str("{}").unwrap()).is_err());
        assert!(parse_id_lists(&serde_json::from_str("[1.5]").unwrap()).is_err());
    }
}
