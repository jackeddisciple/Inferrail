use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extra, Nullable};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ModelList {
    pub fn new(data: Vec<ModelCard>) -> Self {
        Self {
            object: "list".to_owned(),
            data,
            extra: Extra::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub root: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub parent: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub max_model_len: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub permission: Nullable<Vec<Value>>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_and_vllm_model_lists_parse() {
        let openai = json!({"object": "list", "data": [
            {"id": "gpt-4.1", "object": "model", "created": 1744316542, "owned_by": "system"}
        ]});
        let list: ModelList = serde_json::from_value(openai.clone()).unwrap();
        assert!(list.data[0].max_model_len.is_absent());
        assert_eq!(serde_json::to_value(&list).unwrap(), openai);

        let vllm = json!({"object": "list", "data": [{
            "id": "m", "object": "model", "created": 1, "owned_by": "vllm", "root": "m",
            "parent": null, "max_model_len": 8192, "permission": []
        }]});
        let list: ModelList = serde_json::from_value(vllm.clone()).unwrap();
        assert_eq!(list.data[0].max_model_len, Nullable::Value(8192));
        assert_eq!(serde_json::to_value(&list).unwrap(), vllm);
    }

    #[test]
    fn new_list_is_tagged_as_list() {
        assert_eq!(
            serde_json::to_value(ModelList::new(vec![])).unwrap(),
            json!({"object": "list", "data": []})
        );
    }
}
