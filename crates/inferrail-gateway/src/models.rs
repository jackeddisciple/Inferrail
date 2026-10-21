use hyper::Response;
use hyper::header::{self, HeaderValue};
use inferrail_core::openai::{ModelCard, ModelList, Nullable};

use crate::Shared;
use crate::body::ResponseBody;

fn card(id: &str, root: &str, created: u64) -> ModelCard {
    ModelCard {
        id: id.to_owned(),
        object: "model".to_owned(),
        created,
        owned_by: "inferrail".to_owned(),
        root: Nullable::Value(root.to_owned()),
        parent: Nullable::Null,
        max_model_len: Nullable::Absent,
        permission: Nullable::Absent,
        extra: Default::default(),
    }
}

pub(crate) fn list(shared: &Shared) -> Response<ResponseBody> {
    let created = shared.started_unix;
    let mut cards = Vec::new();
    for model in &shared.config.models {
        let name = model.name.as_str();
        cards.push(card(name, name, created));
        for alias in &model.aliases {
            cards.push(card(alias, name, created));
        }
    }
    let body = serde_json::to_vec(&ModelList::new(cards)).unwrap_or_default();
    let mut response = Response::new(ResponseBody::full(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}
