use hyper::Response;
use hyper::header::{self, HeaderValue};
use inferrail_core::openai::{ModelCard, ModelList, Nullable};

use crate::Shared;
use crate::body::ResponseBody;

fn synthesized(id: &str, created: u64) -> ModelCard {
    ModelCard {
        id: id.to_owned(),
        object: "model".to_owned(),
        created,
        owned_by: "inferrail".to_owned(),
        root: Nullable::Value(id.to_owned()),
        parent: Nullable::Null,
        max_model_len: Nullable::Absent,
        permission: Nullable::Absent,
        extra: Default::default(),
    }
}

/// Every configured model and alias in config order, described by the card
/// an engine published for it (routable replicas first) when there is one.
/// Served from memory; listing never calls an engine.
pub(crate) fn list(shared: &Shared) -> Response<ResponseBody> {
    let fleet = shared.registry.load();
    let now = shared.clock.now();
    let mut cards = Vec::new();
    for model in &shared.config.models {
        let name = model.name.as_str();
        let mut serving: Vec<_> = fleet.serving(&model.name).iter().collect();
        serving.sort_by_key(|r| !r.is_routable(now));
        let card = serving
            .iter()
            .filter_map(|r| r.cards())
            .find_map(|known| known.cards.iter().find(|c| c.id == name).cloned())
            .unwrap_or_else(|| synthesized(name, shared.started_unix));
        for alias in &model.aliases {
            cards.push(ModelCard {
                id: alias.clone(),
                ..card.clone()
            });
        }
        cards.insert(cards.len() - model.aliases.len(), card);
    }
    let body = serde_json::to_vec(&ModelList::new(cards)).unwrap_or_default();
    let mut response = Response::new(ResponseBody::full(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}
