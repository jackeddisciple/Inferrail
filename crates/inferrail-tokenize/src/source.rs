//! Where a model's tokenizer comes from: a directory of model files, a
//! repository already in the local Hugging Face cache, or the simulator's
//! built in tokenizer. Nothing is downloaded, so startup never waits on the
//! network and always uses the files an operator put in place.

use std::path::{Path, PathBuf};

use crate::TokenizeError;

pub const SIM: &str = "builtin:sim";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Sim,
    Dir(PathBuf),
}

/// Resolves `spec` against the file system first, then the cache.
pub fn resolve(spec: &str) -> Result<Source, TokenizeError> {
    if spec == SIM {
        return Ok(Source::Sim);
    }
    let path = Path::new(spec);
    if path.is_dir() {
        return Ok(Source::Dir(path.to_owned()));
    }
    if path.is_file() {
        return match (path.file_name().and_then(|n| n.to_str()), path.parent()) {
            (Some("tokenizer.json"), Some(dir)) => Ok(Source::Dir(dir.to_owned())),
            _ => Err(TokenizeError::NotFound(format!(
                "{spec} is a file; point at a model directory or its tokenizer.json"
            ))),
        };
    }
    let (repo, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    if !is_repo_id(repo) || revision.is_empty() {
        return Err(TokenizeError::NotFound(format!(
            "tokenizer {spec:?} is neither a directory nor a Hugging Face repository id"
        )));
    }
    let root = cache_root().ok_or_else(|| {
        TokenizeError::NotFound("no Hugging Face cache directory; set HF_HUB_CACHE".to_owned())
    })?;
    cached(&root, repo, revision)
        .map(Source::Dir)
        .ok_or_else(|| {
            TokenizeError::NotFound(format!(
                "{repo} at {revision} is not in the Hugging Face cache at {}; fetch it with \
                 `hf download {repo} tokenizer.json tokenizer_config.json` or point at a directory",
                root.display()
            ))
        })
}

fn is_repo_id(repo: &str) -> bool {
    let parts: Vec<&str> = repo.split('/').collect();
    (1..=2).contains(&parts.len())
        && parts.iter().all(|p| {
            !p.is_empty()
                && !p.starts_with('.')
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

fn cache_root() -> Option<PathBuf> {
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    var("HF_HUB_CACHE")
        .or_else(|| var("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| var("XDG_CACHE_HOME").map(|c| c.join("huggingface").join("hub")))
        .or_else(|| var("HOME").map(|h| h.join(".cache").join("huggingface").join("hub")))
}

/// `models--org--name/snapshots/<commit>`, where a branch or tag name is
/// looked up in `refs/` and a full commit hash is used as is.
fn cached(root: &Path, repo: &str, revision: &str) -> Option<PathBuf> {
    let base = root.join(format!("models--{}", repo.replace('/', "--")));
    let is_commit = revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit());
    let commit = if is_commit {
        revision.to_owned()
    } else {
        if revision.contains(['/', '\\']) || revision.starts_with('.') {
            return None;
        }
        std::fs::read_to_string(base.join("refs").join(revision))
            .ok()?
            .trim()
            .to_owned()
    };
    if commit.is_empty() || commit.contains(['/', '\\']) || commit.starts_with('.') {
        return None;
    }
    let dir = base.join("snapshots").join(commit);
    dir.join("tokenizer.json").is_file().then_some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("inferrail-src-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn directories_files_and_the_builtin_resolve() {
        let dir = scratch("dir");
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        assert_eq!(resolve(SIM).unwrap(), Source::Sim);
        assert_eq!(
            resolve(dir.to_str().unwrap()).unwrap(),
            Source::Dir(dir.clone())
        );
        let file = dir.join("tokenizer.json");
        assert_eq!(
            resolve(file.to_str().unwrap()).unwrap(),
            Source::Dir(dir.clone())
        );
        let other = dir.join("vocab.txt");
        std::fs::write(&other, "").unwrap();
        assert!(resolve(other.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cached_repositories_resolve_by_branch_or_commit() {
        let root = scratch("cache");
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let base = root.join("models--Org--Model-8B");
        std::fs::create_dir_all(base.join("refs")).unwrap();
        std::fs::create_dir_all(base.join("snapshots").join(commit)).unwrap();
        std::fs::write(base.join("refs").join("main"), format!("{commit}\n")).unwrap();
        std::fs::write(
            base.join("snapshots").join(commit).join("tokenizer.json"),
            "{}",
        )
        .unwrap();
        let snapshot = base.join("snapshots").join(commit);
        assert_eq!(
            cached(&root, "Org/Model-8B", "main"),
            Some(snapshot.clone())
        );
        assert_eq!(cached(&root, "Org/Model-8B", commit), Some(snapshot));
        assert_eq!(cached(&root, "Org/Model-8B", "dev"), None);
        assert_eq!(cached(&root, "Org/Other", "main"), None);
        assert_eq!(cached(&root, "Org/Model-8B", "../main"), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn repository_ids_are_checked() {
        for good in ["Qwen/Qwen3-8B", "gpt2", "org/model.v1_x"] {
            assert!(is_repo_id(good), "{good}");
        }
        for bad in ["", "a/b/c", "../x", "a/", "org/mo del", "/abs"] {
            assert!(!is_repo_id(bad), "{bad}");
        }
        let err = resolve("not a repo id!").unwrap_err();
        assert!(err.to_string().contains("neither"), "{err}");
    }
}
