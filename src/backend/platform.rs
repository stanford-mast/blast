//! Worker execution compatibility classes.

const SNAPSHOT_ABI: &str = "v1";
const PLATFORM_LIMIT: usize = 32;
const CLASSED_PLATFORM_PREFIX: &str = "blast-v1-";

/// Returns the backend, host, and operator compatibility class.
pub fn worker_platform(
    platform_suffix: Option<&str>,
    backend: &crate::config::BackendConfig,
    host_identity: &str,
) -> String {
    let backend_abi = match backend {
        crate::config::BackendConfig::Smolvm { .. } => {
            format!(
                "smolvm-{}",
                crate::backend::bootstrap::SMOLVM_RELEASE_VERSION
            )
        }
        crate::config::BackendConfig::Hypeman { endpoint, .. } => {
            // `Hypeman` snapshots are bound to their hypervisor endpoint.
            use sha2::{Digest, Sha256};
            let affinity = hex::encode(Sha256::digest(
                format!("{host_identity}\0{endpoint}").as_bytes(),
            ));
            format!("hypeman-{}", &affinity[..16])
        }
        crate::config::BackendConfig::Docker => "docker".to_owned(),
    };
    let base = format!(
        "blast-{SNAPSHOT_ABI}-{}-{}-{}",
        normalized(&backend_abi),
        normalized(std::env::consts::OS),
        normalized_arch(std::env::consts::ARCH)
    );
    match platform_suffix
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(suffix) => classed_platform(&base, &normalized(suffix)),
        None => base,
    }
}

fn classed_platform(base: &str, operator_class: &str) -> String {
    use sha2::{Digest, Sha256};

    let digest = hex::encode(Sha256::digest(
        format!("{base}\0{operator_class}").as_bytes(),
    ));
    let digest_len = PLATFORM_LIMIT - CLASSED_PLATFORM_PREFIX.len();
    format!("{CLASSED_PLATFORM_PREFIX}{}", &digest[..digest_len])
}

fn normalized_arch(isa: &str) -> String {
    match isa {
        "x86_64" => "x86".to_string(),
        "aarch64" => "arm".to_string(),
        other => normalized(other),
    }
}

fn normalized(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_is_an_execution_compatibility_boundary() {
        assert_eq!(normalized_arch("x86_64"), "x86");
        assert_eq!(normalized_arch("aarch64"), "arm");
        let smolvm = crate::config::BackendConfig::Smolvm {
            binary: "smolvm".into(),
        };
        let base = worker_platform(None, &smolvm, "test-host");
        assert!(base.contains("-smolvm-"), "{base}");
        assert!(base.contains(std::env::consts::OS), "{base}");
        assert_eq!(worker_platform(Some("  "), &smolvm, "test-host"), base);
        let class_a = worker_platform(Some(" Ice Lake / A "), &smolvm, "test-host");
        assert_eq!(class_a.len(), PLATFORM_LIMIT);
        assert_eq!(
            class_a,
            worker_platform(Some("ice lake / a"), &smolvm, "test-host")
        );
        assert_ne!(
            class_a,
            worker_platform(Some("ice-lake-b"), &smolvm, "test-host")
        );
        assert_ne!(
            worker_platform(None, &crate::config::BackendConfig::Docker, "test-host"),
            base
        );
        assert_ne!(
            class_a,
            worker_platform(
                Some("Ice Lake / A"),
                &crate::config::BackendConfig::Docker,
                "test-host"
            )
        );
    }

    #[test]
    fn classed_platform_stays_bounded_for_arbitrarily_long_classes() {
        let smolvm = crate::config::BackendConfig::Smolvm {
            binary: "smolvm".into(),
        };
        let platform = worker_platform(Some(&"x".repeat(10_000)), &smolvm, "test-host");
        assert_eq!(platform.len(), PLATFORM_LIMIT);
        assert!(platform.starts_with(CLASSED_PLATFORM_PREFIX));
    }
}
