"""Secrets management for BLAST."""

from pathlib import Path
from typing import Dict, Optional, Union
from dotenv import dotenv_values

class SecretsManager:
    """Manages sensitive data loaded from secrets file.
    
    Supports both flat and domain-specific secrets:
    - Flat: All secrets available to all domains
    - Domain-specific: Secrets restricted to specific domains
    """
    
    def __init__(self):
        """Initialize secrets manager."""
        self._secrets: Dict[str, Dict[str, str]] = {}  # Domain-specific secrets
        self._flat_secrets: Dict[str, str] = {}        # Non-domain-specific secrets
        
    def load_secrets(self, secrets_file_path: Optional[str] = None) -> None:
        """Load secrets from file if it exists.
        
        Args:
            secrets_file_path: Optional path to secrets file. If not provided,
                              looks for secrets.env in current directory.
        """
        if secrets_file_path:
            path = Path(secrets_file_path)
        else:
            path = Path("secrets.env")
            
        if path.exists():
            # Load secrets directly from file without affecting os.environ
            env_values = dotenv_values(path)
            
            # Process secrets from file
            for key, value in env_values.items():
                if key.startswith('DOMAIN_'):
                    # Extract domain and secret name
                    # Format: DOMAIN_example.com_secretname=value
                    parts = key.split('_', 2)
                    if len(parts) == 3:
                        domain = parts[1]
                        secret_name = parts[2]
                        
                        # Add protocol if missing
                        if not domain.startswith(('http://', 'https://', 'chrome-extension://')):
                            domain = f'https://{domain}'  # Default to https
                        
                        # Store domain-specific secret
                        if domain not in self._secrets:
                            self._secrets[domain] = {}
                        self._secrets[domain][secret_name] = value
                else:
                    # Store as flat secret
                    self._flat_secrets[key] = value
    
    def get_secrets(self) -> Optional[Dict[str, Union[Dict[str, str], str]]]:
        """Get all loaded secrets in browser-use compatible format.
        
        Returns:
            Dictionary with:
            - Domain keys mapping to secret dicts: "https://*.example.com" -> {"username": "value"}
            - Flat secrets: "key" -> "value"
            Returns None if no secrets are loaded.
        """
        # Return None if no secrets
        if not self._flat_secrets and not self._secrets:
            return None
            
        # Start with flat secrets
        secrets = self._flat_secrets.copy()
        
        # Add domain-specific secrets
        for domain, domain_secrets in self._secrets.items():
            # Convert domain to wildcard format if needed
            if domain.count('.') > 1:  # Has subdomain
                base_domain = '.'.join(domain.split('.')[-2:])  # Get example.com
                domain = domain.replace(base_domain, f'*.{base_domain}')
            secrets[domain] = domain_secrets
            
        return secrets
        