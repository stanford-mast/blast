"""Secrets management for BLAST."""

import os
from pathlib import Path
from typing import Dict, Optional, Union
from dotenv import load_dotenv

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
            # Load secrets using python-dotenv
            load_dotenv(path)
            
            # Process all environment variables
            for key, value in os.environ.items():
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
    
    def get_secrets(self) -> Dict[str, Dict[str, str]]:
        """Get all loaded secrets.
        
        Returns:
            Dictionary mapping domains to their secrets, with flat secrets under None key
        """
        secrets = self._secrets.copy()
        if self._flat_secrets:
            secrets[None] = self._flat_secrets.copy()  # Store flat secrets under None key
        return secrets
        