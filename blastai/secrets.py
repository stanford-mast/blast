"""Secrets management for BLAST."""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

class SecretsManager:
    """Manages sensitive data loaded from secrets file."""
    
    def __init__(self):
        """Initialize secrets manager."""
        self._secrets: Dict[str, str] = {}
        
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
            
            # Store secrets in memory
            self._secrets = {
                key: value for key, value in os.environ.items()
            }
    
    def get_secrets(self) -> Dict[str, str]:
        """Get all loaded secrets.
        
        Returns:
            Dictionary of secret key-value pairs
        """
        return self._secrets.copy()  # Return copy to prevent modification