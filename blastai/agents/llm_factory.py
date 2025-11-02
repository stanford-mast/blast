"""
LLM factory for creating BaseChatModel instances from model names.

This module provides a centralized way to instantiate LLM models across
different providers (OpenAI, Anthropic, Google, Groq, etc.) based on
model name patterns or explicit provider specifications.
"""

import os
from typing import Optional
from browser_use.llm.base import BaseChatModel


class LLMFactory:
    """
    Factory for creating BaseChatModel instances from model names.
    
    Supports:
    - OpenAI (gpt-4o, gpt-4.1, etc.)
    - Anthropic (claude-3-5-sonnet, etc.)
    - Google (gemini-2.0-flash, etc.)
    - Groq (meta-llama/*, qwen/*, etc.)
    
    Usage:
        llm = LLMFactory.create_llm("gpt-4o", temperature=0.5)
        llm = LLMFactory.create_llm("meta-llama/llama-4-maverick-17b-128e-instruct")
        llm = LLMFactory.create_llm("claude-3-5-sonnet-20241022")
    """
    
    @staticmethod
    def detect_provider(model_name: str) -> str:
        """
        Detect the provider from the model name.
        
        Args:
            model_name: The model name or identifier
            
        Returns:
            Provider name: 'openai', 'anthropic', 'google', 'groq'
            
        Raises:
            ValueError: If provider cannot be detected
        """
        model_lower = model_name.lower()
        
        # OpenAI models
        if model_lower.startswith('gpt-') or model_lower.startswith('o1-') or model_lower.startswith('o3-'):
            return 'openai'
        
        # Anthropic models
        if model_lower.startswith('claude-'):
            return 'anthropic'
        
        # Google models
        if model_lower.startswith('gemini-') or model_lower.startswith('gemma-'):
            return 'google'
        
        # Groq models (use namespace prefixes)
        if '/' in model_name:
            namespace = model_name.split('/')[0].lower()
            if namespace in ['meta-llama', 'qwen', 'moonshotai', 'openai', 'google']:
                # These namespaces are typically used by Groq
                return 'groq'
        
        # Default to OpenAI for unknown models (allows custom endpoints)
        return 'openai'
    
    @staticmethod
    def create_llm(
        model_name: str,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance from a model name.
        
        Args:
            model_name: The model name/identifier
            provider: Optional explicit provider ('openai', 'anthropic', 'google', 'groq')
                     If not provided, will auto-detect from model_name
            temperature: Temperature parameter for the model
            api_key: API key for the provider (falls back to env vars)
            base_url: Base URL for the API (for custom endpoints)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            BaseChatModel instance configured for the specified model
            
        Raises:
            ValueError: If provider is unsupported or invalid
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = LLMFactory.detect_provider(model_name)
        
        provider = provider.lower()
        
        # Create appropriate LLM instance based on provider
        if provider == 'openai':
            from browser_use.llm.openai.chat import ChatOpenAI
            
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                **kwargs
            )
        
        elif provider == 'anthropic':
            from browser_use.llm.anthropic.chat import ChatAnthropic
            
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            return ChatAnthropic(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                **kwargs
            )
        
        elif provider == 'google':
            from browser_use.llm.google.chat import ChatGoogle
            
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            
            # Google has different parameter structure
            config_kwargs = {}
            if temperature is not None:
                config_kwargs['temperature'] = temperature
            config_kwargs.update(kwargs)
            
            return ChatGoogle(
                model=model_name,
                api_key=api_key,
                **config_kwargs
            )
        
        elif provider == 'groq':
            from browser_use.llm.groq.chat import ChatGroq
            
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.getenv('GROQ_API_KEY')
            
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: openai, anthropic, google, groq"
            )
    
    @staticmethod
    def create_llm_from_env(
        env_var_prefix: str = "BLASTAI",
        default_model: str = "gpt-4.1"
    ) -> BaseChatModel:
        """
        Create LLM from environment variables.
        
        Looks for:
        - {env_var_prefix}_MODEL: Model name
        - {env_var_prefix}_PROVIDER: Explicit provider (optional)
        - {env_var_prefix}_TEMPERATURE: Temperature (optional)
        - Provider-specific API keys (OPENAI_API_KEY, etc.)
        
        Args:
            env_var_prefix: Prefix for environment variables
            default_model: Default model if not specified in env
            
        Returns:
            BaseChatModel instance
        """
        model_name = os.getenv(f"{env_var_prefix}_MODEL", default_model)
        provider = os.getenv(f"{env_var_prefix}_PROVIDER")  # Optional
        temperature_str = os.getenv(f"{env_var_prefix}_TEMPERATURE")
        
        temperature = None
        if temperature_str:
            try:
                temperature = float(temperature_str)
            except ValueError:
                pass
        
        return LLMFactory.create_llm(
            model_name=model_name,
            provider=provider,
            temperature=temperature
        )
