"""Model-related utilities for BLAST."""

def is_openai_model(model_name: str) -> bool:
    """Check if a model name is from OpenAI.
    
    Args:
        model_name: Name of the model (with optional provider prefix)
        
    Returns:
        True if model is from OpenAI, False otherwise
    """
    model_name = model_name.lower()
    
    # Handle provider prefix (e.g., "openai:gpt-4")
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        if provider == "openai":
            return True
        return False
        
    # Check for OpenAI model patterns
    return any(prefix in model_name for prefix in ['gpt-3', 'gpt-4', 'o1', 'o1-mini', 'o3', 'o3-mini', 'o4', 'o4-mini'])