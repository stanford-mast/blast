"""
Monkey patches for browser-use to improve error handling and fix JavaScript execution issues.
"""

import json
import logging

logger = logging.getLogger(__name__)


def patch_browser_use_evaluate():
    """
    Patch browser-use's evaluate action to:
    1. Auto-wrap async code (fixes "Uncaught" error with await)
    2. Provide better error messages (extract full exception description from CDP)
    
    Since the installed browser-use has poor error extraction,
    we completely replace the evaluate function with an enhanced version.
    """
    from browser_use.tools.service import Tools
    from browser_use import ActionResult
    import json
    
    # Store original __init__
    original_init = Tools.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call original init (this registers all actions including evaluate)
        original_init(self, *args, **kwargs)
        
        # Now find and patch the evaluate action in the registry
        if hasattr(self.registry, 'registry') and hasattr(self.registry.registry, 'actions'):
            actions = self.registry.registry.actions
            
            if 'evaluate' in actions:
                # Completely replace evaluate with enhanced version
                async def enhanced_evaluate(*, params=None, browser_session, **kwargs):
                    """Enhanced evaluate with async auto-wrapping and detailed error messages"""
                    # Extract code from params
                    code = params.code if params else kwargs.get('code', '')
                    original_code = code  # Keep original for error messages
                    
                    # Auto-wrap all code in async IIFE to handle:
                    # 1. Top-level await expressions (only valid in async functions)
                    # 2. Top-level return statements (only valid in functions)
                    # 3. Ensure consistent execution context
                    #
                    # This is more robust than pattern matching and works for all cases.
                    # The async IIFE works for both sync and async code with negligible overhead.
                    stripped = code.strip()
                    
                    # Skip wrapping if already wrapped in a function/IIFE
                    already_wrapped = (
                        stripped.startswith('(function') or
                        stripped.startswith('function') or
                        stripped.startswith('(async') or
                        stripped.startswith('async ') or
                        stripped.startswith('(() =>')
                    )
                    
                    if not already_wrapped:
                        logger.debug("Auto-wrapping code in async IIFE for safe execution")
                        code = f"(async function() {{ {code} }})()"
                    
                    # Execute JavaScript with CDP
                    cdp_session = await browser_session.get_or_create_cdp_session()
                    
                    try:
                        result = await cdp_session.cdp_client.send.Runtime.evaluate(
                            params={'expression': code, 'returnByValue': True, 'awaitPromise': True},
                            session_id=cdp_session.session_id,
                        )
                        
                        # Enhanced error handling with full description
                        if result.get('exceptionDetails'):
                            exception = result['exceptionDetails']
                            
                            # Extract detailed error information
                            error_text = exception.get('text', 'Unknown error')
                            line_number = exception.get('lineNumber', 'unknown')
                            column_number = exception.get('columnNumber', 'unknown')
                            
                            # Get the full description from the exception object
                            exc_obj = exception.get('exception', {})
                            description = exc_obj.get('description', '')
                            
                            # Build detailed error message
                            if description:
                                # Description includes full error with stack trace
                                error_msg = f'JavaScript execution error: {description}'
                            else:
                                error_msg = f'JavaScript execution error: {error_text} at line {line_number}:{column_number}'
                            
                            msg = f'Code: {original_code}\n\nError: {error_msg}'
                            logger.info(msg)
                            return ActionResult(error=msg)
                        
                        # Get the result data
                        result_data = result.get('result', {})
                        
                        # Check for wasThrown flag
                        if result_data.get('wasThrown'):
                            msg = f'Code: {original_code}\n\nError: JavaScript execution failed (wasThrown=true)'
                            logger.info(msg)
                            return ActionResult(error=msg)
                        
                        # Get the actual value
                        value = result_data.get('value')
                        
                        # Handle different value types
                        if value is None:
                            result_text = str(value) if 'value' in result_data else 'undefined'
                        elif isinstance(value, (dict, list)):
                            try:
                                result_text = json.dumps(value, ensure_ascii=False)
                            except (TypeError, ValueError):
                                result_text = str(value)
                        else:
                            result_text = str(value)
                        
                        logger.info(f'Code: {original_code}\n\nResult: {result_text}')
                        return ActionResult(extracted_content=result_text)
                        
                    except Exception as e:
                        error_msg = f'Code: {original_code}\n\nError: Failed to execute JavaScript: {type(e).__name__}: {e}'
                        logger.info(error_msg)
                        return ActionResult(error=error_msg)
                
                # Replace the function in the registry
                actions['evaluate'].function = enhanced_evaluate
                logger.debug("✅ Patched evaluate action with enhanced error messages")
    
    # Monkey patch __init__
    Tools.__init__ = patched_init
    logger.info("✅ Patched browser-use Tools.__init__ to auto-wrap async code and enhance error messages")


def apply_all_patches():
    """Apply all browser-use patches"""
    patch_browser_use_evaluate()
    logger.info("✅ All browser-use patches applied successfully")

