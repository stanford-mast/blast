"""Patch for BrowserSession to simplify tab following functionality."""

import logging
from functools import wraps

# Don't import BrowserSession yet
logger = logging.getLogger(__name__)

def apply_all_patches():
    """Apply all browser session patches."""
    # Import BrowserSession only when needed
    from browser_use.browser import BrowserSession
    
    patch_take_screenshot(BrowserSession)
    logger.info("BrowserSession patches applied")

def patch_take_screenshot(BrowserSession):
    """
    Patch take_screenshot to handle timeouts better while preserving visual quality.
    """
    original_take_screenshot = BrowserSession.take_screenshot
    
    @wraps(original_take_screenshot)
    async def patched_take_screenshot(self, full_page: bool = False) -> str:
        """
        Returns a base64 encoded screenshot of the current page.
        Uses original high-quality screenshot code with better timeout handling.
        """
        import asyncio
        import base64
        
        assert self.agent_current_page is not None, 'Agent current page is not set'
        page = await self.get_current_page()

        try:
            # Wait for page to stabilize, but don't block on it
            try:
                await page.wait_for_load_state(timeout=5000)
            except Exception:
                pass

            # 0. Attempt full-page screenshot first
            if full_page:
                try:
                    screenshot = await asyncio.wait_for(
                        page.screenshot(
                            full_page=True,
                            scale='css',
                            timeout=10000,
                            animations='allow',
                            caret='initial',
                        ),
                        timeout=15000,
                    )

                    screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                    return screenshot_b64
                except Exception as e:
                    logger.warning(
                        f'⚠️ Failed to take full-page screenshot after 10s: {type(e).__name__}: {e} trying with height limit instead...'
                    )

            # Fallback: manually expand viewport and take viewport screenshot

            # 1. Get current page dimensions
            dimensions = await page.evaluate("""() => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio || 1
                };
            }""")

            # 2. Save current viewport state and calculate expanded dimensions
            original_viewport = page.viewport_size
            viewport_expansion = self.browser_profile.viewport_expansion if self.browser_profile.viewport_expansion else 0

            expanded_width = dimensions['width']  # Keep width unchanged
            expanded_height = dimensions['height'] + viewport_expansion

            try:
                # 3. Expand the viewport if we are using one
                if original_viewport:
                    await asyncio.wait_for(
                        page.set_viewport_size({'width': expanded_width, 'height': expanded_height}),
                        timeout=2000
                    )

                # 4. Take full-viewport screenshot with original high-quality settings
                screenshot = await asyncio.wait_for(
                    page.screenshot(
                        full_page=False,
                        scale='css',
                        timeout=10000,
                        clip={'x': 0, 'y': 0, 'width': expanded_width, 'height': expanded_height},
                        animations='allow',
                        caret='initial',
                    ),
                    timeout=15000,
                )

                screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                return screenshot_b64

            except Exception as e:
                logger.error(f'❌ Failed to take screenshot: {type(e).__name__}: {e}')
                # Return a minimal 1x1 transparent PNG if screenshot fails
                return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

            finally:
                # 5. Always restore original viewport state if we expanded it
                if original_viewport:
                    try:
                        await asyncio.wait_for(
                            page.set_viewport_size(original_viewport),
                            timeout=2000
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore viewport (continuing anyway): {e}")

        except Exception as e:
            logger.error(f"Screenshot failed: {type(e).__name__}: {e}")
            # Return a minimal 1x1 transparent PNG if screenshot fails
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # Replace the method
    BrowserSession.take_screenshot = patched_take_screenshot
    return patched_take_screenshot