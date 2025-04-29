"""Cache management for BlastAI."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList

from .utils import get_appdata_dir
from .tools import Tools
from .scheduler import Scheduler
from .config import Constraints

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages both in-memory and on-disk caching of task results and plans."""
    
    def __init__(self, instance_hash: str, persist: bool = True, constraints: Optional[Constraints] = None):
        """Initialize the cache manager.
        
        Args:
            instance_hash: Unique identifier for this engine instance
            persist: Whether to persist cache to disk
        """
        self.instance_hash = instance_hash
        self.persist = persist
        
        # Initialize cache directories
        self.cache_dir = get_appdata_dir() / "cache"
        self.results_dir = self.cache_dir / "results"
        self.plans_dir = self.cache_dir / "plans"
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self._results_cache: Dict[str, AgentHistoryList] = {}
        self._plans_cache: Dict[str, AgentHistoryList] = {}
        
        # Dummy agent for loading history - initialized in load()
        self._dummy_agent: Optional[Agent] = None
        self.constraints = constraints or Constraints()
        
    def load(self, scheduler: Scheduler):
        """Initialize the dummy agent for loading history.
        
        This should be called after scheduler is fully set up.
        
        Args:
            scheduler: Scheduler instance for tools
        """
        if not self._dummy_agent:
            # Create LLM for dummy agent
            llm = ChatOpenAI(model=self.constraints.llm_model_mini or "gpt-4.1-mini")  # Use mini model for cache
            
            # Create fresh Tools instance for dummy agent with LLM
            tools = Tools(scheduler=scheduler, resource_manager=None, llm_model=llm)  # No resource manager needed for cache
            
            # Create dummy agent with tools controller
            self._dummy_agent = Agent(
                task="dummy",
                llm=llm,
                controller=tools.controller
            )
            
    def _load_history_with_output_model(self, cache_file: Path) -> Optional[AgentHistoryList]:
        """Load history from file with proper output model that includes custom actions.
        
        Args:
            cache_file: Path to the cache file to load
            
        Returns:
            Loaded history if successful, None otherwise
        """
        try:
            if not self._dummy_agent:
                raise ValueError("Cache manager not loaded - call load() first")
                
            return AgentHistoryList.load_from_file(cache_file, self._dummy_agent.AgentOutput)
        except Exception as e:
            logger.error(f"Error loading history with output model from {cache_file}: {e}")
            return None
            
    def _parse_cache_control(self, cache_control: str) -> Dict[str, bool]:
        """Parse cache control string into settings.
        
        Args:
            cache_control: Cache control string (e.g. "no-cache,no-cache-plan")
            
        Returns:
            Dict of cache settings
        """
        settings = {
            "cache_results": True,
            "cache_plans": True
        }
        
        if not cache_control:
            return settings
            
        directives = [d.strip() for d in cache_control.split(",")]
        for directive in directives:
            if directive == "no-cache":
                settings["cache_results"] = False
            elif directive == "no-cache-plan":
                settings["cache_plans"] = False
                
        return settings
        
    def _get_cache_key(self, task_lineage: List[str]) -> str:
        """Generate a cache key from task lineage.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
            
        Returns:
            Cache key string
        """
        return "_".join(task_lineage)
        
    def get_result(self, task_lineage: List[str], cache_control: str = "") -> Optional[AgentHistoryList]:
        """Get cached result for a task lineage if available.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
            cache_control: Cache control directives
            
        Returns:
            Cached result if available, None otherwise
        """
        cache_settings = self._parse_cache_control(cache_control)
        if not cache_settings["cache_results"]:
            return None
            
        cache_key = self._get_cache_key(task_lineage)
        
        # Check in-memory cache first
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]
            
        # Check disk cache if persisting
        if self.persist:
            cache_file = self.results_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    return self._load_history_with_output_model(cache_file)
                except Exception as e:
                    logger.error(f"Error loading cache file {cache_file}: {e}")
                    
        return None
        
    def get_plan(self, task_lineage: List[str], cache_control: str = "") -> Optional[AgentHistoryList]:
        """Get cached plan for a task lineage if available.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
            cache_control: Cache control directives
            
        Returns:
            Cached plan if available, None otherwise
        """
        cache_settings = self._parse_cache_control(cache_control)
        if not cache_settings["cache_plans"]:
            return None
            
        cache_key = self._get_cache_key(task_lineage)
        
        # Check in-memory cache first
        if cache_key in self._plans_cache:
            return self._plans_cache[cache_key]
            
        # Check disk cache if persisting
        if self.persist:
            cache_file = self.plans_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    return self._load_history_with_output_model(cache_file)
                except Exception as e:
                    logger.error(f"Error loading cache file {cache_file}: {e}")
                    
        return None
        
    def update_result(self, task_lineage: List[str], result: AgentHistoryList, cache_control: str = ""):
        """Update cached result for a task lineage.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
            result: Result to cache
            cache_control: Cache control directives
        """
        cache_settings = self._parse_cache_control(cache_control)
        if not cache_settings["cache_results"]:
            return
            
        cache_key = self._get_cache_key(task_lineage)
        
        # Update in-memory cache
        self._results_cache[cache_key] = result
        
        # Update disk cache if persisting
        if self.persist:
            cache_file = self.results_dir / f"{cache_key}.json"
            try:
                result.save_to_file(cache_file)
            except Exception as e:
                logger.error(f"Error saving cache file {cache_file}: {e}")
                
    def update_plan(self, task_lineage: List[str], plan: AgentHistoryList, cache_control: str = ""):
        """Update cached plan for a task lineage.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
            plan: Plan to cache
            cache_control: Cache control directives
        """
        cache_settings = self._parse_cache_control(cache_control)
        if not cache_settings["cache_plans"]:
            return
            
        cache_key = self._get_cache_key(task_lineage)
        
        # Update in-memory cache
        self._plans_cache[cache_key] = plan
        
        # Update disk cache if persisting
        if self.persist:
            cache_file = self.plans_dir / f"{cache_key}.json"
            try:
                plan.save_to_file(cache_file)
            except Exception as e:
                logger.error(f"Error saving cache file {cache_file}: {e}")
                
    def remove_task(self, task_lineage: List[str]):
        """Remove all cached data for a task lineage.
        
        Args:
            task_lineage: List of task descriptions representing task ancestry
        """
        cache_key = self._get_cache_key(task_lineage)
        
        # Remove from in-memory caches
        self._results_cache.pop(cache_key, None)
        self._plans_cache.pop(cache_key, None)
        
        # Remove from disk if persisting
        if self.persist:
            results_file = self.results_dir / f"{cache_key}.json"
            plans_file = self.plans_dir / f"{cache_key}.json"
            
            if results_file.exists():
                results_file.unlink()
            if plans_file.exists():
                plans_file.unlink()
                
    def clear(self):
        """Clear all caches."""
        # Clear in-memory caches
        self._results_cache.clear()
        self._plans_cache.clear()
        
        # Clear disk caches if persisting
        if self.persist:
            shutil.rmtree(self.results_dir)
            shutil.rmtree(self.plans_dir)
            self.results_dir.mkdir(parents=True)
            self.plans_dir.mkdir(parents=True)