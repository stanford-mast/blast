"""Task planning for BLAST."""

import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Constraints

logger = logging.getLogger(__name__)

class Planner:
    """Plans task execution with subtask management."""
    
    def __init__(self, constraints: Optional[Constraints] = None):
        """Initialize planner.
        
        Args:
            constraints: Optional constraints for LLM model selection
        """
        self.constraints = constraints or Constraints()
        self.llm = ChatOpenAI(model=self.constraints.llm_model)
        
        # System prompt for planning
        self.system_prompt = """You are a task planner that generates concise 1–2 sentence plans for web browser tasks.  
Focus only on whether, how, and when to launch_subtask and get_subtask_results.  
Use pseudocode to indicate dependencies and parallelism opportunities. For example:  
first find top 10 phones of 2025 --> launch_subtask(check price of $phone on Amazon, <referral URL if provided>) for $phone in <list of phones> --> get_subtask_results(<list of phones>) to compare and summarize.  
Indicate with angle brackets when a value needs to be filled in at execution time.  
Explain reasoning with inline comments, for example: /* because 'top 10' web pages typically have referral URLs */  
Parallelize across unique web pages or websites, but do not parallelize within the same page or website. For example:  
get Kyrie’s stats this season /* not parallelizing because all the stats are likely on the same web page */  
Do not hallucinate additional levels of detail or make assumptions about how the task can be decomposed.  
Launch a subtask only if it is doing some work in parallel with the main task or another task. Otherwise, use the main task.

Examples:  
Example 1:  
Task: Search for 'vegan dessert recipes' on Bing and click the second result  
Plan: Search for 'vegan dessert recipes' on Bing --> click the second result /* fully sequential */  

Example 2:  
Task: What are people asking about under the tags 'python', 'docker', and 'graphql' on StackOverflow?  
Plan: launch_subtask(get top questions from <URL of StackOverflow tag $tag>) for $tag in [python, docker, graphql] /* each tag page is independent */ --> get_subtask_results of all tags to summarize the most common questions  

Example 3:  
Task: Find the 5 highest-grossing films of 2024, then tell me who directed each one  
Plan: first find top 5 highest-grossing films of 2024 --> launch_subtask(find director of $film, <URL of IMDb page for $film>) for $film in <list of 5 films> /* parallelize per film page */ --> get_subtask_results to present each director  

Example 4:  
Task: Search for the top 4 indie coffee shops in Seattle, open each one’s Yelp page, and collect their review counts  
Plan: first search for top 4 indie coffee shops in Seattle --> launch_subtask(open Yelp page for $shop, with initial query for $shop) for $shop in <list of 4 shops> /* parallelize because each Yelp page is separate */ --> get_subtask_results to extract and summarize review counts  

Example 5:  
Task: Lionel Messi: tell me his goals, assists, minutes played, yellow cards, and red cards in the last 5 matches  
Plan: search for Lionel Messi's match stats for his last 5 matches --> get goals, assists, minutes, yellow cards, and red cards /* not parallelizing because each match's box-score is a single page extract */  

Example 6:  
Task: On GitLab: search for trending projects in Ruby, PHP, and C++  
Plan: launch_subtask(search for trending projects in $language on GitLab) for $language in [Ruby, PHP, C++] /* each language query is a unique page */ --> get_subtask_results to summarize the top repos per language  

Example 7:  
Task: Research the latest breakthroughs in renewable energy  
Plan: search for latest breakthroughs in renewable energy --> launch_subtask(summarize paper titled '$title' on <initial search URL>) for $title in <list of papers> /* each paper requires its own page visit */ --> get_subtask_results to aggregate key findings  

Example 8:  
Task: Compare the newest MacBook Air, Dell XPS, and HP Spectre laptop models  
Plan: launch_subtask(evaluate the latest $laptop model specs, initial search query for $laptop) for $laptop in [MacBook Air, Dell XPS, HP Spectre] /* separate spec pages per model */ --> get_subtask_results to compare the three models  

Example 9:  
Task: Compare the last 3 Marvel movies, then check their ticket prices on Fandango and AtomTickets  
Plan: find last 3 Marvel movies --> get_subtask_results(check ticket price of $movie on $site) for $movie in <list of 3 movies> for $site in [Fandango, AtomTickets] /* each ticket-site lookup is separate */ --> get_subtask_results to compare pricing  

Example 10:  
Task: Find the top 8 biotech companies and for each CEO find the undergraduate university they attended  
Plan: search for list of top 8 biotech companies --> get list of CEOs from that list /* CEO names usually listed alongside company details */ --> launch_subtask(find undergraduate university of $CEO, initial search for $CEO) for $CEO in <list of 8 CEOs> /* parallelize each CEO lookup */ --> get_subtask_results to summarize each CEO's alma mater"""
        
    async def plan(self, task_description: str, subtask_depth: int = 0) -> str:
        """Generate a plan for task execution.
        
        Args:
            task_description: Description of the task to plan
            subtask_depth: Current depth in subtask tree (0 for root tasks)
            
        Returns:
            Brief plan focusing on subtask management
        """
        # Check if we're at max depth
        if subtask_depth >= self.constraints.max_parallelism_nesting_depth:
            guidance = "Do not launch subtasks"
            return f"{task_description}\n{guidance}"

        # Build guidance based on allowed parallelism types
        parallelism = self.constraints.allow_parallelism
        guidance_parts = []

        # First check first_of_n
        if parallelism.get("first_of_n", False) and subtask_depth < self.constraints.max_parallelism_nesting_depth:
            guidance_parts.append(
                f"Execute launch_subtask(task=\"{task_description}\", num_copies=3) --> then get_first_subtask_result with the returned subtask IDs"
            )

        # Add task parallelism if allowed
        if parallelism.get("task", False):
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Generate a plan for this task: {task_description}")
            ]
            response = await self.llm.ainvoke(messages)
            plan = response.content.replace('Plan:', '').strip()
            if len(plan.split('\n')) > 2:
                plan = ' '.join(plan.split('\n')[:2])
            guidance_parts.append('Execute: ' + plan)

        # Add data parallelism if allowed
        if parallelism.get("data", False):
            guidance_parts.append(
                "Use extract_content_fast instead of extract_content."
            )

        # Combine guidance or use default
        if guidance_parts:
            guidance = "\n\n".join(guidance_parts)
        else:
            guidance = "Do not launch subtasks."

        return f"{task_description}\n\n{guidance} Do not tell the user about these implementation details."