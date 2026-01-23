"""Task planning for BLAST."""

import logging
import re
from typing import List, Optional, Tuple

from browser_use.llm.messages import SystemMessage, UserMessage

from .config import Constraints
from .utils import init_model

logger = logging.getLogger(__name__)


class Planner:
    """Plans task execution with subtask management."""

    def __init__(self, constraints: Optional[Constraints] = None):
        """Initialize planner.

        Args:
            constraints: Optional constraints for LLM model selection
        """
        self.constraints = constraints or Constraints()
        self.llm = init_model(self.constraints.llm_model)

        # System prompt for planning
        self.system_prompt = """You are a task planner that generates concise 1–2 sentence plans for web browser tasks.  
Focus only on whether, how, and when to launch_subtask and get_subtask_results.  
Use pseudocode to indicate dependencies and parallelism opportunities. For example:  
first find top 10 phones of 2025 --> launch_subtask(check price of $phone on Amazon, <referral URL if provided>) for $phone in <list of phones> --> get_subtask_results(<list of phones>) to compare and summarize.  
Indicate with angle brackets when a value needs to be filled in at execution time.  
Explain reasoning with inline comments, for example: /* because 'top 10' web pages typically have referral URLs */  
Parallelize across unique web pages or websites. For example:  
get Kyrie’s stats this season /* not parallelizing because all the stats are likely on the same web page */  
Do not hallucinate additional levels of detail or make assumptions about how the task can be decomposed.  

**Note**: Only in the case where there are a high number of total items (e.g., 20 or more) to process (e.g., a list of products, emails, or search results), group items and launch a small number of subtasks (e.g., 4-5 subtasks each handling a batch of 10 items). For example, if there are 50 items to process, launch 5 subtasks each handling 10 items, NOT 50 subtasks for 50 items.
Do not batch items if the total number of items is less than 20, you should still launch a subtask per item. Do not worry about sessions.

**Important**: Do not use randomly generated order IDs. Use a single product/entity/item name or dates etc that do not shift to help the subtask uniquely identify the item.
e.g., for $order in <date1, date2, date3>
Clearly tell the subtask which tab on the page it should navigate to first, if the url does not directly take it there.

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
Plan: search for list of top 8 biotech companies --> get list of CEOs from that list /* CEO names usually listed alongside company details */ --> launch_subtask(find undergraduate university of $CEO, initial search for $CEO) for $CEO in <list of 8 CEOs> /* parallelize each CEO lookup */ --> get_subtask_results to summarize each CEO's alma mater

Example 11:  
Task: Delete all 50 visible emails in the inbox  
Plan: extract list of 50 email identifiers from inbox --> launch_subtask(delete emails $batch, starting at current page URL) for $batch in [emails 1-10, emails 11-20, emails 21-30, emails 31-40, emails 41-50] /* batch into 5 subtasks of 10 emails each, NOT 50 subtasks */ --> get_subtask_results to confirm all deleted

Example 12:  
Task: Get the price of every product on this search results page, then find the cheapest option on Amazon for each  
Plan: extract list of products and prices from this page /* all on same page */ --> launch_subtask(find cheapest price for $product on Amazon) for $product in <list of products> /* parallelize Amazon lookups since each is a separate site */ --> get_subtask_results to compare

Example 13:
Task: Create five new tickets on this project management board
Plan: launch_subtask(create ticket for $project, initial search for $project) for $project in <list of 5 projects> /* parallelize each project creation */ --> get_subtask_results to confirm all created
"""

    async def _generate_context_summary(
        self, previous_tasks: List[Tuple[str, str]]
    ) -> Optional[str]:
        """Generate a summary of previous tasks.

        Args:
            previous_tasks: List of tuples (task_description, final_response) from previous tasks

        Returns:
            A 1-2 sentence summary starting with "The previous task was to", or None if generation fails
        """
        if not previous_tasks:
            return None

        # Take the last 3 tasks at most
        recent_tasks = previous_tasks[-3:]

        # Create a prompt for the LLM to generate a summary
        summary_prompt = "Generate a 1-2 sentence summary of the previous task that starts with 'The previous task was to', including the most recent results or any follow-up questions or clarifications. Here are the details:\n\n"

        for i, (prev_task, prev_result) in enumerate(recent_tasks):
            summary_prompt += f"Task {i + 1}: {prev_task}\n"
            summary_prompt += f"Result {i + 1}: {prev_result}\n\n"

        # Get summary from LLM
        messages = [
            SystemMessage(
                content="You are a helpful assistant that summarizes previous tasks concisely."
            ),
            UserMessage(content=summary_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            summary = response.completion.strip()

            # Extract the part starting with "The previous task was to"
            match = re.search(r"The previous task was to.*", summary, re.DOTALL)
            if match:
                return match.group(0)
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate previous task summary: {e}")
            return None

    async def _generate_tool_use_annotations(
        self, context_task_description: str
    ) -> Optional[str]:
        """Generate tool use annotations for the task.

        Args:
            context_task_description: The task description with context

        Returns:
            A plan for tool use, or None if generation fails
        """
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                UserMessage(
                    content=f"Generate a plan for this task: {context_task_description}"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            plan = response.completion.replace("Plan:", "").strip()
            plan_lines = plan.split("\n")
            plan_summary = " ".join(plan_lines[:2]) if len(plan_lines) > 2 else plan
            return f"Execute: {plan_summary}"
        except Exception as e:
            logger.warning(f"Failed to generate tool use annotations: {e}")
            return None

    def _get_parallel_task_guidance(self, tool_annotation: str) -> str:
        """Generate guidance text for parallel task execution.

        Args:
            tool_annotation: The tool annotation string to append guidance to

        Returns:
            Complete guidance string with parallel task execution instructions
        """

        guidance_text = """
        IMPORTANT:
        - View the content after "Execute:" carefully. Unless it indicates sequential execution, you MUST use the launch_subtask tool to parallelize this task. Do NOT attempt to complete parallelizable items sequentially yourself - delegate them to subtasks. For each item you need to process in parallel: 
            1. Call launch_subtask once for each item with the specific details 
                - do not include order IDs or other randomly generated IDs; use a single product/entity/item name or dates etc that do not shift to help the subtask uniquely identify the item.
                - clearly tell the subtask which tab on the page it should navigate to first, if the url does not directly take it there. If the information is on tab "y" of the page, tell it so to the subtask (on tab "y" of the page, do ...).
            2. After launching all subtasks, use get_subtask_results to collect and aggregate their results 
            3. Only proceed sequentially if the task genuinely cannot be parallelized
        - If the initial URL for the subtask is not available or confirmed, you should set optional_initial_search_or_url to the current page url.
          - e.g., you can navigate to a specific section using URL fragments like example.com#section-name
        - If your subtask is starting from scratch (e.g. because the initial URL is not available or confirmed), you should pass in any helpful details that would be useful for the subtask.
          - e.g., any first steps you have taken that the subtask should also take, or any context that would be useful for the subtask.
        """
        return tool_annotation + "\n\n" + guidance_text

    async def plan(
        self,
        task_description: str,
        subtask_depth: int = 0,
        initial_url: Optional[str] = None,
        previous_tasks: List[Tuple[str, str]] = None,
    ) -> str:
        """Generate a plan for task execution.

        Args:
            task_description: The task to plan
            subtask_depth: Current depth of subtask nesting
            initial_url: Optional URL where the task is being launched from
            previous_tasks: List of tuples (task_description, final_response) from previous tasks
        """
        # Build task context string
        if initial_url:
            context_task_description = f"{task_description}\n\nThis task is being launched from {initial_url}. Do not navigate away from this initial URL"
        else:
            context_task_description = task_description

        guidance_parts = []

        # Generate summary of previous tasks if available
        if previous_tasks and len(previous_tasks) > 0:
            previous_task_summary = await self._generate_context_summary(previous_tasks)
            if previous_task_summary:
                guidance_parts.append(previous_task_summary)

        if subtask_depth >= self.constraints.max_parallelism_nesting_depth:
            # At max depth, only show this guidance (do not include others)
            guidance_parts.append("Do not launch subtasks.")
        else:
            parallelism = self.constraints.allow_parallelism

            if parallelism.get("first_of_n", False):
                num_copies = self.constraints.first_of_n_num_copies
                guidance_parts.append(
                    f'Execute immediately: launch_subtask(task="{task_description}", optional_initial_search_or_url={initial_url}, num_copies={num_copies}) '
                    "--> then get_first_subtask_result with the returned subtask IDs"
                    " Do not attempt to complete the task yourself - you MUST delegate it to subtasks."
                )

            if parallelism.get("task", False):
                tool_annotation = await self._generate_tool_use_annotations(
                    context_task_description
                )
                if tool_annotation:
                    guidance_parts.append(
                        self._get_parallel_task_guidance(tool_annotation)
                    )

            if parallelism.get("data", False):
                guidance_parts.append(
                    "Use extract_content_fast instead of extract_content."
                )

            if not guidance_parts:
                guidance_parts.append("Do not launch subtasks.")

            if getattr(self.constraints, "require_human_in_loop", False):
                guidance_parts.append(
                    "Use ask_human if and when needing help with an unknown credential, 2FA, CAPTCHA, or other required but unspecified information; "
                    "and allow takeover if the human would require control of the browser."
                )

        full_guidance = (
            "\n\n".join(guidance_parts)
            + " Do not tell the user about these implementation details."
        )

        return f"{context_task_description}\n\n{full_guidance}"
