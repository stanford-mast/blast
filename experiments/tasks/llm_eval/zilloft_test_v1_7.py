"""
Zilloft Test v1-7: Count Manufactured Homes

Task: How many "Manufactured" homes are available in San Francisco under $1,000,000?

Expected answer: 0 homes (none are in San Francisco; https://real-zilloft-46k595h3n-real-sites.vercel.app/homes?q=San+Francisco%2C+CA&homeTypes=Manufactured)
Reference: experiments/agisdk/src/agisdk/REAL/browsergym/webclones/v1/tasks/zilloft-7.json
"""

from pydantic import BaseModel, Field

from experiments.tasks.base import TaskValidator


class ZilloftTestV1_7Output(BaseModel):
    """Expected output schema for zilloft-test-v1-7 task"""

    num_homes: int = Field(
        description="Number of 'Manufactured' homes available in San Francisco under $1,000,000"
    )


# Ground truth: 0 manufactured homes
EXPECTED_NUM_HOMES = 0


class ZilloftTestV1_7Validator(TaskValidator):
    """Validator for Zilloft Test v1-7: Count Manufactured Homes"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return ZilloftTestV1_7Output

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """
        Check if the parsed output contains the correct number of manufactured homes.

        Args:
            parsed_output: Output parsed into the task's schema

        Returns:
            True if output is correct, False otherwise
        """
        if not isinstance(parsed_output, ZilloftTestV1_7Output):
            return False

        return parsed_output.num_homes == EXPECTED_NUM_HOMES

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """
        Check partial correctness as a percentage.

        For this task, it's binary - either correct (1.0) or incorrect (0.0).
        """
        return 1.0 if self.check_correctness(parsed_output) else 0.0


# Create singleton validator instance
validator = ZilloftTestV1_7Validator()
