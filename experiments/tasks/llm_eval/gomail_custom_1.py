"""
GoMail Custom Task 1: Unread Email Count

Task: How many unread emails are in the Inbox?

Expected Answer: 17 unread emails in the Inbox
(Based on v1/gomail-1 task)
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import TaskValidator


class GomailCustom1Output(BaseModel):
    """Expected output schema for gomail-custom-1 task"""

    unread_email_count: int = Field(description="Number of unread emails in the Inbox")
    reasoning: Optional[str] = Field(
        default=None, description="Optional explanation of how the count was determined"
    )


# Ground truth from original gomail-1 eval
EXPECTED_UNREAD_COUNT = 17


class GomailCustom1Validator(TaskValidator):
    """Validator for GoMail Custom Task 1"""

    @property
    def output_schema(self) -> type[BaseModel]:
        return GomailCustom1Output

    def check_correctness_pct(self, parsed_output: BaseModel) -> float:
        """Check if the unread email count is correct."""
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(parsed_output, GomailCustom1Output):
            logger.warning(
                f"check_correctness_pct: unexpected type {type(parsed_output)}"
            )
            return 0.0

        is_correct = parsed_output.unread_email_count == EXPECTED_UNREAD_COUNT
        logger.info(
            f"check_correctness_pct: expected={EXPECTED_UNREAD_COUNT}, "
            f"got={parsed_output.unread_email_count}, correct={is_correct}"
        )

        return 1.0 if is_correct else 0.0

    def check_correctness(self, parsed_output: BaseModel) -> bool:
        """Check if the unread email count is correct."""
        return self.check_correctness_pct(parsed_output) == 1.0


# Create singleton validator instance
validator = GomailCustom1Validator()
