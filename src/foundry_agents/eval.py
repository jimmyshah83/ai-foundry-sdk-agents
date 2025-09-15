"""
Evaluate Canadian Emergency Room Triage scenario for the Canadian Triage and Acuity Scale (CTAS).
"""

import os
import logging
from anyio import Path
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Evaluation:
    """Evaluation for Canadian ER Triage Agent using CTAS."""

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        endpoint = os.getenv("AZURE_AI_PROJECTS_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECTS_ENDPOINT environment variable is required")
        self._client = AIProjectClient(credential=self._credential, endpoint=endpoint)
        logger.info("Initialized AIProjectClient %s", self._client)

    def run_evaluation(self) -> None:
        """Run the evaluation for the Canadian ER triage assessment."""
        self._client.datasets.upload_file(
            name="Canadian_ER_Triage_Assessment",
            version="1.0",
            file_path=str(Path(__file__).parent / "config" / "evaluation_data.jsonl"),
        )


def main() -> None:
    """Main entry point for the evaluation."""
    evaluation = Evaluation()
    evaluation.run_evaluation()
