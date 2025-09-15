"""
Evaluate Canadian Emergency Room Triage scenario for the Canadian Triage and Acuity Scale (CTAS).
"""

import os
import logging
from pathlib import Path
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from azure.ai.projects.models import (
    EvaluatorIds, 
    Evaluation, 
    InputDataset, 
    EvaluatorConfiguration
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

class CloudEvaluation:
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
        data_id = self._client.datasets.get("Canadian_ER_Triage_Assessment", "1.0").id
        if not data_id:
            data_id = self._client.datasets.upload_file(
                name="Canadian_ER_Triage_Assessment",
                version="1.0",
                file_path=str(Path(__file__).parent / "config" / "evaluation_data.jsonl"),
            )
        evaluators = {
            "relevance": EvaluatorConfiguration(
                id=EvaluatorIds.RELEVANCE.value,
                init_params={"deployment_name": deployment_name},
                data_mapping={
                    "query": "${data.input}",
                    "response": "${data.expected_output}",
                },
            ),
            "groundedness": EvaluatorConfiguration(
                id=EvaluatorIds.GROUNDEDNESS.value,
                init_params={"deployment_name": deployment_name},
                data_mapping={
                    "query": "${data.input}",
                    "response": "${data.expected_output}",
                },
            ),
            "hateunfairness": EvaluatorConfiguration(
                id=EvaluatorIds.HATE_UNFAIRNESS.value,
                init_params={"azure_ai_project": os.getenv("AZURE_AI_PROJECTS_ENDPOINT")},
                data_mapping={
                    "query": "${data.input}",
                    "response": "${data.expected_output}",
                },
            ),
            "violence": EvaluatorConfiguration(
                id=EvaluatorIds.VIOLENCE.value,
                init_params={"azure_ai_project": os.getenv("AZURE_AI_PROJECTS_ENDPOINT")},
                data_mapping={
                    "query": "${data.input}",
                    "response": "${data.expected_output}",
                },
            ),
            "bluescore": EvaluatorConfiguration(
                id=EvaluatorIds.BLEU_SCORE.value,
                init_params={},
                data_mapping={
                    "y_pred": "${data.expected_output}",
                    "y_true": "${data.ground_truth}",
                },
            ),
            "coherence": EvaluatorConfiguration(
                id=EvaluatorIds.COHERENCE.value,
                init_params={"deployment_name": deployment_name},
                data_mapping={
                    "query": "${data.input}",
                    "response": "${data.expected_output}",
                },
            ),
        }
        evaluation = Evaluation(
            display_name="Canadian ER Triage Evaluation",
            description="Evaluation for Canadian ER Triage Agent using CTAS",
            data=InputDataset(id=data_id),
            evaluators=evaluators,
        )
        evaluation_response = self._client.evaluations.create(
            evaluation,
            headers={
                "model-endpoint": os.getenv("AZURE_ENDPOINT"),
                "api-key": os.getenv("AZURE_API_KEY"),
            }
        )
        logger.info("Created evaluation: %s", evaluation_response.name)

def main() -> None:
    """Main entry point for the evaluation."""
    evaluation = CloudEvaluation()
    evaluation.run_evaluation()


if __name__ == "__main__":
    main()