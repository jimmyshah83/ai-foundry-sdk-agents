"""
Red teaming agents for evaluating AI models.

Red teaming is the process of rigorously testing and challenging AI models to identify
vulnerabilities, biases, and potential failure points. This is crucial for ensuring
the robustness, reliability, and safety of AI systems before they are deployed in
real-world applications.
"""

import os
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv
from azure.ai.projects.models import (
    RedTeam,
    AzureOpenAIModelConfiguration,
    AttackStrategy,
    RiskCategory,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

model_endpoint = os.environ["AZURE_ENDPOINT"]  # Sample: https://<account_name>.openai.azure.com
model_api_key = os.environ["AZURE_API_KEY"]
model_deployment_name = os.environ["AZURE_DEPLOYMENT_NAME"]


class RedTeamAgent:
    """A class representing a red team agent for evaluating AI models."""

    def __init__(self, model: str) -> None:
        """Initialize the red team agent with the specified model."""
        self._credential = DefaultAzureCredential()
        endpoint = os.getenv("AZURE_AI_PROJECTS_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECTS_ENDPOINT environment variable is required")
        self._client = AIProjectClient(credential=self._credential, endpoint=endpoint)
        logger.info("Initialized AIProjectClient %s", self._client)

    def evaluate(self, test_data: str) -> None:
        """Evaluate the model using the provided test data."""
        target_config = AzureOpenAIModelConfiguration(model_deployment_name=model_deployment_name)

        red_team = RedTeam(
            attack_strategies=[AttackStrategy.BASE64],
            risk_categories=[RiskCategory.HATE_UNFAIRNESS, RiskCategory.VIOLENCE],
            display_name="red-team-cloud-run",
            target=target_config,
        )

        headers = {"model-endpoint": model_endpoint, "api-key": model_api_key}

        red_team_response = self._client.red_teams.create(red_team=red_team, headers=headers)

        logger.info("Created red team agent: %s", red_team_response.name)


def main() -> None:
    """Main entry point for the red team evaluation."""
    red_team = RedTeamAgent(model=model_deployment_name)
    test_data = "Test data for evaluation"
    red_team.evaluate(test_data)


if __name__ == "__main__":
    main()
