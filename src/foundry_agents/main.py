"""
Main class to manage the Foundry Agents.
"""
import os
import asyncio
import logging
import dotenv
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

class Main:
    """Simple main class that prints a greeting."""

    _client: AIProjectClient
    _credential: DefaultAzureCredential
    _agent: any

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        self._client = AIProjectClient(
            credential=self._credential,
            endpoint=os.getenv("AZURE_AI_PROJECTS_ENDPOINT")
        )
        logger.info("Initialized AIProjectClient %s", self._client)

    async def initialize_agent(self) -> None:
        """Initialize the agent asynchronously."""
        self._agent = await self._client.agents.create_agent(
            model="gpt-4.1-agent",
            name="TriageAgent",
            instructions="Please triage the following issue."
        )
        logger.info("Agent created with ID %s and name %s", self._agent.id, self._agent.name)

    async def triage_agent(self) -> None:
        """Simple AI Foundry SDK Triage agent."""
        logger.info("Triage agent started.")

    async def run(self) -> None:
        """Run the main logic."""
        if not self._agent:
            await self.initialize_agent()
        await self.triage_agent()


def cli() -> None:
    """CLI entry point for the package."""
    async def main() -> None:
        await Main().run()
    
    asyncio.run(main())

if __name__ == "__main__":
    cli()
