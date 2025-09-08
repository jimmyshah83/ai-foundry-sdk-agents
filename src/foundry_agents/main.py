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

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        self._client = AIProjectClient(
            credential=self._credential,
            endpoint=os.getenv("AZURE_AI_PROJECTS_ENDPOINT"),
        )
        logger.info("Initialized AIProjectClient.")

    async def run(self) -> None:
        """Run the main logic."""
        logger.info("hello world")

def cli() -> None:
    """CLI entry point for the package."""
    async def main() -> None:
        await Main().run()
    
    asyncio.run(main())

if __name__ == "__main__":
    cli()
