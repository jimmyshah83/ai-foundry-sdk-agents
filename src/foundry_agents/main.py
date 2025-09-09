"""
Canadian Emergency Room Triage Agent using the Canadian Triage and Acuity Scale (CTAS).
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
    """Canadian Emergency Room Triage Agent using CTAS (Canadian Triage and Acuity Scale)."""

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
        """Initialize the Canadian ER triage agent asynchronously."""
        instructions = """You are a Canadian Emergency Room triage nurse following the Canadian Triage and Acuity Scale (CTAS).
        
        Assess patients using the 5-level CTAS system:
        - Level 1 (Resuscitation): Life-threatening, immediate intervention required
        - Level 2 (Emergent): Potential threat to life/limb, within 15 minutes
        - Level 3 (Urgent): Potentially serious, within 30 minutes
        - Level 4 (Less Urgent): Conditions related to patient age, distress, potential complications, within 60 minutes
        - Level 5 (Non-urgent): Non-urgent conditions, within 120 minutes
        
        For each patient, provide:
        1. CTAS level (1-5) with rationale
        2. Recommended immediate actions
        3. Estimated wait time based on current ER capacity
        4. Any red flags requiring immediate attention
        
        Always prioritize patient safety and follow Canadian healthcare protocols."""
        
        self._agent = await self._client.agents.create_agent(
            model="gpt-4.1-agent",
            name="CanadianERTriageAgent",
            instructions=instructions
        )
        logger.info("Canadian ER Triage Agent created with ID %s and name %s", self._agent.id, self._agent.name)

    async def triage_patient(self, patient_info: str) -> str:
        """Triage a patient using Canadian ER protocols."""
        logger.info("Starting patient triage assessment.")

        thread = await self._client.agents.threads.create()
        message = await self._client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please assess this patient for Canadian ER triage: {patient_info}"
        )
        logger.debug("Sent patient info message ID %s", message.content)
        
        # Run the triage assessment
        run = await self._client.agents.runs.create(
            thread_id=thread.id,
            assistant_id=self._agent.id
        )
        
        # Wait for completion and get response
        await self._client.agents.runs.get(
            thread_id=thread.id,
            run_id=run.id
        )
        
        # Get the triage assessment
        messages = await self._client.agents.messages.list(thread_id=thread.id)
        triage_result = messages.data[0].content[0].text.value
        
        logger.info("Patient triage completed.")
        return triage_result

    async def run(self, patient_info: str = None) -> str:
        """Run the Canadian ER triage assessment."""
        if not self._agent:
            await self.initialize_agent()
        
        # Use sample patient if none provided
        if not patient_info:
            patient_info = ("45-year-old male presenting with chest pain, shortness of breath, "
                          "sweating, and nausea. Onset 30 minutes ago. No known cardiac history. "
                          "Vital signs: BP 150/90, HR 110, RR 22, O2 sat 96%")
        
        return await self.triage_patient(patient_info)


def cli() -> None:
    """CLI entry point for the Canadian ER triage agent."""
    async def main() -> None:
        triage_agent = Main()
        result = await triage_agent.run()
        print("\n" + "="*60)
        print("CANADIAN ER TRIAGE ASSESSMENT")
        print("="*60)
        print(result)
        print("="*60)
    
    asyncio.run(main())

if __name__ == "__main__":
    cli()
