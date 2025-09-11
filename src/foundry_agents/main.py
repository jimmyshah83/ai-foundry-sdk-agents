"""
Canadian Emergency Room Triage Agent using the Canadian Triage and Acuity Scale (CTAS).
"""
import os
import logging
import dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import ConnectionType
from azure.ai.agents.models import (
    Agent,
    MessageRole,
    ListSortOrder,
    AzureAISearchTool,
    AzureAISearchQueryType,
)

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

class Main:
    """Canadian Emergency Room Triage Agent using CTAS (Canadian Triage and Acuity Scale)."""

    _client: AIProjectClient
    _credential: DefaultAzureCredential
    _triage_agent: Agent
    _triage_agent_name: str = "CanadianERTriageAgent"
    _patient_history_agent_name: str = "CanadianERPatientHistoryAgent"
    _patient_history_agent: Agent
    _search_idx_name: str = "patient-records-idx"
    _instructions = """You are a Canadian Emergency Room triage nurse following the Canadian Triage and Acuity Scale (CTAS).
        
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

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        self._client = AIProjectClient(
            credential=self._credential,
            endpoint=os.getenv("AZURE_AI_PROJECTS_ENDPOINT")
        )
        logger.info("Initialized AIProjectClient %s", self._client)

    def execute_agent(self, agent: Agent, content: str) -> None:
        """Triage a patient using Canadian ER protocols."""
        logger.info("Starting patient triage assessment.")

        thread = self._client.agents.threads.create()
        message = self._client.agents.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=content,
        )
        logger.info("Sent patient info message ID %s", message.content)
        
        # Run the triage assessment
        self._client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id
        )
        
        # Get the triage assessment
        messages = self._client.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for msg in messages:
            if msg.text_messages:
                last_text = msg.text_messages[-1]
                text = last_text.text.value.replace("\u3010", "[").replace("\u3011", "]")
                logger.info("Received patient triage %s", text)

    def run(self, patient_info: str = None) -> str:
        """Run the Canadian ER triage assessment."""

        existing_agents = self._client.agents.list_agents()

        # Create a triage agent if it doesn't exist
        existing_triage_agent = None
        for agent in existing_agents:
            if agent.name == self._triage_agent_name:
                existing_triage_agent = agent
                self._triage_agent = existing_triage_agent
                logger.info("Found existing agent with name %s and ID %s", self._triage_agent.name, self._triage_agent.id)
                break

        if not existing_triage_agent:
            logger.info("Creating new agent with name %s", self._triage_agent_name)
            self._triage_agent = self._client.agents.create_agent(
                model="gpt-4.1-agent",
                name=self._triage_agent_name,
                instructions=self._instructions
            )
            logger.debug("Canadian ER Triage Agent created with ID %s and name %s", self._triage_agent.id, self._triage_agent.name)

        #Define AI search connection
        search_connection_id = self._client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
        search_tool = AzureAISearchTool(
            index_connection_id=search_connection_id,
            index_name=self._search_idx_name,
            query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
            top_k=3,
        )

        # Create a patient history agent if it does not exist
        existing_patient_history_agent = None
        for agent in existing_agents:
            if agent.name == self._patient_history_agent_name:
                existing_patient_history_agent = agent
                self._patient_history_agent = existing_patient_history_agent
                logger.info("Found existing agent with name %s and ID %s", self._patient_history_agent.name, self._patient_history_agent.id)
                break

        if not existing_patient_history_agent:
            logger.info("Creating new agent with name %s", self._patient_history_agent_name)
            self._patient_history_agent = self._client.agents.create_agent(
                model="gpt-4.1-agent",
                name=self._patient_history_agent_name,
                instructions=self._instructions,
                tools=[search_tool.definitions] if search_tool else None,
                tool_resources=[search_tool.resources] if search_tool else None,
            )
            logger.debug("Canadian ER Patient History Agent created with ID %s and name %s", self._patient_history_agent.id, self._patient_history_agent.name)

        # Validate triage agent
        if not patient_info:
            patient_info = ("45-year-old male presenting with chest pain, shortness of breath, "
                          "sweating, and nausea. Onset 30 minutes ago. No known cardiac history. "
                          "Vital signs: BP 150/90, HR 110, RR 22, O2 sat 96%")

        triage_agent_content: str = f"Please assess this patient for Canadian ER triage: {patient_info}"
        self.execute_agent(agent=self._triage_agent, content=triage_agent_content)

def cli() -> None:
    """CLI entry point for the Canadian ER triage agent."""
    def main() -> None:
        Main().run()

    main()

if __name__ == "__main__":
    cli()
