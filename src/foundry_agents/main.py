"""
Canadian Emergency Room Triage Agent using the Canadian Triage and Acuity Scale (CTAS).
"""
import os
import logging
from importlib import resources
from pathlib import Path
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
	ConnectedAgentTool,
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
	_conversation_agent: Agent
	_triage_agent: Agent
	_patient_history_agent: Agent
	_search_tool: AzureAISearchTool
	_conversation_agent_name: str = "CanadianERConversationAgent"
	_triage_agent_name: str = "CanadianERTriageAgent"
	_patient_history_agent_name: str = "CanadianERPatientHistoryAgent"	
	_search_idx_name: str = "idx-patient-data"

	@property
	def _triage_instructions(self) -> str:
		"""Load triage instructions from config file."""
		try:
			with resources.files('foundry_agents.config').joinpath('triage_instructions.txt').open('r') as f:
				return f.read()
		except FileNotFoundError:
			config_file = Path(__file__).parent / 'config' / 'triage_instructions.txt'
			return config_file.read_text()

	@property
	def _patient_history_instructions(self) -> str:
		"""Load patient history instructions from config file."""
		try:
			with resources.files('foundry_agents.config').joinpath('patient_history_instructions.txt').open('r') as f:
				return f.read()
		except FileNotFoundError:
			config_file = Path(__file__).parent / 'config' / 'patient_history_instructions.txt'
			return config_file.read_text()

	@property
	def _conversation_instructions(self) -> str:
		"""Load conversation instructions from config file."""
		try:
			with resources.files('foundry_agents.config').joinpath('conversation_instructions.txt').open('r') as f:
				return f.read()
		except FileNotFoundError:
			config_file = Path(__file__).parent / 'config' / 'conversation_instructions.txt'
			return config_file.read_text()

	@property
	def _user_prompt(self) -> str:
		"""Load patient history content from config file."""
		try:
			with resources.files('foundry_agents.config').joinpath('user_prompt.txt').open('r') as f:
				return f.read()
		except FileNotFoundError:
			config_file = Path(__file__).parent / 'config' / 'user_prompt.txt'
			return config_file.read_text()

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

		self._client.agents.runs.create_and_process(
			thread_id=thread.id,
			agent_id=agent.id
		)

		messages = self._client.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
		for msg in messages:
			if msg.text_messages:
				last_text = msg.text_messages[-1]
				text = last_text.text.value.replace("\u3010", "[").replace("\u3011", "]")
				logger.info("%s", text)

	def initialize_search_tool(self) -> AzureAISearchTool:
		"""Get the Azure AI search tool for querying patient records."""
		search_connection_id = self._client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
		self._search_tool = AzureAISearchTool(
			index_connection_id=search_connection_id,
			index_name=self._search_idx_name,
			query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
			top_k=3,
		)
		
	def initialize_triage_agent(self, existing_agents: list) -> None:
		"""Initialize The triage agent"""
		logger.info("Found %d existing agents: %s", len(existing_agents), [agent.name for agent in existing_agents])
		self._triage_agent = next(
			(agent for agent in existing_agents if agent.name == self._triage_agent_name), None
		)
		if self._triage_agent:
			logger.info("Found existing triage agent with name %s and ID %s", self._triage_agent.name, self._triage_agent.id)
		else:
			logger.info("Creating new agent with name %s", self._triage_agent_name)
			self._triage_agent = self._client.agents.create_agent(
				model="gpt-4.1-agent",
				description="Agent to perform Canadian ER triage assessments using CTAS",
				name=self._triage_agent_name,
				instructions=self._triage_instructions
			)
			logger.debug("Canadian ER Triage Agent created with ID %s and name %s", self._triage_agent.id, self._triage_agent.name)

	def initialize_patient_history_agent(self, existing_agents: list) -> None:
		"""Initialize The Patient History agent"""
		self._patient_history_agent = next(
			(agent for agent in existing_agents if agent.name == self._patient_history_agent_name), None
		)
		if self._patient_history_agent:
			logger.info("Found existing patient history agent with name %s and ID %s", self._patient_history_agent.name, self._patient_history_agent.id)
		else:
			logger.info("Creating new agent with name %s", self._patient_history_agent_name)
			self._patient_history_agent = self._client.agents.create_agent(
				model="gpt-4.1-agent",
				name=self._patient_history_agent_name,
				instructions=self._patient_history_instructions,
				description="Agent to retrieve patient Immunization and DiagnosticReport history using AI search tool",
				tools=self._search_tool.definitions if self._search_tool else None,
				tool_resources=self._search_tool.resources if self._search_tool else None,
			)
			logger.debug("Canadian ER Patient History Agent created with ID %s and name %s", self._patient_history_agent.id, self._patient_history_agent.name)

	def initialize_conversation_agent(self, existing_agents: list) -> None:
		"""Main conversation agent that takes in user prompt"""
		self._conversation_agent = next(
			(agent for agent in existing_agents if agent.name == self._conversation_agent_name), None
		)
		if self._conversation_agent:
			logger.info("Found existing conversation agent with name %s and ID %s", self._conversation_agent.name, self._conversation_agent.id)
		else:
			logger.info("Creating new agent with name %s", self._conversation_agent_name)
			triage_connected_agent = ConnectedAgentTool(
				id=self._triage_agent.id, name=self._triage_agent.name, description="Triage the patient based on CTAS"
			)
			patient_history_connected_agent = ConnectedAgentTool(
				id=self._patient_history_agent.id, name=self._patient_history_agent.name, description="Retrieve patient history"
			)
			
			self._conversation_agent = self._client.agents.create_agent(
				model="gpt-4.1-agent",
				description="Main conversation agent for Canadian ER triage",
				name=self._conversation_agent_name,
				instructions=self._conversation_instructions,
				tools=[
					triage_connected_agent.definitions[0],
					patient_history_connected_agent.definitions[0]
				]
			)
			logger.debug("Canadian ER Conversation Agent created with ID %s and name %s", self._conversation_agent.id, self._conversation_agent.name)

	def run(self) -> str:
		"""Run the Canadian ER triage assessment."""

		existing_agents = list(self._client.agents.list_agents())

		# Initialize search tool first as it's needed by patient history agent
		self.initialize_search_tool()

		# Initialize specialized agents first before conversation agent
		self.initialize_triage_agent(existing_agents)
		self.initialize_patient_history_agent(existing_agents)

		# Initialize conversation agent last as it needs references to other agents
		self.initialize_conversation_agent(existing_agents)

		self.execute_agent(agent=self._conversation_agent, content=self._user_prompt)

def cli() -> None:
	"""CLI entry point for the Canadian ER triage agent."""
	def main() -> None:
		Main().run()

	main()

if __name__ == "__main__":
	cli()
