import os

current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')
adaptation_config_file = os.path.join(config_dir, 'config.json')

PATTERNS = [
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
]
USE_RAG = True
AGENT_LOG_FILE = os.environ.get("AGENT_LOG_FILE", os.path.join(os.getcwd(), "logs", "ai_agent_decisions.txt"))
PERFORMANCE_DIR = os.environ.get("AP4FED_PERFORMANCE_DIR", os.path.join(os.getcwd(), "performance"))
RATIONALE_CSV_FILE = os.environ.get(
    "AP4FED_RATIONALE_CSV_FILE",
    os.path.join(PERFORMANCE_DIR, "FLwithAP_adaptation_rationales.csv"),
)
