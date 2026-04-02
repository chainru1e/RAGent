# ════════════════════════════════════════════════════════════════════════════
# config/settings.py - Global Configuration Settings
# ════════════════════════════════════════════════════════════════════════════

#
# ENCODING CONFIGURATION
#
DEFAULT_ENCODING = "utf-8"

#
# CHUNKING CONFIGURATION
#
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
CHUNKING_STRATEGY = "semantic"

#
# EMBEDDING CONFIGURATION
#
DENSE_MODEL = "all-MiniLM-L6-v2"
DENSE_DIMENSION = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

#
# HYBRID SEARCH CONFIGURATION
#
ALPHA = 0.6

#
# QDRANT CONFIGURATION
#
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "documents"
QDRANT_COLLECTION = "documents"
QDRANT_PATH = "./qdrant_storage"
QDRANT_MODE = "local"
QDRANT_VECTOR_SIZE = 384

#
# RETRIEVAL CONFIGURATION
#
SEARCH_SCORE_THRESHOLD = 0.5
SEARCH_TOP_K = 5

#
# DATA CONFIGURATION
# 
DATA_DIR = "./data"

#
# LOGGING CONFIGURATION
#
LOG_LEVEL = "INFO"