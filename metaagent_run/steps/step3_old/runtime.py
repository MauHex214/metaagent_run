import importlib
import importlib.util
import logging

try:
    has_text = importlib.util.find_spec("sklearn.feature_extraction.text") is not None
    has_pairwise = importlib.util.find_spec("sklearn.metrics.pairwise") is not None
    if has_text and has_pairwise:
        TfidfVectorizer = importlib.import_module(
            "sklearn.feature_extraction.text"
        ).TfidfVectorizer
        cosine_similarity = importlib.import_module(
            "sklearn.metrics.pairwise"
        ).cosine_similarity
    else:
        TfidfVectorizer = None
        cosine_similarity = None
except (ImportError, ModuleNotFoundError, AttributeError):
    TfidfVectorizer = None
    cosine_similarity = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

STOP_SENTINEL = "</json>"
LOGGER = logging.getLogger(__name__)
