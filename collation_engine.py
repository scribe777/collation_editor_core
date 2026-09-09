"""Abstract collation engine and plugin registry.

Provides the base class for collation engines and a registry mechanism so
that engines can be added without modifying core code. Deployment concerns
(where settings come from, where logs go) and engine-family concerns (e.g.
what an LLM-backed engine has to validate) belong in subclasses or mixins,
which the hooks below are there for.
"""

import json
import sys
import time
from abc import ABC, abstractmethod


class CollationResult:
    """Container for collation engine output."""

    def __init__(self):
        self.table = []
        self.witnesses = []
        self.regularization_suggestions = []
        self.feedback = {
            'comments': '',
            'alignment_table': '',
            'processing_duration': None,
            'engine_usage': None,
        }

    @staticmethod
    def parse_collatex_json(payload):
        """Parse a CollateX-shaped JSON document (bytes, str or dict) into (table, witnesses)."""
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload.get('table', []), payload.get('witnesses', [])

    def to_output_dict(self):
        """Return a dict suitable for use as the 'output' block."""
        d = {
            'witnesses': self.witnesses,
            'table': self.table,
        }
        if self.regularization_suggestions:
            d['regularization_suggestions'] = self.regularization_suggestions
        # include only non-empty feedback entries
        feedback = {k: v for k, v in self.feedback.items() if v is not None and v != ''}
        if feedback:
            d['collation_feedback'] = feedback
        return d


class CollationEngine(ABC):
    """Abstract base class for collation engines.

    Subclasses implement name() and collate(). run() wraps collate() with
    timing and usage bookkeeping and then serialises the result. Two hooks
    exist for subclasses and mixins: obtain_result() (around the collate
    call, e.g. for caching) and post_process() (on the output dict before
    it is serialised, e.g. for validation or enrichment).
    """

    # Subclasses should define these to register their models and metadata.
    _engine_meta = {
        'display_name': 'My Collation Engine',
        'model_override_key': 'my_engine_algorithm',
    }
    _models = [
        # e.g.,
        #    {'id': 'dekker',
        #     'name': 'Dekker',
        #     'default': True
        #     'max_tokens': 0,  # any additional model propeties the engine wants to keep per model for its own use
        #    },
        #    {'id': 'needleman-wunsch', 'name': 'Needleman-Wunsch', 'max_tokens': 0},
    ]

    @classmethod
    def get_model_names(cls):
        """Return {model_id: display_name} dict."""
        return {m['id']: m['name'] for m in cls._models}

    @classmethod
    def get_model_max_tokens(cls):
        """Return {model_id: max_tokens} dict."""
        return {m['id']: m['max_tokens'] for m in cls._models}

    @classmethod
    def get_model_reasoning_tokens(cls):
        """Return {model_id: reasoning_tokens} dict.

        If a model does not define reasoning_tokens, it defaults to max_tokens.
        """
        return {m['id']: m.get('reasoning_tokens', m['max_tokens']) for m in cls._models}

    @classmethod
    def get_default_model(cls):
        """Return the default model ID, or the first model if none marked."""
        for m in cls._models:
            if m.get('default'):
                return m['id']
        return cls._models[0]['id'] if cls._models else None

    @classmethod
    def available(cls):
        """Whether this engine can run here (e.g. its optional dependency is installed)."""
        return True

    @classmethod
    def get_engine_registry(cls):
        """Return this engine's metadata and models in registry format."""
        meta = dict(cls._engine_meta)
        meta['models'] = cls._models
        return meta

    def __init__(self, algorithm_settings, display_settings=None):
        self.algorithm_settings = algorithm_settings or {}
        self.display_settings = display_settings or {}

    def get_setting(self, key, default=None):
        """Return algorithm_settings[key] unless absent or empty, else default.

        Deployments that keep settings elsewhere override this.
        """
        val = self.algorithm_settings.get(key)
        return val if val not in (None, '') else default

    def log(self, entry):
        """Report engine progress; a string or a list of strings. Defaults to stderr."""
        if isinstance(entry, list):
            entry = '\n'.join(entry)
        print(entry, file=sys.stderr)

    @abstractmethod
    def name(self):
        """Return the engine identifier string (e.g. 'collatex')."""

    @abstractmethod
    def collate(self, data, options, basetext_siglum):
        """Perform collation and return a CollationResult.

        Args:
            data: dict with 'witnesses' list and 'algorithm' key
            options: dict with 'outputFormat', 'algorithm', 'tokenComparator'
            basetext_siglum: the siglum of the base text witness

        Returns:
            CollationResult with table and witnesses populated
        """

    def obtain_result(self, data, options, basetext_siglum):
        """Hook around collate(); override to reuse a previous result, retry, etc."""
        return self.collate(data, options, basetext_siglum)

    def post_process(self, output, data):
        """Hook on the output dict before serialisation; override to validate or enrich it."""
        return output

    def run(self, data, options, basetext_siglum):
        """Collate with timing and usage bookkeeping, then serialise the result."""
        start_time = time.time()
        result = self.obtain_result(data, options, basetext_siglum)
        elapsed = round(time.time() - start_time, 1)

        if result.feedback.get('processing_duration') is None:
            result.feedback['processing_duration'] = elapsed
        if result.feedback.get('engine_usage') is None:
            result.feedback['engine_usage'] = {}
        usage = result.feedback['engine_usage']
        usage.setdefault('engine', self.name())
        usage.setdefault('algorithm', options.get('algorithm', ''))
        usage.setdefault('duration_seconds', elapsed)
        usage.setdefault('summary', '{} | {}s'.format(self.name(), elapsed))

        return self.process_result(result, data)

    def process_result(self, result, data):
        """Serialise a CollationResult, giving post_process() a chance at the dict first.

        Returns:
            JSON string of the output dict, or the engine's raw response
            when it supplied one (e.g. CollateX returning bytes directly).
        """
        if getattr(result, '_raw_response', None):
            return result._raw_response
        output = self.post_process(result.to_output_dict(), data)
        return json.dumps(output, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

# Imported here rather than at the top: the engines subclass CollationEngine,
# so top-level imports would be circular.
from collation.core.engines.collate_service import CollateServiceEngine  # noqa: E402
from collation.core.engines.collatex_microservice import CollatexEngine  # noqa: E402
from collation.core.engines.collatex_python import CollatexPythonEngine  # noqa: E402

_engine_registry = {}
_default_engine = CollatexEngine


def register_engine(name, engine_class, default=False):
    """Register a collation engine class by algorithm name.

    If default=True, this engine handles any algorithm name not
    explicitly registered (e.g. CollateX handles dekker, needleman-wunsch, etc.).
    """
    _engine_registry[name] = engine_class
    if default:
        global _default_engine
        _default_engine = engine_class


def get_engine(name, algorithm_settings, display_settings=None):
    """Look up and instantiate a registered engine, or fall back to the default."""
    cls = _engine_registry.get(name, _default_engine)
    if cls is not None:
        return cls(algorithm_settings, display_settings=display_settings)
    return None


def list_engines():
    """Return list of registered engine names."""
    return list(_engine_registry.keys())


def get_engine_registry():
    """Return model metadata from every registered engine that is available and has metadata."""
    engines = {}
    for name, cls in _engine_registry.items():
        if getattr(cls, '_engine_meta', None) and cls.available():
            engines[name] = cls.get_engine_registry()
    return {'engines': engines}


# The engines shipped with the core. Services may register more, or re-register
# these names with their own classes.
register_engine('collatex', CollatexEngine)
register_engine('collatex-python', CollatexPythonEngine)
register_engine('local', CollateServiceEngine)
