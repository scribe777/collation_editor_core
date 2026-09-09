# -*- coding: utf-8 -*-
"""The collation engines shipped with the core.

CollatexEngine        the CollateX Java microservice over HTTP (the default)
CollatexPythonEngine  the collatex Python package, in-process (optional dependency)
LocalFunctionEngine   the legacy localCollationFunction hook: a project-supplied
                      python_file / class_name / function returning CollateX JSON
"""

import importlib
import importlib.util
import json
import sys
import urllib.request

from collation.core.collation_engine import CollationEngine, CollationResult

# Seconds to wait for the CollateX microservice. Applies to connect AND to each
# socket read -- and since CollateX sends nothing at all until the collation is
# finished, in practice this is the whole budget for the run.
#
# Deliberately very generous. A verse-sized collation returns in milliseconds,
# but a large one -- e.g. John.11 against all available witnesses, approaching
# 1000 -- can legitimately grind for a long time, and timing out a real result
# after the editor has waited that long is far worse than waiting longer. The
# point of the bound is only to stop an UNBOUNDED process leak when the service
# is dead (see the note at the urlopen call), not to police slow collations.
# Override per project with the collatex_timeout algorithm setting.
DEFAULT_COLLATEX_TIMEOUT = 3600


class CollatexEngine(CollationEngine):
    """Collation engine backed by the CollateX Java microservice."""

    _engine_meta = {
        'display_name': 'CollateX',
        'model_override_key': 'collatex_algorithm',
    }

    _models = [
        {'id': 'dekker', 'name': 'Dekker', 'max_tokens': 0, 'default': True},
        {'id': 'needleman-wunsch', 'name': 'Needleman-Wunsch', 'max_tokens': 0},
    ]

    def name(self):
        """Return the registry name of this engine."""
        return 'collatex'

    def collate(self, data, options, basetext_siglum):
        """POST the witnesses to CollateX and return its JSON alignment table."""
        host = self.algorithm_settings.get('collatexHost', 'http://localhost:7369/collate')
        algorithm = self.algorithm_settings.get('collatex_algorithm') or options.get('algorithm', 'dekker')

        witnesses = data.get('witnesses', [])
        word_counts = [len(w.get('tokens', [])) for w in witnesses]
        self.log(
            'algorithm={} host={}\n{} witnesses, {} to {} words each'.format(
                algorithm, host, len(witnesses), min(word_counts), max(word_counts)
            )
        )

        if 'algorithm' in options:
            data['algorithm'] = options['algorithm']
        if 'tokenComparator' in options:
            data['tokenComparator'] = options['tokenComparator']

        json_witnesses = json.dumps(data)
        if 'outputFormat' in options:
            accept_header = self._convert_header(options['outputFormat'])
        else:
            accept_header = 'application/json'

        req = urllib.request.Request(host)
        req.add_header('content-type', 'application/json')
        req.add_header('Accept', accept_header)

        # Without an explicit timeout urllib blocks in recv() indefinitely. On
        # 2026-08-20 a JVM upgrade left the CollateX service accepting connections
        # but never replying, and because this call could not time out, 28
        # collate_cli.py processes accumulated over 16 hours -- one per request,
        # each pinning a socket and ~36 MB -- until the host ran short of memory.
        # A bounded wait turns a dead service into a prompt, visible error.
        try:
            timeout = float(self.algorithm_settings.get('collatex_timeout') or DEFAULT_COLLATEX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = DEFAULT_COLLATEX_TIMEOUT

        try:
            response = urllib.request.urlopen(req, json_witnesses.encode('utf-8'), timeout=timeout)
        except Exception as e:
            self.log('+++ ERROR: CollateX service unavailable after {}s: {} +++'.format(timeout, e))
            raise

        response_body = response.read()

        algorithm_name = self.get_model_names().get(algorithm, algorithm)
        fuzzy = 'with' if self.algorithm_settings.get('fuzzy_match') else 'without'

        result = CollationResult()
        try:
            response_json = json.loads(response_body)
            result.table = response_json.get('table', [])
            result.witnesses = response_json.get('witnesses', [])
            self.log('+++ SUCCESS: {} CGs, {} witnesses +++'.format(len(result.table), len(result.witnesses)))
            result.feedback['comments'] = (
                'CollateX {} {} fuzzy match: {} column groups, {} witnesses with between {} and {} words each'.format(
                    algorithm_name, fuzzy, len(result.table), len(result.witnesses), min(word_counts), max(word_counts)
                )
            )
            result.feedback['engine_usage'] = {
                'engine': 'collatex',
                'model': algorithm,
                'model_name': algorithm_name,
                'summary': 'CollateX {}'.format(algorithm_name),
            }
        except Exception as e:
            print('======= error parsing CollateX result as json: ' + str(e), file=sys.stderr)
            self.log('+++ JSON PARSE ERROR: {} +++'.format(e))
            result.feedback['comments'] = 'Error parsing CollateX response: {}'.format(e)
            # return raw response as-is for backward compatibility
            result._raw_response = response_body
        return result

    @staticmethod
    def _convert_header(accept):
        if accept == 'json' or accept == 'lcs':
            return 'application/json'
        elif accept == 'tei':
            return 'application/tei+xml'
        elif accept == 'graphml':
            return 'application/graphml+xml'
        elif accept == 'dot':
            return 'text/plain'
        elif accept == 'svg':
            return 'image/svg+xml'
        return 'application/json'


def _table_from_json(payload):
    """Parse a CollateX JSON document (bytes, str or dict) into (table, witnesses)."""
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8')
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload.get('table', []), payload.get('witnesses', [])


class CollatexPythonEngine(CollationEngine):
    """Collation engine backed by the collatex Python package, run in-process.

    Requires the optional ``collatex`` package (and its Levenshtein dependency);
    available() reports whether it is importable so the registry can leave the
    engine out of menus where it cannot run. The package's JSON table is
    witness-major with ``null`` for gaps and carries its own bookkeeping keys on
    every token; collate() normalises that to the column-major, empty-list-gap
    shape the postprocessor expects, so both CollateX routes look identical
    downstream.
    """

    _engine_meta = {
        'display_name': 'CollateX (Python)',
        'model_override_key': 'collatex_python_algorithm',
    }

    _models = [
        {'id': 'dekker', 'name': 'Dekker', 'max_tokens': 0, 'default': True},
    ]

    _PRIVATE_TOKEN_KEYS = ('_sigil', '_token_array_position')

    @classmethod
    def available(cls):
        """True when the collatex package can be imported."""
        return importlib.util.find_spec('collatex') is not None

    def name(self):
        """Return the registry name of this engine."""
        return 'collatex-python'

    def collate(self, data, options, basetext_siglum):
        """Collate in-process with the collatex package and return the normalised table."""
        collatex = importlib.import_module('collatex')

        witnesses = data.get('witnesses', [])
        word_counts = [len(w.get('tokens', [])) for w in witnesses] or [0]
        comparator = options.get('tokenComparator') or {}
        near_match = comparator.get('type') == 'levenshtein'
        detect_transpositions = bool(self.algorithm_settings.get('detect_transpositions'))
        self.log(
            'collatex (python) near_match={} detect_transpositions={}\n{} witnesses, {} to {} words each'.format(
                near_match, detect_transpositions, len(witnesses), min(word_counts), max(word_counts)
            )
        )

        collation = collatex.Collation()
        for witness in witnesses:
            collation.add_witness({'id': witness['id'], 'tokens': witness.get('tokens', [])})
        response = collatex.collate(
            collation,
            output='json',
            segmentation=False,
            near_match=near_match,
            detect_transpositions=detect_transpositions,
        )
        rows, witness_ids = _table_from_json(response)

        result = CollationResult()
        result.witnesses = witness_ids
        result.table = self._to_column_major(rows)
        fuzzy = 'with' if near_match else 'without'
        self.log('+++ SUCCESS: {} CGs, {} witnesses +++'.format(len(result.table), len(result.witnesses)))
        result.feedback['comments'] = (
            'CollateX (Python) {} fuzzy match: {} column groups, {} witnesses with between {} and {} words each'.format(
                fuzzy, len(result.table), len(result.witnesses), min(word_counts), max(word_counts)
            )
        )
        result.feedback['engine_usage'] = {
            'engine': self.name(),
            'model': 'dekker',
            'model_name': 'Dekker',
            'summary': 'CollateX (Python) Dekker',
        }
        return result

    @classmethod
    def _to_column_major(cls, rows):
        """Transpose witness-major rows into column-major cells; None gaps become []; private keys go."""
        if not rows:
            return []
        columns = []
        for col in range(len(rows[0])):
            cell_per_witness = []
            for row in rows:
                cell = row[col] if col < len(row) else None
                if cell is None:
                    cell_per_witness.append([])
                else:
                    cell_per_witness.append(
                        [{k: v for k, v in token.items() if k not in cls._PRIVATE_TOKEN_KEYS} for token in cell]
                    )
            columns.append(cell_per_witness)
        return columns


class LocalFunctionEngine(CollationEngine):
    """The legacy localCollationFunction hook as an engine.

    algorithm_settings['local_collation_function'] holds the project's
    ``python_file`` / ``class_name`` / ``function``; the method is called with
    (data, options) and must return CollateX-shaped JSON (bytes, str or dict),
    exactly as documented for the services-file variable. It is not listed in
    engine menus: the preprocessor selects it whenever the hook is configured.
    """

    _engine_meta = {}

    def name(self):
        """Return the registry name of this engine."""
        return 'local'

    def collate(self, data, options, basetext_siglum):
        """Call the configured project function and wrap whatever it returns."""
        config = self.algorithm_settings.get('local_collation_function') or {}
        module = importlib.import_module(config['python_file'])
        instance = getattr(module, config['class_name'])()
        self.log(
            'local collation function {}.{}.{}'.format(config['python_file'], config['class_name'], config['function'])
        )
        response = getattr(instance, config['function'])(data, options)

        result = CollationResult()
        try:
            result.table, result.witnesses = _table_from_json(response)
            result.feedback['engine_usage'] = {'engine': self.name(), 'summary': config['function']}
        except Exception as e:
            self.log('+++ could not parse the local function result as CollateX JSON: {} +++'.format(e))
            result._raw_response = response
        return result
