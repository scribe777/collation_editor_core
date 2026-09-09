"""CollateX collation engine: the collatex Python package, run in-process (optional dependency)."""

import importlib
import importlib.util

from collation.core.collation_engine import CollationEngine, CollationResult


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
        rows, witness_ids = CollationResult.parse_collatex_json(response)

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
