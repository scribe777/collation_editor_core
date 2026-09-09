"""Collation engine for a project-supplied collation service: the legacy localCollationFunction hook."""

import importlib

from collation.core.collation_engine import CollationEngine, CollationResult


class CollateServiceEngine(CollationEngine):
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
            result.table, result.witnesses = CollationResult.parse_collatex_json(response)
            result.feedback['engine_usage'] = {'engine': self.name(), 'summary': config['function']}
        except Exception as e:
            self.log('+++ could not parse the local function result as CollateX JSON: {} +++'.format(e))
            result._raw_response = response
        return result
