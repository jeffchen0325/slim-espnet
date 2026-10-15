import logging
import re
from omegaconf import DictConfig, ListConfig, OmegaConf

# Used to rewrite relative paths in resolver expressions such as
# `${load_line:conf/tokens.txt}` or `${config_path:../src/file.md}` during
# config loading.
RELATIVE_RESOLVER_PATTERN = re.compile(
    r"\$\{(?P<resolver>load_line|config_path):(?P<path>[^,}]+)"
)


def load_line(path):
    """Load lines from a text file and return as a list of strings.

    This function is used as a custom resolver in OmegaConf,
    allowing YAML files to reference external text line files
    dynamically via `${load_line:some/file.txt}`.

    This resolver is intended to load vocab file in configuration.

    Args:
        path (str or Path): Path to the file.

    Returns:
        List[str]: A list of stripped lines from the file.
    """
    try:
        with open(path, "r") as f:
            return ListConfig([line.strip() for line in f.readlines()])
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except PermissionError:
        logging.error(f"Permission denied when accessing file: {path}")
        raise


def self_name(path):
    """Return the current config stem for `${self_name:}` interpolation.

    This resolver supports config values such as `exp_tag: ${self_name:}`.
    During config loading, `load_config_with_defaults()` rewrites that
    interpolation to the stem of the YAML file currently being processed.

    The resolver itself is intentionally a passthrough because OmegaConf only
    passes explicit resolver arguments to callbacks. The actual config-name
    injection happens earlier in `_normalize_relative_resolver_paths()`, which
    replaces `${self_name:}` with the plain config stem before resolution.

    Args:
        path (str): Config stem passed explicitly to the resolver, such as
            `training` or `inference`.

    Returns:
        str: The injected config stem unchanged.

    Notes:
        This resolver is intended for ESPnet config composition. It is
        primarily consumed from YAML rather than called directly from Python.

    Examples:
        In a config file named `training.yaml`:

            exp_tag: ${self_name:}

        The expression is rewritten during loading to:

            training
    """
    return path


def config_path(path):
    """Return an absolute path rewritten by ``_normalize_relative_resolver_paths``.

    Used as ``${config_path:../src/file.md}`` in YAML configs. The relative
    path is resolved to an absolute path at load time by
    ``_normalize_relative_resolver_paths``, so this resolver simply returns
    its argument unchanged.

    Args:
        path (str): Absolute path injected during config loading.

    Returns:
        str: The path unchanged.

    Examples:
        In a config file at ``conf/demo.yaml``:

            pack:
              readme: ${config_path:../src/hf_demo_readme.md}

        Resolves to the absolute path of ``src/hf_demo_readme.md`` relative
        to the directory containing ``conf/``.
    """
    return path


def set_corpus_and_system(value: str) -> str:
    """Passthrough for ``${set_corpus_and_system:}`` after load-time rewriting.

    This resolver is used as ``${set_corpus_and_system:}`` in YAML configs.
    ``_normalize_relative_resolver_paths`` rewrites it to
    ``${set_corpus_and_system:<corpus>_<system>}`` (e.g.
    ``${set_corpus_and_system:mini_an4_asr}``) by extracting the corpus and
    system names from the ``egs3/<corpus>/<system>/`` portion of the config
    file path. The resolver itself simply returns that injected value.

    If OmegaConf ever resolves this expression before the rewrite has run
    (which should not happen in normal usage), it raises ``RuntimeError`` with
    a clear message.

    Args:
        value (str): The ``<corpus>_<system>`` string injected at load time.

    Returns:
        str: The injected value unchanged.

    Raises:
        RuntimeError: If called with an empty string, meaning the rewrite did
            not run before OmegaConf resolution.

    Examples:
        In ``egs3/mini_an4/asr/conf/publication.yaml``:

            upload_model:
              hf_repo: espnet/${set_corpus_and_system:}_${exp_tag}

        After loading via ``load_and_merge_config``, resolves to::

            espnet/mini_an4_asr_<exp_tag>
    """
    if not value:
        raise RuntimeError(
            "${set_corpus_and_system:} was not rewritten at load time. "
            "Load configs via load_and_merge_config with a recipe path "
            "under egs3/<corpus>/<system>/..."
        )
    return value


OMEGACONF_ESPNET3_RESOLVER = {
    "load_line": load_line,
    "self_name": self_name,
    "config_path": config_path,
    "set_corpus_and_system": set_corpus_and_system,
}
for name, resolver in OMEGACONF_ESPNET3_RESOLVER.items():
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)
        logging.info(f"Registered ESPnet-3 OmegaConf Resolver: {name}")
    else:
        logging.debug(f"Resolver '{name}' already registered, skipping.")