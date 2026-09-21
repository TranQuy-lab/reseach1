from . import cleaning, sample, schema, splitting

# NOTE: `pipeline` is deliberately NOT imported here. It is meant to be run
# as a script (`python -m nids_preprocessing.pipeline ...`); importing it
# eagerly at package-init time causes Python to warn that the module was
# "found in sys.modules ... prior to execution of ... this may result in
# unpredictable behaviour" when run with `-m`. Import it explicitly
# (`from nids_preprocessing import pipeline`) if you need it as a library.

__all__ = ["cleaning", "sample", "schema", "splitting"]
