"""Case-neutral scientific planning registries and display strategy.

The package stays import-lazy.  Canonical consumers import the concrete
submodule they need so loading one registry does not pull the whole planning
surface (or the pipeline) into memory.
"""
