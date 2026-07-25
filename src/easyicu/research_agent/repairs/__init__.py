"""Science-neutral repair transport, routing, and source-mutation helpers.

The package intentionally performs no eager imports.  Individual repair
modules have different authority and dependency boundaries, and callers must
name the boundary they consume rather than loading the whole repair surface.
"""
