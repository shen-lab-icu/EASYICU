"""Typed research-context construction and prompt projection.

The package is intentionally lazy; callers import ``builder``, ``typed`` or
``prompt_scope`` directly so context construction does not pull orchestration
or provider code into a schema-only consumer.
"""
