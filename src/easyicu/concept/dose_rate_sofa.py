"""SOFA cardiovascular worst doses with the existing >=60-minute gate."""
from __future__ import annotations

from ..table import ICUTable


def cardio_max_tables(tables, ctx):
    """Use max companions for dose, and *60 solely for qualification.

    Both SOFA generations keep their duration, threshold and support rules.
    A missing maximum leaves the existing qualification dose available; zero
    and negative qualification values never become active doses.
    Direct legacy callback callers without companions keep their old interface;
    the shipped dictionaries always declare all four max dependencies.
    """
    out = dict(tables)
    if ctx.concept_name not in {'sofa_cardio', 'sofa2_cardio'}:
        return out
    for drug in ('dopa', 'dobu', 'epi', 'norepi'):
        name, companion = f'{drug}60', f'{drug}_rate_max'
        maximum = out.pop(companion, None)
        qualified = out.get(name)
        if maximum is None or qualified is None:
            continue
        if qualified.data.empty:
            continue
        ids = list(qualified.id_columns)
        index = qualified.index_column
        keys = [*ids, index]
        if maximum.data.empty:
            continue
        else:
            dose = maximum.data.rename(columns={maximum.index_column: index})
            frame = qualified.data[keys + [name]].merge(dose[keys + [companion]], on=keys, how='left')
            frame[name] = frame[companion].where(frame[name].gt(0) & frame[companion].notna(), frame[name])
        out[name] = ICUTable(frame[keys + [name]], id_columns=ids, index_column=index, value_column=name)
    return out
