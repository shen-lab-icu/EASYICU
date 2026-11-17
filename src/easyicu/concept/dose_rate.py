"""Active dose-rate intervals and their declared time-weighted hourly summary.

Source owners convert units and supply exact ICU-relative spans, the same
representation underlying the native minute rates. Sweep all sources together:
concurrent infusions add before integration, and the denominator is the union
of active time. No sampling or rounding of interval endpoints is needed.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import pandas as pd

DOSE_RATE_CONCEPTS = frozenset({
    'norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate', 'phn_rate',
    'angiotensin_ii_rate', 'milrinone_rate', 'enoximone_rate', 'propofol_rate',
    'midazolam_rate', 'fentanyl_rate', 'remifentanil_rate',
})
RATE_MAX_CONCEPTS = tuple(f'{name}_max' for name in sorted(DOSE_RATE_CONCEPTS)
                          if name not in {'propofol_rate', 'midazolam_rate',
                                          'fentanyl_rate', 'remifentanil_rate'})
START, END, POINT = '_dose_start', '_dose_end', '_dose_point'
CHANNEL = '_dose_channel'
HOURLY_BOUNDED = '_dose_hourly_bounds_checked'
SPAN_COLUMNS = (START, END, POINT, CHANNEL, HOURLY_BOUNDED)
# None: direct utility caller; False: legacy minute/*60 path; True: capture
# spans until all sources are combined; str: the direct source concept name.
_CAPTURE: ContextVar[bool | str | None] = ContextVar('easyicu_dose_spans', default=None)


def base_rate(name):
    return name[:-4] if name in RATE_MAX_CONCEPTS else name


def enabled(name, *, generic=False):
    from .minute_resolution import active_minute_rate_grid

    scoped = _CAPTURE.get()
    if scoped is not None:
        return scoped is not False
    return (base_rate(name) in DOSE_RATE_CONCEPTS or (generic and name in {'rate', 'value'})) and active_minute_rate_grid() is None


@contextmanager
def source_rate_scope(name, interval, *, capture=True):
    """Keep legacy minute dependencies (including vaso60) on their existing path."""
    hourly = interval is None or pd.to_timedelta(interval) >= pd.Timedelta(hours=1)
    token = _CAPTURE.set((True if capture else name) if base_rate(name) in DOSE_RATE_CONCEPTS and hourly else False)
    try:
        yield
    finally:
        _CAPTURE.reset(token)


def _finish(frame, ids, index, value, scale):
    if _CAPTURE.get() is True:
        return frame
    return collapse_spans(frame, ids=ids, index=index, value=value, native_scale=scale,
                          maximum_rate=_CAPTURE.get() in RATE_MAX_CONCEPTS)


def _channel_keys(frame, column):
    if not column or column not in frame:
        return ''
    # Integral ids can arrive as either Arrow integers or nullable floats.
    return frame[column].astype('string').str.replace(r'^(-?\d+)\.0+$', r'\1', regex=True).fillna('')


def interval_rates(frame, *, ids, index, value, starts, ends, native_scale=1., channel=None):
    """Publish exact spans or summarise them for a direct utility caller.

    ``starts``/``ends`` are ICU-relative hours; ``index`` stays on its source
    clock until normal resolver alignment. ``native_scale`` is source units/h.
    """
    out = frame.copy()
    out[START] = np.asarray(starts, dtype=float)
    out[END] = np.asarray(ends, dtype=float)
    out[POINT] = False
    out[CHANNEL] = _channel_keys(out, channel)
    return _finish(out, ids, index, value, native_scale)


def point_rates(frame, *, ids, index, value, hours, channel=None, native_scale=1., gap_carry_hours=None):
    """Carry settings to the next point within five hours, with a one-hour tail.

    Terminal positive settings retain the ricu one-hour overhang. An explicit
    gap carry cap preserves the source owner's coverage before a later setting
    across a long gap (eICU: six hours). Stops and invalid settings only close.
    Keep points until source concat so unit/label changes share one sequence.
    """
    out = frame.copy()
    out[START] = np.asarray(hours, dtype=float)
    out[END] = np.nan if gap_carry_hours is None else out[START] + gap_carry_hours
    out[POINT] = True
    channel = channel or next((c for c in ('infusionid', 'linkorderid', 'orderid') if c in out), None)
    out[CHANNEL] = _channel_keys(out, channel)
    out[value] = pd.to_numeric(out[value], errors='coerce').fillna(0.)
    return _finish(out, ids, index, value, native_scale)


def _point_flags(frame):
    # An object column of True/False/None would make ``~`` a bitwise integer
    # negation under pandas 3; the flag is always read as a strict boolean.
    return frame[POINT].fillna(False).astype(bool)


def _point_spans(frame, ids, value):
    point = _point_flags(frame)
    points = frame.loc[point].copy()
    if not points.empty:
        keys = [*ids, CHANNEL]
        # At an exact tie, stop wins; otherwise use the greatest setting.
        points['_stop'] = points[value].le(0)
        points = points.sort_values([*keys, START, '_stop', value], kind='stable')
        points = points.drop_duplicates([*keys, START], keep='last')
        nxt = points.groupby(keys, dropna=False)[START].shift(-1)
        gap = nxt - points[START]
        tail = points[START] + 1.
        carry = points[END].notna() & nxt.notna()
        tail = tail.where(~carry, np.minimum(nxt, points[END]))
        points[END] = nxt.where(gap.gt(0) & gap.le(5.), tail)
    return pd.concat([frame.loc[~point], points], ignore_index=True)


def _supersede_intervals(frame, ids, value):
    identified = ~_point_flags(frame) & frame[CHANNEL].fillna('').ne('')
    intervals = frame.loc[identified].copy()
    if not intervals.empty:
        keys = [*ids, CHANNEL]
        intervals['_stop'] = intervals[value].le(0)
        intervals = intervals.sort_values([*keys, START, '_stop', value], kind='stable')
        intervals = intervals.drop_duplicates([*keys, START], keep='last')
        nxt = intervals.groupby(keys, dropna=False)[START].shift(-1)
        intervals[END] = np.minimum(intervals[END], nxt.fillna(np.inf))
    return pd.concat([frame.loc[~identified], intervals], ignore_index=True)


def _anchor_values(out, anchor, ids, column):
    if len(ids) == 1:
        return out[ids[0]].map(anchor[column])
    keys = pd.MultiIndex.from_frame(out[ids])
    return pd.Series(anchor[column].reindex(keys).to_numpy(), index=out.index)


def collapse_spans(frame, *, ids, index, value, native_scale=None, minimum=None, maximum=None, maximum_rate=False):
    """Sweep concurrent rates, then integrate over active time or take its max.

    Called after source concat. A numeric ``native_scale`` restores the
    source clock (e.g. eICU minutes) for normal episode alignment and clipping.
    Frames without span metadata belong to another extraction path.
    """
    if START not in frame:
        return frame
    units = [c for c in ('rateuom', 'doseunit', 'unit', 'doserateunit') if c in frame]
    columns = [*ids, index, value, *units]
    if frame.empty:
        return frame.reindex(columns=columns)
    anchor = frame.loc[frame[index].notna() & np.isfinite(frame[START])].drop_duplicates(ids).set_index(ids)
    work = _supersede_intervals(_point_spans(frame, ids, value), ids, value)
    valid = np.isfinite(work[START]) & np.isfinite(work[END]) & work[END].gt(work[START])
    valid &= pd.to_numeric(work[value], errors='coerce').gt(0)
    bounded = work[HOURLY_BOUNDED].eq(True) if HOURLY_BOUNDED in work else pd.Series(False, index=work.index)
    if minimum is not None:
        valid &= bounded | work[value].ge(minimum)
    if maximum is not None:
        valid &= bounded | work[value].le(maximum)
    work = work.loc[valid]
    if work.empty:
        return frame[columns].iloc[:0].copy()
    opening = work[[*ids, START, value]].rename(columns={START: '_t', value: '_delta'})
    opening['_count'] = 1
    closing = work[[*ids, END, value]].rename(columns={END: '_t', value: '_delta'})
    closing['_delta'] *= -1
    closing['_count'] = -1
    events = pd.concat([opening, closing], ignore_index=True)
    events = events.sort_values([*ids, '_t', '_delta', '_count'], kind='stable')
    events = events.groupby([*ids, '_t'], as_index=False, dropna=False)[['_delta', '_count']].sum()
    events = events.sort_values([*ids, '_t'])
    groups = events.groupby(ids, dropna=False, sort=False)
    events['_rate'] = groups['_delta'].cumsum()
    events['_active'] = groups['_count'].cumsum()
    events['_end'] = groups['_t'].shift(-1)
    segments = events.loc[events['_active'].gt(0) & events['_end'].gt(events['_t'])]
    starts = segments['_t'].to_numpy()
    ends = segments['_end'].to_numpy()
    counts = (np.ceil(ends) - np.floor(starts)).astype(int)
    rows = np.repeat(np.arange(len(segments)), counts)
    offset = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    hours = np.floor(starts)[rows] + offset
    overlap = np.minimum(ends[rows], hours + 1) - np.maximum(starts[rows], hours)
    parts = segments.iloc[rows][ids].reset_index(drop=True)
    parts[index] = hours
    parts['_dt'] = overlap
    parts['_rate'] = segments['_rate'].to_numpy()[rows]
    parts['_area'] = parts['_rate'] * overlap
    grouped = parts.groupby([*ids, index], as_index=False, dropna=False)
    if value in RATE_MAX_CONCEPTS or maximum_rate:
        out = grouped['_rate'].max().rename(columns={'_rate': value})
    else:
        out = grouped[['_area', '_dt']].sum()
        out[value] = out.pop('_area') / out.pop('_dt')
    for unit in units:
        out[unit] = _anchor_values(out, anchor, ids, unit)
    if HOURLY_BOUNDED in work and work[HOURLY_BOUNDED].eq(True).all():
        # These spans have already been clipped to the source's accepted
        # legacy hourly grid. Bounding their new active-time mean again would
        # change coverage solely because its denominator changed.
        out[HOURLY_BOUNDED] = True
        columns.append(HOURLY_BOUNDED)
    if native_scale is not None:
        original = _anchor_values(out, anchor, ids, index)
        origin_h = _anchor_values(out, anchor, ids, START)
        delta_h = out[index] - origin_h
        if pd.api.types.is_datetime64_any_dtype(original):
            out[index] = original + pd.to_timedelta(np.rint(delta_h * 3.6e12).astype('int64'), unit='ns')
        else:
            origin = original - origin_h * native_scale
            origin = origin.where(origin.abs().gt(1e-9), 0.)
            out[index] = origin + out[index] * native_scale
    out.attrs.update(frame.attrs)
    return out[columns]


def datetime_hours(frame, index, ids, admission_times):
    """Exact relative hours for MIMIC point settings, without absolute flooring."""
    if pd.api.types.is_numeric_dtype(frame[index]):
        return pd.to_numeric(frame[index], errors='coerce')
    origins = pd.Timestamp('1970-01-01', tz='UTC')
    if admission_times is not None:
        lookup = admission_times.drop_duplicates(ids[0]).set_index(ids[0])['intime']
        lookup = pd.to_datetime(lookup, errors='coerce', utc=True)
        # Reindex rather than Series.map: pandas 3 maps through a float cast that
        # a datetime lookup cannot take.
        origins = lookup.reindex(frame[ids[0]].to_numpy())
        origins.index = frame.index
    return (pd.to_datetime(frame[index], errors='coerce', utc=True) - origins).dt.total_seconds() / 3600.


def _hirid_hourly_bounds(frame, *, id_col, index, value, output, weights, minimum, maximum):
    """Keep dev9's hourly mass/weight bound mask before changing denominators."""
    from ..utils.callback_utils import _hirid_apportion_hourly

    legacy = _hirid_apportion_hourly(frame, index, value)
    legacy = legacy.groupby([id_col, '_time_bin'], as_index=False)[value].sum()
    legacy[value] /= 60. * legacy[id_col].map(weights)
    valid = legacy[value].gt(0)
    if minimum is not None:
        valid &= legacy[value].ge(minimum)
    if maximum is not None:
        valid &= legacy[value].le(maximum)
    from ..source_bounds import record_read_counts, scope_active

    if scope_active():
        record_read_counts(database='hirid', concept=output, table='pharma', stays=legacy[id_col].unique(),
                           frame=pd.DataFrame({'stay': legacy[id_col],
                                               'raw': legacy[value].notna().astype('int64'),
                                               'bounded': valid.astype('int64')}),
                           id_column='stay', raw_column='raw', bounded_column='bounded')
    return legacy.loc[valid, [id_col, '_time_bin']].rename(columns={'_time_bin': '_dose_hour'})


def _clip_hirid_to_bounded_hours(frame, allowed, *, id_col, index):
    """Clip exact spans to accepted hours, retaining their active rate."""
    valid = np.isfinite(frame[index]) & np.isfinite(frame[END]) & frame[END].gt(frame[index])
    frame = frame.loc[valid]
    starts, ends = frame[index].to_numpy(), frame[END].to_numpy()
    counts = (np.ceil(ends) - np.floor(starts)).astype(int)
    rows = np.repeat(np.arange(len(frame)), counts)
    offset = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    hours = np.floor(starts)[rows] + offset
    out = frame.iloc[rows].reset_index(drop=True).copy()
    out[index] = np.maximum(starts[rows], hours)
    out[END] = np.minimum(ends[rows], hours + 1.)
    out['_dose_hour'] = hours
    out = out.merge(allowed, on=[id_col, '_dose_hour'], how='inner', validate='many_to_one')
    out[HOURLY_BOUNDED] = True
    return out.drop(columns='_dose_hour')


def hirid_amount_spans(frame, *, id_col, index, value, output, unit_col, per_kg=False, output_unit=None, minimum=None, maximum=None):
    """Reuse HiRID's preceding-interval dose representation (DEV-RICUC-04).

    Untimed amounts keep the established one-hour-bin fallback; known spans
    carry amount / exact elapsed minutes, rather than amount / occupied hour.
    """
    from ..utils.callback_utils import HIRID_APPORTION_MAX_HOURS

    out = frame.reset_index(drop=True).copy()
    end = pd.to_numeric(out[index], errors='coerce')
    start = pd.to_numeric(out.get('_prev_t', pd.Series(np.nan, index=out.index)), errors='coerce')
    length = end - start
    timed = length.gt(0) & length.le(HIRID_APPORTION_MAX_HOURS)
    start = start.where(timed, np.floor(end))
    end = end.where(timed, np.floor(end) + 1.)
    allowed = None
    if per_kg:
        weights = out.dropna(subset=['weight']).groupby(id_col)['weight'].first()
        if minimum is not None or maximum is not None:
            allowed = _hirid_hourly_bounds(out, id_col=id_col, index=index, value=value,
                                          output=output, weights=weights, minimum=minimum, maximum=maximum)
    out[output] = out[value] / ((end - start) * 60.)
    if per_kg:
        # Preserve the existing first usable stay weight in this callback.
        out[output] /= out[id_col].map(weights)
    if unit_col:
        out[unit_col] = output_unit or ('mcg/kg/min' if per_kg else f'{out[unit_col].iloc[0]}/min')
    out[index] = start
    if allowed is not None:
        out[END] = end
        out = _clip_hirid_to_bounded_hours(out, allowed, id_col=id_col, index=index)
        start, end = out[index], out[END]
    else:
        if minimum is not None:
            out = out.loc[out[output].ge(minimum)]
        if maximum is not None:
            out = out.loc[out[output].le(maximum)]
    return interval_rates(out, ids=[id_col], index=index, value=output, starts=start.loc[out.index], ends=end.loc[out.index])


def resolver_summary(frame, ids, index, value, definition, *, database=''):
    """Keep all resolver entry points free of internal span columns."""
    relative = database in {'mimic', 'mimic_demo'} and index in frame and pd.api.types.is_numeric_dtype(frame[index])
    return collapse_spans(
        frame, ids=ids, index=index, value=value,
        native_scale=None if relative else (60. if database in {'aumc', 'eicu', 'eicu_demo'} else 1.),
        minimum=definition.minimum, maximum=definition.maximum,
    )
