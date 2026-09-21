# Third-party notices

## ricu clinical configuration

EasyICU includes modified clinical configuration derived from
[ricu](https://github.com/eth-mds/ricu), an R package for working with ICU
datasets.

- Upstream project: `eth-mds/ricu`
- Upstream license: GNU General Public License, version 3 (`GPL-3.0-only`)
- Upstream authors listed by ricu: Nicolas Bennett, Drago Plecko, and Ida-Fong
  Ukor
- EasyICU files containing adapted material:
  `src/easyicu/data/concept-dict.json` and
  `src/easyicu/data/data-sources.json`
- Nature of the changes: the configuration was ported for EasyICU's Python
  runtime, expanded with additional concepts and data-source support, and
  subsequently modified for database-specific mappings, bounds, callbacks,
  clinical contracts, and extraction behavior.
- Modification period: 2026, with the current repository history recording
  the individual changes.

The complete GPL version 3 text is provided in [COPYING](COPYING). EasyICU is
not affiliated with or endorsed by the ricu authors or ETH Zurich.

The EasyICU-authored portions were originally released under the MIT License;
the original notice is retained in [LICENSES/MIT.txt](LICENSES/MIT.txt). The
combined EasyICU distribution is conveyed under `GPL-3.0-only` because it
contains the modified ricu-derived configuration described above.
