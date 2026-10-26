/* Owner: display names of the ICU databases EasyICU reads. */
/* Every screen names a database through this owner. The keys are the backend
   profile keys; aliases are other spellings a scan, study context or older
   record may carry; a "<key>_demo" key is that database's public demo. */
(function () {
  'use strict';

  const DATABASES = [
    { key: 'miiv', name: 'MIMIC-IV', abbr: 'M-IV', release: 'MIMIC-IV 3.1', aliases: ['mimiciv', 'mimic_iv'] },
    { key: 'eicu', name: 'eICU-CRD', abbr: 'eICU', release: 'eICU 2.0', aliases: ['eicu_crd'] },
    { key: 'aumc', name: 'AmsterdamUMCdb', abbr: 'AUMC', release: 'AmsterdamUMCdb', aliases: ['amsterdamumcdb'] },
    { key: 'hirid', name: 'HiRID', abbr: 'HiRID', release: 'HiRID 1.1.1', aliases: [] },
    { key: 'mimic', name: 'MIMIC-III', abbr: 'M-III', release: 'MIMIC-III 1.4', aliases: ['miii', 'mimiciii', 'mimic_iii'] },
    { key: 'sic', name: 'SICdb', abbr: 'SIC', release: 'SICdb 1.0.6', aliases: ['sicdb'] },
  ];
  const BY_SPELLING = new Map();
  DATABASES.forEach(row => [row.key, ...row.aliases].forEach(spelling => BY_SPELLING.set(spelling, row)));

  function parse(value) {
    const raw = String(value == null ? '' : value).trim().toLowerCase();
    const demo = /_demo$/.test(raw);
    const row = BY_SPELLING.get(demo ? raw.replace(/_demo$/, '') : raw);
    return row ? { row, demo } : null;
  }

  window.EU_DATABASE_NAMES = Object.freeze({
    /** Profile keys in display order. */
    keys: () => DATABASES.map(row => row.key),
    /** The profile key for any known spelling ('eicu_demo' gives 'eicu'), or ''. */
    key(value) { const hit = parse(value); return hit ? hit.row.key : ''; },
    isDemo(value) { const hit = parse(value); return Boolean(hit && hit.demo); },
    /** "MIMIC-IV", "eICU-CRD Demo"; '' for a key this owner does not know. */
    name(value) { const hit = parse(value); return hit ? `${hit.row.name}${hit.demo ? ' Demo' : ''}` : ''; },
    /** Compact form for chips and narrow columns: "M-IV", "eICU". */
    abbr(value) { const hit = parse(value); return hit ? hit.row.abbr : ''; },
    /** The release an extraction expects: "MIMIC-IV 3.1". */
    release(value) { const hit = parse(value); return hit ? hit.row.release : ''; },
  });
})();
