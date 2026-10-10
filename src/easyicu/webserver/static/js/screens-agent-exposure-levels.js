/* Owner: a run's exposure groups, named in the study's words wherever a
   reader shows a grouped value.

   A study can group a concept's value into named levels; the analysis models
   each level by a code (1, 2, 3), and its tables, estimates and plan carry
   the codes.  The host projects the run's groupings (`exposure_groups` of
   run_context.json and of the plan review summary, owned by
   pi_copilot/exposure_group_notes.py): status `shown` with each grouping's
   variable, concept names and levels (code, the study's label), or status
   `unavailable` when the grouping record cannot be read.

   `reader(source, tr)` answers, for one grouped variable, its reader name
   ("乳酸分组" / "Lactate groups") and the label of one of its codes.  A
   value is named only when the record labels that variable's code; anything
   else keeps its own text, so no code is given a meaning here.  An
   unreadable record names nothing and says why (`unreadableText`).
   `tableRows` names the grouped cells of one result table: a column that is
   a grouped variable, the level columns of a row whose exposure is one, and
   the group columns of a Table 1 grouped by one (`group_by`), and the
   exposure cell that names a grouped variable.  `inText`
   names a grouped variable where plan prose quotes it as a whole identifier;
   the plan's own wording stays where the plan reader shows it. */
(function () {
  'use strict';

  const CODE = /^-?(?:0|[1-9]\d*)(?:\.0+)?$/;
  const LEVEL_COLUMNS = ['exposure_level', 'reference_level'];
  const ROW_VARIABLE_COLUMNS = ['exposure', 'exposure_column'];
  const GROUP_COLUMNS = ['group', 'reference_group', 'comparison_group'];
  const CONTRAST = /^\s*(\S+)\s+vs\.?\s+(\S+)\s*$/i;
  // The run artifacts whose readers name grouped values or columns; a viewer
  // of one loads the run's context beside it.
  const NAMED_IN = new Set([
    'agent_plan.json', 'result_tables.json', 'question_requirements.json', 'question_requirements_review.json',
  ]);

  function unreadableText(tr) {
    return tr(
      'The grouping record cannot be read, so the groups are shown by their codes and what each code means cannot be checked.',
      '分组记录读不出，分组按编码显示，各编码的含义无法核对。',
    );
  }

  function reader(source, tr) {
    const record = source && typeof source === 'object' && source.exposure_groups
      && typeof source.exposure_groups === 'object' ? source.exposure_groups : null;
    const unreadable = !!record && record.status === 'unavailable';
    const groupings = new Map();
    (record && record.status === 'shown' && Array.isArray(record.groupings) ? record.groupings : [])
      .filter(row => row && typeof row.variable === 'string' && row.variable && Array.isArray(row.levels))
      .forEach(row => groupings.set(row.variable, row));
    const grouping = variable => groupings.get(String(variable == null ? '' : variable).trim()) || null;

    function variableName(variable) {
      const row = grouping(variable);
      if (!row) return '';
      const concept = String(row.concept || row.variable);
      const en = String(row.concept_label_en || concept);
      const zh = String(row.concept_label_zh || en);
      return tr(`${en} groups`, `${zh}分组`);
    }

    function levelName(variable, value) {
      const row = grouping(variable);
      const text = String(value == null ? '' : value).trim();
      if (!row || row.status !== 'labelled' || !CODE.test(text)) return '';
      const level = row.levels.find(item => item && item.code === Number(text));
      return level && typeof level.label === 'string' ? level.label : '';
    }

    // "2 vs 1" names both sides, or neither.
    function contrastName(variable, value) {
      const match = CONTRAST.exec(String(value == null ? '' : value));
      if (!match) return '';
      const left = levelName(variable, match[1]);
      const right = levelName(variable, match[2]);
      return left && right ? `${left} vs ${right}` : '';
    }

    // Only a whole identifier: `lact_max_group_n` is another column.
    function inText(text) {
      let named = String(text == null ? '' : text);
      groupings.forEach((_row, variable) => {
        const quoted = variable.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        named = named.replace(new RegExp(`(?<![A-Za-z0-9_])${quoted}(?![A-Za-z0-9_])`, 'g'), () => variableName(variable));
      });
      return named;
    }

    function tableRows(table, headers, rows) {
      const names = (Array.isArray(headers) ? headers : []).map(String);
      const index = column => names.indexOf(column);
      const groupBy = table && typeof table.group_by === 'string' && grouping(table.group_by) ? table.group_by : '';
      const rowVariable = ROW_VARIABLE_COLUMNS.map(index).find(position => position >= 0);
      return (Array.isArray(rows) ? rows : []).map(row => {
        if (!Array.isArray(row)) return row;
        const variable = rowVariable != null && rowVariable >= 0 ? row[rowVariable] : '';
        return row.map((cell, position) => {
          const header = names[position];
          let name = '';
          if (grouping(header)) name = levelName(header, cell);
          else if (LEVEL_COLUMNS.includes(header) && grouping(variable)) name = levelName(variable, cell);
          else if (header === 'contrast' && grouping(variable)) name = contrastName(variable, cell);
          else if (ROW_VARIABLE_COLUMNS.includes(header)) name = variableName(cell);
          else if (GROUP_COLUMNS.includes(header) && groupBy) name = levelName(groupBy, cell);
          return name || cell;
        });
      });
    }

    return {
      unreadable,
      has: variable => !!grouping(variable),
      names: () => Object.fromEntries([...groupings.keys()].map(variable => [variable, variableName(variable)])),
      variableName,
      levelName,
      contrastName,
      inText,
      tableRows,
      unreadableText: () => (unreadable ? unreadableText(tr) : ''),
    };
  }

  window.AGENT_EXPOSURE_LEVELS = { reader, unreadableText, namedIn: name => NAMED_IN.has(String(name || '')) };
})();
