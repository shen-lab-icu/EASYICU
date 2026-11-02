"""Real metadata-only registry route over synthetic registered/discovered exports."""
from fastapi.testclient import TestClient
from easyicu.webserver import sources
from easyicu.webserver.app import app
from tests.support.typed_export import typed_export
from tests.support.workspace_registry_no_read import forbid_source_table_reads


def test_registry_registered_and_autodiscovered_typed_exports_do_not_open_tables(tmp_path, monkeypatch):
    export_root = tmp_path / 'exports'
    export_root.mkdir()
    registered = typed_export(export_root / 'registered')
    discovered = typed_export(export_root / 'discovered')
    monkeypatch.setattr(sources, '_CONFIG_DIR', tmp_path / 'cfg')
    monkeypatch.setattr(sources, '_CONFIG_PATH', tmp_path / 'cfg' / 'sources.json')
    monkeypatch.setattr(sources.settings_store, 'load_settings', lambda: {'export_dir': str(export_root)})
    client = TestClient(app)
    response = client.post('/api/workspaces/register', json={'path':str(registered)})
    assert response.status_code == 200
    with monkeypatch.context() as guarded:
        accesses = forbid_source_table_reads(guarded, roots=[registered, discovered])
        response = client.get('/api/workspaces/registry')
    assert accesses == []
    assert response.status_code == 200
    rows = {item['path']: item for item in response.json()['sources']}
    assert rows[str(registered)]['ok'] is True
    assert rows[str(discovered)]['ok'] is True
    assert rows[str(registered)]['file_count'] == 2
    assert rows[str(discovered)]['file_count'] == 2
