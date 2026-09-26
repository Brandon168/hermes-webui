import copy
import time
import pytest
from api import config as c
from api import model_manager as mm

@pytest.fixture
def env(tmp_path, monkeypatch):
    path=tmp_path/'settings.json'; path.write_text('{}')
    monkeypatch.setattr(c,'SETTINGS_FILE',path)
    monkeypatch.setattr(c,'invalidate_models_cache',lambda:None)
    groups=[{'provider_id':p,'provider':p,'models':[{'id':x,'label':x} for x in ['vendor/model','vendor/model-fast','vendor/new','vendor/image']]} for p in ['one','two']]
    def inventory(**kwargs):
        assert kwargs == {'include_hidden':True}
        return {'groups':copy.deepcopy(groups)}
    monkeypatch.setattr(c,'get_available_models',inventory)
    c.save_settings({'models_hidden':{'absent':['old/id'],'one':['vendor/model-fast']},'models_meta':{'vendor/image':{'type':'image'},'vendor/model':{'type':'language','released':123}},'models_meta_refreshed_at':int(time.time()),'models_meta_refresh_attempted_at':int(time.time())})
    return groups

def test_global_override_new_provider_and_preservation(env):
    original=c._model_excluded_ids_for_provider
    r=mm.mutate({'model_id':'vendor/model','hidden':True})
    assert r['global_hidden']==['vendor/model']
    assert c._model_excluded_ids_for_provider is original
    assert 'old/id' in r['hidden_map']['absent']
    assert 'vendor/model-fast' in r['hidden_map']['one']
    mm.mutate({'model_id':'vendor/model','provider_id':'two','hidden':False})
    env.append({'provider_id':'three','models':[{'id':'vendor/model'},{'id':'vendor/new'}]})
    r=mm.catalog()
    assert 'vendor/model' in r['hidden_map']['three']
    assert 'vendor/model' not in r['hidden_map']['two']
    assert 'vendor/new' not in r['hidden_map']['three']
    r=mm.mutate({'model_id':'vendor/model','hidden':False})
    assert not r['global_hidden'] and not r['provider_overrides']
    assert all('vendor/model' not in a for a in r['hidden_map'].values())
    assert 'vendor/model-fast' in r['hidden_map']['one']

def test_nonchat_and_legacy_hidden_inventory(env):
    r=mm.catalog()
    assert r['global_hidden']==[]
    assert 'vendor/image' in r['hidden_map']['one']
    one=next(g for g in r['groups'] if g['provider_id']=='one')
    assert next(m for m in one['models'] if m['id']=='vendor/model-fast')['hidden']
    absent=next(g for g in r['groups'] if g['provider_id']=='absent')
    assert absent['models'][0]['available'] is False

def test_refresh_failure_and_invalid_actions(env,monkeypatch):
    monkeypatch.setattr(c,'refresh_models_meta',lambda:(_ for _ in ()).throw(OSError('offline')))
    assert mm.catalog(refresh=True)['meta_fresh'] is False
    before=c._read_raw_settings_file()
    with pytest.raises(ValueError):mm.mutate({'model_id':'missing','hidden':True})
    with pytest.raises(ValueError):mm.mutate({'model_id':'vendor/model','hidden':'false'})
    assert c._read_raw_settings_file()==before

def test_commandcode_bare_ids_join_namespaced_meta_and_nest(tmp_path, monkeypatch):
    path = tmp_path / 'settings.json'
    path.write_text('{}')
    monkeypatch.setattr(c, 'SETTINGS_FILE', path)
    monkeypatch.setattr(c, 'invalidate_models_cache', lambda: None)
    groups = [
        {'provider_id': 'commandcode', 'provider': 'commandcode',
         'models': [{'id': 'gpt-6-luna', 'label': 'gpt-6-luna'},
                    {'id': 'claude-opus-5', 'label': 'claude-opus-5'},
                    {'id': 'xiaomi/mimo-v2.6-pro', 'label': 'mimo'}]},
        {'provider_id': 'vercel-brandon-pro', 'provider': 'vercel',
         'models': [{'id': 'openai/gpt-6-luna', 'label': 'luna'}]},
    ]
    monkeypatch.setattr(c, 'get_available_models', lambda **kwargs: {'groups': copy.deepcopy(groups)})
    c.save_settings({'models_meta': {
        'openai/gpt-6-luna': {'type': 'language', 'released': 1790035200},
        'anthropic/claude-opus-5': {'type': 'language', 'released': 1790035200},
    }, 'models_meta_refreshed_at': int(time.time()), 'models_meta_refresh_attempted_at': int(time.time())})
    # Meta join: bare commandcode rows resolve through the vendor alias.
    _meta = c._load_gateway_models_meta()
    _luna = c._manager_meta_for('gpt-6-luna', _meta, 'commandcode')
    assert _luna is not None and _luna['released'] == 1790035200
    _opus = c._manager_meta_for('claude-opus-5', _meta, 'commandcode')
    assert _opus is not None and _opus['type'] == 'language'
    # No inheritance: variants and unrelated families stay unknown.
    assert c._manager_meta_for('gpt-6-luna-fast', _meta, 'commandcode') is None
    assert c._manager_meta_for('xiaomi/mimo-v2.6-pro', _meta, 'commandcode') is None
    # Grouping: bare + namespaced rows share one canonical identity.
    assert mm.identity('gpt-6-luna', 'commandcode') == 'openai/gpt-6-luna'
    assert mm.identity('claude-opus-5', 'commandcode') == 'anthropic/claude-opus-5'
    assert mm.identity('openai/gpt-6-luna', 'vercel-brandon-pro') == 'openai/gpt-6-luna'
    # Provider toggle accepts either spelling, stores canonical.
    r = mm.mutate({'model_id': 'gpt-6-luna', 'provider_id': 'commandcode', 'hidden': True})
    assert r['provider_overrides']['openai/gpt-6-luna']['commandcode'] == 'hide'
    cmd = next(g for g in r['groups'] if g['provider_id'] == 'commandcode')
    assert next(m for m in cmd['models'] if m['id'] == 'gpt-6-luna')['hidden'] is True
    assert next(m for m in cmd['models'] if m['id'] == 'gpt-6-luna')['meta']['type'] == 'language'
    # Stored bare intent keeps applying after canonicalization.
    c.save_settings({'models_provider_overrides': {'gpt-6-luna': {'commandcode': 'hide'}}})
    r = mm.catalog()
    cmd = next(g for g in r['groups'] if g['provider_id'] == 'commandcode')
    assert next(m for m in cmd['models'] if m['id'] == 'gpt-6-luna')['hidden'] is True


def test_private_inventory_dedup_does_not_filter(env):
    groups=copy.deepcopy(env)
    c._deduplicate_model_ids(groups,include_hidden=True)
    assert len(groups[0]['models'])==4
    assert len(groups[1]['models'])==4
    c._deduplicate_model_ids(groups)
    assert len(groups[0]['models'])==3
