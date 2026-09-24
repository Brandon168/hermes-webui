"""Model-manager policy; private inventory never enters the picker cache."""
import copy
import threading
from api import config as c

_lock = threading.RLock()


def identity(model, provider):
    """Return the exact canonical identity used for grouping and mutations.

    CommandCode serves bare multi-vendor wire ids while the gateway metadata
    feed (and Vercel rows) namespace them (``openai/gpt-…``,
    ``anthropic/claude-…``). The alias nests the same model under one card;
    anything else stays exact.
    """
    key = c._model_storage_id(model)
    if '/' not in key:
        if provider == 'openai-codex':
            return 'openai/' + key
        if provider in {
            'commandcode',
            'commandcode-chat',
            'commandcode-anthropic',
            'commandcode-claude',
        }:
            lowered = key.lower()
            if lowered.startswith('gpt-'):
                return 'openai/' + key
            if lowered.startswith('claude-'):
                return 'anthropic/' + key
    return key


def _override_state(overrides, key, provider_id):
    """Return the explicit show/hide state for one provider.

    Canonical-first with a bare-id fallback so visibility intent stored
    before the vendor alias existed keeps applying after it.
    """
    state = (overrides.get(key) or {}).get(provider_id)
    if state in {'show', 'hide'}:
        return state
    bare = key.split('/', 1)[1] if '/' in key else key
    if bare != key:
        state = (overrides.get(bare.lower()) or {}).get(provider_id)
        if state in {'show', 'hide'}:
            return state
    return None


def _snapshot(refresh=False):
    fresh = not refresh
    if refresh:
        try:
            c.refresh_gateway_models_meta()
            fresh = True
        except Exception:
            fresh = False
    # This explicit argument takes a synchronous private build path, with no
    # shared function mutation and no picker cache reads/publications.
    raw = c.get_available_models(include_hidden=True)
    stored = c._read_raw_settings_file()
    meta = c._load_gateway_models_meta()
    groups = copy.deepcopy(raw.get('groups') or [])
    by_provider = {g['provider_id']: g for g in groups if g.get('provider_id')}
    # Keep absent routes editable, but don't claim these are live discoveries.
    for pid, ids in (stored.get('models_hidden') or {}).items():
        if not isinstance(ids, list):
            continue
        group = by_provider.get(pid)
        if group is None:
            group = {'provider_id': pid, 'provider': pid + ' (not currently discovered)', 'models': [], 'available': False}
            groups.append(group)
            by_provider[pid] = group
        seen = {c._model_visibility_key(m.get('id')) for bucket in ('models', 'extra_models') for m in group.get(bucket, [])}
        for mid in ids:
            if c._model_visibility_key(mid) not in seen:
                group.setdefault('models', []).append({'id': mid, 'label': c._model_storage_id(mid), 'available': False})
                seen.add(c._model_visibility_key(mid))
    return groups, stored, meta, fresh


def _global_hidden_match(key, global_ids):
    """True when a global-hide entry covers *key* (canonical or bare)."""
    if key in global_ids:
        return True
    bare = key.split('/', 1)[1] if '/' in key else key
    return bare != key and bare in global_ids


def _materialize(groups, stored, meta):
    hidden = copy.deepcopy(stored.get('models_hidden') or {})
    if not isinstance(hidden, dict):
        raise ValueError('Legacy hidden-list settings must be migrated before using the manager')
    global_ids = {x.lower() for x in c._models_global_hidden_ids(stored)}
    overrides = c._models_provider_overrides(stored)
    for g in groups:
        pid = g['provider_id']
        arr = list(hidden.get(pid, []))
        for bucket in ('models', 'extra_models'):
            for row in g.get(bucket, []):
                mid = row['id']
                key = identity(mid, pid).lower()
                info = c._manager_meta_for(mid, meta, pid) or {}
                state = _override_state(overrides, key, pid)
                policy = True if info.get('type') in c._NON_CHAT_GATEWAY_TYPES else (
                    state == 'hide' if state else (True if _global_hidden_match(key, global_ids) else None))
                if policy is None:
                    continue  # untouched legacy selections remain untouched
                arr = [x for x in arr if not c._model_ids_match(x, mid)]
                if policy:
                    arr.append(c._model_storage_id(mid))
        if arr or pid in hidden:
            hidden[pid] = arr
    return hidden


def _response(groups, stored, meta, fresh):
    out = []
    for group in groups:
        pid = group['provider_id']
        hidden = c._effective_model_exclusions_for_provider(pid, stored)
        rows = []
        seen = set()
        for bucket in ('models', 'extra_models'):
            for m in group.get(bucket, []):
                mid = m['id']
                bare = c._model_storage_id(mid)
                if bare in seen:
                    continue
                seen.add(bare)
                info = c._manager_meta_for(mid, meta, pid)
                rows.append({'id': mid, 'canonical_model_id': identity(mid, pid),
                             'label': bare, 'hidden': any(c._model_ids_match(mid, x) for x in hidden),
                             'meta': info, 'available': m.get('available', group.get('available', True))})
        out.append({'provider_id': pid, 'provider': group.get('provider', pid), 'models': rows})
    return {'groups': out, 'hidden_map': stored.get('models_hidden', {}),
            'global_hidden': c._models_global_hidden_ids(stored),
            'provider_overrides': c._models_provider_overrides(stored),
            'meta_count': len(meta), 'meta_fresh': fresh}


def catalog(refresh=False):
    with _lock:
        groups, stored, meta, fresh = _snapshot(refresh)
        hidden = _materialize(groups, stored, meta)
        if hidden != stored.get('models_hidden', {}):
            c.save_settings({'models_hidden': hidden})
            stored['models_hidden'] = hidden
            c.invalidate_models_cache()
        return _response(groups, stored, meta, fresh)


def _targets_for(target):
    """Return the identity spellings one model_id can match (bare/canonical)."""
    lowered = target.lower()
    out = {lowered}
    bare = lowered.split('/', 1)[1] if '/' in lowered else lowered
    out.add(bare)
    out.add(f"openai/{bare}")
    out.add(f"anthropic/{bare}")
    return out


def _row_matches_target(mid, pid, targets):
    if identity(mid, pid).lower() in targets:
        return True
    return c._model_storage_id(mid).lower() in targets


def mutate(body):
    if not isinstance(body.get('model_id'), str) or not body['model_id'].strip() or type(body.get('hidden')) is not bool:
        raise ValueError('model_id and boolean hidden are required')
    if body.get('provider_id') is not None and not isinstance(body['provider_id'], str):
        raise ValueError('provider_id must be a string')
    with _lock:
        groups, stored, meta, fresh = _snapshot(False)
        target = c._model_storage_id(body['model_id'])
        pid = c._stored_provider_id(body.get('provider_id'))
        targets = _targets_for(target)
        matches = [(g, m) for g in groups for bucket in ('models', 'extra_models') for m in g.get(bucket, [])
                   if _row_matches_target(m['id'], g['provider_id'], targets)
                   and (not pid or g['provider_id'] == pid)]
        if not matches:
            raise ValueError('Model/provider not in the manager inventory')
        # Store provider overrides under the canonical spelling so the card's
        # router entries (bare commandcode id, namespaced vercel id) share it.
        canonical = identity(target, pid).lower() if pid else target.lower()
        global_ids = c._models_global_hidden_ids(stored)
        overrides = c._models_provider_overrides(stored)
        hidden = copy.deepcopy(stored.get('models_hidden') or {})
        if pid:
            overrides.setdefault(canonical, {})[pid] = 'hide' if body['hidden'] else 'show'
            # Drop stale bare spellings now covered by the canonical key.
            bare = canonical.split('/', 1)[1] if '/' in canonical else canonical
            if bare != canonical and bare in overrides and pid in overrides[bare]:
                del overrides[bare][pid]
                if not overrides[bare]:
                    del overrides[bare]
        else:
            global_ids = [x for x in global_ids if x.lower() not in targets]
            if body['hidden']:
                global_ids.append(canonical)
            for key in list(overrides):
                if key in targets:
                    del overrides[key]
            # Global show clears exact identity in absent stored providers too.
            for provider, arr in hidden.items():
                hidden[provider] = [x for x in arr if c._model_storage_id(x).lower() not in targets]
            for g, m in matches:
                if body['hidden']:
                    hidden.setdefault(g['provider_id'], []).append(c._model_storage_id(m['id']))
        stored.update(models_hidden=hidden, models_global_hidden=global_ids, models_provider_overrides=overrides)
        stored['models_hidden'] = _materialize(groups, stored, meta)
        c.save_settings({k: stored[k] for k in ('models_hidden', 'models_global_hidden', 'models_provider_overrides')})
        c.invalidate_models_cache()
        return _response(groups, stored, meta, fresh)
