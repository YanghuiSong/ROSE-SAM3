"""Evaluate the recorded clean-response evidence gates before residual DAG search."""
import json

import torch


def run_search(cache_dir, pool_path, candidates_path, classes_path, output_dir, prompt_out):
    spec = json.loads(classes_path.read_text())
    classes, native = spec['classes'], spec['native_bank']
    candidates = json.loads(candidates_path.read_text())
    candidates = candidates.get('candidate_pool', candidates)
    count = len(json.loads(pool_path.read_text())['images'])
    phrases = list(dict.fromkeys(p for c in classes for p in native[c] + candidates.get(c, [])))
    values = {p: {'fused': [], 'presence': []} for p in phrases}
    for index in range(count):
        path = cache_dir / f'{index:04d}.pt'
        if not path.is_file():
            path = cache_dir / f'{index:05d}.pt'
        record = torch.load(path, map_location='cpu', weights_only=False)
        for phrase in phrases:
            response = record['clean'][phrase]
            values[phrase]['fused'].append(response['fused'].float()[::8, ::8])
            values[phrase]['presence'].append(response['presence'].float())
    response = {p: {k: torch.stack(v) for k, v in branches.items()} for p, branches in values.items()}
    out_bank = {c: list(native[c]) for c in classes}
    reports = []
    for c in classes:
        base = torch.stack([response[p]['fused'] for p in native[c]]).amax(0)
        other = torch.stack([torch.stack([response[p]['fused'] for p in native[d]]).amax(0)
                             for d in classes if d != c], 1).amax(1)
        rows = []
        for phrase in candidates.get(c, []):
            if phrase in native[c]:
                continue
            x = response[phrase]['fused']
            presence = float(response[phrase]['presence'].mean().item())
            strength = float(x.amax((-1, -2)).mean().item())
            denominator = (base > .5).float().sum().item() + 1e-6
            preserve = float((((base > .5) & (x >= base - .02)).float().sum().item()) / denominator)
            incremental_mass = float((x - base).clamp_min(0).mean().item())
            conflict = float(((x > base + .25) & (other > .5)).float().mean().item())
            score = .40 * presence + .25 * strength + .25 * preserve + .10 * min(incremental_mass / .05, 1.) - .75 * conflict
            safe = bool(preserve >= .90 and incremental_mass <= .05 and conflict <= .02 and presence >= .50)
            rows.append({'candidate': phrase, 'score': score, 'presence': presence,
                         'strength': strength, 'native_preservation': preserve,
                         'incremental_mass': incremental_mass, 'cross_class_conflict': conflict, 'safe': safe})
        rows.sort(key=lambda row: (-row['safe'], -row['score'], row['candidate']))
        selected = rows[0] if rows and rows[0]['safe'] else None
        if selected:
            out_bank[c].append(selected['candidate'])
        reports.append({'class': c, 'selected': selected, 'candidates': rows})
    bank = [out_bank[c] for c in classes]
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_out.write_bytes(('\n'.join(','.join(row) for row in bank) + '\n').encode())
    summary = {'method': 'recorded clean-response evidence gates',
               'protocol': {'gt_free': True, 'training_free': True, 'clean_only': True, 'metric_stride': 8},
               'calibration_image_count': count, 'class_reports': reports,
               'final_bank': bank, 'final_query_count': sum(map(len, bank)), 'outcome': 'selection_completed'}
    (output_dir / 'search_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
