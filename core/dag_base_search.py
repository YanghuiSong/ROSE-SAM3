#!/usr/bin/env python3
"""Select a base PromptBank using concept DAG metrics and proposal consensus."""
from __future__ import annotations
import csv, itertools, json, math
from pathlib import Path
from collections import defaultdict
import torch
from core.concept_dag import build_finite_concept_dag
from core.concept_search_metrics import load_response_cache, class_maps, teacher, native_core, evaluate_candidate, prepare_native_partition, ConceptMetrics
from core.pareto_concept_search import best_first_search, OBJECTIVES

CLASSES = []
PROPOSAL_MARGIN = 0.005

def read_partitions(path):
    x=json.loads(Path(path).read_text()); return x.get('partitions', x)

def path_count(dag, roots):
    memo={}
    def f(n):
        if n in memo:return memo[n]
        ch=dag.edges.get(n,[]); memo[n]=1 if not ch else sum(f(c) for c in ch)
        return memo[n]
    return sum(f(r) for r in roots)

def subset(x, idx): return {k:v[idx] for k,v in x.items()}

def protection_for_bank(native_t, native_f, native_b, native_bf, pred_c, pred_f):
    out={}; refs={}
    for j,cls in enumerate(CLASSES):
        core,w=native_core(native_t,native_f,native_b,native_bf,j)
        union=(native_t['pred']==j)|(native_f['pred']==j)
        den=(w*union.float()).sum().item()+1e-6
        refs[cls]=float((w*union.float()*core.float()).sum().item()/den)
        kept=(pred_c['pred']==j)&(pred_f['pred']==j)
        out[cls]=float((w*union.float()*kept.float()).sum().item()/den)
    return out,refs

def global_bank_eval(response, bank, native_bank, idx, quality):
    nc=class_maps(response,native_bank,'clean'); nf=class_maps(response,native_bank,'flip_aligned')
    nc=[subset(x,idx) for x in nc]; nf=[subset(x,idx) for x in nf]
    nt, nff=teacher(nc), teacher(nf)
    nb={k:torch.stack([x[k] for x in nc],1) for k in ('semantic','instance')}; nbf={k:torch.stack([x[k] for x in nf],1) for k in ('semantic','instance')}
    bc=class_maps(response,bank,'clean'); bf=class_maps(response,bank,'flip_aligned')
    bc=[subset(x,idx) for x in bc]; bf=[subset(x,idx) for x in bf]
    st,sf=teacher(bc),teacher(bf)
    prot,refs=protection_for_bank(nt,nff,nb,nbf,st,sf)
    rescue_total=drop_total=0.0
    for j in range(len(CLASSES)):
        core,w=native_core(nt,nff,nb,nbf,j); stable=core
        rescue=((nt['pred']!=j)|(nff['pred']!=j))&(st['pred']==j)&(sf['pred']==j)
        drop=stable&~((st['pred']==j)&(sf['pred']==j))
        rescue_total += float((w*rescue.float()).mean().item()); drop_total += float((w*drop.float()).mean().item())
    safe=all(prot[c]>=refs[c]-1e-6 for c in CLASSES)
    return {'bank':[list(x) for x in bank], 'queries':sum(map(len,bank)), 'protection':prot, 'native_reference':refs,
            'min_protection':min(prot.values()), 'stable_rescue':rescue_total, 'class_drop':drop_total,
            'global_net_gain':rescue_total-drop_total, 'safe':bool(safe),
            'mean_quality':float(sum(quality.get((i,tuple(bank[i])),0.0) for i in range(len(CLASSES)))/len(CLASSES)),
            'modified_classes':sum(tuple(bank[i])!=tuple(native_bank[i]) for i in range(len(CLASSES)))}

def run_search(settings):
    """Evaluate the recorded finite concept and complete-bank candidates."""
    a = settings
    global CLASSES
    spec=json.loads(Path(a.classes_json).read_text()); CLASSES=list(spec.get('classes',list(spec['native_bank']))); NATIVE={k:list(v) for k,v in spec['native_bank'].items()}
    import core.concept_search_metrics as _csm; _csm.CLASSES=CLASSES
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pool=json.loads(Path(a.pool).read_text()); parts=read_partitions(a.partitions); candidates=json.loads(Path(a.candidates).read_text())
    if 'candidate_pool' in candidates: candidates=candidates['candidate_pool']
    prompts,response=load_response_cache(
        a.cache_dir, len(pool['images'])
    )
    missing=sorted(set(sum(candidates.values(),[]))-set(prompts))
    if missing: raise RuntimeError(f'cache misses require SAM3 evaluation; refusing implicit rerun: {missing[:5]}')
    dag=build_finite_concept_dag(candidates,CLASSES); roots=[dag._key(c,candidates[c][0]) for c in CLASSES]
    # Per-partition node metrics.  All tensors remain frozen CPU responses.
    per_node=defaultdict(list); partition_reports=[]
    native_bank=[NATIVE[c] for c in CLASSES]
    for pname,ids in parts.items():
        idx=list(ids)
        prepared = prepare_native_partition(response, native_bank, idx)
        safe=set()
        for nid,node in list(dag.nodes.items()):
            if node.phrase not in response: continue
            if node.class_id==0:
                m=ConceptMetrics(1,1,1,1,{c:1.0 for c in CLASSES[1:]},1.0,safety_pass=True,cache_hit=True)
            else:
                m=evaluate_candidate(response,native_bank,node.phrase,node.class_id,prepared)
            per_node[nid].append(m.to_dict())
            if node.relation=='native' or m.safety_pass: safe.add(nid)
        partition_reports.append({'partition':pname,'safe_nodes':len(safe),'unsafe_nodes':len(dag.nodes)-len(safe)})
    # Aggregate metrics and 4/5 robustness.
    metrics_by_node={}; safe_global=set()
    for nid,node in dag.nodes.items():
        rows=per_node.get(nid,[])
        if not rows: continue
        passed=sum(bool(r['safety_pass']) or node.relation=='native' for r in rows)
        agg={k:float(sum(r[k] for r in rows)/len(rows)) for k in ('pseudo_reference_fidelity','domain_matchability','visual_concept_agreement','native_core_preservation','min_competitor_protection','own_gain','non_target_damage')}
        agg['competitor_protection']={c:float(sum(r['competitor_protection'].get(c,0) for r in rows)/len(rows)) for c in CLASSES if c!=node.class_name}
        agg['partition_robustness']=passed/len(parts); agg['safety_pass']=passed>=4; agg['cache_hit']=True
        node.metrics=agg; node.status='safe' if agg['safety_pass'] else ('native' if node.relation=='native' else 'unsafe')
        metrics_by_node[nid]=agg
        if agg['safety_pass'] or node.relation=='native': safe_global.add(nid)
    result=best_first_search(dag,metrics_by_node,safe_global,roots)
    for ar in result.archives.values():
        for n in ar.nodes: n.status='pareto'
    # Build class-local global options from safe Pareto concepts; Native is always present.
    options=[]; quality={}
    for ci,c in enumerate(CLASSES):
        native=tuple(native_bank[ci]); opts=[native]
        if ci>0:
            for n in result.archives[c].as_list():
                action=(n.phrase,)
                if action not in opts and sum(len(x) for x in native_bank)-len(native)+1 <= 11: opts.append(action)
                quality[(ci,action)]=sum(float(n.metrics.get(k,0)) for k in OBJECTIVES)/len(OBJECTIVES)
        # Keep the exact Cartesian search tractable.  If a class archive is
        # large, retain Native plus its highest Pareto-quality state;
        # the discarded states remain in the class archive and are reported.
        if len(opts) > 3:
            ranked_opts = sorted(opts[1:], key=lambda z: (-quality.get((ci, z), 0.0), tuple(z)))
            opts = [opts[0]] + ranked_opts[:1]
        options.append(opts)
    cart=math.prod(len(x) for x in options)
    base_bank_candidates=[]
    raw_candidates=json.loads(Path(a.base_bank_candidates).read_text())
    base_bank_candidates=raw_candidates.get('banks',raw_candidates)
    global_reports=[]; winners=[]
    for pname,ids in parts.items():
        idx=torch.as_tensor(ids,dtype=torch.long); ranked=[]
        combos=itertools.product(*options) if cart<=100000 else itertools.product(*[x[:min(len(x),4)] for x in options])
        for combo in combos:
            item=global_bank_eval(response,list(combo),native_bank,idx,quality)
            if item['safe']: ranked.append(item)
        ranked.sort(key=lambda z:(-z['min_protection'],-z['stable_rescue'],z['class_drop'],-z['mean_quality'],z['queries'],z['modified_classes'],json.dumps(z['bank'])))
        proposal_ranked=[]
        for proposal in base_bank_candidates:
            bank=[tuple(x) for x in proposal]
            if len(bank)!=len(CLASSES) or sum(map(len,bank))>a.query_budget: continue
            item=global_bank_eval(response,bank,native_bank,idx,quality); item['source']='base_bank_proposal'; proposal_ranked.append(item)
            item['strict_safe']=item['safe']
            if PROPOSAL_MARGIN>0.0:
                # Apply the configured protection and net-gain tolerances.
                item['soft_safe']=bool(
                    all(item['protection'][c] >= item['native_reference'][c]-PROPOSAL_MARGIN-1e-6 for c in CLASSES)
                    and item['global_net_gain'] >= -PROPOSAL_MARGIN
                )
                item['safe']=bool(item['strict_safe'] or item['soft_safe'])
        all_ranked=ranked+proposal_ranked
        all_ranked.sort(key=lambda z:(not z['safe'],-z['min_protection'],-z['stable_rescue'],z['class_drop'],-z['mean_quality'],z['queries'],z['modified_classes'],json.dumps(z['bank'])))
        best=all_ranked[0] if all_ranked else global_bank_eval(response,native_bank,native_bank,idx,quality)
        winners.append(best['bank']); global_reports.append({'partition':pname,'cartesian_size':cart,'evaluated_global_banks':len(ranked),'evaluated_base_bank_proposals':len(proposal_ranked),'winner':best,'top_banks':ranked[:20],'base_bank_proposals':proposal_ranked})
    counts={json.dumps(w,ensure_ascii=False):sum(x==w for x in winners) for w in winners}; native_serial=[list(x) for x in native_bank]
    # Select complete-bank proposals using the configured partition consensus.
    proposal_consensus=[]
    for proposal_index,proposal in enumerate(base_bank_candidates):
        key=json.dumps([[str(x) for x in row] for row in proposal],ensure_ascii=False)
        passed=sum(1 for report in global_reports for item in report.get('base_bank_proposals',[]) if json.dumps(item['bank'],ensure_ascii=False)==key and item.get('soft_safe',item.get('safe',False)))
        proposal_consensus.append((passed,proposal_index))
    best_proposal=max(proposal_consensus,default=(0,-1))
    if best_proposal[0]>=a.proposal_min_consensus:
        final=[[str(x) for x in row] for row in base_bank_candidates[best_proposal[1]]]
    else:
        key=max(counts,key=counts.get); final=json.loads(key) if counts[key]>=4 else native_serial
    prompt=Path(a.prompt_out); prompt.parent.mkdir(parents=True,exist_ok=True); prompt.write_bytes(('\n'.join(','.join(x) for x in final)+'\n').encode('utf-8'))
    summary={'method':'Pareto-Guided Competition-Aware Concept DAG Search','protocol':{'training_free':True,'gt_free':True,'labels_used':False,'no_tta':True,'offline_flip_probe_only':True,'validation_used':a.calibration_split=='val','validation_images_used':a.calibration_split=='val','validation_labels_used':False,'calibration_split':a.calibration_split,'fixed_partitions':list(parts),'metric_stride':4},
      'raw_tree_paths':path_count(dag,roots),'unique_dag_nodes':len(dag.nodes),
      'unsafe_pruned':sum(1 for n in dag.nodes.values() if n.status=='unsafe'),
      'pareto_pruned':sum(1 for n in dag.nodes.values() if 'pareto_dominated' in n.prune_reasons),'bound_pruned':len(result.bound_pruned),
      'expanded_nodes':len(result.expanded),'branch_pruned':len(result.branch_pruned),'cached_evaluations':len(set(prompts)),'actual_new_sam3_evaluations':0,
      'cache_hit_ratio':1.0,'flat_exhaustive_states':len(dag.nodes),'tree_baseline_paths':path_count(dag,roots),'dag_unique_states':len(dag.nodes),
      'search_diagnostics':{'flat':len(dag.nodes),'tree':path_count(dag,roots),'dag_unique_states':len(dag.nodes),'pareto_archive_states':sum(len(x.nodes) for x in result.archives.values()),'expanded_states':len(result.expanded)},
      'pareto_archives':{c:[n.phrase for n in result.archives[c].as_list()] for c in CLASSES},'partition_reports':partition_reports,
      'global_promptbank':{'cartesian_size':cart,'base_bank_candidates':base_bank_candidates,'query_budget':a.query_budget,'proposal_soft_margin':PROPOSAL_MARGIN,'proposal_min_consensus':a.proposal_min_consensus,'proposal_consensus':proposal_consensus,'partition_reports':global_reports,'winner_counts':counts,'final_bank':final,'final_query_count':sum(map(len,final)),'fallback':'Native unless a complete base-bank proposal reaches the configured unlabeled consensus'},
      'outcome':'selection_completed'}
    (out/'search_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); (out/'concept_dag.json').write_text(json.dumps(dag.serialize(),indent=2,ensure_ascii=False))
    with (out/'search_trace.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['step','class','node_id','phrase','depth','relation','action','priority']+list(OBJECTIVES)+['prune_reason']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for step,nid in enumerate(result.expanded):
            n=dag.nodes[nid]; row={'step':step,'class':n.class_name,'node_id':nid,'phrase':n.phrase,'depth':n.depth,'relation':n.relation,'action':'expand','priority':0,'prune_reason':';'.join(n.prune_reasons)}; row.update({k:n.metrics.get(k,0) for k in OBJECTIVES}); w.writerow(row)
    print(json.dumps({'raw_tree_paths':summary['raw_tree_paths'],'unique_dag_nodes':summary['unique_dag_nodes'],'unsafe_pruned':summary['unsafe_pruned'],'pareto_pruned':summary['pareto_pruned'],'bound_pruned':summary['bound_pruned'],'expanded_nodes':summary['expanded_nodes'],'final_bank':final,'final_query_count':summary['global_promptbank']['final_query_count'],'outcome':'selection_completed'},indent=2,ensure_ascii=False))
