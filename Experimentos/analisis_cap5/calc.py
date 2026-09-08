import pandas as pd, numpy as np, itertools, os

def _cm(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    return (int(((y==1)&(p==1)).sum()),int(((y==0)&(p==1)).sum()),
            int(((y==1)&(p==0)).sum()),int(((y==0)&(p==0)).sum()))
def accuracy_score(y,p):
    tp,fp,fn,tn=_cm(y,p); return (tp+tn)/(tp+fp+fn+tn)
def precision_score(y,p,pos_label=1,zero_division=0):
    tp,fp,fn,tn=_cm(y,p)
    if pos_label==0: tp,fp=tn,fn
    return tp/(tp+fp) if tp+fp else zero_division
def recall_score(y,p,pos_label=1,zero_division=0):
    tp,fp,fn,tn=_cm(y,p)
    if pos_label==0: tp,fn=tn,fp
    return tp/(tp+fn) if tp+fn else zero_division
def _f1(y,p,pl,zd):
    pr=precision_score(y,p,pl,zd); rc=recall_score(y,p,pl,zd)
    return 2*pr*rc/(pr+rc) if pr+rc else zd
def f1_score(y,p,pos_label=1,average=None,zero_division=0):
    if average=='macro': return (_f1(y,p,0,zero_division)+_f1(y,p,1,zero_division))/2
    return _f1(y,p,pos_label,zero_division)
def cohen_kappa_score(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int); n=len(y)
    po=(y==p).mean()
    pe=sum((y==k).mean()*(p==k).mean() for k in (0,1))
    return (po-pe)/(1-pe) if pe!=1 else float('nan')


BASE='/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos/experiments/experimento_21_agentskills'
CL='/home/jggomez/Desktop/IRIS/iris-uc3m/CLUSTER/experimento_21_cluster'
OUT='/tmp/claude-3073/-home-jggomez-Desktop-IRIS-iris-uc3m/4c1285f9-0db2-43d4-9389-d70665385d87/scratchpad'
V=['lenguaje_sexista','masc_generico','sexismo_discurso','asimetria_mujer_hombre','denominacion_sexualizada']
CODE={'lenguaje_sexista':'V25','masc_generico':'V26','sexismo_discurso':'V30','asimetria_mujer_hombre':'V33','denominacion_sexualizada':'V35'}

def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',','.',regex=False).str.strip(), errors='coerce')

def binar(s):
    x=num(s)
    return x.map(lambda v: 0 if v==1 else (1 if v in (2,3) else np.nan))

gt=pd.read_csv(f'{BASE}/real1315_corpus.csv',low_memory=False)
G=pd.DataFrame({'IdNoticia':gt.IdNoticia})
for v in V: G[v]=binar(gt[v])
n0=len(G); G=G.dropna(); 
print('GT filas:',n0,'-> validas en las 5 variables:',len(G),'descartadas:',n0-len(G))
IDS=set(G.IdNoticia)
G=G.set_index('IdNoticia')

# --- carga de predicciones ---
API=['gemini-3.1-flash-lite','gpt-4o-mini','gpt-5.4-nano']
sources={}   # (modelo, config) -> DataFrame idx IdNoticia, cols V (0/1/nan)
notes=[]
def load(path, model, config, errsuffix=False):
    if not os.path.exists(path):
        notes.append(f'FALTA {path}'); return
    d=pd.read_csv(path,low_memory=False)
    nraw=len(d)
    d=d[d.IdNoticia.isin(IDS)].drop_duplicates('IdNoticia').set_index('IdNoticia')
    out=pd.DataFrame(index=d.index)
    fails={}
    for v in V:
        c='modelo_'+v
        if c not in d.columns: out[v]=np.nan; fails[v]='col ausente'; continue
        b=binar(d[c])
        ec=v+'_error'
        if ec in d.columns:
            b=b.where(d[ec].isna())
        out[v]=b
        fails[v]=int(b.isna().sum())
    # Las respuestas que el modelo no llego a emitir en formato valido, y las piezas
    # que no aparecen en el fichero, se contabilizan como negativas. Asi todas las
    # metricas del capitulo se calculan sobre las mismas 1.313 piezas.
    out=out.reindex(sorted(IDS)).fillna(0)
    sources[(model,config)]=out
    notes.append(f'{model}|{config}: filas fichero={nraw}, usadas={len(out)}, faltan_vs_1313={1313-len(d)}, nulas/fallidas por var={fails}')

for m in API:
    load(f'{BASE}/results/b0_{m}/exp21_{m}.csv',m,'B0')
    load(f'{BASE}/results/bench_{m}/exp21_{m}.csv',m,'B1')
    for a,tag in [('abl_minimo','abl_minimo'),('abl_singuia','abl_singuia'),('abl_sinres','abl_sinres')]:
        load(f'{BASE}/results/{a}_{m}/exp21_{m}.csv',m,tag)
load(f'{CL}/results_b0/FULL_dedup.csv','gemma4:e4b','B0')
load(f'{CL}/results_b1_completo/FULL.csv','gemma4:e4b','B1')
load(f'{CL}/results_b1_minimo/FULL.csv','gemma4:e4b','abl_minimo')
load(f'{CL}/results_b1_singuia/FULL.csv','gemma4:e4b','abl_singuia')
load(f'{CL}/results_b1_sinres/FULL.csv','gemma4:e4b','abl_sinres')

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    return dict(n=len(y),
        prev_real=y.mean(), prev_pred=p.mean(),
        exactitud=accuracy_score(y,p),
        precision=precision_score(y,p,pos_label=1,zero_division=0),
        recall=recall_score(y,p,pos_label=1,zero_division=0),
        f1_pos=f1_score(y,p,pos_label=1,zero_division=0),
        f1_macro=f1_score(y,p,average='macro',zero_division=0),
        kappa=cohen_kappa_score(y,p),
        TP=int(((y==1)&(p==1)).sum()),FP=int(((y==0)&(p==1)).sum()),
        FN=int(((y==1)&(p==0)).sum()),TN=int(((y==0)&(p==0)).sum()))

rows=[]
for (m,cfg),df in sources.items():
    for v in V:
        s=df[v].dropna()
        idx=s.index.intersection(G.index)
        y=G.loc[idx,v]; p=s.loc[idx]
        r=metrics(y,p); r.update(modelo=m,configuracion=cfg,variable=v,codigo=CODE[v],
                                 n_descartadas_pred=int(len(G)-len(idx)))
        rows.append(r)
R=pd.DataFrame(rows)
cols=['codigo','variable','modelo','configuracion','n','prev_real','prev_pred','exactitud','precision','recall','f1_pos','f1_macro','kappa','TP','FP','FN','TN']
R=R[cols].sort_values(['codigo','configuracion','modelo'])
(R[R.configuracion.isin(['B0','B1'])].rename(columns={'configuracion':'nivel'})
   .to_csv(f'{OUT}/metricas_por_variable.csv',index=False))
R[R.configuracion.isin(['B1','abl_minimo','abl_singuia','abl_sinres'])].to_csv(f'{OUT}/ablacion_por_variable.csv',index=False)
R.to_csv(f'{OUT}/_todas_metricas.csv',index=False)

# --- kappa entre modelos ---
kr=[]
MODELS=API+['gemma4:e4b']
for lvl in ['B0','B1']:
    for a,b in itertools.combinations(MODELS,2):
        if (a,lvl) not in sources or (b,lvl) not in sources: continue
        for v in V:
            sa=sources[(a,lvl)][v].dropna(); sb=sources[(b,lvl)][v].dropna()
            idx=sa.index.intersection(sb.index)
            x=sa.loc[idx].astype(int); y=sb.loc[idx].astype(int)
            kr.append(dict(codigo=CODE[v],variable=v,nivel=lvl,modelo_a=a,modelo_b=b,n=len(idx),
                           acuerdo=(x.values==y.values).mean(),
                           kappa=cohen_kappa_score(x,y)))
K=pd.DataFrame(kr); K.to_csv(f'{OUT}/kappa_entre_modelos_por_variable.csv',index=False)

# --- voto mayoria B1 ---
vr=[]
sub={m:sources[(m,'B1')] for m in MODELS}
tie_info={}
for v in V:
    mat=pd.DataFrame({m:sub[m][v] for m in MODELS})
    mat=mat.reindex(G.index)
    valid=mat.notna().sum(axis=1)
    pos=mat.fillna(0).sum(axis=1)
    keep=valid>0
    frac=pos[keep]/valid[keep]
    ties=int((frac==0.5).sum()); tie_info[v]=(ties,int(keep.sum()))
    for regla,val in [('empate=Si',(frac>=0.5)),('empate=No',(frac>0.5))]:
        p=val.astype(int); y=G.loc[p.index,v]
        r=metrics(y,p); r.update(variable=v,codigo=CODE[v],modelo='voto_mayoria_4',
                                 configuracion='B1',regla_empate=regla,n_empates=ties)
        vr.append(r)
VT=pd.DataFrame(vr)
best=R[(R.configuracion=='B1')].sort_values('f1_pos').groupby('variable').tail(1)
for _,b in best.iterrows():
    d=b.to_dict(); d['regla_empate']='-'; d['n_empates']=np.nan
    d['modelo']='MEJOR INDIVIDUAL: '+b['modelo']
    vr.append(d)
VT=pd.DataFrame(vr)
c2=['codigo','variable','modelo','regla_empate','n','n_empates','prev_real','prev_pred','exactitud','precision','recall','f1_pos','f1_macro','kappa','TP','FP','FN','TN']
VT=VT[c2].sort_values(['codigo','modelo'])
VT.to_csv(f'{OUT}/voto_mayoria.csv',index=False)
# --- equipos de anotacion (nivel B1) ---
# El equipo se deduce del campo no_NombreUsuario del corpus anotado.
equipo=(gt.set_index('IdNoticia')['no_NombreUsuario'].astype(str)
          .str.contains('ndexa',case=False).map({True:'Indexa',False:'UCM3'}))
er=[]
for v in V:
    for e in ['Indexa','UCM3']:
        idx=[i for i in G.index if equipo.get(i)==e]
        y=G.loc[idx,v].astype(int)
        row=dict(codigo=CODE[v],equipo=e,n=len(idx),prev=y.mean())
        for m in MODELS:
            p_=sources[(m,'B1')].loc[idx,v].astype(int)
            row['recall_'+m]=recall_score(y,p_,pos_label=1,zero_division=0)
            row['kappa_'+m]=cohen_kappa_score(y,p_)
        er.append(row)
E=pd.DataFrame(er)
E=E[['codigo','equipo','n','prev']+[f'{k}_{m}' for m in MODELS for k in ('recall','kappa')]]
E.to_csv(f'{OUT}/equipos_por_variable.csv',index=False)

print('\n'.join(notes))
print('EMPATES por variable:',tie_info)
