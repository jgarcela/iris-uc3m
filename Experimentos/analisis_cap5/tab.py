import pandas as pd,numpy as np
OUT="/tmp/claude-3073/-home-jggomez-Desktop-IRIS-iris-uc3m/4c1285f9-0db2-43d4-9389-d70665385d87/scratchpad"
def f(x):
    if pd.isna(x): return '--'
    return f'{x:.3f}'.replace('.',',')
V=['lenguaje_sexista','masc_generico','sexismo_discurso','asimetria_mujer_hombre','denominacion_sexualizada']
CODE={'lenguaje_sexista':'V25','masc_generico':'V26','sexismo_discurso':'V30','asimetria_mujer_hombre':'V33','denominacion_sexualizada':'V35'}
R=pd.read_csv(f'{OUT}/_todas_metricas.csv')
def tab(df,cols,hdr):
    print('| '+' | '.join(hdr)+' |'); print('|'+'---|'*len(hdr))
    for _,r in df.iterrows():
        cells=[]
        for c in cols:
            v=r[c]
            cells.append(f(v) if isinstance(v,(float,np.floating)) else str(v))
        print('| '+' | '.join(cells)+' |')
    print()

print('## 1. B0 vs B1 por variable\n')
for v in V:
    d=R[(R.variable==v)&(R.configuracion.isin(['B0','B1']))].sort_values(['configuracion','modelo'])
    print(f'### {CODE[v]} — {v} (prevalencia real = {f(d.prev_real.iloc[0])})\n')
    tab(d,['modelo','configuracion','n','prev_pred','exactitud','precision','recall','f1_pos','f1_macro','kappa'],
        ['Modelo','Nivel','N','Prev. pred.','Exactitud','Precisión','Recall','F1 (Sí)','F1 macro','Kappa'])

print('\n## 2. Ablaciones (B1 completo = bench) por variable\n')
for v in V:
    d=R[(R.variable==v)&(R.configuracion.isin(['B1','abl_minimo','abl_singuia','abl_sinres']))].sort_values(['modelo','configuracion'])
    print(f'### {CODE[v]} — {v} (prevalencia real = {f(d.prev_real.iloc[0])})\n')
    tab(d,['modelo','configuracion','n','prev_pred','exactitud','precision','recall','f1_pos','f1_macro','kappa'],
        ['Modelo','Config.','N','Prev. pred.','Exactitud','Precisión','Recall','F1 (Sí)','F1 macro','Kappa'])

print('\n## 3. Kappa entre modelos por variable\n')
K=pd.read_csv(f'{OUT}/kappa_entre_modelos_por_variable.csv')
for lvl in ['B0','B1']:
    print(f'### Nivel {lvl}\n')
    for v in V:
        d=K[(K.variable==v)&(K.nivel==lvl)]
        print(f'**{CODE[v]} — {v}**\n')
        tab(d,['modelo_a','modelo_b','n','acuerdo_bruto','kappa'],['Modelo A','Modelo B','N','Acuerdo bruto','Kappa'])

print('\n## 4. Voto por mayoría (4 modelos, B1)\n')
VT=pd.read_csv(f'{OUT}/voto_mayoria.csv')
for v in V:
    d=VT[VT.variable==v]
    print(f'### {CODE[v]} — {v} (prevalencia real = {f(d.prev_real.iloc[0])})\n')
    tab(d,['modelo','regla_empate','n_empates','prev_pred','exactitud','precision','recall','f1_pos','f1_macro','kappa'],
        ['Sistema','Regla empate','Empates','Prev. pred.','Exactitud','Precisión','Recall','F1 (Sí)','F1 macro','Kappa'])
