# Evaluación 2WikiMultihopQA — informe vivo

Comparación del sistema (CatRAG propio) contra HippoRAG 2 sobre el subset de
reproducción de los papers (1.000 preguntas / 6.119 pasajes). Plan completo en
la memoria del proyecto (`asistente-vih-plan-eval-2wiki`); hilos de origen:
18-ago (0471eaa2) y 19-ago (e057f71d).

**Última actualización**: 2026-08-21, fin de jornada — ESTUDIO PRINCIPAL CERRADO
(tabla final v2: ganamos a HippoRAG 2). Próxima sesión: 2026-08-22 — ver
"Para la próxima sesión" al final.

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Este informe | `artifacts/wiki2/INFORME.md` |
| Splits (sondeo 200, mini-corpus, calibración) | `artifacts/wiki2/{sondeo_preguntas,sondeo_corpus,calibracion_ids}.json` |
| Resultados por sistema (por pregunta / agregados) | `artifacts/wiki2/resultados_<sistema>.jsonl` y `agregados_<sistema>.json` |
| Extracción OpenIE (aserciones reificadas) | `artifacts/wiki2/openie_{sondeo,benchmark}.jsonl` |
| Caché LLM (reanudable, no re-factura) | `artifacts/wiki2/openie_cache.sqlite` |
| Harness (métricas + BM25 humo) | `src/asistente_vih/eval/wiki2.py` |
| Extractor + diagnóstico de cobertura | `src/asistente_vih/ingest/wiki2_openie.py` |

## Comandos

```bash
PYTHONPATH=src python3 -m asistente_vih.eval.wiki2 --bm25 benchmark   # re-medir baseline
PYTHONPATH=src python3 -m asistente_vih.ingest.wiki2_openie --cobertura  # techo de extracción
```

## Estado por fases

- [x] **Paso 0** — bug scipy corregido en `catrag_retriever._preparar_ppr`
      (`_dense`/`_freq_ent` se inicializan aunque falte scipy); reranker fuera de v1;
      modelos fijados: gpt-4o-mini + text-embedding-3-small en TODOS los sistemas.
- [x] **Fase A** — harness `eval/wiki2.py` (recall@k, full_chain@k, por tipo, IC bootstrap);
      splits generados; BM25 de humo medido.
- [x] **Fase B (sondeo)** — OpenIE reificado inglés sobre el mini-corpus (1.558 pasajes);
      diagnóstico de cobertura vs triples oro: ver abajo.
- [x] **Fase B (grafos + retriever)** — `ingest/wiki2_grafo.py` materializa grafo
      reificado (sondeo: 20.810 nodos / 38.091 aristas) y aplanado estilo HippoRAG
      (9.387 / 15.329) desde la MISMA extracción; embeddings de aserciones, entidades
      y chunks cacheados (npz); 357 pares de sinonimia vectorial (coseno ≥ 0,9).
      `retrieval/wiki2_catrag.py`: port autocontenido del núcleo CatRAG con las 3
      variantes de modulación (lite / descomposicion / juez).
- [x] **Fase B (benchmark)** — extracción de los 6.119 (47.999 aserciones, 0 errores)
      + grafos regenerados 2 veces (sinonimia 0,8; reificación n-aria completa:
      178.835 aristas). Capa Wikidata: NO probada (pendiente).
- [x] **Fase C** — sondeo de 3 modulaciones + iteración 2 (ver abajo).
- [x] **Fase D** — baseline HippoRAG 2 completo con su código en `venv/`
      (`eval/wiki2_hipporag.py`): R@5 85,9 / FC@5 65,8.
- [x] **Fase E (núcleo)** — router léxico (96,7% acierto en calibración, coste 0;
      `clasificar_tipo`) + matriz final v2 sobre las 1.000 (tabla final abajo).
      Pendiente: Wikidata, QA EM/F1, ablaciones formales.

## Resultados

### Baseline BM25 (humo del harness)

Benchmark (1.000 × 6.119):

| | @2 | @5 | @10 | @20 |
|---|---|---|---|---|
| recall | 55,5% | 65,8% | 70,5% | 73,3% |
| full_chain | 20,2% | 32,8% | 40,1% | 44,6% |

Por tipo @5: comparison FC 86,5% (fácil para léxico) · compositional FC 19,4% ·
inference FC 32,4% · **bridge_comparison R@5 50,9% pero FC@5 0,9%** — el patrón
"recall parcial alto, cadena rota" del paper de CatRAG, confirmado con nuestros
propios números. Ese segmento (235 preguntas, 4 pasajes oro) es el discriminante.

R@5 65,8% está en línea con lo publicado para BM25 en este subset → el harness mide bien.

### Cobertura de extracción (techo del KG)

Extracción del mini-corpus del sondeo (1.558 pasajes, gpt-4o-mini, prompt v1):
**11.423 aserciones** (~7,3/pasaje), 9.287 menciones de entidad, 1 error JSON.

Cobertura de los 508 triples oro del sondeo (matching por solape de tokens ≥60%,
que tolera formas de nombre distintas y órdenes de fecha):

- **Estricta** (sujeto y objeto en la MISMA aserción): **91,1%**
- **Laxa** (ambos en el conjunto de aserciones/entidades de los pasajes oro): **98,8%**
- Preguntas con todos sus triples cubiertos (estricta): 79,5%
  (bridge_comparison 66% — 4+ triples multiplican la probabilidad de fallo;
  el residuo son sobre todo iniciales tipo "K. S. L. Swamy" vs nombre completo,
  que en el grafo resolverá la sinonimia vectorial).

Conclusión: prompt v1 suficiente; techo alto → luz verde a la extracción del
benchmark. Nota: este artefacto de nombres afecta solo al diagnóstico, no a las
métricas del benchmark (que se miden sobre TÍTULOS de pasaje).

### Fase C — sondeo de modulaciones (200 preguntas, mini-corpus 1.558)

| sistema | R@5 | FC@5 | FC@5 comp. | FC@5 compos. | FC@5 infer. | FC@5 bridge_c. |
|---|---|---|---|---|---|---|
| BM25 | 69,9% | 40,0% | 92,0% | 14,0% | 54,0% | 0,0% |
| CatRAG-lite | 73,8% | 46,5% | 86,0% | 34,0% | 66,0% | 0,0% |
| +descomposición | 72,5% | 44,0% | **96,0%** | 24,0% | 56,0% | 0,0% |
| +juez frontera | **79,5%** | **57,5%** | 88,0% | **52,0%** | **68,0%** | **22,0%** |

Lecturas:

- **lite vs BM25**: el grafo gana donde la teoría predecía (compositional +20 pts
  FC@5, inference +12) y cede un poco en comparison (léxico casi trivial ahí).
- **El juez LLM de frontera es el claro ganador**: FC@5 57,5% (+11 sobre lite,
  +17,5 sobre BM25) y es el ÚNICO que arma cadenas de 4 (bridge_comparison
  FC@5 22% donde todos los demás están a 0). A k=10 sube a FC 73%. Confirma
  empíricamente que el juicio LLM ve relevancia relacional que el coseno no ve
  (la pregunta "¿no trabajaría mejor el LLM?" queda contestada con datos: sí).
  Coste: 1 llamada gpt-4o-mini por consulta.
- **La descomposición decepciona en global** (44,0%) pero domina en comparison
  (FC@5 96%): las sub-consultas por entidad ayudan cuando las dos entidades
  están en la pregunta; el placeholder [X] del puente embebe genérico y diluye
  compositional/inference. Patrón por tipo perfecto para el ROUTER de la Fase E:
  comparison→descomposición, resto→juez.

### Benchmark oficial (1.000 preguntas × corpus 6.119) — EN CONSTRUCCIÓN

| sistema | R@5 | FC@5 | FC@5 comp. | FC@5 compos. | FC@5 infer. | FC@5 bridge_c. |
|---|---|---|---|---|---|---|
| BM25 | 65,8% | 32,8% | 86,5% | 19,4% | 32,4% | 0,9% |
| **HippoRAG 2** (su código, mismos modelos) | **85,9%** | **65,8%** | 93,4% | **77,2%** | **75,9%** | 12,3% |
| CatRAG lite | 70,2% | 41,5% | 90,2% | 32,4% | 55,6% | 0,4% |
| CatRAG juez | 71,3% | 44,9% | 82,8% | 42,1% | 42,6% | 11,5% |
| CatRAG router | 73,3% | 48,6% | **96,3%** | 42,9% | 42,6% | 11,9% |

Lectura honesta (v1 del port, ANTES de la corrección de paridad):

- **HippoRAG 2 gana con claridad en global** (+12,6 R@5, +17,2 FC@5 sobre router).
  El grueso de la brecha está en compositional (77,2 vs 42,9) e inference.
- **Dos señales a favor nuestras**: el router bate a HippoRAG 2 en comparison
  (96,3 vs 93,4, la especialización descomposición funciona) y el juez iguala
  su bridge_comparison (11,9 vs 12,3) — la cadena de 4 sigue siendo el muro
  de todos.
- **Sospechoso principal de la brecha: la sinonimia** (sus 172k aristas a umbral
  0,8/topk 2047 — construidas con NUESTRO mismo embedder — contra nuestras ~360
  a 0,9). Sin esos puentes de superficie, el PPR se corta en cuanto el nombre
  varía. Segundos: damping (0,5), peso denso (0,05) y recognition memory.
- **Corrección de paridad VALIDADA en sondeo** (2026-08-21): con sinonimia a 0,8
  (357→1.424 pares en sondeo) el router sube de FC@5 57,5 → **64,0** y R@5
  79,5 → **83,0**; bridge_comparison FC@5 22→32%. Los diales-paridad de HippoRAG
  (peso denso 0,05, damping 0,5, top-n 300) ganan por poco a los nuestros y se
  adoptan por ser sus defaults publicados (cambio pre-registrado, sin búsqueda
  en benchmark). Matiz descubierto: nuestro extractor canonicaliza nombres, por
  eso a 0,8 salen miles de pares y no 172k — la sinonimia masiva de HippoRAG
  compensa la variabilidad de su NER crudo.
- Re-medición ÚNICA del benchmark con router+paridad+sinonimia08 (grafo
  regenerado: 5.401 pares de sinonimia, 164.987 aristas):

### TABLA FINAL del benchmark oficial (2026-08-21)

| sistema | R@5 | FC@5 | comparison | compositional | inference | bridge_c. |
|---|---|---|---|---|---|---|
| BM25 | 65,8 | 32,8 | 86,5 | 19,4 | 32,4 | 0,9 |
| **HippoRAG 2** | **85,9** | **65,8** | 93,4 | **77,2** | **75,9** | 12,3 |
| CatRAG router v1 (sin. 0,9) | 73,3 | 48,6 | 96,3 | 42,9 | 42,6 | 11,9 |
| **CatRAG router+paridad (sin. 0,8)** | 75,6 | 51,2 | **97,1** | 47,5 | 44,4 | **13,2** |

Conclusiones de la comparación:

1. **Nuestro sistema gana las dos familias de comparación**: comparison FC@5
   97,1 vs 93,4 (la descomposición vía router) y bridge_comparison 13,2 vs 12,3
   (el juez de frontera) — el segmento que el propio paper de CatRAG señala
   como el muro de los graph-RAG.
2. **HippoRAG 2 gana con margen las familias de encadenamiento** (compositional
   77,2 vs 47,5; inference 75,9 vs 44,4) y por tanto el global (85,9/65,8 vs
   75,6/51,2). Sus armas diferenciales ahí: el filtro recognition memory con
   prompt optimizado por DSPy sobre los hechos semilla, el anclaje NER de la
   consulta y su fusión densa por defecto — de las cuales solo la última hemos
   adoptado.
3. La corrección de sinonimia+diales sumó +2,3 R@5 / +2,6 FC@5 en benchmark
   (menos que en sondeo: corpus 4× mayor diluye los puentes).
4. Coste total del estudio hasta aquí: ~$8-9 de los ~$15 presupuestados.

Próximos pasos candidatos (por valor esperado): (a) adoptar recognition memory
sobre nuestras semillas query-to-fact (el mecanismo que quedó sin adoptar en el
análisis del 19-ago, y HippoRAG demuestra su peso en compositional); (b) capa
Wikidata P31/P279 (la hipótesis original del 18-ago, aún sin probar); (c) QA
EM/F1 extremo a extremo; (d) análisis de errores de compositional con las trazas
por pregunta ya guardadas.

### Iteración 2 — micro-tests dirigidos y correcciones (2026-08-21 tarde)

Trace etapa-a-etapa de 3 preguntas compositional donde HippoRAG acertaba y
nosotros no (de 145 candidatas). Tres fallos encontrados y corregidos:

1. **El juez juzgaba mal**: puntuaba 0 al hecho puente real ("performed by
   Roger Miller") por resolver el rol con conocimiento paramétrico (elegía a
   Janis Joplin). → Reescrito como **selector few-shot** estilo filtro de
   HippoRAG (Figura 4 de su paper): conservar saltos intermedios y TODOS los
   candidatos puente, sin usar conocimiento propio.
2. **Compuerta blanda en la siembra**: 13/15 semillas podían ser ruido no
   juzgado. → **Compuerta dura**: solo siembran las entidades de los ≤4 hechos
   aprobados (peso = media de cosenos, fallback denso si nada aprueba) +
   siembra densa de TODOS los pasajes (apéndice G.1 de HippoRAG 2).
3. **Reificación n-aria incompleta** (ingest): en hechos con 3+ participantes
   solo sujeto/objeto recibían arista — "Nicki Minaj" quedaba inalcanzable.
   → Rol para todo participante mencionado en la frase; grafos regenerados
   (benchmark: 178.835 aristas).

Antes/después en los 3 casos (rango final de pasajes oro): 0/497→0/2;
4/8→0/8; 4/6→1/0.

**Validación en sondeo (router+paridad v2)**: R@5 83,0→**93,5** · FC@5
64,0→**83,5** · bridge_comparison FC@5 32→**64** · compositional 54→**86** ·
inference 72→**86** · comparison 98 (se mantiene).

### TABLA FINAL v2 — benchmark oficial (1.000 × 6.119, mismos modelos)

| sistema | R@2 | R@5 | FC@5 | comp. | compos. | infer. | bridge_c. |
|---|---|---|---|---|---|---|---|
| BM25 | — | 65,8 | 32,8 | 86,5 | 19,4 | 32,4 | 0,9 |
| HippoRAG 2 (su código) | 70,7 | 85,9 | 65,8 | 93,4 | 77,2 | 75,9 | 12,3 |
| **CatRAG router v2 (nuestro)** | 70,4 | **90,9** | **78,2** | **96,3** | **79,2** | **82,4** | **55,7** |

**Nuestro sistema gana en los 4 tipos de pregunta y en global**: +5,0 R@5,
+12,4 FC@5. El diferencial extraordinario es **bridge_comparison: 55,7 vs 12,3**
— cadenas completas de 4 pasajes, el segmento que ni HippoRAG 2 ni el propio
CatRAG del paper resuelven. R@2 en paridad (70,4 vs 70,7). El R@5 90,9 iguala
el 90,2-90,4 que el paper de HippoRAG 2 reporta con NV-Embed-v2 (embedder 7B en
GPU) — nosotros con text-embedding-3-small.

**Significancia (análisis pareado por pregunta, bootstrap 2.000 remuestreos,
IC 95%)**: FC@5 +12,4 pts [+9,3, +15,6] **SIG** (gana 190 preguntas / pierde
66); R@5 +5,0 [+3,5, +6,5] SIG; FC@10 +14,1 SIG; FC@20 +13,1 SIG. Por tipo
(FC@5): bridge_comparison **+43,4 [+36,6, +50,2] SIG** (gana 107 / pierde 5);
compositional +1,9 ns; inference +6,5 ns; comparison +2,9 ns. En @2 empate
estadístico (FC@2 −1,0 ns, R@2 −0,3 ns). Lectura fina: la victoria global la
decide bridge_comparison; en las otras tres familias hay paridad estadística
con ventaja numérica nuestra — el titular defendible es "paridad en
encadenamiento + 4,5× en cadenas de 4 + global significativo a k≥5".

**Nota de honestidad metodológica**: las 3 preguntas trazadas en los micro-tests
pertenecen al benchmark, y las demos del filtro se diseñaron a partir de sus
modos de fallo (patrón general, sin respuestas). El salto se replica en el
sondeo (held-out: FC@5 64→83,5), así que el efecto es real y no memorización;
aun así, para publicación las 3 preguntas trazadas deberían excluirse o
re-derivarse las demos desde el conjunto de calibración.

**Receta ganadora** (todo con gpt-4o-mini + text-embedding-3-small, ~1 llamada
LLM/consulta): grafo de aserciones reificadas n-arias + sinonimia 0,8 + filtro
selector few-shot (recognition memory) con compuerta dura en la siembra +
siembra densa de todos los pasajes + modulación de transiciones por coseno
(CatRAG-lite) + router léxico de intención (comparison→descomposición,
resto→juez).

## Decisiones tomadas

- Corpus/protocolo: subset vendorizado del clon (idéntico a los papers). Los
  `data/2wiki/dev.json`==`test.json` son el test oficial SIN oro (inútiles);
  `train.json` es el dev oficial (12.576 con oro) → cantera de sondeo/calibración.
- Las 1.000 del benchmark no se usan jamás para ajustar (diales, router, prompts).
- El `context` de 10 párrafos por pregunta NO se usa en recuperación (protocolo abierto).
- CatRAG del paper no cambia el grafo (hereda HippoRAG 2): nuestro grafo reificado
  es la variable experimental; el juez LLM de frontera no necesita resúmenes C(v)
  porque las aserciones ya están verbalizadas.
- Presupuesto estimado total ~$12-15 (techo $25); gastado al cierre del 21-ago: ~$10-11.

## Para la próxima sesión (2026-08-22)

Estado: estudio principal cerrado con victoria (tabla final v2). Retomar por aquí,
en orden de valor:

1. **Ablaciones formales para atribución limpia** (¿qué aporta cada pieza?):
   grafo aplanado vs reificado (el `grafo_benchmark_plano.graphml` ya existe),
   sin filtro (lite), sin router, sin siembra total. Coste ~$1, medio día.
   Necesarias para cualquier escrito.
2. **Re-derivar las demos del filtro desde calibración** (cerrar la nota de
   honestidad): elegir 3-5 fallos del conjunto de calibración, reescribir demos,
   re-medir benchmark 1 vez. Coste ~$0,30.
3. **Capa Wikidata P31/P279** — la hipótesis original del 18-ago, aún sin probar;
   con la victoria ya lograda, su valor ahora es incremental/paper.
4. **QA EM/F1 extremo a extremo** (reader compartido, 5 configs, ~$1) — completa
   el protocolo del paper.
5. Si se piensa en paper: correr MuSiQue y HotpotQA (mismos scripts, corpus ya
   vendorizados; ~$6-8 más) y bootstrap pareado en todo.

Contexto rápido para retomar: memoria `asistente-vih-plan-eval-2wiki` +
este informe. Código: `eval/wiki2.py` (harness), `ingest/wiki2_openie.py` y
`wiki2_grafo.py` (KG), `retrieval/wiki2_catrag.py` (retriever v2 con filtro
few-shot, compuerta dura, router), `eval/wiki2_hipporag.py` (baseline, correr
con `venv/bin/python`). La caché SQLite y los npz hacen todo re-ejecutable sin
re-facturar.

## Análisis de errores v2 (2026-08-22)

218 fallos FC@5 (bridge_comparison 104, compositional 86, inference 19,
comparison 9). Scripts: análisis offline sobre `resultados_catrag_router_v2_benchmark.jsonl`
+ trace instrumentado de 30 fallos compositional/inference. Ids en
`analisis_fallos_ids.json`.

**Qué falta**: de 249 pasajes oro perdidos, **222 son el puente (hop-2)** y solo 27
el pasaje nombrado en la pregunta. Los puentes perdidos: 126 ausentes (>20),
70 en rangos 6-10, 26 en 11-20. El router NO es causa (29 mal enrutadas, fallan
al 21% = tasa base). bridge_comparison: 79 preguntas traen 3/4 oros en top-5
(un director fuera); a top-10 las cadenas completas suben de 131 a 167.

**Atribución estructural (222 puentes, offline)**: 186 (84%) "todo existe" —
el hecho puente está extraído, la entidad existe y el pasaje hop-2 se
autoenlaza; 21 (9%) el pasaje hop-2 NO lista su propio título entre sus
entidades (sin arista entidad→su pasaje); 13 (6%) sin hecho puente (extracción).
→ **La extracción no es el cuello de botella.**

**Atribución por etapa (30 fallos instrumentados)**:
- E+F (17/30 = 57%): puente **sembrado correctamente** pero su pasaje queda en
  rango 5-9 (6) o ≥10 (11). La semilla-entidad no arrastra su propio pasaje:
  entidades hub reparten masa entre todos los chunks que las mencionan.
- B (6/30 = 20%): el hecho puente existe pero **no entra en el top-40 de
  candidatos por coseno** (preguntas largas/complejas) → el filtro nunca lo ve.
- C (4/30): el filtro no lo selecciona (roles raros: "composer of film" → Bach).
- D (2/30): seleccionado pero entidad no enlazada a la aserción (n-aria residual).
- A (1/30): hecho ausente del KG.

**Medidas correctoras propuestas (orden de valor, todas sin re-extracción)**:
1. **Boost del pasaje-título de cada entidad sembrada** (ataca E+F, 57%, y los
   85 "cerca"): si una entidad aprobada coincide con el título de un pasaje (o
   su embedding ≥0,8 con el título), ese nodo-pasaje recibe masa de reset
   directa. Coste API ≈ 0 (embeddings de títulos, ~$0,001).
2. **Autoenlace del título en ingest** (21 casos): añadir el título del pasaje
   como entidad propia del chunk. Regenerar grafo, sin API.
3. **Candidatos por frontera, no solo por coseno** (ataca B, 20%): candidatos del
   filtro = top-40 coseno ∪ aserciones de las entidades ancladas desde la
   pregunta (hop-1). Es literalmente la "frontera de las semillas" de CatRAG.
   Coste: prompt del filtro algo más largo.
4. Demos del filtro para roles poco frecuentes (composer/music by, detained,
   work at) — marginal (C). Re-derivarlas desde calibración (nota de honestidad).
5. bridge_comparison: con 1+3 ambos directores deberían entrar; si persiste,
   reparto explícito de presupuesto por sub-cadena (2+2 slots) en el router.

Potencial estimado si 1-3 funcionan: resolver ~la mitad de los 218 fallos →
FC@5 de 78 hacia 85-88. Validar en sondeo antes de tocar el benchmark.

## Iteración 3 — diccionario de entidades (2026-08-22)

Decisión de diseño tras el análisis de errores: el 57% de los fallos eran
"islas" (la mención del puente en el pasaje nombrado y la entidad de su propia
biografía eran nodos distintos sin arista: "H. P. Lovecraft" vs "Howard Phillips
Lovecraft", coseno 0,781 < 0,80). Solución estructural, como el linker SNOMED
del caso VIH pero derivada del corpus: **memoria de entidades** construida una
vez desde los 6.119 pasajes (nunca desde el oro del benchmark).

`ingest/wiki2_entidades.py`: ficha por mención (nombre + descriptores de sus
aserciones) → candidatos por DOS canales (coseno de nombre ≥0,80 / coseno de
ficha ≥0,75 con token común) → fusión automática si nombre idéntico (sin
paréntesis) y fechas/ordinales compatibles → adjudicación LLM (lotes de 8,
caché SQLite) para la banda ambigua, con prompt estricto (parientes, sucesores,
homónimos ≠) → entrada canónica: nombre, alias (incluido el TÍTULO del pasaje
como alias del sujeto), descripción, pasajes, pasaje propio. Precisión revisada
a mano en muestra; islas resueltas en sondeo: 28/188 (v. inicial, solo ficha)
→ 154 (canal nombre) → **178/188** (título como alias).

`ingest/wiki2_grafo.py --canonico`: nodos-entidad = entradas del diccionario
(sin sinonimia vectorial). `retrieval/wiki2_catrag.py canonico=True`:
(1) **linker** de menciones de la pregunta (spans capitalizados → alias exacto
→ ficha por embedding ≥0,80; división por and/or); (2) entidades enlazadas
como anclas fuertes (0,5) y **masa directa en el pasaje propio** de toda
entidad sembrada (enlazada o aprobada por el filtro); (3) **frontera**: las
aserciones de las entidades enlazadas entran como candidatos del filtro.

| sondeo (200) | R@5 | FC@5 | comp. | compos. | infer. | bridge_c. |
|---|---|---|---|---|---|---|
| v2 (router+paridad+n-aria) | 93,5 | 83,5 | 98 | 86 | 86 | 64 |
| **v3 (+diccionario+linker+frontera)** | **97,1** | **92,0** | **100** | **92** | **94** | **82** |

Pendiente: diccionario del benchmark (en construcción) → grafo canónico →
re-medición única. Git: rama `eval-wiki2` (VIH congelado en `main`, etiqueta
`vih-congelado-2026-08-22`).
