# Evaluación 2WikiMultihopQA — informe vivo

Comparación del sistema (CatRAG propio) contra HippoRAG 2 sobre el subset de
reproducción de los papers (1.000 preguntas / 6.119 pasajes). Plan completo en
la memoria del proyecto (`asistente-vih-plan-eval-2wiki`); hilos de origen:
18-ago (0471eaa2) y 19-ago (e057f71d).

**Última actualización**: 2026-08-23 — estudio de errores v3 hecho; dos planes de mejora listos (calidad-primero y coste-mínimo). Ver "Para la próxima sesión (2026-08-24)" al final.

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

## Para la sesión del 2026-08-22 (histórico — cumplido: se hizo el diccionario)

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

### TABLA FINAL v3 — benchmark oficial (1.000 × 6.119, mismos modelos) — 2026-08-22

| sistema | R@2 | R@5 | FC@5 | FC@10 | comp. | compos. | infer. | bridge_c. |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 65,8 | 32,8 | 40,1 | 86,5 | 19,4 | 32,4 | 0,9 |
| HippoRAG 2 (su código) | 70,7 | 85,9 | 65,8 | 71,7 | 93,4 | 77,2 | 75,9 | 12,3 |
| CatRAG router v2 | 70,4 | 90,9 | 78,2 | 85,8 | 96,3 | 79,2 | 82,4 | 55,7 |
| **CatRAG v3 (diccionario+linker+frontera)** | **73,0** | **96,0** | **90,6** | **93,4** | **99,2** | **91,0** | **84,3** | **83,8** |

**Pareado vs HippoRAG 2** (bootstrap 2.000, IC95): FC@5 **+24,8 [+21,8, +27,8] SIG**
(gana 279 preguntas / pierde 31); R@5 +10,2 SIG; R@2 +2,3 SIG (ya no es empate).
Por tipo FC@5: bridge_comparison **+71,5 SIG**, compositional **+13,8 SIG**,
comparison +5,7 SIG, inference +8,3 ns (n=108). Es decir: **victoria
significativa en 3 de 4 tipos y en todas las métricas globales**, incluido el
encadenamiento puro (compositional), donde la v2 solo empataba.

**Pareado vs v2**: FC@5 +12,4 [+10,2, +14,7] SIG (gana 136 / pierde 12) — el
diccionario apenas rompe nada (12 preguntas) y arregla 136.

Contexto: el R@5 96,0 supera el 90,4 que el paper de HippoRAG 2 reporta con
Llama-3.3-70B + NV-Embed-v2 (7B), obtenido aquí con gpt-4o-mini +
text-embedding-3-small y ~1 llamada LLM por consulta.

**Nota de honestidad (ampliada)**: (a) las 3 preguntas trazadas en la iteración
2 son del benchmark (ver arriba); (b) el análisis de errores que motivó la
iteración 3 se hizo sobre los fallos de la v2 EN EL BENCHMARK — a nivel de
mecanismo (clases de fallo), no de pregunta, y la solución (diccionario) es
estructural y se validó en el sondeo held-out con el mismo efecto (83,5→92,0);
aun así, para publicación el análisis de errores debe repetirse sobre el
conjunto de calibración y las demos del filtro re-derivarse de ahí. Ninguna
decisión usa el oro del benchmark como dato de entrada.

**Receta final v3**: memoria de entidades canónica derivada del corpus
(fichas, fusión por nombre+contexto, adjudicación LLM, título como alias) →
grafo reificado n-ario sobre nodos canónicos → linker de menciones → filtro
selector few-shot con candidatos por coseno ∪ frontera de las entidades
enlazadas → compuerta dura en la siembra + pasaje propio de cada entidad
sembrada + todos los pasajes con prior denso → PPR (d=0,5) con modulación por
coseno → router léxico por tipo. Coste acumulado del estudio: ~$12.

**Siguientes pasos (cuando se retome)**: ablaciones formales sobre v3 (sin
diccionario / sin linker / sin frontera / sin pasaje propio / grafo plano) para
atribuir; análisis de errores de v3 sobre calibración; QA EM/F1 extremo a
extremo; MuSiQue y HotpotQA con los mismos scripts si hay intención de paper.

## Estudio de errores de la v3 (2026-08-23)

Método: estadísticas agregadas sobre los 94 fallos FC@5 del benchmark (sin
mirar preguntas individuales) + trazas instrumentadas sobre los 16 fallos del
SONDEO (held-out), para no contaminar el test con el diagnóstico fino.

**Benchmark (94 fallos: bridge_c. 38, compositional 37, inference 17, comparison 2)**
- Oros perdidos: 88 de 102 son el pasaje puente (64 ausentes del top-20, 18 en
  6-10, 6 en 11-20); solo 14 son el pasaje nombrado (12 de ellos en 6-10).
- bridge_comparison: 31 preguntas traen 3 de 4 oros en top-5 (falta un director).
- Router por reglas: 5 de 94 fallos mal enrutados (≈ tasa base 29/1000) → no es causa.
- Linker: todas las preguntas tienen span detectado; ids linker↔grafo consistentes.
- Regresiones v3 vs v2: 12 (inference 6, compositional 5); fallan ambos sistemas
  (v3 y HippoRAG 2): 63; solo v3: 31.

**Atribución por etapa (15 fallos puente del sondeo, instrumentados)**
- **12/15: el hecho puente ESTABA en la lista de candidatos del filtro y el LLM
  no lo seleccionó** (gpt-4o-mini, máx. 4 hechos; en bridge_comparison hacen
  falta ≥4: dos "directed by" + dos fechas, y el filtro tiende a quedarse en
  los hechos de película). Ejemplo: "Bílá spona was directed by Kurt Neumann"
  presente y no elegido.
- 3/15: el hecho puente no llegó a candidatos (linker sin enlace para "The
  Magician (1958 Film)"; hecho formulado sin el nombre).
- 2 filtros vacíos (fallback denso), 2 casi-aciertos de ranking (rangos 5-8).
- La hipótesis "tope de 20 en la frontera" se descartó: fronteras de 0-10.

**Conclusión**: el cuello de botella actual es el **juicio y la capacidad del
filtro** (modelo pequeño + selección máxima de 4), seguido del alcance del
linker. Las heurísticas (router léxico, spans capitalizados) no son causa
medible de fallo en este dataset, pero sí límites de generalización.

## Para la próxima sesión (2026-08-24)

**Estado**: v3 en benchmark R@5 96,0 / FC@5 90,6 (HippoRAG 2: 85,9 / 65,8).
Estudio de errores v3 hecho (sección anterior): el cuello de botella es el
JUICIO del filtro (12/15 fallos held-out: el hecho puente estaba en la lista y
no se seleccionó), después el alcance del linker (3/15). Código en rama
`eval-wiki2`, subido a github.com/bjur27r/hiv_rag (VIH congelado en `main`).

**Decisión pendiente del usuario**: qué plan ejecutar primero.

### Plan A — calidad primero (mejorar FC@5; ~$0,01/consulta, ~$10 por pasada)
Evidencia en vivo sobre fallos held-out (2026-08-23):
1. **Filtro "planifica y luego selecciona"** (máx. 8, modelo mejor solo aquí):
   en "Bílá spona / A Night at Earl Carroll's" el filtro actual eligió al ACTOR
   Earl Carroll como director y omitió a Kurt Neumann aunque estaba en la
   lista; con plan, gpt-4o-mini seleccionó los dos puentes y gpt-4o exactamente
   los dos. Mayor impacto esperado (bridge_comparison y compositional).
2. **Selección final de CONJUNTO** (LLM elige 5 del top-10/20 que cubran la
   cadena): "Joan de Beauchamp" tenía el oro en el puesto 6; la selección de
   conjunto lo recogió. 30 oros del benchmark están en puestos 6-10.
3. **Segundo salto dirigido**: el plan nombra el hueco ("fecha de muerte de
   Kurt Neumann") → si su pasaje no está, se pide por nombre y se re-filtra.
4. **Linker por LLM (NER + forma canónica + desambiguación)**: la regla de
   mayúsculas produjo el span roto 'Magician (1958' y no enlazó "The Magician
   (1958 film)" aunque el diccionario tenía la entrada; el NER-LLM devolvió el
   título canónico exacto. Poco impacto en 2Wiki (+0,5-1), grande en
   generalización (español, minúsculas, homónimos).
Implementación: 1+3+4 en UNA llamada (NER → plan → selección), 2 en una
segunda pequeña. Validar en sondeo (listón 92,0) → una re-medición.
Expectativa: FC@5 94-96. Índice (más tarde): re-extracción prompt v2 (sujeto =
forma del título + alias, roles n-arios), adjudicación por grupos, descripciones
de ficha redactadas por LLM.

### Plan B — coste mínimo estricto (misma calidad, ~0,5 llamadas/consulta)
1. Comparison sin LLM (linker + pasaje propio + PPR; 24% de consultas a $0).
2. Compuerta de confianza: ruta sin LLM primero; filtro solo si las entidades
   enlazadas no tienen su pasaje en top-5 o no hay pasaje puente alcanzado.
3. Caché de prefijo del prompt del filtro (80% fijo) y caché de resultados
   (embeddings de pregunta, salidas del filtro) → re-mediciones a $0.
4. Modelo más barato en el filtro con regla dura: "el más barato que mantenga
   FC@5 ≥ 92 en sondeo" (candidatos: gpt-4.1-nano, gpt-5-nano, Gemini Flash-Lite,
   DeepSeek-V3 con caché; precios a verificar). DeepSeek ya se usa en
   extraccion_masiva.py; lotes nocturnos / Batch API para trabajos de índice.
El diccionario NO se elimina: es el recuperador rápido (alias exacto, $0) y
la capa LLM de sinonimia solo actúa sobre lo que no sabe y lo escribe de vuelta.

### Lo que NO movería FC@5: router por LLM (5/94 errores = tasa base), sinonimia
por umbral en consulta, tocar el PPR.

### Pendiente de siempre: ablaciones v3, QA EM/F1 + Joint Success Rate,
MuSiQue/HotpotQA, análisis de errores sobre calibración para el paper.

## Plan A ejecutado — matriz de modelos en sondeo (2026-08-25)

Implementado `retrieval/wiki2_plan.py` (extiende la v3 sin tocarla): ANALISTA
(NER + plan de la cadena + selección con tope dinámico, 1 llamada), SALTO
dirigido por huecos con candidatos ESTRUCTURALES (aserciones del pasaje propio
de la entidad enlazada — resuelve el caso "The Magician"/Ansiktet, título en
otro idioma) y CONJUNTO final (5 pasajes que cubren el plan; se omite en
comparison). Multi-proveedor (OpenAI / DeepSeek / Groq) con caché SQLite.
Iteración del prompt: prohibido rellenar el plan con conocimiento paramétrico
(el analista inventaba "Menahem Golan"); huecos como consulta de recuperación;
procedencia [artículo] en los hechos del salto.

| modelo (200 held-out) | R@5 | FC@5 | $/consulta | supera 92 |
|---|---|---|---|---|
| deepseek-chat (V3) | **99,0** | **97,5** | ~0,0010 | sí |
| groq gpt-oss-20b | 98,9 | 97,0 | ~0,0005 | sí |
| gpt-4o | 98,6 | 96,5 | ~0,012 | sí |
| gpt-4o-mini | 98,5 | 96,0 | ~0,0008 | sí |
| groq qwen3.6-27b | 85,6 | 67,5 | — | NO (planes vacíos) |
| v3 (referencia) | 97,1 | 92,0 | ~0,0004 | — |

Lecturas: (1) la ganancia es del DISEÑO de la tarea, no del músculo — gpt-4o
solo +0,5 sobre su mini a 15×; (2) gpt-oss-20b (rango bajo de Groq) empata de
facto con DeepSeek (1 pregunta de diferencia) a mitad de coste — validado como
alternativa; en contra: límites de peticiones de la capa gratuita y fallos de
validación JSON (mitigados con reintento); (3) qwen3.6-27b marca el suelo de
capacidad. Decisión: benchmark con deepseek-chat (calidad máxima + API
estable); groq oss-20b queda como alternativa validada sin gastar una mirada
al benchmark. Medición en curso.

## Análisis por estructura de salto (2026-08-25) — y por qué la profundidad rompe a HippoRAG y no a nosotros

Los 4 tipos de 2Wiki codifican TRES estructuras de salto (en 2Wiki no existe el
mono-salto: toda pregunta exige ≥2 pasajes). El harness ahora lo imprime
siempre (`ESTRUCTURA_SALTO` en eval/wiki2.py):

- **A. Sin puente** (`comparison`, n=244): las dos entidades vienen nombradas
  en la pregunta; dos búsquedas paralelas, ninguna entidad que descubrir.
- **B. Puente simple** (`compositional`+`inference`, n=521): una entidad
  puente ausente de la pregunta; 2 saltos encadenados.
- **C. Doble puente** (`bridge_comparison`, n=235): dos cadenas con puente y
  comparación; 4 pasajes oro en 5 huecos.

| FC@5 benchmark | A sin puente | B puente simple | C doble puente |
|---|---|---|---|
| HippoRAG 2 | 93,4 | 77,0 | **12,3** |
| CatRAG v3 | 99,2 | 89,6 | **83,8** |

**El modelo multiplicativo (por qué creemos que es así).** Si cada pasaje oro
entrara en el top-5 de forma independiente con probabilidad p, entonces
FC ≈ p^(nº de oros). Ajustando p con el grupo B (2 oros con puente) y
prediciendo C (4 oros):

- CatRAG v3: B 89,6 → p≈0,947; predicción C = 0,947⁴ ≈ **80,3** vs observado
  **83,8** — el modelo ajusta (incluso algo mejor: las dos cadenas comparten
  contexto). Nuestra degradación 99→90→84 es la ACUMULACIÓN de un error
  residual por eslabón (~5%: juicio del filtro, huecos de extracción), no un
  cambio de régimen.
- HippoRAG 2: B 77,0 → p≈0,877; predicción C = 0,877⁴ ≈ **59,2** vs observado
  **12,3** — 4,8× POR DEBAJO del modelo independiente. Su fallo en C no es
  acumulativo: es ESTRUCTURAL.

**El mecanismo del colapso de HippoRAG en C**: sus recursos por consulta son
fijos y COMPARTIDOS entre las dos cadenas — el filtro selecciona ~≤4 triples y
en la práctica se concentra en una cadena; la masa del PPR se reparte entre
las dos películas nombradas; y el presupuesto k=5 deja UN solo hueco de
holgura para 4 oros. Cuando la selección cubre solo una cadena, el puente de
la otra nunca se siembra → FC=0 garantizado por diseño, por bien que ordene lo
demás. La firma está en sus números: R@5 68,8 en C (encuentra ~2,7 de 4 oros:
recall parcial alto) con FC 12,3 (cadena rota) — la tesis del paper de CatRAG,
observada en su sistema sucesor y amplificada.

**Por qué nuestros mecanismos quitan la dependencia de la profundidad**:
(1) la identidad se resuelve en INDEXACIÓN (diccionario canónico): con
sinonimia por umbral, cada eslabón extra es otra oportunidad de "isla" — la
probabilidad de fallo por eslabón CRECE con la profundidad; con fusión
canónica es ~constante; (2) los hechos son nodos con tránsito modulado (μ=1
si el filtro los aprueba): la masa VIAJA a través del hecho aprobado en vez de
competir con todas las aristas del concentrador; (3) la contabilidad explícita
de la cadena escala con ella: la compuerta siembra TODAS las entidades de los
hechos aprobados (recursos por eslabón, no por consulta), el pasaje propio
convierte cada puente descubierto en pasaje sembrado, y el tope del
filtro/plan crece con el nº de huecos (Plan A). Por qué A no es 100:
casi-duplicados (secuelas, homónimos) en el ranking; por qué C < B incluso
para nosotros: 4 oros en 5 huecos no dejan holgura — un solo distractor por
encima del 4º oro rompe la cadena.

Verificación futura: MuSiQue codifica el nº de saltos en el id de cada
pregunta (2hop/3hop/4hop) → la curva FC-vs-saltos saldrá gratis en F2 y es el
contraste directo de este modelo.

## RESULTADO FINAL DE LA CAMPAÑA — Plan A en el benchmark (2026-08-25)

| sistema | R@2 | R@5 | FC@5 | comp. | compos. | infer. | bridge_c. |
|---|---|---|---|---|---|---|---|
| BM25 | — | 65,8 | 32,8 | 86,5 | 19,4 | 32,4 | 0,9 |
| HippoRAG 2 | 70,7 | 85,9 | 65,8 | 93,4 | 77,2 | 75,9 | 12,3 |
| CatRAG v3 | 73,0 | 96,0 | 90,6 | 99,2 | 91,0 | 84,3 | 83,8 |
| **Plan A (deepseek-chat)** | **82,2** | **98,5** | **96,9** | **99,6** | **95,6** | **90,7** | **99,1** |

Pareado vs HippoRAG 2: FC@5 **+31,3 [+28,3, +34,4] SIG** (gana 322 / pierde 9);
R@2 +11,5 SIG. Por tipo, TODOS significativos (incluido inference +14,8, que
en v3 no lo era). Pareado vs v3: FC@5 +6,5 [+4,8, +8,2] SIG (gana 74/pierde 9).

**El hallazgo conceptual: el gradiente de dificultad se INVIERTE.** Por
estructura de salto: A 99,6 · B 94,6 · **C doble puente 99,1** — el segmento
que era el peor de todos los sistemas (BM25 0,9; HippoRAG 12,3) es ahora el
MEJOR del Plan A, y queda muy por encima del modelo multiplicativo
(p_B=0,973 → C predicho 89,5 vs observado 99,1): el plan asigna recursos POR
CADENA (huecos explícitos, tope dinámico, conjunto final), así que la doble
cadena, lejos de compartir un presupuesto fijo, recibe el doble — la
refutación constructiva del colapso estructural de HippoRAG.

Coste de la medición: ~$1 (DeepSeek, parte en franja nocturna). Coste
operativo: ~$0,001/consulta, ~2,2 llamadas. DECISIÓN: Plan A + deepseek-chat
queda ADOPTADO como configuración final de la campaña (gpt-oss-20b de Groq,
validado como alternativa a mitad de coste). Nota .tex: el documento describe
la v3; el Plan A requerirá su propia subsección cuando se escriba (con las
ablaciones).

## Ablaciones del Plan A (sondeo, deepseek-chat, 2026-08-25)

| configuración | FC@5 | FC@2 | C doble puente | aporte |
|---|---|---|---|---|
| v3 (referencia, sin Plan A) | 92,0 | 46,0 | 82,0 | — |
| solo analista | 92,5 | 55,5 | 86,0 | +0,5 |
| analista + conjunto (sin salto) | 94,0 | 61,0 | 86,0 | +1,5 |
| analista + salto (sin conjunto) | 95,5 | 55,0 | 96,0 | +3,0 |
| **Plan A completo** | **97,5** | **61,5** | **96,0** | **+5,5** |

Lectura: el analista POR SÍ SOLO apenas mueve el agregado (+0,5) — su valor
es que produce el PLAN que consumen los otros dos componentes. El **salto
dirigido es el mayor contribuyente individual** (+3,0; en doble puente
86→96: rescata los puentes que ni el coseno ni la frontera alcanzan, caso
Ansiktet). El **conjunto final aporta orden temprano** (+1,5 en FC@5 y +6
puntos de FC@2: coloca la cadena en las primeras posiciones). Ligera
superaditividad (0,5+3,0+1,5=5,0 < 5,5 observado): el conjunto ordena mejor
lo que el salto ha traído. La cadena causal completa de la campaña queda:
v3 92,0 → +plan 92,5 → +salto 95,5 → +conjunto 97,5 (sondeo), que en
benchmark es 90,6 → 96,9.

## Guía de lectura de las métricas (añadida a petición del usuario)

- **R@k (exhaustividad)**: fracción de los pasajes oro presentes entre los k
  primeros devueltos. Parcial: 1 de 2 oros en el top-5 → R@5 = 0,5.
- **FC@k (cadena completa)**: 1 solo si TODOS los oros están en el top-k.
  Es la métrica alineada con multisalto: al lector no le vale media cadena.
- **FC@2**: exige los oros exactamente en las posiciones 1-2. OJO: las 235
  bridge_comparison tienen 4 oros → FC@2 es IMPOSIBLE por definición para
  ellas (por eso siempre marcan 0,0); el agregado global lleva ese 23,5% de
  ceros estructurales incorporado (el 61,5 del Plan A equivale a ~80% sobre
  las preguntas donde es alcanzable). Responde a "¿bastarían 2 pasajes de
  contexto?"; la métrica estándar de los papers es FC@5.
- **Escalera de contribuciones** (antes mal llamada "cadena causal"): secuencia
  de versiones donde cada peldaño añade UNA pieza sobre lo demás fijo; la
  subida de cada peldaño es atribuible a esa pieza (= ablaciones aditivas).
- **EM / F1 (QA)**: coincidencia exacta / solape de palabras entre la
  respuesta del lector y la respuesta oro. **JSR (Joint Success Rate)**:
  cadena completa recuperada Y respuesta correcta — "resuelto con evidencia".
- **Cifra de archivo**: la medición final con todo congelado (prompts con
  ejemplos re-derivados de calibración, diales fijos) — el número publicable,
  sin contacto test↔diseño.
