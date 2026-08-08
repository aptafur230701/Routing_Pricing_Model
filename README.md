# AM-PPO Routing & Pricing Model

Agente de ruteo basado en un **Attention Model (Transformer encoder-decoder)**
entrenado con **PPO (Proximal Policy Optimization)**, para un problema de
ruteo estocástico con revelación post-decisión de disponibilidad de carga
(Stochastic DVRP). El agente decide, en cada nodo, a qué nodo moverse a
continuación para maximizar la recompensa acumulada (margen de flete) dentro
de una ventana de duración máxima.

Este README es la guía de la estructura del proyecto después de una limpieza
(rama `clean-structure`) que eliminó scripts de debugging, tests puntuales y
sweeps de tuning ya aplicados, dejando solo el pipeline del modelo y su
validación final.

## Cómo correrlo

```bash
pip install -r requirements.txt
python main.py          # entrena (o evalúa) y guarda resultados en results/<N>nodes/
python inference.py     # inferencia interactiva con un checkpoint ya entrenado
```

Todo lo que hay que tocar para un run distinto está al principio de
`main()` en [main.py](main.py): `NUM_NODES`, `EVAL_ONLY`, `USE_TRANSFER`,
`N_DAYS_PER_NODE`. Los hiperparámetros del modelo y del PPO están todos en
[config.py](config.py).

## Arquitectura del modelo

```
problem_data.py ──► matrices (tiempo, tarifa, carga, distancia, diesel, LTR, camiones)
                              │
                              ▼
                    routing_env.py (RoutingEnv / VectorRoutingEnv)
                              │  state_features.py construye el vector de estado
                              ▼
        ┌─────────────────────────────────────────────┐
        │                am_agent.py                   │
        │  AttentionEncoder  (attention_encoder.py)     │
        │      → embeddings por nodo + embedding global │
        │  ContextNetwork    (attention_encoder.py)     │
        │      → vector de contexto h_t                 │
        │  AttentionDecoder  (attention_decoder.py)      │
        │      → distribución π(a|s) sobre nodos válidos │
        │  CriticHead        (critic_head.py)             │
        │      → V(s_t) para la ventaja PPO                │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                    am_training.py  (loop PPO-Clip + GAE)
                              │
                              ▼
                checkpoints/am_checkpoint_<N>nodes.pt
                              │
                              ▼
                    evaluation.py (compara contra Solvers.py)
                              │
                              ▼
                results/<N>nodes/  (Excel + gráficos + training log)
```

**Por qué el modelo escala sin reentrenar desde cero**: ninguna capa de
`AMRoutingAgent` ni de `CriticHead` depende de `num_nodes` — el Transformer
opera sobre conjuntos de tamaño variable. Esto es lo que permite el transfer
learning en `transfer_learning.py` (cadena 10 → 20 → 35 → 50 → 75 → 100
nodos, ver `NODE_SEQUENCE`).

## Estructura de archivos

### Pipeline y modelo (raíz)

| Archivo | Rol |
|---|---|
| [main.py](main.py) | Punto de entrada: carga datos → entrena → evalúa → guarda resultados. |
| [config.py](config.py) | Todas las constantes globales e hiperparámetros (arquitectura, PPO, reward shaping). |
| [problem_data.py](problem_data.py) | Carga de `datos/` y construcción de matrices de recompensa por día. |
| [routing_env.py](routing_env.py) | Entorno Gymnasium (`RoutingEnv`, `VectorRoutingEnv`) para el DVRP estocástico. |
| [state_features.py](state_features.py) | Construcción del vector de estado que consume el agente. |
| [attention_encoder.py](attention_encoder.py) | `AttentionEncoder` (Transformer) + `ContextNetwork`. |
| [attention_decoder.py](attention_decoder.py) | `AttentionDecoder` (pointer network) con glimpse + pointer attention. |
| [critic_head.py](critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. |
| [am_agent.py](am_agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; expone `generate_route`, `act`, `act_with_value`. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE. |
| [debug_utils.py](debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [Solvers.py](Solvers.py) | Baselines de comparación: heurística greedy, 2-Opt, LNS, GA, HGA-LNS, rolling-horizon (determinista y estocástico), MC-Rollout, label-setting exacto/oráculo. |
| [inference.py](inference.py) | Inferencia standalone: carga un checkpoint y genera una ruta desde un nodo dado. |

### Validación final del modelo

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Corre la comparación DRL vs. todos los solvers de `Solvers.py`, genera diagnósticos y guarda resultados. |
| [excel_report.py](excel_report.py) | Formato del Excel de resultados (hojas Summary / Per Node). |
| [convergence_check.py](convergence_check.py) | Chequeo automático de convergencia del entrenamiento PPO (reward, explained variance, KL, entropía) sobre `training_log`. |
| [stats_analysis.py](stats_analysis.py) | Robustez estadística: descriptivos + test de Wilcoxon pareado de DRL contra cada baseline estocástico. |

Estos cuatro son los únicos módulos de análisis que sobrevivieron a la
limpieza porque `main.py` los invoca como parte del pipeline de validación
final. Todo lo demás que existía (sweeps de hiperparámetros de decodificación,
diagnóstico de profundidad de lookahead, microbenchmarks de rendimiento,
`pytest` de regresión puntual) fue exploración de una sola vez ya aplicada al
modelo actual, y se eliminó — sigue disponible en el historial de git de las
ramas anteriores si hace falta consultarlo.

### Datos y artefactos

| Carpeta | Contenido |
|---|---|
| `datos/` | Matrices fuente (`.csv`/`.npy`): distancia, duración, tarifa, diesel, disponibilidad de carga, LTR, camiones. |
| `checkpoints/` | Pesos entrenados (`am_checkpoint_<N>nodes.pt`), compartidos entre tamaños para transfer learning. Ignorado por git (`.gitignore`). |
| `results/<N>nodes/` | Salida de cada run: `DRL_Routing_Summary_AM.xlsx`, diagnósticos PPO/training (`.png`), `training_log_<N>nodes.pkl`/`.xlsx`. |
| `results/10nodes/best/` | Snapshot curado manualmente del mejor run de 10 nodos (no se sobreescribe con cada entrenamiento). |
| `docs/architecture.html` / `.png` | Diagrama explicativo del pipeline AM-PPO. |
| `runs/` | Logs de TensorBoard (si se generan). Ignorado por git. |

## Flujo de trabajo típico

1. **Entrenar un tamaño nuevo**: fijar `NUM_NODES` en `main.py`, dejar
   `USE_TRANSFER = True` para partir del checkpoint del tamaño anterior en
   `NODE_SEQUENCE` (o `False` para pesos aleatorios). `python main.py`.
2. **Solo evaluar** un checkpoint existente sin reentrenar: `EVAL_ONLY = True`.
3. **Revisar convergencia**: la consola imprime el reporte de
   `convergence_check.py` al final del entrenamiento; también queda una hoja
   en el Excel de resultados.
4. **Comparar contra baselines**: `results/<N>nodes/DRL_Routing_Summary_AM.xlsx`
   trae el gap de DRL contra cada solver de `Solvers.py`, más las hojas de
   estadística de `stats_analysis.py` (Wilcoxon pareado).
5. **Inferencia puntual**: `python inference.py` con un checkpoint ya
   guardado en `checkpoints/`.

## Requisitos

Ver [requirements.txt](requirements.txt). Entorno principal: PyTorch,
Gymnasium, NumPy/Pandas, Optuna (tuning opcional), PuLP (solvers exactos).
