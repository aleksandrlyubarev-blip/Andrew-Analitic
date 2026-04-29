# Robo QC — 3D AI Inspection: Техническое задание

**Версия:** 0.1 (draft)
**Ветка:** `claude/add-3d-ai-inspection-5xFKd`
**Репозитории:** `aleksandrlyubarev-blip/Andrew-Analitic` (Python backend),
`aleksandrlyubarev-blip/Romeo_PHD` (TS/React UI + Express API)
**Источник вдохновения:** in-line 3D AI инспекция качества, показанная Леонидом Пилчем
(CAD-aligned digital twin, sub-mm точность, обычные камеры, 100 % покрытие линии).

---

## 1. Цели

Добавить в экосистему Andrew/Romeo отдельный модуль **Robo QC**, который выполняет
in-line 3D AI инспекцию качества и размеров детали:

1. Реконструирует 3D-облако точек детали по N синхронным кадрам с **обычных
   камер** (без лазерных сканеров, без структурированного света).
2. Совмещает скан с **CAD-моделью** (digital twin) методом ICP/feature-based
   registration.
3. Выполняет **dimensional inspection** (GD&T-lite: расстояния, диаметры,
   плоскостность, концентричность) и **defect detection** (царапины, вмятины,
   missing features, недосвар, заусенцы).
4. Принимает решение PASS / FAIL / HITL по детерминированным quality gates
   с уже существующим механизмом **HITL escalation** в Andrew.
5. Поддерживает 100 % проверку каждой детали в реальном времени и логирует
   результаты в общую memory/telemetry-инфраструктуру (Romeo PHD telemetry,
   Andrew memory store).

## 2. Нефункциональные требования (NFR)

| ID | Требование | Целевое значение | Способ верификации |
|----|------------|------------------|--------------------|
| NFR-1 | Точность измерений | ≤ 1 мм @ 95 % CI на эталонной детали 100×100×50 мм | Калибровочный тест с CMM-эталоном, 30 прогонов |
| NFR-2 | Надёжность классификации | TPR ≥ 99.7 %, FPR ≤ 0.3 % | Adversarial test set ≥ 1000 OK + 1000 NOK деталей |
| NFR-3 | Latency end-to-end | ≤ 1.5 с на деталь @ 8 камер 12 MP | Бенчмарк p95 на reference hardware |
| NFR-4 | 100 % coverage линии | 0 пропусков при cycle time ≥ 2 с | Soak-тест 8 ч × 1800 деталей/ч |
| NFR-5 | Footprint оборудования | ≤ 0.5 м² на станцию (только камеры + LED + edge box) | Bill of Materials review |
| NFR-6 | Стоимость станции | ≤ $10k BOM (без робота) | Список компонентов в `docs/ROBO_QC_BOM.md` |
| NFR-7 | Reproducibility | Один и тот же снимок → один и тот же вердикт (deterministic) | `tests/test_robo_qc_determinism.py` |

## 3. Архитектура

```
                      ┌─────────────────────────────────────────┐
   PLC/conveyor       │       Robo QC station (edge box)         │
   trigger ────────►  │                                          │
                      │  Camera array (4–8× обычных USB3/GigE)   │
                      │             │                            │
                      │             ▼                            │
                      │  ┌────────────────────┐                  │
                      │  │ capture sync (ROS2 │                  │
                      │  │ / Python asyncio)  │                  │
                      │  └─────────┬──────────┘                  │
                      │            │ N×RGB frames                │
                      │            ▼                             │
                      │  ┌────────────────────┐  CAD model       │
                      │  │ Robo QC core       │◄─ STEP/STL       │
                      │  │ (core/robo_qc)     │                  │
                      │  │  • calibration     │                  │
                      │  │  • reconstruction  │                  │
                      │  │  • CAD alignment   │                  │
                      │  │  • dim. analysis   │                  │
                      │  │  • defect detect   │                  │
                      │  │  • quality gate    │                  │
                      │  └─────────┬──────────┘                  │
                      │            │ verdict + report            │
                      └────────────┼─────────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────┐
                │  Andrew bridge (FastAPI :8100)   │
                │   POST /qc/inspect               │
                │   GET  /qc/runs/:id              │
                │   GET  /qc/metrics               │
                │   POST /qc/hitl/:id/respond      │
                └──────────┬───────────────────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
     Andrew memory store         Romeo PHD UI
     (results + drift)           (artifacts/romeo-phd
                                  /src/pages/robo-qc.tsx)
```

### 3.1. Принципы

- **Pure-numpy ядро.** Базовая реализация (триангуляция, ICP, dimensional
  checks) использует только `numpy`/`scipy` (уже в `requirements.txt`).
  Open3D / OpenCV — **опциональные** ускорители, импорты под `try/except`,
  с работающим fallback.
- **Детерминизм.** Все стохастические шаги (RANSAC, KMeans) принимают `seed`.
  Quality gates чисто функциональны: `(scan, cad, thresholds) → verdict`.
- **HITL by default.** Решения с `confidence < HITL_CONFIDENCE_THRESHOLD`
  (env var, уже существует в Andrew) возвращают HTTP 202 +
  `hitl_required=true` — оператор подтверждает в Romeo PHD UI.
- **Совместимость с существующей памятью.** Каждый прогон сохраняется как
  `MemoryRecord` в `core/memory.py` (extend схему: `record_kind="qc_run"`).
  Это даёт Andrew возможность аналитических запросов «покажи дрейф диаметра
  отверстия за смену».

## 4. Структура кода

### 4.1. Andrew-Analitic (Python)

```
core/robo_qc/
  __init__.py
  types.py              # dataclasses: CameraIntrinsics, CameraExtrinsics,
                        # PointCloud, CADModel, Defect, QCRun, Verdict
  calibration.py        # pinhole model, checkerboard calib loader,
                        # extrinsic calibration via known fiducials
  reconstruction.py     # multi-view triangulation → PointCloud
                        # + dense MVS hook (опционально через Open3D)
  cad_alignment.py      # ICP (point-to-point + point-to-plane),
                        # coarse FPFH-based registration (опционально)
  measurements.py       # GD&T-lite: distance, diameter (RANSAC circle),
                        # planarity, perpendicularity, profile tolerance
  defects.py            # deviation map (scan vs CAD), CC-based defect
                        # blob extraction, severity classifier
                        # + AI-hook: PyTorch/ONNX runtime, gracefully optional
  quality_gate.py       # PASS/FAIL/HITL логика; принимает порогкарту
                        # из CAD spec (tolerance.json в CAD bundle)
  pipeline.py           # orchestrator: frames + cad → QCRun
                        # async, batched, deterministic
  cad_loaders.py        # STL/STEP loader (trimesh, опционально через
                        # OCP/cadquery — fallback: STL only)
  metrics.py            # Prometheus-friendly counters: pass_rate,
                        # avg_confidence, drift по ключевым размерам

bridge/
  qc_routes.py          # FastAPI router /qc/*; mounted в moltis_bridge.py

tests/
  test_robo_qc_calibration.py
  test_robo_qc_reconstruction.py     # синтетический куб → точность ≤ 0.1 мм
  test_robo_qc_alignment.py          # известный transform → recover < 1e-3
  test_robo_qc_measurements.py
  test_robo_qc_defects.py
  test_robo_qc_pipeline.py           # end-to-end на synthetic dataset
  test_robo_qc_determinism.py        # NFR-7
  test_robo_qc_api.py                # FastAPI contract tests

docs/
  ROBO_QC_SPEC.md       # этот файл
  ROBO_QC_ARCHITECTURE.md
  ROBO_QC_BOM.md        # рекомендованное оборудование (NFR-5/6)
  ROBO_QC_CALIBRATION.md
```

**Зависимости (добавить в `requirements.txt`):**
```
trimesh>=4.0.0           # STL loading, mesh ops (pure-Python core)
scipy>=1.13.0            # spatial.cKDTree для ICP
# Опциональные (extras):
# open3d>=0.18.0         # FPFH, fast ICP
# opencv-python>=4.10    # checkerboard calib, undistort
# onnxruntime>=1.18      # AI-defect inference
```

### 4.2. Romeo_PHD (TS / React / Express)

```
artifacts/api-server/src/
  routes/robo-qc/
    index.ts             # GET /api/qc/runs, POST /api/qc/inspect (proxy → Andrew)
    schemas.ts           # zod схемы (runs, verdict, hitl)
  lib/qc-client.ts       # httpx-стиль клиент к Andrew /qc/*

artifacts/romeo-phd/src/
  pages/
    robo-qc.tsx          # дашборд: live verdict feed, pass-rate, drift charts
    robo-qc-run.tsx      # детальный отчёт по прогону: 3D-вьюер
                          # (three.js: scan vs CAD overlay, deviation heatmap)
  components/qc/
    QCFeed.tsx
    DeviationViewer.tsx
    HITLApprovalCard.tsx  # переиспользуем consultations queue UI
  hooks/
    useQCRuns.ts          # React Query: список прогонов
    useQCMetrics.ts

lib/api-spec/
  qc.openapi.yaml         # contract /api/qc/* — feeds api-zod + api-client-react

docs/
  robo-qc.md              # как поднять и подключить
```

## 5. Контракты API

### 5.1. `POST /qc/inspect` (Andrew)

Запрос:
```json
{
  "run_id": "uuid",
  "part_number": "PN-12345",
  "cad_bundle_id": "cad/PN-12345/v3",
  "frames": [
    {"camera_id": "cam0", "image_b64": "...", "timestamp": "..."},
    {"camera_id": "cam1", "image_b64": "...", "timestamp": "..."}
  ],
  "calibration_id": "station-A-2026-04-01"
}
```

Ответ (200 PASS / 200 FAIL):
```json
{
  "run_id": "uuid",
  "verdict": "PASS|FAIL",
  "confidence": 0.998,
  "measurements": [
    {"name": "hole_d1", "nominal_mm": 8.000, "actual_mm": 8.012,
     "tolerance_mm": [-0.05, 0.05], "in_spec": true}
  ],
  "defects": [
    {"type": "scratch", "severity": "minor", "location_mm": [12.4, 5.6, 0.0],
     "size_mm": 1.2, "confidence": 0.91}
  ],
  "deviation_map_url": "/qc/runs/uuid/deviation.png",
  "scan_pointcloud_url": "/qc/runs/uuid/scan.ply",
  "cad_aligned_url": "/qc/runs/uuid/cad_aligned.ply",
  "elapsed_ms": 1180
}
```

Ответ 202 (HITL):
```json
{
  "run_id": "uuid",
  "verdict": "HITL_REQUIRED",
  "confidence": 0.31,
  "reason": "low_alignment_score",
  "hitl_id": "hitl-uuid"
}
```

### 5.2. `GET /qc/metrics`
Возвращает ту же схему, что `/metrics` Andrew, плюс `qc_pass_rate`,
`qc_avg_confidence`, `qc_drift[measurement_name]`.

### 5.3. `POST /qc/hitl/:id/respond`
Совместимо с уже существующим HITL escalation flow (Sprint 7).

## 6. Спринты (план реализации)

| Спринт | Содержание | DoD |
|-------|------------|-----|
| **S1 — Core skeleton** | `core/robo_qc/types.py`, `calibration.py`, `cad_loaders.py` (STL), `pipeline.py` stub. Synthetic data generator. | Импортируется, `pytest tests/test_robo_qc_calibration.py` зелёный |
| **S2 — Reconstruction** | Multi-view linear triangulation, RANSAC outlier rejection. Synthetic куб тест ≤ 0.1 мм. | NFR-1 на синтетике |
| **S3 — CAD alignment** | ICP point-to-point, point-to-plane. Тест: random rigid transform → recover. | `test_robo_qc_alignment.py` зелёный |
| **S4 — Measurements + defects** | GD&T-lite (4 типа), deviation-map defect detection (severity по distance threshold). | `test_robo_qc_measurements.py`, `test_robo_qc_defects.py` зелёные |
| **S5 — Quality gate + HITL** | Threshold loader из `tolerance.json`, интеграция с существующим `hitl_escalate`. | NFR-2 на synthetic adversarial set ≥ 200 |
| **S6 — FastAPI routes + tests** | `bridge/qc_routes.py`, contract tests. Mount в `moltis_bridge.py`. | `pytest tests/test_robo_qc_api.py` зелёный |
| **S7 — Romeo PHD UI** | `pages/robo-qc.tsx` + 3D viewer (three.js), feed, HITL approval. OpenAPI → zod → React Query. | UI поднимается в Codespaces, прогон демо-данных видно |
| **S8 — AI defect model (опционально)** | ONNX hook, обучение на public dataset (MVTec AD). | TPR ≥ 99.7 % на validation split |
| **S9 — BOM + калибровка на железе** | Документ `ROBO_QC_BOM.md`, инструкция калибровки, soak-тест. | NFR-3, NFR-4, NFR-5 на reference station |

## 7. Acceptance criteria для S1–S6 (MVP)

1. На synthetic dataset (программно сгенерированный куб + добавленный дефект)
   pipeline возвращает корректный `verdict` за < 1.5 с.
2. Все NFR проверены unit/integration-тестами там, где это возможно без
   реального железа (NFR-1 на синтетике, NFR-2 на synthetic adversarial set,
   NFR-3 как latency-ассерт в pipeline-тесте, NFR-7 как determinism-тест).
3. CI зелёный на ветке `claude/add-3d-ai-inspection-5xFKd`, без сломанных
   существующих 166 тестов.
4. Andrew `/health` отдаёт `qc_ready: true` если все опциональные модули
   подгружены, или `qc_ready: false` + список missing с понятным сообщением
   (нет лазерного сканера и ONNX — это нормально, fallback работает).

## 8. Технический стек (итог)

| Слой | Выбор | Обоснование |
|------|-------|-------------|
| Camera capture | Python asyncio + `pylonpy`/`pyrealsense2`/USB3 (опционально) | Совместимо с edge box, не требует ROS |
| 3D ядро | `numpy` + `scipy.spatial` (KDTree) + `trimesh` | Уже в стеке Python 3.11; чистый Python core |
| Ускорители (опц.) | `open3d`, `opencv-python`, `onnxruntime` | Импорты под `try/except`, gracefully degrade |
| API | FastAPI (уже используется) | Один процесс с Andrew bridge |
| UI | React + three.js (`@react-three/fiber`) | Уже есть React/Vite в Romeo PHD |
| Хранение | Andrew memory + Postgres (Drizzle в Romeo) | Существующая инфраструктура |

## 9. Риски и митигации

| Риск | Митигация |
|------|-----------|
| Sub-mm точность с обычными камерами требует жёсткой калибровки | S9 — отдельная процедура калибровки + чек-лист в `ROBO_QC_CALIBRATION.md`; deviation от этой процедуры → HITL |
| MVS (dense reconstruction) тяжёлый для edge box | MVP использует sparse triangulation + photometric refinement только в зонах GD&T; dense MVS — за фичефлагом |
| STEP-loader на Python ограничен | MVP принимает STL; STEP конвертируется офлайн через FreeCAD CLI (это уже умеет Buxter в Romeo PHD) |
| AI-defect датасет специфичен под клиента | S8 опциональный; до него работает чисто геометрический detector (deviation > tolerance → defect) |
| Регрессия в существующем 166-тестовом suite | Новый код изолирован в `core/robo_qc/`; роуты mounted под `/qc/*`, не пересекаются |

## 10. Открытые вопросы (требуют ответа от заказчика до S2)

1. **Какие детали проверяем первыми?** (металл / пластик / сборка)
   От этого зависит освещение и тип defect detector.
2. **Cycle time линии?** (NFR-4 калибруется под него)
3. **Есть ли существующие CAD-файлы и в каком формате?** (STEP / STL / IGES)
4. **Робот-манипулятор уже есть?** Если да, какой? (UR / Fanuc / KUKA / cobot)
   От этого зависит, нужен ли Robo QC как отдельная станция или встраивается
   в trajectory робота.
5. **Где уже работает Robo QC?** (или это greenfield?) — если есть legacy
   2D vision, нужно зафиксировать API совместимости.

---

**Следующий шаг.** После согласования этого ТЗ запускаю спринт S1 в той же
ветке `claude/add-3d-ai-inspection-5xFKd`, отдельным коммитом per-sprint.
