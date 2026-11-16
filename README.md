# AI-Driven Taxi Fleet Management System

Predictive taxi rebalancing for a Dubai-like city. A spatio-temporal GNN forecasts zone demand
every 15 minutes and conformal intervals quantify its uncertainty. A learned zone-level policy
plus a min-cost-flow LP then repositions idle taxis. The design principle: **learning sets the
*targets*, optimisation executes the *moves*** — rather than per-vehicle MARL acting end to end.

## Headline results (simulated city, held-out test days)

| | No rebalancing | Robust LP rebalancing |
|---|---|---|
| Requests served (fleet of 1,500) | 79.4% | **96.9%** |
| Mean passenger wait | 7.7 min | **3.7 min** |
| 90th-percentile wait | 16.8 min | **10.8 min** |

* 7 demand forecasters benchmarked (historical average, last value, GCN-GRU, GCN-GRU + adaptive
  embeddings, AGCRN, Graph WaveNet, STAEformer); best MAE ≈ 2.08 pickups/zone/15 min vs 2.27 for the
  historical average. Split conformal intervals per horizon and zone reach ≈ 90% coverage.
* Stress days (Ramadan, concert, New Year's Eve), NYC TLC → Dubai transfer, OpenCity zero-shot,
  decision-focused fine-tuning, vehicle-level simulation and SUMO/TraCI replay.
* Full result tables are in [`results/`](results/). The city and demand are synthetic (NYC TLC is used
  for pretraining), so all numbers are simulation results.

> Data, trained weights and generated outputs are not included (several GB). `python main.py data`
> regenerates the synthetic city; NYC TLC Parquet files and the OpenCity source/weights go in
> `data/raw/nyc/` and `external/` respectively. API keys (optional, for the video renders only) are
> read from a local `secrets/` folder that is git-ignored.

## Quick start

```bash
pip install -r requirements.txt
python main.py all          # every stage below (~2 h on a laptop GPU)
```

| Command | What it does | Main output |
|---|---|---|
| `data` | generate the synthetic city (or aggregate NYC TLC) | `data/processed/` |
| `forecast` | train and benchmark 7 forecasters, export forecasts with conformal intervals | `forecast_metrics.md` |
| `train-rl` | bilevel and mean-field rebalancing policies (BC warm start + A2C) | `policy_*.pt` |
| `evaluate` / `sweep` | all controllers on held-out days; fleet sizes 1k–3k | `control_metrics.md`, `fleet_sweep.md` |
| `dfl` | decision-focused fine-tuning of the forecaster through the LP | `dfl_*_metrics.md` |
| `event` | vehicle-level simulator: learn V(zone, slot); rebalancers × matchers | `event_metrics.md` |
| `stress` | Ramadan, concert and New Year's Eve stress days | `stress_metrics.md` |
| `transfer` | pretrain on NYC TLC; zero/few-shot to Dubai; OpenCity zero-shot | `transfer_metrics.md` |
| `sumo` | replay scaled peak windows in SUMO with TraCI-dispatched taxis; fleet sweep | `sumo_metrics.md` |

Each command takes `--methods` to restrict which models or controllers run. For example,
`python main.py evaluate --methods none lp bilevel_rl --fleet 2000`, or
`python main.py transfer --methods opencity` to recompute only the OpenCity rows.

## Architecture

```
data/                         processed city arrays; data/raw/nyc/ holds TLC Parquet + taxi_zones.zip
external/                     OpenCity source (HKUDS) and published weights (OpenCity-base/plus.pth)
src/
  ingestion/  synthetic.py    Dubai-like corridor city + demand generator, Ramadan/concert/NYE scenario days
              nyc_tlc.py      NYC TLC Parquet -> 15-min zone series; zone centroids from the TLC shapefile
              graph.py        ZoneGraph G=(V,E): distances, adjacency, speed profile
              dataset.py      CityData, chronological splits (+ 'stress' split), DemandDataset
  models/     forecasting.py  HA, last-value, GCN+GRU (blueprint ST-GNN), GCN+GRU + adaptive embeddings
              sota_forecasters.py  AGCRN, Graph WaveNet, STAEformer (official default hyperparameters)
              conformal.py    split conformal intervals per horizon x zone
              decision_focused.py  forecaster fine-tuning on LP decision cost (Gaussian-smoothed gradient)
              transfer.py     zone-count-agnostic GNN (Laplacian eigenvectors + instance norm), NYC -> Dubai
              opencity_adapter.py  zero-shot inference with the published OpenCity foundation model
              policies.py     BilevelPolicy (GNN -> Dirichlet target), MeanFieldPolicy (per-zone splits)
              value_table.py  V(zone, slot) by backward-induction policy evaluation (DiDi learning-and-planning)
              train_*.py      forecaster training/export; RL behaviour cloning + A2C (GAE)
  simulation/ fleet_sim.py    batched zone-level fluid simulator (training + fast evaluation)
              event_sim.py    vehicle-level minute-step simulator (individual taxis and riders)
              sumo_bridge.py  SUMO grid network + TraCI taxi dispatch/rebalancing (libsumo when available)
  control/    flow_solver.py  min-cost rebalancing LP (scipy HiGHS)
              baselines.py    none, greedy-neighbour, LP, robust LP, 1-step oracle LP
              matching.py     greedy FCFS, Hungarian, value-augmented matching
  evaluation/ metrics.py, run_eval.py, event_eval.py, stress.py, report.py
config.yaml                   every hyperparameter, zone, cost and scenario setting
main.py                       pipeline orchestrator
```

**Decision loop (every 15-min slot):**
1. Serve requests with idle taxis.
2. Forecast the next hour from data up to now.
3. The controller sets where leftover idle taxis should be.
4. The LP computes the cheapest feasible zone-to-zone moves.

In the vehicle-level and SUMO simulators, matching runs every minute (vehicle-level) or every 30 s (SUMO) between those decisions.

### Control methods

| Method | What it does |
|---|---|
| `none` | No repositioning; taxis cruise where they dropped off. Lower bound. |
| `greedy` | Each surplus zone sends taxis to its most under-supplied neighbour. |
| `lp` | Min-cost flow toward forecast demand (point forecast). |
| `lp_robust` | Same LP toward the conformal **upper** bound (uncertainty-aware). |
| `lp_dfl`, `lp_robust_dfl` | Same LPs fed by the decision-focused forecaster. |
| `bilevel_rl` | GNN policy outputs a target supply distribution; the LP executes it. Warm-started by imitating `lp_robust`, then fine-tuned with A2C on profit. |
| `residual_rl` | Same GNN, but its output rescales robust LP's planned post-move supply per zone (zero-initialised, so it starts exactly as `lp_robust`); A2C learns only the correction. `python main.py train-rl --methods residual_rl`. |
| `meanfield_rl` | Per-zone policy splits idle taxis over stay + 8 neighbours, with **no LP**. This is the end-to-end ablation. |
| `oracle_lp` | LP toward the *true* next-slot demand. It is myopic (one slot ahead), so it is **not** an upper bound. |

Matchers in the vehicle-level simulator:
- `greedy`: oldest rider first, nearest taxi.
- `hungarian`: minimum total pickup distance.
- `value`: fare + γᵏ·V(drop-off zone, t+k) − V(taxi zone, t) − Ω·pickup km, where k counts pickup ETA plus trip time.

## Data

* **Synthetic (default).** No public trip-level Dubai data exists; RTA open data has aggregates only.
  The generator builds a 4×10 corridor of 3 km zones from Jebel Ali to DXB. It includes zone types,
  hourly profiles, a Sat–Sun weekend, day-level noise and evening event spikes. Demand is about 36k
  trips/day for 1,500 taxis, roughly 1/8 of Dubai's ~308k/day.
* **Stress days.** Appended after the test period, generated from the same random stream, so the
  stored normal days stay identical.
  * **Ramadan:** the day shifts later. Midnight–03:00 runs at ×2, there is an iftar trough 18:30–19:30 (×0.4),
    and taxis take a 1-hour iftar break. The daily total is held at 95% of normal.
  * **Concert:** drop-offs at the venue run ×2.5 from 17:00–20:00, then a ×5 egress spike that decays with τ = 40 min.
  * **New Year's Eve:** pickups around Downtown fall to ×0.35 because of road closures, followed by a ×6 spike in
    adjacent zones from 00:30–03:00.
  * All magnitudes are assumptions shaped by Gulf News/Careem timings and RTA NYE totals. No published
    Ramadan taxi volumes exist.
* **NYC TLC.** Yellow Taxi 2024-01..03 Parquet files and `taxi_zones.zip` go in `data/raw/nyc/`.
  Zone centroids are computed from the shapefile; only per-zone series are stored (the full OD would be ~2.4 GB).

## Watching the simulation (SUMO GUI)

```bash
python main.py sumo --gui --network osm --methods bilevel_rl      # real Dubai roads, RL controller
python main.py sumo --gui --network osm --methods lp_robust --fleet 150
python main.py sumo --gui --network osm --3d                     # same, in the 3D view
python main.py sumo --gui                                        # synthetic grid city
```

One hour (07:00-08:00) plays live in sumo-gui; nothing is written to `outputs/`.

**Satellite replay viewer.** `python main.py sumo --replay` records the same hour on the Dubai map for
`bilevel_rl`, `lp_robust` and `none`, using the same riders. It writes `outputs/sumo_viewer/`; open `index.html` in a browser.
* The viewer plays the taxis back over Esri satellite imagery or the OpenStreetMap street map.
* Click a taxi to see its status and route trail, a zone to see what the controller saw and sent, or any spot for Street View and Google Maps links.
* The method menu switches controllers on the same riders.

**Promo video.** `python -m src.simulation.make_video` renders a 77 s, 1080p30 commercial-style film from the
`residual_rl` replay. It writes `outputs/sumo_viewer/dubai_ai_taxi.mp4`, plus a smaller `_share.mp4` when encoded by hand.
* **Story:** a title card, then the 200-taxi overview. The 5-min forecast follows as red / blue districts, then the
  AI's recommendation, shown as arrows. Next comes a street-level follow of taxi 1, moved at 07:05, matched to a
  rider at 07:09, picking up 2 min 15 s later, with drop-off near Al Garhoud. It ends on the results cards (−13 % wait,
  100 % served) and an end card.
* **How it's made:** `src/simulation/director.js` makes every frame a pure function of its index: camera path, sim
  time and captions. `make_video.py` drives headless Edge through Playwright, waits for the satellite tiles, takes
  a screenshot, and encodes with the ffmpeg bundled in `imageio-ffmpeg`.
* **Edits and controls:** change the story in the `SCENES` list. `--frames 300,900` renders preview stills. A rerun
  resumes from the frames already on disk; `--fresh` starts over.

**3D film with voice-over** (`outputs/video3d/`): `python -m src.simulation.make_video3d --fresh`.
* **Picture:** 93 s at 1080p30. `src/simulation/video3d.html` uses MapLibre GL to render Esri satellite imagery,
  110,834 OSM buildings extruded to their tagged height or floor count (landmarks set by hand, e.g. Burj Khalifa
  828 m), and 3D taxis. The camera tilts and orbits, then follows taxi 1 in a chase view.
* **Voice:** Microsoft neural voice via `edge-tts`. The script is in `outputs/video3d/narration.json`; scene lengths
  follow the voice in `timeline.json`.
* **Music:** an original score synthesised by `src/simulation/video_music.py`, so no licensing is needed. It is
  ducked under the voice with ffmpeg sidechain compression.
* **Outputs:** `dubai_ai_taxi_3d.mp4` (full quality) and `dubai_ai_taxi_3d_share.mp4`.
  `linkedin_post.md` holds a ready-to-paste post.
* **Changing the narration:** edit `narration.json`, regenerate the clips, then rebuild `timeline.json` and `music.wav`.
  `--mix-only` re-mixes the audio without re-rendering.
* **Night look:** the film grades from day to blue hour to night (`applyTheme` / `NIGHT` in `video3d.html`). The
  satellite layer dims, towers turn to blue glass, and the 34k OSM roads glow (`roads.geojson`, built from the SUMO
  network). Taxis leave 60-second light trails, and ffmpeg adds a light grade and film grain.
* **Photoreal edition:** `python -m src.simulation.build_video3d_google` writes `video3d_google.html`, the same film on
  Google Photorealistic 3D Tiles via CesiumJS.
  * Zones are draped on the 3D city, moves are drawn as arcs, and taxis glow with light trails.
  * Render it with `python -m src.simulation.make_video3d --google --vo vo_eleven --tag _photoreal`.
  * It needs a Google Maps key with the Map Tiles API enabled, kept in `secrets/google_maps.key` and passed to the
    page only during the render.
  * The Google logo and data credits stay on screen, as the Map Tiles API policy requires for promotional videos.
* **Voice:** ElevenLabs "Brian" (`vo_eleven/`, key in `secrets/elevenlabs.key`) or edge-tts "Andrew" (`vo/`), chosen
  with `--vo`.
* **AI B-roll opening:** `python -m src.simulation.make_broll generate` makes three illustrative shots (skyline, waiting
  rider, taxi driver).
  * They are made locally with LTX-Video 2B (FP8) in ComfyUI portable at `C:\ComfyUI_windows_portable`; on an RTX 4050
    each 4 s clip takes about 35 s.
  * `make_broll cut --take skyline=1,rider=1,driver=1 --film photoreal` puts them under the opening line and
    crossfades into the film, giving `dubai_ai_taxi_photoreal_broll*.mp4` (98 s).
  * The B-roll is AI-generated mood footage, not simulation output.
* **More realism:** the buildings are untextured. Google Photorealistic 3D Tiles would add real textured buildings,
  but need a Google Maps Platform API key.

* **Map** (`--network osm`, `src/simulation/osm_map.py`):
  * Real Dubai roads, junctions and guessed traffic lights from OpenStreetMap: 41.9k edges, 216 signals.
  * Also 110k buildings, parks, water, and the Gulf with the Palm and World islands.
  * The 40 model zones lie along a 30 x 12 km corridor from Dubai Marina toward Deira, following Sheikh Zayed Road.
  * Gulf-side zones that are open sea borrow riders from the nearest zone on land.
  * The first run downloads about 120 MB from the Overpass API into `data/raw/osm/` (data (c) OpenStreetMap contributors, ODbL).
* **Taxis:** green = parked, orange = going to a pickup, red = carrying a rider, blue = being repositioned.
* **Zones:** shaded every 15 min by forecast demand minus parked and arriving taxis (red = shortage, blue = surplus).
  Labels show requests forecast, idle taxis and taxis arriving.
* **Rebalancing:** lines show the controller's moves, purple for RL and teal for LP. The status line shows the clock, served / waiting / abandoned riders and the last decision.
* **3D view** (`--3d`): starts sumo-gui in its OpenSceneGraph view, which renders slower. Adding a 3D view next to the 2D one crashes sumo-gui 1.28 on Windows, so the window is one or the other.

## Modelling assumptions (read before quoting numbers)

* **Three simulators, three fidelity levels:**
  * Fluid zone-level: fast, used for training.
  * Vehicle-level minute-step: individual taxis, a 6 km pickup radius, and 10-minute rider patience.
  * SUMO: a microscopic road grid at 50 km/h, with demand ×0.25 for speed (`sumo.scale`) and the fleet
    set separately (`sumo.fleet_scale`, or absolute sizes in `sumo.fleet_sweep`).

  Absolute KPIs differ between them, so compare methods *within* a simulator.
* **In SUMO, rebalancing gives a consistent gain when there are idle taxis to move.**
  * **Setup:** `python main.py sumo --methods none lp_robust residual_rl bilevel_rl` runs 150/200/250/300 taxis in
    two windows, 07:00–09:00 (1,080 riders) and 17:00–20:00 (2,125 riders). Riders are identical across runs, and
    all 32 episodes had zero TraCI/SUMO errors (`outputs/sumo_metrics.md`).
  * **Defaults:** rebalancing decisions every 5 min, and LP move costs equal to real SUMO road travel times
    (see the ablation below).
  * **What the controller sees:** en-route supply. Every taxi that is not parked (picking up, occupied or
    repositioning) counts at its route's last edge, in the slot its free-flow remaining time ends.
  * **Fleet of 250–300** (33–125 parked taxis per decision):
    * Robust LP cuts mean wait by 0.75–1.05 min (13–17 %) and p90 wait by 1.0–1.5 min, with up to +0.9 pts served.
    * Residual RL matches it: −0.76 to −1.09 min.
    * Bilevel RL manages only about −0.1 min.
    * Empty driving per served trip rises by about 5–8 %.
  * **Fleet of 150–200:** few taxis are free (5–29). Robust LP still adds up to +1.7 pts served (evening, 200 taxis),
    while wait changes stay within ±0.5 min.
  * **Dubai roads** (OSM, 07:00–08:00, 200 taxis, 559 riders, `python main.py sumo --replay`):
    * Residual RL and robust LP both serve 100 %, with 4.52 / 4.53 min mean wait and 8.0 min p90.
    * `none` serves 99.5 % with 5.19 min and 10.0 min p90 (−13 % mean wait for the LP).
    * Bilevel RL serves 99.8 % with 5.16 min.
  * **Why the gain is smaller than in the vehicle-level simulator** (likely explanations, not tested):
    * The vehicle-level simulator's gain is 6.23 → 4.26 min wait with Hungarian matching. It runs a full day at
      ~35 % utilisation, so there are many idle taxis to move.
    * The SUMO windows are short peaks at ×0.25 demand.
    * Hungarian matching every 30 s over a 6 km radius (two 3 km zones) closes most local gaps first.
  * **Earlier result was a bug:** a previous version reported no effect because repositioned taxis never moved.
    The `stop` idle algorithm re-parks any empty taxi without a stop where it stands, so a bare `changeTarget` +
    `resume` was undone at once. Each move now adds a held stop at the target zone, released when the dispatcher
    assigns the taxi (`send_to` / `release` in `sumo_bridge.py`).

* **Control upgrades, one at a time** (`python main.py sumo --ablation` → `outputs/sumo_ablation.md`;
  robust LP, grid network, 48 runs, zero errors):
  * **5-min instead of 15-min decisions: the biggest gain.** The wait reduction over `none` roughly doubles, e.g.
    −0.46 → −0.99 min at 300 taxis (evening) and −0.53 → −0.77 min at 250 (morning). It serves up to +1.3 pts at
    150 taxis.
  * **LP costs from SUMO fastest-route times instead of centroid distance: small, mostly positive.** For example
    +1.7 pts served at 200 taxis (evening) and −1.05 min wait at 300. Road times between neighbouring 3 km zones come
    out at 7.5 min, against 3.6 min for a straight 3 km at free-flow speed, so the distance LP underestimates move
    costs.
  * **Value-aware dispatch (Xu et al. 2018) instead of Hungarian: mixed.** It is slightly better when taxis are
    scarce and adds up to 0.2 min of wait at 250–300 in the morning, so `hungarian` stays the default.
* **Residual RL** (`residual_rl`, fluid simulator, 1,500 taxis, test days, `outputs/residual_rl_metrics.md`):
  * It ties robust LP: 96.99 % vs 96.93 % served, 3.70 vs 3.70 min mean wait, and +457 AED/day profit, all
    within the 95 % intervals.
  * That beats the old bilevel RL (92.1 % served, 4.92 min).
  * A2C raised validation profit by only 0.04 % over the starting point, which is robust LP itself. This is
    consistent with robust LP already sitting at or above the oracle LP: there is little left for RL to correct.
* **Smaller zones** (`config_z160.yaml`: 160 zones of 1.5 km, own `data_z160/` and `outputs_z160/` folders; data
  regenerated, forecaster retrained):
  * **Forecast:** GCN+GRU beats historical average at this resolution too (MAE 1.03 vs 1.09).
  * **Fluid simulator, robust LP vs the 40-zone setup:**
    * Mean wait 2.87 vs 3.70 min and p90 wait 5.6 vs 10.8 min.
    * Profit 900k vs 843k AED/day, empty-km share 31.1 vs 34.1 %.
    * Served 95.6 vs 96.9 %.
    * Caveat: the fluid simulator matches only within neighbouring zones, so smaller zones also make `none` worse
      (64 vs 79 %).
  * **SUMO** (riders differ between the two zonings, so compare each zoning's gain over its own `none`): robust LP
    gains more at 200–250 taxis, e.g. −0.69 vs −0.36 min wait (morning, 200) and +1.4 vs +0.9 pts served
    (evening, 250). Elsewhere the gain is about the same.
  * RL was not retrained at 160 zones: A2C took about 17 min per 25 episodes, about 3 h for both policies.

  Implementation notes:
  * Taxis start and riders appear mid-edge. SUMO cannot re-place an idle stop within a car length of the
    edge start when such a taxi is dispatched.
  * A taxi that still ends up parked there (for example after a teleport) is moved to another edge of its zone.
    This is counted in `unparked` in `sumo_metrics.json`, and is 0 in every run of the current sweep.
  * SUMO's own error log is counted with the TraCI errors.
* Profit = fares (12 AED flag + 2 AED/km) − 0.6 AED per empty km (pickup, rebalancing, idle cruising).
* Every headline percentage must name its baseline. Report empty km **per served trip** alongside
  served rate and waits, because empty-km share can fall while total empty km rises.
* The value-matching pickup penalty Ω = 8 AED/km was picked from {4, 8, 15} on two test days. Treat the
  value-matcher result as lightly tuned.
* OpenCity is evaluated at its native 30-min taxi resolution, and each 30-min forecast is split evenly into
  two 15-min slots. NYC data was in OpenCity's pretraining corpus, so only Dubai is a true zero-shot.

## Extending

* New forecaster: subclass `Forecaster` and register it in `models/forecasting.build_forecaster`.
* New controller: subclass `Controller` in `control/baselines.py` and register it in `evaluation/run_eval.make_controller`.
* New stress scenario: add a branch to `ingestion/synthetic._apply_scenario` and list it in `config.yaml: stress.schedule`.
