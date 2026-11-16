// Cinematic "director" mode for the replay viewer: open index.html#director=1&m=residual_rl.
// Every video frame is a pure function of its index (camera, sim time, captions), so a capture script can
// render frame f, wait for the satellite tiles, screenshot, and move on -- no real-time playback involved.
// The story follows taxi1 in the residual_rl replay: moved by the AI at 07:05, matched to a rider at 07:09,
// pickup at 07:11, drop-off near Al Garhoud at 07:24.
(() => {
  const FPS = 30;
  const STAR = "taxi1";
  const ease = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  const lerp = (a, b, f) => a + (b - a) * f;
  const clamp01 = x => Math.max(0, Math.min(1, x));
  const fade = (t, a, b, edge = 0.6) => clamp01(Math.min((t - a) / edge, (b - t) / edge));

  // Camera keys are {lat, lon, z} or "star" (follow the hero taxi). Times are video seconds.
  const OVERVIEW = { lat: 25.175, lon: 55.245, z: 11.15 };
  const CITY = { lat: 25.19, lon: 55.27, z: 12.6 };
  const DECISION = { lat: 25.192, lon: 55.282, z: 13.6 };
  const SCENES = [
    { t: [0, 5.5], sim: [0, 20], cam: [{ lat: 25.16, lon: 55.22, z: 10.6 }, OVERVIEW], zones: false, title: "intro" },
    { t: [5.5, 13], sim: [20, 290], cam: [OVERVIEW, { ...OVERVIEW, z: 11.5 }], zones: false,
      cap: ["200 taxis on Dubai's real road network", "Morning peak, simulated car by car in SUMO on OpenStreetMap roads"], stats: true },
    { t: [13, 21.5], sim: [302, 345], cam: [{ ...OVERVIEW, z: 11.5 }, CITY], zones: true,
      cap: ["Every 5 minutes, AI forecasts demand", "A graph neural network predicts requests per district. Red: more riders expected than taxis nearby"],
      tag: "1 · FORECAST" },
    { t: [21.5, 30], sim: [345, 400], cam: [CITY, DECISION], zones: true, arrows: true, rec: true,
      cap: ["Then it moves idle taxis ahead of demand", "Robust LP + RL recommend the cheapest moves. Taxi 1: Business Bay → MBR – Al Merkad"],
      tag: "2 · POSITION" },
    { t: [30, 38], sim: [400, 560], cam: [DECISION, "star:16.6"], zones: true, links: true, sel: true,
      cap: ["Taxi 1 heads toward the demand", "Before any rider has booked: the blue line is its new waiting area"], tag: "2 · POSITION" },
    { t: [38, 45], sim: [560, 600], cam: ["star:16.6", "mid:16.4"], links: true, sel: true, zones: true,
      cap: ["07:09 · A rider near Downtown books a ride", "Matched in under 30 seconds to the nearest free taxi (orange line)"], tag: "3 · DISPATCH" },
    { t: [45, 53], sim: [600, 712], cam: ["mid:16.4", "star:17.4"], links: true, sel: true, zones: true,
      cap: ["Pickup in 2 min 15 s", "The car drives the real streets: junctions, traffic lights and turns"], tag: "3 · DISPATCH" },
    { t: [53, 61], sim: [712, 1040], cam: ["star:17.4", "star:14.2"], links: true, sel: true, zones: true,
      cap: ["Rider on board, heading to Al Garhoud", "Meanwhile the platform keeps rebalancing the other 199 taxis"], tag: "4 · RIDE" },
    { t: [61, 71], sim: [1040, 1600], cam: ["star:14.2", { ...OVERVIEW, z: 11.35 }], zones: true, arrows: true, results: true },
    { t: [71, 77], sim: [1600, 1700], cam: [{ ...OVERVIEW, z: 11.35 }, { ...OVERVIEW, z: 11.0 }], zones: false, title: "end" },
  ];
  const DURATION = SCENES[SCENES.length - 1].t[1];

  let star = -1, starRec = -1, overlay;
  function camPoint(k, simT) {
    if (typeof k !== "string") return k;
    const [kind, z] = k.split(":");
    const p = taxiAt(star, simT);
    if (kind === "star") return { lat: p.lat, lon: p.lon, z: +z };
    const tg = targetAt(star, simT);                             // midpoint between taxi and the rider it is going to
    const q = tg ? ll([tg[3], tg[4]]) : [p.lat, p.lon];
    return { lat: (p.lat + q[0]) / 2, lon: (p.lon + q[1]) / 2, z: +z };
  }

  function el(html) { const d = document.createElement("div"); d.innerHTML = html; return d.firstElementChild; }
  window.initDirector = () => {
    document.body.classList.add("director");
    map.options.zoomSnap = 0; map.options.fadeAnimation = false; map.options.zoomAnimation = false;
    map.getContainer().classList.remove("leaflet-fade-anim", "leaflet-zoom-anim");
    map.invalidateSize();
    star = R.taxis.indexOf(STAR);
    window.HERO = star;
    const d0 = R.decisions.find(d => d.taxis.some(m => m[0] === star));
    const mv = d0 && d0.taxis.find(m => m[0] === star);
    starRec = mv ? d0.moves.findIndex(m => m[0] === mv[1] && m[1] === mv[2]) : -1;
    overlay = el(`<div id="dir">
      <div class="vignette"></div>
      <div class="brand"><span class="dot"></span>DUBAI AI TAXI FLEET</div>
      <div class="clockchip"><span id="dClock">07:00</span><small id="dTag"></small></div>
      <div class="title" id="dTitle"></div>
      <div class="lower" id="dLower"><div class="bar"></div><div><div class="h" id="dH"></div><div class="s" id="dS"></div></div></div>
      <div class="stats" id="dStats"></div>
      <div class="results" id="dResults"></div>
      <div class="credit">Simulation: SUMO on OpenStreetMap roads · Imagery © Esri, Maxar, Earthstar Geographics</div>
    </div>`);
    document.body.appendChild(overlay);
    window.directorInfo = { fps: FPS, frames: Math.round(DURATION * FPS) };
    return window.directorInfo;
  };

  window.directorFrame = f => {
    const t = f / FPS;
    const sc = SCENES.find(s => t >= s.t[0] && t < s.t[1]) || SCENES[SCENES.length - 1];
    const u = clamp01((t - sc.t[0]) / (sc.t[1] - sc.t[0])), e = ease(u);
    simT = lerp(sc.sim[0], sc.sim[1], u);
    const a = camPoint(sc.cam[0], simT), b = camPoint(sc.cam[1], simT);
    map.setView([lerp(a.lat, b.lat, e), lerp(a.lon, b.lon, e)], lerp(a.z, b.z, e), { animate: false });
    $("showZones").checked = !!sc.zones; $("showLinks").checked = !!sc.links;
    for (const lay of [zoneLayer, labelLayer]) { if (sc.zones && !map.hasLayer(lay)) lay.addTo(map); if (!sc.zones && map.hasLayer(lay)) map.removeLayer(lay); }
    sel = sc.sel ? { taxi: star } : null;
    recSel = sc.rec ? starRec : -1;
    drawn = recKey = undefined;
    const [d] = decisionAt(simT);
    if (!sc.arrows && !sc.rec) moveLayer.clearLayers();
    drawDecision(sc.arrows || sc.rec ? d : (sc.zones ? d : null), sc.arrows || sc.rec ? decisionAt(simT)[1] : -2);
    if (!sc.arrows && !sc.rec) moveLayer.clearLayers();
    draw();
    // ---- overlays
    const [k0] = frameAt(simT), [served, waiting, , req] = R.stats[k0];
    $("dClock").textContent = clock(simT);
    $("dTag").textContent = sc.tag || "";
    const tt = $("dTitle");
    if (sc.title === "intro") {
      tt.innerHTML = `<div class="kicker">AI-DRIVEN FLEET MANAGEMENT</div><h1>Every taxi in the right place,<br>before the rider asks.</h1>
        <p>Forecast demand · Position the fleet · Dispatch in seconds</p>`;
      tt.style.opacity = fade(t, sc.t[0] - 1, sc.t[1], 0.9);
    } else if (sc.title === "end") {
      tt.innerHTML = `<div class="kicker">DUBAI AI TAXI FLEET</div><h1>Forecast. Position. Dispatch.</h1>
        <p>Graph neural network forecasting · robust optimisation · reinforcement learning · tested in SUMO on Dubai's roads</p>`;
      tt.style.opacity = fade(t, sc.t[0], sc.t[1] + 1, 0.9);
    } else tt.style.opacity = 0;
    const lw = $("dLower");
    if (sc.cap) { $("dH").textContent = sc.cap[0]; $("dS").textContent = sc.cap[1]; lw.style.opacity = fade(t, sc.t[0], sc.t[1]);
                  lw.style.transform = `translateX(${(1 - ease(clamp01((t - sc.t[0]) / 0.6))) * -40}px)`; }
    else lw.style.opacity = 0;
    const st = $("dStats");
    st.style.opacity = sc.stats || sc.zones && !sc.results ? fade(t, sc.t[0], sc.t[1]) : 0;
    st.innerHTML = `<div><b>${req}</b><span>requests</span></div><div><b>${served}</b><span>served</span></div><div><b>${waiting}</b><span>waiting</span></div>`;
    const rs = $("dResults");
    if (sc.results) {
      const g = k => clamp01((t - sc.t[0] - 0.6 - 0.5 * k) / 0.8);
      const num = (v, k) => (v * ease(g(k))).toFixed(v % 1 ? 1 : 0);
      rs.innerHTML = `<div class="rh">Same riders, same roads: AI rebalancing vs none</div>
        <div class="cards">
          <div class="card" style="opacity:${g(0)}"><b>−${num(13, 0)}%</b><span>average wait</span><small>4.5 vs 5.2 min</small></div>
          <div class="card" style="opacity:${g(1)}"><b>${num(100, 1)}%</b><span>riders served</span><small>vs 99.5% with no rebalancing</small></div>
          <div class="card" style="opacity:${g(2)}"><b>−${num(2, 2)} min</b><span>slowest 10% of waits</span><small>8.0 vs 10.0 min</small></div>
        </div><div class="rf">SUMO on Dubai's OpenStreetMap roads, 07:00–08:00, 200 taxis, 559 riders (scaled demand)</div>`;
      rs.style.opacity = fade(t, sc.t[0], sc.t[1]);
    } else rs.style.opacity = 0;
  };

  // True when every visible satellite / label tile has finished loading.
  window.directorReady = () => !sat.isLoading() && !(map.hasLayer(places) && places.isLoading()) &&
    [...document.querySelectorAll(".leaflet-tile-container img")].every(im => im.complete && im.naturalWidth > 0);
})();
