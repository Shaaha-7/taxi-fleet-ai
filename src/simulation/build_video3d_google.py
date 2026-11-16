"""Build video3d_google.html: the same film as video3d.html (story, captions, timeline, replay data),
rendered on Google Photorealistic 3D Tiles with CesiumJS instead of MapLibre + OSM extrusions.

    python -m src.simulation.build_video3d_google

Google attribution: Cesium's credit display (Google logo + data providers) stays on screen for the whole film,
as the Map Tiles API policies require for promotional videos.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent
html = (SRC / "video3d.html").read_text(encoding="utf-8")

# ---- head and DOM
html = html.replace('<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">',
                    '<link href="https://cdn.jsdelivr.net/npm/cesium@1.120.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet">')
html = html.replace('<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>',
                    '<script>window.CESIUM_BASE_URL = "https://cdn.jsdelivr.net/npm/cesium@1.120.0/Build/Cesium/";</script>\n'
                    '<script src="https://cdn.jsdelivr.net/npm/cesium@1.120.0/Build/Cesium/Cesium.js"></script>\n<script src="gkey.js"></script>')
html = html.replace('<div id="map"></div>', '<div id="map"></div><div id="gcredit"></div>')
html = html.replace("  .maplibregl-ctrl-attrib { display: none; }",
                    "  #gcredit { position: fixed; left: 56px; bottom: 14px; z-index: 12; font: 12px Montserrat, sans-serif; color: #fff; text-shadow: 0 1px 6px rgba(0,0,0,.9); white-space: nowrap; }\n"
                    "  #gcredit * { display: inline !important; position: static !important; white-space: nowrap; }\n"
                    "  .lower { bottom: 118px !important; } .stats { bottom: 118px !important; } .step { bottom: 254px !important; }\n"
                    "  .credit { bottom: 14px !important; }\n"
                    "  #gcredit img { height: 18px; vertical-align: middle; }\n"
                    "  #gcredit .cesium-credit-lightbox-overlay, #gcredit .cesium-credit-expand-link { display: none !important; }\n"
                    "  .cesium-viewer-bottom { display: none; }")
html = html.replace("Simulation: SUMO on OpenStreetMap roads &amp; buildings · Imagery © Esri, Maxar, Earthstar Geographics",
                    "Simulation: SUMO on OpenStreetMap roads · 3D city: Google")

# ---- replace the MapLibre-specific JS (geometry builders, map, frame loop, place) with Cesium
a = html.index("// ------------------------------------------------------------------ geometry builders")
b = html.index("// ------------------------------------------------------------------ the film")
CESIUM = r'''// ------------------------------------------------------------------ Cesium scene (Google Photorealistic 3D Tiles)
const D2R = Math.PI / 180;
Cesium.Ion.defaultAccessToken = "";
const viewer = new Cesium.Viewer("map", {
  globe: false, baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
  animation: false, timeline: false, fullscreenButton: false, infoBox: false, selectionIndicator: false, baseLayer: false,
  creditContainer: document.getElementById("gcredit"), requestRenderMode: false,
  contextOptions: { webgl: { preserveDrawingBuffer: true, alpha: false } },
});
const scene = viewer.scene, cam = viewer.camera;
scene.skyAtmosphere.show = true; scene.fog.enabled = true; scene.fog.density = 1.2e-4;
viewer.clock.currentTime = Cesium.JulianDate.fromIso8601("2026-01-15T13:10:00Z");    // ~17:10 Dubai: low golden sun
viewer.clock.shouldAnimate = false;
// Match MapLibre's framing: its vertical FOV is 36.87 deg; Cesium's fov is the wider (horizontal) angle.
cam.frustum.fov = 2 * Math.atan(Math.tan(18.435 * D2R) * window.innerWidth / window.innerHeight);
// Warm cinematic grade over the photoreal city.
scene.postProcessStages.add(new Cesium.PostProcessStage({ fragmentShader: `
  uniform sampler2D colorTexture; in vec2 v_textureCoordinates;
  void main() { vec3 c = texture(colorTexture, v_textureCoordinates).rgb;
    c = pow(c, vec3(0.96)); c = (c - 0.5) * 1.07 + 0.5; c *= vec3(1.045, 1.0, 0.94);
    float l = dot(c, vec3(0.299, 0.587, 0.114)); c = mix(vec3(l), c, 1.12);
    vec2 q = v_textureCoordinates - 0.5; c *= 1.0 - 0.32 * dot(q, q);
    out_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0); }` }));
let tileset = null, GROUND = -20;                                                 // ellipsoid height of the streets (refined from tiles)
(async () => {
  tileset = await Cesium.createGooglePhotorealistic3DTileset(window.GKEY, { maximumScreenSpaceError: 10, showCreditsOnScreen: true });
  scene.primitives.add(tileset);
  window.ready3d = true;
})();

const points = scene.primitives.add(new Cesium.PointPrimitiveCollection());
const glow = scene.primitives.add(new Cesium.PointPrimitiveCollection());
const trails = scene.primitives.add(new Cesium.PolylineCollection());
const C = h => Cesium.Color.fromCssColorString(h);
const NOCLIP = Number.POSITIVE_INFINITY;
const taxiPts = R.taxis.map((_, i) => points.add({ pixelSize: i === HERO ? 16 : 7, color: C(COL[0]), outlineColor: C("#000000").withAlpha(0.7),
  outlineWidth: i === HERO ? 3 : 1.2, disableDepthTestDistance: NOCLIP, scaleByDistance: new Cesium.NearFarScalar(400, 1.8, 25000, 0.55) }));
const taxiGlow = R.taxis.map((_, i) => glow.add({ pixelSize: i === HERO ? 46 : 20, color: C(COL[0]).withAlpha(0.28),
  disableDepthTestDistance: NOCLIP, scaleByDistance: new Cesium.NearFarScalar(400, 1.6, 25000, 0.5) }));
const riderPts = []; for (let k = 0; k < 160; k++) riderPts.push(points.add({ show: false, pixelSize: 9, color: C("#ffffff"), outlineColor: C("#c0392b"),
  outlineWidth: 2.5, disableDepthTestDistance: NOCLIP, scaleByDistance: new Cesium.NearFarScalar(400, 1.7, 25000, 0.6) }));
const P = (lon, lat, h = 0) => Cesium.Cartesian3.fromDegrees(lon, lat, GROUND + h);

let zonePrims = [], moveEnts = [], tripEnts = [];
function setZones(d, show) {
  for (const p of zonePrims) scene.primitives.remove(p);
  zonePrims = [];
  if (!d || !show) return;
  R.zones.forEach((pts, z) => {
    if (seaZone(z)) return;
    const g = Math.max(-1, Math.min(1, gapOf(d, z) / 5));
    const col = (g > 0 ? C("#e63c3c") : C("#3c6ee6")).withAlpha(0.18 + 0.32 * Math.abs(g));
    const inst = new Cesium.GeometryInstance({ geometry: new Cesium.PolygonGeometry({
        polygonHierarchy: new Cesium.PolygonHierarchy(pts.map(q => { const [lo, la] = ll(q); return Cesium.Cartesian3.fromDegrees(lo, la, -200); })),
        height: -200, extrudedHeight: 900 }),
      attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(col) } });
    // One primitive per zone: a ClassificationPrimitive allows a single colour across its instances.
    zonePrims.push(scene.primitives.add(new Cesium.ClassificationPrimitive({ geometryInstances: inst, asynchronous: false,
      classificationType: Cesium.ClassificationType.CESIUM_3D_TILE })));
  });
}
function setMoves(d, mode) {
  for (const e of moveEnts) viewer.entities.remove(e);
  moveEnts = [];
  if (!d || !mode) return;
  for (const [i, j, n] of d.moves) {
    const isHero = d.taxis.some(m => m[0] === HERO && m[1] === i && m[2] === j);
    if (mode === 2 && !isHero) continue;
    const [ax, ay] = ll(R.centres[i]), [bx, by] = ll(R.centres[j]);
    const arc = [0, 0.25, 0.5, 0.75, 1].map(f => P(lerp(ax, bx, f), lerp(ay, by, f), 120 + 420 * Math.sin(Math.PI * f)));
    moveEnts.push(viewer.entities.add({ polyline: { positions: arc, width: 14 + 3 * Math.min(n, 6), arcType: Cesium.ArcType.NONE,
      material: new Cesium.PolylineArrowMaterialProperty(isHero ? C("#ffc400") : C("#b06cff")) } }));
  }
}
function setTrip(t, others) {
  for (const e of tripEnts) viewer.entities.remove(e);
  tripEnts = [];
  const add = (i, w) => {
    const tg = targetAt(i, t), p = taxiAt(i, t); if (!tg || !p) return;
    if (tg[2] === 2 && i !== HERO) return;
    const [qx, qy] = ll([tg[3], tg[4]]);
    tripEnts.push(viewer.entities.add({ polyline: { positions: [P(p.lon, p.lat, 4), P(qx, qy, 4)], width: w, arcType: Cesium.ArcType.NONE,
      material: new Cesium.PolylineDashMaterialProperty({ color: C(tg[2] === 1 ? "#ff9f1a" : tg[2] === 2 ? "#ff4d4d" : "#5aa2ff"), dashLength: 18 }),
      depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: C(tg[2] === 1 ? "#ff9f1a" : "#ff4d4d").withAlpha(0.7), dashLength: 18 }) } }));
  };
  if (others) R.taxis.forEach((_, i) => i !== HERO && add(i, 2));
  add(HERO, 5);
}
function updateFleet(t, name) {
  const showHero = !["intro", "end", "overview"].includes(name);
  R.taxis.forEach((_, i) => {
    const p = taxiAt(i, t), pt = taxiPts[i], g = taxiGlow[i];
    if (!p) { pt.show = g.show = false; return; }
    const pos = P(p.lon, p.lat, i === HERO ? 3 : 2);
    const col = i === HERO && showHero ? "#ffc400" : COL[p.state];
    pt.position = g.position = pos; pt.show = g.show = name !== "intro";
    pt.color = C(col); g.color = C(col).withAlpha(i === HERO && showHero ? 0.25 + 0.15 * Math.sin(t * 1.8) : 0.22);
  });
  const [k0] = frameAt(t);
  riderPts.forEach((rp, k) => {
    const q = R.riders[k0][k];
    if (!q || name === "intro") { rp.show = false; return; }
    const [lo, la] = ll(q); rp.show = true; rp.position = P(lo, la, 2);
    rp.color = C(q[2] ? "#ffe14d" : "#ffffff"); rp.outlineColor = C(q[2] ? "#6b5800" : "#c0392b");
  });
  trails.removeAll();
  if (name === "intro") return;
  R.taxis.forEach((_, i) => {
    const now = taxiAt(i, t); if (!now) return;
    const pts = [];
    for (let s = 60; s >= 0; s -= 4) { const p = taxiAt(i, Math.max(0, t - s)); if (p) pts.push(P(p.lon, p.lat, 2)); }
    if (pts.length < 2) return;
    const hero = i === HERO && showHero;
    trails.add({ positions: pts, width: hero ? 10 : 5, material: Cesium.Material.fromType("PolylineGlow",
      { color: C(hero ? "#ffd84d" : COL[now.state]).withAlpha(hero ? 0.95 : 0.7), glowPower: 0.25, taperPower: 0.6 }) });
  });
}
// Same camera parameters as the MapLibre film: centre, web-mercator zoom, pitch from vertical, bearing.
let CUR_Z = 12;
function setCam(c) {
  CUR_Z = c.z;
  const mpp = 78271.52 * Math.cos(c.lat * D2R) / Math.pow(2, c.z);             // MapLibre zoom uses 512 px tiles
  const range = (mpp * window.innerHeight / 2) / Math.tan(18.435 * D2R);
  cam.lookAt(Cesium.Cartesian3.fromDegrees(c.lon, c.lat, GROUND),
             new Cesium.HeadingPitchRange(c.bearing * D2R, -(90 - c.pitch) * D2R, range));
  cam.lookAtTransform(Cesium.Matrix4.IDENTITY);
}
function refineGround(lon, lat) {                                                // street height under the camera target
  const h = scene.sampleHeight ? scene.sampleHeight(Cesium.Cartographic.fromDegrees(lon, lat), [points, glow, trails]) : undefined;
  if (h !== undefined && isFinite(h) && h > -80 && h < 80) GROUND = lerp(GROUND, h, 0.5);
}

'''
html = html[:a] + CESIUM + html[b:]

# frame loop
a = html.index("window.frame3d = async f => {")
b = html.index("function overlays(t, name, tl, S, simT, d) {")
FRAME = r'''window.frame3d = async f => {
  const t = f / FPS;
  const name = Object.keys(SCN).find(n => TL[n] && t >= TL[n].start && t < TL[n].end) || "end";
  const S = SCN[name], tl = TL[name], u = clamp01((t - tl.start) / (tl.end - tl.start)), e = ease(u);
  const simT = lerp(S.sim[0], S.sim[1], u);
  const a = camAt(S.cam[0], simT), b = camAt(S.cam[1], simT);
  const c = { lon: lerp(a.lon, b.lon, e), lat: lerp(a.lat, b.lat, e), z: lerp(a.z, b.z, e), pitch: lerp(a.pitch, b.pitch, e), bearing: lerpAng(a.bearing, b.bearing, e) };
  setCam(c);
  for (const zp of zonePrims) zp.show = c.z < 14.0;                              // no ground tint at street level
  updateFleet(simT, name);
  setTrip(simT, S.others && S.trips ? 1 : 0); if (!S.trips) { for (const x of tripEnts) viewer.entities.remove(x); tripEnts = []; }
  const d = decisionAt(simT), key = `${name}|${d ? d.t : -1}`;
  if (key !== lastDecisionKey) { lastDecisionKey = key; setZones(d, !!S.zones); setMoves(S.moves ? d : null, S.moves); }
  for (const zp of zonePrims) zp.show = c.z < 14.0;
  // Render until the photoreal tiles for this view are in (or give up after ~25 s).
  let stable = 0;
  for (let k = 0; k < 250 && stable < 3; k++) {
    scene.render();
    if (k === 2) refineGround(c.lon, c.lat);
    await new Promise(r => setTimeout(r, k < 3 ? 15 : 100));
    stable = tileset && tileset.tilesLoaded ? stable + 1 : 0;
  }
  setCam(c); scene.render();
  overlays(t, name, tl, S, simT, d);
};
function place(cls, lonlat, text, lift = 0) {
  const w = Cesium.SceneTransforms.worldToWindowCoordinates(scene, P(lonlat[0], lonlat[1], 3));
  if (!w) return;
  const el = document.createElement("div"); el.className = cls; el.textContent = text;
  el.style.left = w.x + "px"; el.style.top = (w.y - lift) + "px"; $("labels").appendChild(el);
}
'''
html = html[:a] + FRAME + html[b:]
html = html.replace("  const z = map.getZoom();\n", "  const z = CUR_Z;\n")
html = html.replace("const NIGHT = { intro: [0.0, 0.55], overview: [0.55, 0.9], end: [0.9, 0.85] };\n", "")
# photoreal shots look best a little closer to the ground than the OSM version
html = html.replace("<title>Dubai AI Taxi 3D</title>", "<title>Dubai AI Taxi Photoreal</title>")
(SRC / "video3d_google.html").write_text(html, encoding="utf-8")
print("wrote", SRC / "video3d_google.html")
