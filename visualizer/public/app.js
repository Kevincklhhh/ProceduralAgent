// ProceduralAgent task visualizer frontend.
// Talks to the video/data server on :4002 (same split as self_data/annotator).

// API port can be overridden with ?api=<port> if 4002 is taken.
const API_PORT = new URLSearchParams(window.location.search).get('api') || '4002';
const API = `http://${window.location.hostname}:${API_PORT}`;

const PALETTE = [
  '#4f8cc9', '#52a675', '#b07cc6', '#c9824f', '#5bb5b0',
  '#8a9a4f', '#c95f8a', '#7a82d6', '#4faac9', '#a68b52',
];

const els = {
  viewNav: document.getElementById('view-nav'),
  videoView: document.getElementById('video-view'),
  jsonView: document.getElementById('json-view'),
  jsonFileList: document.getElementById('json-file-list'),
  jsonFilename: document.getElementById('json-filename'),
  jsonTree: document.getElementById('json-tree'),
  jsonLinks: document.getElementById('json-links'),
  recipeSelect: document.getElementById('recipe-select'),
  errorsOnly: document.getElementById('errors-only'),
  recMeta: document.getElementById('rec-meta'),
  stepsHeader: document.getElementById('steps-header'),
  remindersHeader: document.getElementById('reminders-header'),
  select: document.getElementById('video-select'),
  meta: document.getElementById('video-meta'),
  player: document.getElementById('player'),
  layerGt: document.getElementById('layer-gt'),
  layerBaseline: document.getElementById('layer-baseline'),
  layerProposed: document.getElementById('layer-proposed'),
  layerQualcomm: document.getElementById('layer-qualcomm'),
  axis: document.getElementById('time-axis'),
  tracks: document.getElementById('tracks'),
  taskTitle: document.getElementById('task-title'),
  taskSteps: document.getElementById('task-steps'),
  taskReminders: document.getElementById('task-reminders'),
  activeEvents: document.getElementById('active-events'),
  tooltip: document.getElementById('tooltip'),
  vlmContext: document.getElementById('vlm-context'),
  armSelect: document.getElementById('arm-select'),
  vlmCallMeta: document.getElementById('vlm-call-meta'),
  vlmFrames: document.getElementById('vlm-frames'),
  vlmPrompt: document.getElementById('vlm-prompt'),
  monContext: document.getElementById('monitor-context'),
  monSelect: document.getElementById('mon-select'),
  replayBtn: document.getElementById('replay-btn'),
  monCallMeta: document.getElementById('mon-call-meta'),
  monState: document.getElementById('mon-state'),
  monWhy: document.getElementById('mon-why'),
  monPoll: document.getElementById('mon-poll'),
  // T1 / T2 split view
  t1t2View: document.getElementById('t1t2-view'),
  player2: document.getElementById('player2'),
  qcArmSelect: document.getElementById('qc-arm-select'),
  qcArmMeta: document.getElementById('qc-arm-meta'),
  t2ShowQualcomm: document.getElementById('t2-show-qualcomm'),
  t1Axis: document.getElementById('t1-axis'),
  t1Tracks: document.getElementById('t1-tracks'),
  t2Axis: document.getElementById('t2-axis'),
  t2Tracks: document.getElementById('t2-tracks'),
  t2GtList: document.getElementById('t2-gt-list'),
  t2PredList: document.getElementById('t2-pred-list'),
};

let state = {
  videos: [],
  current: null,    // video entry
  timeline: null,   // /api/timeline response
  duration: 0,
  stepColors: {},   // step_id -> color
  taskFiles: [],    // /api/taskfiles response
  currentFile: null,
  baseline: null,   // /api/baseline response {arms:[...]}
  armIdx: 0,        // selected baseline arm
  monitor: null,    // /api/monitor response {plan, arms:[...]}
  monArmIdx: 0,     // selected monitor arm
  qualcrun: null,   // /api/qualcrun response {arms:[...]} (turn/stream T1+T2 predictions)
  qcArmIdx: 0,      // selected qualcrun arm
  timelineLayers: { gt: true, baseline: true, proposed: true, qualcomm: false },
  replaying: false, // virtual-clock replay (when no video is driving the playhead)
  virtualT: 0,
  replayTimer: null,
};

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(0).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

async function init() {
  setupViewNav();
  setupTimelineLayerControls();
  loadTaskFiles();
  state.stepDesc = await (await fetch(`${API}/api/stepdescriptions`)).json().catch(() => ({}));
  state.videos = await (await fetch(`${API}/api/videos`)).json();

  // Recipe filter dropdown: "All recipes" + each recipe with its video count.
  const recipes = [...new Set(state.videos.map(v => v.activity_name).filter(Boolean))].sort();
  els.recipeSelect.innerHTML =
    `<option value="">All recipes (${state.videos.length})</option>` +
    recipes.map(r => {
      const n = state.videos.filter(v => v.activity_name === r).length;
      return `<option value="${escapeAttr(r)}">${r} (${n})</option>`;
    }).join('');

  els.recipeSelect.addEventListener('change', () => populateVideoSelect());
  els.errorsOnly.addEventListener('change', () => populateVideoSelect());
  els.select.addEventListener('change', () => loadVideo(els.select.value));
  els.t2ShowQualcomm?.addEventListener('change', () => renderT1T2());

  populateVideoSelect();
}

function setupTimelineLayerControls() {
  const pairs = [
    ['gt', els.layerGt],
    ['baseline', els.layerBaseline],
    ['proposed', els.layerProposed],
    ['qualcomm', els.layerQualcomm],
  ];
  for (const [key, input] of pairs) {
    if (!input) continue;
    input.checked = !!state.timelineLayers[key];
    input.addEventListener('change', () => {
      state.timelineLayers[key] = input.checked;
      syncLayerPanels();
      renderTimeline();
      updatePlayhead();
    });
  }
}

function syncLayerPanels() {
  const hasBaseline = !!(state.baseline?.arms || []).length;
  const hasMonitor = !!(state.monitor?.arms || []).length;
  els.vlmContext.hidden = !hasBaseline || !state.timelineLayers.baseline;
  els.monContext.hidden = !hasMonitor || !state.timelineLayers.proposed;
}

// Build the video dropdown from the current recipe / errors-only filters,
// grouped into <optgroup>s by recipe so 384 entries stay navigable.
function populateVideoSelect() {
  const recipe = els.recipeSelect.value;
  const errorsOnly = els.errorsOnly.checked;
  let list = state.videos.filter(v =>
    (!recipe || v.activity_name === recipe) &&
    (!errorsOnly || v.is_error));

  // Group by recipe.
  const groups = {};
  for (const v of list) {
    const key = v.activity_name || 'Other';
    (groups[key] = groups[key] || []).push(v);
  }
  const html = Object.keys(groups).sort().map(g => {
    const opts = groups[g].map(v => {
      const flags = [v.is_error ? `⚠${v.num_error_steps}` : null,
                     v.source === 'videos_480p' ? '480p' : null,
                     v.has_annotation ? 'ann' : null,
                     v.has_qualcomm ? `qc${v.num_qc_mistakes ? '⚠' + v.num_qc_mistakes : ''}` : null]
                    .filter(Boolean).join(' ');
      return `<option value="${escapeAttr(v.id)}">${v.id}${flags ? '  · ' + flags : ''}</option>`;
    }).join('');
    return `<optgroup label="${escapeAttr(g)} (${groups[g].length})">${opts}</optgroup>`;
  }).join('');
  els.select.innerHTML = html || '<option>(no videos match)</option>';

  if (list.length) {
    const keep = list.find(v => v.id === state.current?.id);
    loadVideo(keep ? keep.id : list[0].id);
  }
}

async function loadVideo(videoId) {
  const video = state.videos.find(v => v.id === videoId);
  if (!video) return;
  state.current = video;
  state.duration = video.duration || 0;
  els.select.value = videoId;

  els.player.src = `${API}/videos/${video.source}/${encodeURIComponent(video.filename)}`;

  const resp = await fetch(`${API}/api/timeline/${encodeURIComponent(videoId)}`);
  state.timeline = resp.ok ? await resp.json() : null;
  state._propKey = {};   // proposed stage -> CC4D color-key cache (per recording)

  const bresp = await fetch(`${API}/api/baseline/${encodeURIComponent(videoId)}`);
  state.baseline = bresp.ok ? await bresp.json() : { arms: [] };
  state.armIdx = 0;
  populateArmSelect();

  const mresp = await fetch(`${API}/api/monitor/${encodeURIComponent(videoId)}`);
  state.monitor = mresp.ok ? await mresp.json() : { plan: null, arms: [] };
  state.monArmIdx = 0;
  stopReplay();
  populateMonArmSelect();

  // T1/T2 view: same video in its own player + the turn/stream Qualcomm run.
  els.player2.src = els.player.src;
  const qresp = await fetch(`${API}/api/qualcrun/${encodeURIComponent(videoId)}`);
  state.qualcrun = qresp.ok ? await qresp.json() : { arms: [] };
  // Default to a turn-based arm when present (the wired-in result of interest).
  const qarms = state.qualcrun.arms || [];
  const turnIdx = qarms.findIndex(a => /turn/.test(a.arm));
  state.qcArmIdx = turnIdx >= 0 ? turnIdx : 0;
  populateQcArmSelect();

  const m = state.timeline?.meta;
  const bits = [fmtTime(video.duration), `[${video.source}]`];
  if (m) {
    bits.push(`P${m.person}`, `env${m.environment}`);
    if (m.is_error) bits.push(`⚠ ${video.num_error_steps} error step${video.num_error_steps === 1 ? '' : 's'}`);
  }
  els.meta.textContent = bits.join(' · ');

  assignColors();
  renderTimeline();
  renderSidePanel();
  renderT1T2();
}

function escapeAttr(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

function assignColors() {
  state.stepColors = {};
  state.stepColors['other'] = '#55565f';
  const task = state.timeline?.task;
  if (task?.steps) {
    // Hand-written task (HD-EPIC clip / activity-8 ann tracks): color by its semantic
    // step_id. CC4D GT/Pred/Qualcomm tracks instead key by numeric step_id (see below).
    task.steps.forEach((s, i) => { state.stepColors[s.step_id] = PALETTE[i % PALETTE.length]; });
  }
  // Seed stable colors for CC4D GT steps by their numeric step_id, in
  // first-performed order so the palette walks the timeline left-to-right.
  for (const s of state.timeline?.gt?.steps || []) {
    const key = String(s.step_id);
    if (!(key in state.stepColors)) {
      state.stepColors[key] = PALETTE[Object.keys(state.stepColors).length % PALETTE.length];
    }
  }
}

function colorFor(stepId) {
  if (!(stepId in state.stepColors)) {
    state.stepColors[stepId] = PALETTE[Object.keys(state.stepColors).length % PALETTE.length];
  }
  return state.stepColors[stepId];
}

// ---- Unified step labels (one scheme shared by GT / Pred / Qualcomm tracks) ----
// Keyed by CC4D step_id, so the SAME step renders the same abbreviation + color on
// every track. The abbreviation ("fill_milk", "add_cinnamon") is our own derivation
// from the CC4D description (verb + most distinctive noun) — a display label only;
// the canonical step_id and full description are kept in tooltips (stepDescOf).
const _STOP = new Set(['a', 'an', 'the', 'of', 'to', 'with', 'into', 'in', 'on', 'for',
  'and', 'until', 'about', 'your', 'from', 'over', 'at', 'then', 'or']);
const _UNIT = new Set(['tsp', 'teaspoon', 'teaspoons', 'tbsp', 'tablespoon', 'tablespoons',
  'oz', 'ounce', 'ounces', 'minute', 'minutes', 'second', 'seconds', 'piece', 'pieces',
  'inch', 'g', 'ml', 'gram', 'grams', 'cup', 'cups']);
const _GENERIC = new Set(['mug', 'bowl', 'pan', 'cup', 'contents', 'mixture', 'skillet',
  'plate', 'pot', 'dish', 'container', 'pieces', 'piece']);

function abbreviate(desc) {
  const clean = cleanDesc(desc);
  const toks = (clean.toLowerCase().match(/[a-z][a-z-]*/g) || [])
    .filter(w => !_STOP.has(w) && !_UNIT.has(w));
  if (!toks.length) return clean.split(/\s+/)[0] || String(desc);
  const verb = toks[0];
  let noun = null;
  for (let i = toks.length - 1; i > 0; i--) {
    if (!_GENERIC.has(toks[i])) { noun = toks[i]; break; }
  }
  return noun && noun !== verb ? `${verb}_${noun}` : verb;
}

// step_id -> unified display label (abbreviation for CC4D ids; raw id otherwise).
function stepLabel(id) {
  const k = String(id);
  if (k === 'other') return 'other';
  const desc = state.stepDesc && state.stepDesc[k];
  return desc ? abbreviate(desc) : k;
}

// step_id -> cleaned CC4D description for tooltips (empty for non-CC4D ids).
function stepDescOf(id) {
  const desc = state.stepDesc && state.stepDesc[String(id)];
  return desc ? cleanDesc(desc) : '';
}

// Map a free-text Qualcomm instruction to the best-matching performed GT step_id
// (token overlap), so its span can share the unified color + label. null if weak.
function _tokset(s) {
  return new Set((cleanDesc(s).toLowerCase().match(/[a-z]+/g) || []).filter(w => w.length > 2));
}
function matchStepId(text, gtSteps) {
  const w = _tokset(text);
  if (!w.size) return null;
  let best = null, bestScore = 0;
  for (const s of gtSteps || []) {
    if (s.start_time < 0) continue;
    const sw = _tokset(s.description);
    let inter = 0;
    for (const x of w) if (sw.has(x)) inter++;
    const sc = inter / Math.min(w.size, sw.size || 1);
    if (sc > bestScore) { bestScore = sc; best = String(s.step_id); }
  }
  return bestScore >= 0.5 ? best : null;
}

// Proposed-monitor stages are keyed by monitor stage names (fill_milk,
// add_cinnamon, ...), but GT and Baseline color by numeric CC4D step_id. Resolve
// a stage name to the CC4D step_id it represents so the Proposed lane shares the
// unified palette with the matching GT/Baseline lane. Prefer an explicit cc4d
// mapping from the compiled plan; otherwise token-match the unit instruction
// against the performed GT steps (same heuristic as the Qualcomm layer). Blocks
// (e.g. quiet_middle) span several CC4D steps, so they keep their own key and
// land on a distinct color. Cached per recording (reset on timeline load).
function proposedColorKey(stageId) {
  const k = String(stageId);
  const cache = state._propKey || (state._propKey = {});
  if (k in cache) return cache[k];
  const u = planUnitById(stageId);
  let key = k;
  if (u && u.kind !== 'block') {
    const explicit = (u.cc4d != null) ? u.cc4d : u.cc4d_step_id;
    if (explicit != null) {
      key = String(explicit);
    } else if (u.instruction) {
      const mid = matchStepId(u.instruction, state.timeline?.gt?.steps);
      if (mid) key = mid;
    }
  }
  cache[k] = key;
  return key;
}

// Baseline arm selector (one entry per experiments/t1_baseline/<arm>/ with a
// result for this recording). Hides the whole VLM-context panel if none.
function populateArmSelect() {
  const arms = state.baseline?.arms || [];
  syncLayerPanels();
  if (!arms.length) return;
  els.armSelect.innerHTML = arms.map((a, i) => {
    const c = a.cost?.vlm_calls != null ? ` · ${a.cost.vlm_calls} calls` : '';
    return `<option value="${i}">${escapeAttr(a.arm)}${c}</option>`;
  }).join('');
  els.armSelect.value = String(state.armIdx);
  els.armSelect.onchange = () => {
    state.armIdx = parseInt(els.armSelect.value, 10) || 0;
    renderTimeline();
    updatePlayhead();
  };
}

// ---------------------------------------------------------------------------
// Procedure-monitor layer: tracks + replay panel (from /api/monitor)
// ---------------------------------------------------------------------------

function monArm() { return state.monitor?.arms?.[state.monArmIdx] || null; }
function planUnitById(id) {
  const units = state.monitor?.plan?.units || [];
  const direct = units.find(u => u.id === id);
  if (direct) return direct;
  // a VLM-labeled sub-stage is a block member: inherit the block's sensing context
  // (role, primitives, sleep, requires/produces) so the schedule + panel resolve it.
  for (const u of units) {
    const m = (u.members || []).find(mm => mm.id === id);
    if (m) return { id, kind: 'member', parent: u.id, members: [], cc4d: m.cc4d,
      requires: u.requires, produces: u.produces, sensing_role: u.sensing_role,
      primitives: u.primitives, sleep: u.sleep, instruction: m.instruction };
  }
  return null;
}
function activeUnitId(arm, t) {
  for (const iv of arm.stage_intervals || []) {
    if (iv.start_s <= t && t < iv.end_s) return iv.stage;
  }
  return null;
}

const SENSOR_NAMES = {
  D1: 'D1 microwave cycle',
  D2: 'D2 appliance motor',
  D3: 'D3 cook end',
  D4: 'D4 cook start',
  D5: 'D5 water flow',
  D6: 'D6 timer logic',
  VLM: 'VLM',
};

function sensorName(p) {
  return SENSOR_NAMES[p] || p;
}

// Monitor arm selector (one entry per experiments/proposed_system/<arm>/ with a
// result for this recording). Hides the monitor panel if none.
function populateMonArmSelect() {
  const arms = state.monitor?.arms || [];
  syncLayerPanels();
  if (!arms.length) return;
  els.monSelect.innerHTML = arms.map((a, i) => {
    const c = a.cost?.vlm_calls != null ? ` · ${a.cost.vlm_calls} VLM` : '';
    return `<option value="${i}">${escapeAttr(a.arm)}${c}</option>`;
  }).join('');
  els.monSelect.value = String(state.monArmIdx);
  els.monSelect.onchange = () => {
    state.monArmIdx = parseInt(els.monSelect.value, 10) || 0;
    renderTimeline();
    updatePlayhead();
  };
}

// The four monitor tracks: sensor schedule (energy), predicted stages, sensor
// events, and state transitions. Called from renderTimeline().
function renderMonitorTracks() {
  const arm = monArm();
  if (!arm) return false;

  // 1. sensor schedule — one lane per primitive. Audio lanes shade ON over the
  //    intervals where the active step binds them (cheap, green); the VLM lane is
  //    discrete poll markers (red) over a faint duty-cycle region — the energy story.
  const prims = [];
  for (const u of (state.monitor?.plan?.units || []))
    for (const p of u.primitives) if (!prims.includes(p)) prims.push(p);
  prims.sort((a, b) => (a === 'VLM') - (b === 'VLM') || a.localeCompare(b));
  for (const p of prims) {
    const label = p === 'VLM' ? 'Proposed: sparse VLM' : `Proposed: ${sensorName(p)}`;
    const { track, lane } = makeTrack(label);
    if (p === 'VLM') {
      for (const iv of arm.stage_intervals || []) {
        const u = planUnitById(iv.stage);
        if (u && u.primitives.includes('VLM')) {
          lane.appendChild(makeSegment(iv.start_s, iv.end_s, '', '#e5484d22',
            `VLM duty-cycled region (${iv.stage})`, 'sensor-region'));
        }
      }
      for (const poll of arm.polls || []) {
        const v = poll.verdict;
        lane.appendChild(makeSegment(poll.t, poll.t + 2.0, '◆', '#e5484d',
          `VLM poll @${fmtTime(poll.t)}\n${v ? v.step_id + ' (' + (v.status || '') + ')' : 'no member match'}` +
          `\n${poll.n_frames} frames · ${poll.latency_s}s`, 'marker vlm-poll'));
      }
    } else {
      for (const iv of arm.stage_intervals || []) {
        const u = planUnitById(iv.stage);
        if (u && u.primitives.includes(p)) {
          lane.appendChild(makeSegment(iv.start_s, iv.end_s, p, '#52a675',
            `${sensorName(p)} ON during ${iv.stage}\n${fmtTime(iv.start_s)} – ${fmtTime(iv.end_s)}`, 'sensor-on'));
        }
      }
    }
    els.tracks.appendChild(track);
  }

  // 2. predicted stages
  {
    const { track, lane } = makeTrack(`Proposed: stages (${arm.arm})`);
    for (const iv of arm.stage_intervals || []) {
      lane.appendChild(makeSegment(iv.start_s, iv.end_s, stepLabel(iv.stage),
        colorFor(proposedColorKey(iv.stage)),
        `${iv.stage}\n${fmtTime(iv.start_s)} – ${fmtTime(iv.end_s)}`, 'pred'));
    }
    els.tracks.appendChild(track);
  }

  // 3. sensor events (point markers)
  {
    const { track, lane } = makeTrack('Proposed: events');
    for (const e of arm.sensor_events || []) {
      lane.appendChild(makeSegment(e.t_s, e.t_s + 1.5, '●', '#4f8cc9',
        `${sensorName(e.primitive)}.${e.event} @${fmtTime(e.t_s)}` +
        (e.step_id ? `\nstep ${e.step_id}` : ''), 'marker'));
    }
    els.tracks.appendChild(track);
  }

  // 4. transitions (start/complete with the rule that fired)
  {
    const { track, lane } = makeTrack('Proposed: transitions');
    for (const tr of arm.transition_trace || []) {
      const sym = tr.transition === 'start' ? '▸' : '◼';
      lane.appendChild(makeSegment(tr.t_s, tr.t_s + 1.5, sym,
        tr.transition === 'start' ? '#5bb5b0' : '#b07cc6',
        `${tr.transition} ${tr.step_or_block} @${fmtTime(tr.t_s)}\nrule: ${tr.rule_id}` +
        `\nevidence: ${(tr.evidence || []).join(', ') || '—'}`, 'marker'));
    }
    els.tracks.appendChild(track);
  }
  return true;
}

// Replay panel at the playhead: active unit (requires/produces + which detectors
// are ON vs asleep), the step-state ledger, the last rule fired, and the last VLM
// poll verdict.
function updateMonitorContext(t) {
  const arm = monArm();
  if (!arm || els.monContext.hidden) return;
  const uid = activeUnitId(arm, t);
  const u = uid ? planUnitById(uid) : null;
  els.monCallMeta.textContent = `${arm.arm} · ${fmtTime(t)}`;

  if (u) {
    els.monState.innerHTML =
      `<div><b style="color:${colorFor(proposedColorKey(uid))}">${escapeHtml(stepLabel(uid))}</b> ` +
      `<span class="role">${escapeHtml(u.sensing_role || '')}</span></div>` +
      `<div class="muted">requires: ${escapeHtml(u.requires.join(', ') || '—')} · ` +
      `produces: ${escapeHtml(u.produces.join(', ') || '—')}</div>` +
      `<div>sensors <span class="sensor-tag on">ON: ${escapeHtml(u.primitives.join(', ') || '—')}</span> ` +
      `<span class="sensor-tag asleep">ASLEEP: ${escapeHtml(u.sleep.join(', ') || '—')}</span></div>`;
  } else {
    els.monState.innerHTML = '<span class="muted">(no active step)</span>';
  }

  // step-state ledger reconstructed from transitions up to t
  const st = {};
  for (const tr of arm.transition_trace || []) {
    if (tr.t_s > t) continue;
    st[tr.step_or_block] = tr.transition === 'complete' ? 'complete' : 'active';
  }
  const ledger = (state.monitor?.plan?.units || []).map(un => {
    const s = st[un.id] || (un.requires.every(r => st[r] === 'complete') ? 'eligible' : 'not_started');
    return `<span class="ledger ${s}" title="${s}">${escapeHtml(stepLabel(un.id))}</span>`;
  }).join(' ');
  let why = null;
  for (const tr of arm.transition_trace || []) if (tr.t_s <= t) why = tr;
  const whyHtml = why
    ? `<b>last:</b> ${why.transition} ${escapeHtml(why.step_or_block)} — rule <code>${escapeHtml(why.rule_id)}</code>` +
      ((why.evidence || []).length ? ` · ${escapeHtml(why.evidence.join(', '))}` : '')
    : '<span class="muted">—</span>';
  els.monWhy.innerHTML = `<div class="ledger-row">${ledger}</div><div class="why muted">${whyHtml}</div>`;

  let poll = null;
  for (const p of arm.polls || []) if (p.t <= t) poll = p;
  if (poll) {
    const v = poll.verdict;
    els.monPoll.innerHTML =
      `<b>VLM poll @${fmtTime(poll.t)}</b> · ${poll.n_frames} frames · ${poll.latency_s}s · ` +
      (v ? `<span style="color:${colorFor(proposedColorKey(v.step_id))}">${escapeHtml(stepLabel(v.step_id))}</span> ` +
           `(${escapeHtml(v.status || '?')})<div class="muted">${escapeHtml(v.evidence || '')}</div>`
         : '<span class="muted">no member match</span>');
  } else {
    els.monPoll.innerHTML = (arm.polls && arm.polls.length)
      ? '<span class="muted">(no VLM poll yet at this time)</span>'
      : '<span class="muted">(audio-only arm — no VLM polls)</span>';
  }
}

// Virtual-clock replay so the run animates even without the video playing.
function currentT() {
  return state.replaying ? state.virtualT : (els.player.currentTime || 0);
}
function toggleReplay() {
  if (state.replaying) return stopReplay();
  state.replaying = true;
  state.virtualT = 0;
  els.replayBtn.textContent = '⏹ stop';
  try { els.player.pause(); } catch { /* no video */ }
  const stepS = Math.max(1, state.duration / 120);
  state.replayTimer = setInterval(() => {
    state.virtualT += stepS;
    if (state.virtualT >= state.duration) { state.virtualT = state.duration; updatePlayhead(); return stopReplay(); }
    updatePlayhead();
  }, 200);
}
function stopReplay() {
  state.replaying = false;
  if (state.replayTimer) clearInterval(state.replayTimer);
  state.replayTimer = null;
  if (els.replayBtn) els.replayBtn.textContent = '▶ replay';
}

// ---------------------------------------------------------------------------
// View switching + Task JSON browser
// ---------------------------------------------------------------------------

function setupViewNav() {
  els.viewNav.addEventListener('click', e => {
    const btn = e.target.closest('button[data-view]');
    if (btn) switchView(btn.dataset.view);
  });
  document.getElementById('json-expand-all').addEventListener('click', () => {
    els.jsonTree.querySelectorAll('details').forEach(d => d.open = true);
  });
  document.getElementById('json-collapse-all').addEventListener('click', () => {
    els.jsonTree.querySelectorAll('details').forEach((d, i) => d.open = d.parentElement === els.jsonTree);
  });
}

function switchView(view) {
  for (const btn of els.viewNav.querySelectorAll('button')) {
    btn.classList.toggle('active', btn.dataset.view === view);
  }
  els.videoView.hidden = view !== 'video';
  els.jsonView.hidden = view !== 'json';
  els.t1t2View.hidden = view !== 't1t2';
  // The recording picker is shared by the Videos and T1/T2 views.
  const showVideoCtrls = view === 'video' || view === 't1t2';
  els.select.hidden = !showVideoCtrls;
  els.meta.hidden = !showVideoCtrls;
  els.recipeSelect.hidden = !showVideoCtrls;
  document.getElementById('errors-only-label').hidden = !showVideoCtrls;
  if (view !== 'video') els.player.pause?.();
  if (view !== 't1t2') els.player2.pause?.();
  if (view === 't1t2') renderT1T2();
}

async function loadTaskFiles() {
  state.taskFiles = await (await fetch(`${API}/api/taskfiles`)).json();
  els.jsonFileList.innerHTML = '';
  for (const f of state.taskFiles) {
    const div = document.createElement('div');
    div.className = 'json-file-item';
    div.dataset.filename = f.filename;
    const sub = f.kind === 'task'
      ? `task_id: ${f.task_id}${f.title ? ' — ' + f.title : ''}`
      : f.kind === 'annotation' ? `video_id: ${f.video_id}`
      : 'fill-in template for new annotations';
    div.innerHTML = `<div><span class="kind ${f.kind}">${f.kind}</span>${f.filename}</div>
      <div class="sub">${sub}</div>`;
    div.addEventListener('click', () => openTaskFile(f.filename));
    els.jsonFileList.appendChild(div);
  }
}

async function openTaskFile(filename, view) {
  if (view !== false) switchView('json');
  state.currentFile = filename;
  for (const div of els.jsonFileList.querySelectorAll('.json-file-item')) {
    div.classList.toggle('active', div.dataset.filename === filename);
  }
  const resp = await fetch(`${API}/api/taskfiles/${encodeURIComponent(filename)}`);
  els.jsonFilename.textContent = `tasks/${filename}`;
  els.jsonTree.innerHTML = '';
  if (!resp.ok) {
    els.jsonTree.innerHTML = '<div class="muted">Failed to load file.</div>';
    return;
  }
  const data = await resp.json();
  els.jsonTree.appendChild(renderJsonNode(data, null, true));
}

// Build a collapsible <details>-based tree for an arbitrary JSON value.
function renderJsonNode(value, key, open) {
  const keyHtml = key !== null
    ? `<span class="json-key">"${escapeHtml(key)}"</span><span class="json-punct">: </span>`
    : '';

  if (Array.isArray(value) || (value !== null && typeof value === 'object')) {
    const isArr = Array.isArray(value);
    const entries = isArr ? value.map((v, i) => [i, v]) : Object.entries(value);
    const details = document.createElement('details');
    if (open) details.open = true;
    const summary = document.createElement('summary');
    const preview = isArr
      ? `<span class="json-punct">[</span><span class="json-count">${entries.length} item${entries.length === 1 ? '' : 's'}</span><span class="json-punct">]</span>`
      : `<span class="json-punct">{</span><span class="json-count">${entries.length} key${entries.length === 1 ? '' : 's'}</span><span class="json-punct">}</span>`;
    summary.innerHTML = keyHtml + preview;
    details.appendChild(summary);
    for (const [k, v] of entries) {
      details.appendChild(renderJsonNode(v, isArr ? String(k) : k, false));
    }
    return details;
  }

  const div = document.createElement('div');
  div.className = 'json-leaf';
  let valHtml;
  if (typeof value === 'string') valHtml = `<span class="json-string">"${escapeHtml(value)}"</span>`;
  else if (typeof value === 'number') valHtml = `<span class="json-number">${value}</span>`;
  else if (typeof value === 'boolean') valHtml = `<span class="json-bool">${value}</span>`;
  else valHtml = `<span class="json-null">null</span>`;
  div.innerHTML = keyHtml + valHtml;
  return div;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Timeline rendering
// ---------------------------------------------------------------------------

function pct(t) {
  return state.duration > 0 ? (100 * t / state.duration) : 0;
}

// `player` defaults to the main Videos-view player; the T1/T2 view passes its own
// <video> so clicks there seek that player instead.
function makeSegment(startS, endS, label, color, tooltipText, extraClass, player) {
  const div = document.createElement('div');
  div.className = 'segment' + (extraClass ? ' ' + extraClass : '');
  const left = pct(startS);
  const width = Math.max(pct(endS) - left, 0.3);
  div.style.left = left + '%';
  div.style.width = width + '%';
  div.style.background = color;
  div.textContent = label;
  div.addEventListener('mousemove', e => showTooltip(e, tooltipText));
  div.addEventListener('mouseleave', hideTooltip);
  div.addEventListener('click', e => {
    e.stopPropagation();
    const p = player || els.player;
    p.currentTime = startS;
    p.play();
  });
  return div;
}

function makeTrack(labelText, player) {
  const track = document.createElement('div');
  track.className = 'track';
  const label = document.createElement('div');
  label.className = 'track-label';
  label.textContent = labelText;
  const lane = document.createElement('div');
  lane.className = 'track-lane';
  // Click empty lane space to seek.
  lane.addEventListener('click', e => {
    const rect = lane.getBoundingClientRect();
    (player || els.player).currentTime = state.duration * (e.clientX - rect.left) / rect.width;
  });
  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.style.left = '0%';
  lane.appendChild(playhead);
  track.appendChild(label);
  track.appendChild(lane);
  return { track, lane };
}

function renderTimeline() {
  els.tracks.innerHTML = '';
  renderAxis();
  const tl = state.timeline;
  if (!tl) {
    const note = document.createElement('div');
    note.className = 'muted';
    note.style.marginLeft = '170px';
    note.textContent = 'No annotation or ground truth for this video.';
    els.tracks.appendChild(note);
    return;
  }

  const ann = tl.annotation;
  let renderedAny = false;
  if (state.timelineLayers.gt && ann) {
    // Stage segments
    const { track, lane } = makeTrack('GT: task stages');
    for (const seg of ann.stage_segments || []) {
      lane.appendChild(makeSegment(
        seg.start_s, seg.end_s, seg.step_id, colorFor(seg.step_id),
        `${seg.step_id}\n${fmtTime(seg.start_s)} – ${fmtTime(seg.end_s)}`
      ));
    }
    els.tracks.appendChild(track);
    renderedAny = true;

    if ((ann.reminder_windows || []).length) {
      const { track, lane } = makeTrack('GT: reminders');
      for (const w of ann.reminder_windows || []) {
        lane.appendChild(makeSegment(
          w.start_s, w.end_s, w.reminder_id, '#f5a623',
          `${w.reminder_id}\nexpected: ${w.expected_action}\n${fmtTime(w.start_s)} – ${fmtTime(w.end_s)}`
        ));
      }
      els.tracks.appendChild(track);
      renderedAny = true;
    }

    if ((ann.mistake_events || []).length) {
      const { track, lane } = makeTrack('GT: mistakes');
      for (const m of ann.mistake_events || []) {
        lane.appendChild(makeSegment(
          m.start_s, m.end_s, m.event_id, '#e5484d',
          `${m.event_id} (${m.type})\nexpected: ${m.expected_action}\n${fmtTime(m.start_s)} – ${fmtTime(m.end_s)}`,
          'marker'
        ));
      }
      els.tracks.appendChild(track);
      renderedAny = true;
    }
  }

  if (state.timelineLayers.gt && tl.gt) {
    const { track, lane } = makeTrack('GT: steps');
    for (const s of tl.gt.steps || []) {
      if (s.start_time < 0) continue; // step not performed
      const stepId = String(s.step_id);
      const label = stepLabel(stepId);
      const errText = (s.errors || []).map(e => `⚠ ${e.tag}: ${e.description}`).join('\n');
      lane.appendChild(makeSegment(
        s.start_time, s.end_time, label, colorFor(stepId),
        `${cleanDesc(s.description)}\n${fmtTime(s.start_time)} – ${fmtTime(s.end_time)}` +
          (s.modified_description ? `\nactual: ${s.modified_description}` : '') +
          (errText ? '\n' + errText : ''),
        s.errors && s.errors.length ? 'error-step' : ''
      ));
    }
    els.tracks.appendChild(track);
    renderedAny = true;

    const errSteps = (tl.gt.steps || []).filter(s => s.errors && s.errors.length && s.start_time >= 0);
    if (errSteps.length) {
      const { track, lane } = makeTrack('GT: errors');
      for (const s of errSteps) {
        for (const e of s.errors) {
          lane.appendChild(makeSegment(
            s.start_time, s.end_time, e.tag, '#e5484d',
            `${e.tag}: ${e.description}\nstep: ${s.description}\n${fmtTime(s.start_time)} – ${fmtTime(s.end_time)}`,
            'marker'
          ));
        }
      }
      els.tracks.appendChild(track);
      renderedAny = true;
    }
  }

  if (state.timelineLayers.baseline) renderedAny = renderPredictedTrack() || renderedAny;
  if (state.timelineLayers.proposed) renderedAny = renderMonitorTracks() || renderedAny;
  if (state.timelineLayers.qualcomm) renderedAny = renderQualcommTracks(tl.qualcomm) || renderedAny;
  if (!renderedAny) {
    const note = document.createElement('div');
    note.className = 'muted';
    note.style.marginLeft = '170px';
    note.textContent = 'Select at least one timeline layer above.';
    els.tracks.appendChild(note);
  }
}

// Baseline predicted-step track for the selected arm. Same palette as GT steps
// (color keyed by stringified step_id), so the predicted lane lines up visually
// against the GT-steps lane directly above it.
function renderPredictedTrack() {
  const arm = state.baseline?.arms?.[state.armIdx];
  if (!arm) return false;
  const { track, lane } = makeTrack(`Baseline VLM: stages (${arm.arm})`);
  for (const iv of arm.stage_intervals || []) {
    const sid = String(iv.stage);
    const desc = stepDescOf(sid);
    lane.appendChild(makeSegment(
      iv.start_s, iv.end_s, stepLabel(sid), colorFor(sid),
      `pred: step ${sid}${desc ? ' — ' + desc : ''}\n${fmtTime(iv.start_s)} – ${fmtTime(iv.end_s)}`, 'pred'
    ));
  }
  els.tracks.appendChild(track);
  renderBaselineCallTrack(arm);
  return true;
}

function renderBaselineCallTrack(arm) {
  const calls = arm.calls || [];
  const events = arm.events || [];
  if (!calls.length && !events.length) return;
  const { track, lane } = makeTrack('Baseline VLM: calls/actions');
  for (const c of calls) {
    const action = c.action?.type && c.action.type !== 'none' ? `\naction: ${c.action.type} — ${c.action.message || ''}` : '';
    const nFrames = c.n_frames ?? (c.frame_urls || []).length;
    lane.appendChild(makeSegment(c.t, c.t + 1.5, '◆', '#e5484d',
      `VLM call @${fmtTime(c.t)}\npred: ${stepLabel(c.pred_step)} (${c.pred_status || '?'})` +
      `\n${nFrames} frames · ${c.latency_s || '?'}s latency${action}`, 'marker vlm-poll'));
  }
  for (const e of events) {
    const t = e.t ?? e.timestamp_s;
    lane.appendChild(makeSegment(t, t + 2.0, '!', '#f5a623',
      `${e.type || e.action_type || 'action'} @${fmtTime(t)}\n${e.message || e.id || ''}`, 'marker'));
  }
  els.tracks.appendChild(track);
}

// Show the VLM call whose context window covers the playhead: the frames sent,
// the predicted step/status, and the exact prompt the model received.
function updateVlmContext(t) {
  const arm = state.baseline?.arms?.[state.armIdx];
  if (els.vlmContext.hidden || !arm) return;
  const calls = arm.calls || [];
  if (!calls.length) {
    els.vlmCallMeta.textContent = '(no per-call trace — re-run that arm with --trace)';
    els.vlmFrames.innerHTML = '';
    els.vlmPrompt.innerHTML = '';
    return;
  }
  let c = calls.find(x => x.start_s <= t && t < x.end_s);
  if (!c) c = calls.reduce((b, x) => Math.abs(x.t - t) < Math.abs(b.t - t) ? x : b, calls[0]);
  const col = colorFor(String(c.pred_step));
  const pdesc = stepDescOf(c.pred_step);
  const nFrames = c.n_frames ?? (c.frame_urls || []).length;
  els.vlmCallMeta.innerHTML =
    `call @ ${fmtTime(c.t)} · window ${fmtTime(c.start_s)}–${fmtTime(c.end_s)} · ` +
    `pred <b style="color:${col}">${escapeHtml(stepLabel(c.pred_step))}</b> ` +
    `<span class="muted">step ${escapeHtml(String(c.pred_step))}${pdesc ? ' — ' + escapeHtml(pdesc) : ''}</span> ` +
    `(${escapeHtml(c.pred_status || '?')}) · ${nFrames} frames · ${c.latency_s}s`;
  els.vlmFrames.innerHTML = (c.frame_urls || [])
    .map(u => `<img src="${API}${u}" loading="lazy" title="${escapeAttr(u.split('/').pop())}" />`).join('');
  // Only the per-call-varying parts: the previous-responses history fed this call
  // and the raw response. Guideline + system prompt are static; frames are above.
  const prev = c.prev_responses || [];
  let histHtml;
  if (prev.length && typeof prev[0] === 'object') {
    // new format: prior VLM responses {t, step, status, evidence}
    histHtml = prev.map(h =>
      `<div class="prev-resp"><span class="hist-step" style="border-color:${colorFor(String(h.step))}">` +
      `${fmtTime(h.t)} · ${escapeHtml(stepLabel(h.step))}</span> ` +
      `<span class="muted">(${escapeHtml(h.status || '?')})</span> ${escapeHtml(h.evidence || '')}</div>`).join('');
  } else {
    // old format: list of completed step ids
    histHtml = prev.map(id =>
      `<span class="hist-step" style="border-color:${colorFor(String(id))}">${escapeHtml(stepLabel(id))}</span>`).join(' ');
  }
  els.vlmPrompt.innerHTML =
    `<div class="vlm-hist"><b>Previous observations sent this call:</b><br>` +
    (histHtml || '<span class="muted">none yet</span>') + `</div>` +
    `<div class="vlm-raw"><b>raw model response</b><pre>${escapeHtml(c.raw || '')}</pre></div>`;
}

// The Qualcomm Interactive Cooking layer (the live-guidance annotation behind
// LiveMamba). Two tracks: instruction-to-instruction guided-step spans, and a
// feedback track of point markers (✓ success confirmations + ⚠ typed mistakes,
// whose timestamp is when the mistake first becomes visible).
function renderQualcommTracks(qc) {
  if (!qc) return false;
  let rendered = false;
  const instr = qc.instructions || [];
  if (instr.length) {
    const { track, lane } = makeTrack('Qualcomm steps');
    const endBound = qc.finish || qc.end || state.duration;
    const gtSteps = state.timeline?.gt?.steps;
    instr.forEach((ins, i) => {
      const end = i + 1 < instr.length ? instr[i + 1].t : endBound;
      // Join the free-text instruction to a CC4D step_id so it shares the
      // unified color + label with the GT/Pred tracks (grey fallback if no match).
      const mid = matchStepId(ins.text, gtSteps);
      const label = mid ? stepLabel(mid) : '·';
      const color = mid ? colorFor(mid) : '#55565f';
      lane.appendChild(makeSegment(
        ins.t, end, label, color,
        `${ins.text}` + (mid ? `\n→ step ${mid}` : '\n(no step match)') +
          `\n${fmtTime(ins.t)} – ${fmtTime(end)}`
      ));
    });
    els.tracks.appendChild(track);
    rendered = true;
  }

  const successes = qc.successes || [];
  const mistakes = qc.mistakes || [];
  if (successes.length || mistakes.length) {
    const { track, lane } = makeTrack('Qualcomm feedback');
    for (const s of successes) {
      lane.appendChild(makeSegment(
        s.t, s.t, '', '#52a675',
        `✓ ${fmtTime(s.t)}\n${s.text}`, 'marker qc-success'
      ));
    }
    for (const m of mistakes) {
      lane.appendChild(makeSegment(
        m.t, m.t, m.class, '#e5484d',
        `⚠ ${m.class} mistake — visible at ${fmtTime(m.t)}\n${m.text}`, 'marker'
      ));
    }
    els.tracks.appendChild(track);
    rendered = true;
  }
  return rendered;
}

// ---------------------------------------------------------------------------
// T1 / T2 split view
// ---------------------------------------------------------------------------

// Qualcomm-run arm selector (one entry per experiments/qualcomm_run/<arm>/ with a
// result for this recording: qwen36_zs_turn, qwen36_zs_stream, ...).
function populateQcArmSelect() {
  const arms = state.qualcrun?.arms || [];
  if (!els.qcArmSelect) return;
  els.qcArmSelect.innerHTML = arms.length
    ? arms.map((a, i) => {
        const c = a.cost?.vlm_calls != null ? ` · ${a.cost.vlm_calls} calls` : '';
        return `<option value="${i}">${escapeAttr(a.arm)}${c}</option>`;
      }).join('')
    : '<option>(no qualcomm_run predictions)</option>';
  els.qcArmSelect.value = String(state.qcArmIdx);
  els.qcArmSelect.onchange = () => {
    state.qcArmIdx = parseInt(els.qcArmSelect.value, 10) || 0;
    renderT1T2();
  };
}

// Reminder color by subtype, shared by the T2 GT and predicted tracks/lists so a
// technique-error GT reminder and a technique-error prediction read the same hue.
const REMINDER_COLORS = {
  order: '#b07cc6', missing_step: '#c9824f', technique: '#e5484d',
  preparation: '#c95f8a', measurement: '#4f8cc9', timing: '#5bb5b0',
  temperature: '#a68b52',
};
function reminderColor(subtype) { return REMINDER_COLORS[subtype] || '#8a8a95'; }
function reminderLabel(ev) { return ev.subtype === 'missing_step' ? 'missing' : (ev.subtype || ev.cls || ev.class || '?'); }

// Human-readable content for a GT reminder event: prefer the original CC4D error
// text (joined on the anchor step server-side), then order pivot / missing
// reminder_id / anchor description.
function reminderContent(ev) {
  if (ev.content) return ev.content;
  if (ev.subtype === 'order' && ev.pivot) return `out of order (pivot: ${ev.pivot})`;
  if (ev.reminder_id) return ev.reminder_id;
  if (ev.anchor_desc) return cleanDesc(ev.anchor_desc);
  return ev.subtype || ev.cls || '';
}

// The split T1/T2 timelines. T1 = GT steps vs predicted steps (stage_intervals);
// T2 = GT proactive reminders (from CC4D, windowed) vs predicted reminder events.
function renderT1T2() {
  if (!els.t1Tracks) return;
  els.t1Tracks.innerHTML = '';
  els.t2Tracks.innerHTML = '';
  renderAxisInto(els.t1Axis);
  renderAxisInto(els.t2Axis);

  const tl = state.timeline;
  const arm = state.qualcrun?.arms?.[state.qcArmIdx] || null;
  const p2 = els.player2;
  els.qcArmMeta.textContent = arm
    ? `${(arm.stage_intervals || []).length} steps · ${(arm.events || []).length} reminders · ${arm.meta?.mode || ''}`
    : '';

  // ---- T1: step localization ----
  {
    const { track, lane } = makeTrack('GT: steps', p2);
    for (const s of tl?.gt?.steps || []) {
      if (s.start_time < 0) continue;
      const sid = String(s.step_id);
      const errText = (s.errors || []).map(e => `⚠ ${e.tag}: ${e.description}`).join('\n');
      lane.appendChild(makeSegment(
        s.start_time, s.end_time, stepLabel(sid), colorFor(sid),
        `${cleanDesc(s.description)}\n${fmtTime(s.start_time)} – ${fmtTime(s.end_time)}` + (errText ? '\n' + errText : ''),
        s.errors && s.errors.length ? 'error-step' : '', p2));
    }
    els.t1Tracks.appendChild(track);
  }
  {
    const { track, lane } = makeTrack(arm ? `Predicted: steps (${arm.arm})` : 'Predicted: steps', p2);
    for (const iv of arm?.stage_intervals || []) {
      const sid = String(iv.stage);
      const desc = stepDescOf(sid);
      lane.appendChild(makeSegment(
        iv.start_s, iv.end_s, stepLabel(sid), colorFor(sid),
        `pred: step ${sid}${desc ? ' — ' + desc : ''}\n${fmtTime(iv.start_s)} – ${fmtTime(iv.end_s)}`, 'pred', p2));
    }
    els.t1Tracks.appendChild(track);
  }

  // ---- T2: proactive reminders ----
  const faEvents = (tl?.family_a?.events) || [];
  {
    const { track, lane } = makeTrack('GT: reminders', p2);
    for (const ev of faEvents) {
      const col = reminderColor(ev.subtype);
      const pt = ev.start_s === ev.end_s;
      const when = pt ? `@${fmtTime(ev.t ?? ev.start_s)}` : `${fmtTime(ev.start_s)} – ${fmtTime(ev.end_s)}`;
      lane.appendChild(makeSegment(
        ev.start_s, ev.end_s, reminderLabel(ev), col,
        `${ev.cls} / ${ev.subtype}\n${reminderContent(ev)}\n${when}`,
        pt ? 'marker reminder-gt' : 'reminder-gt', p2));
    }
    els.t2Tracks.appendChild(track);
  }
  // Optional reference lane: the ORIGINAL Qualcomm mistake feedback (execution-only,
  // no order/missing) — the un-extended benchmark target, for comparison with our GT.
  if (els.t2ShowQualcomm?.checked) {
    const mistakes = tl?.qualcomm?.mistakes || [];
    const { track, lane } = makeTrack('Qualcomm (original)', p2);
    for (const m of mistakes) {
      lane.appendChild(makeSegment(
        m.t, m.t, m.class, reminderColor(m.class),
        `Qualcomm mistake / ${m.class} @${fmtTime(m.t)}\n${m.text || ''}`, 'marker', p2));
    }
    els.t2Tracks.appendChild(track);
  }
  {
    const { track, lane } = makeTrack(arm ? `Predicted: reminders (${arm.arm})` : 'Predicted: reminders', p2);
    for (const ev of arm?.events || []) {
      const sub = ev.subtype || ev.class;
      lane.appendChild(makeSegment(
        ev.t, ev.t, sub, reminderColor(ev.subtype),
        `${ev.class} / ${ev.subtype} @${fmtTime(ev.t)}\n${ev.message || ''}`, 'marker', p2));
    }
    els.t2Tracks.appendChild(track);
  }

  // ---- VLM calls + context windows: WHEN the model was invoked to check for a mistake and
  // WHAT it saw. This is a T2 (reminder-detection) diagnostic: each call is a mistake check.
  // Two elements per call so dense tick-based arms stay readable: (1) a FAINT band over the
  // sampled context window [win_start, win_end] (these tile/overlap on ungated arms — that is
  // the point: near-continuous coverage), and (2) a CRISP tick at the call's decision moment t
  // (red = fired a reminder, grey = silent). Hover either for full call detail.
  if ((arm?.calls || []).length) {
    const { track, lane } = makeTrack(`VLM calls + context (${(arm.calls || []).length})`, p2);
    for (const c of arm.calls) {
      const fired = (c.fired || []).length > 0;
      const ws = c.win_start ?? c.t, we = c.win_end ?? c.t;
      const tip = `VLM ${c.kind || 'mistake'} call @${fmtTime(c.t)}\n` +
        `context window ${fmtTime(ws)} – ${fmtTime(we)} · ${c.n_frames ?? '?'} frames` +
        (c.latency_s != null ? ` · ${c.latency_s}s` : '') + `\nstep ${c.step_id}\n` +
        (fired ? `FIRED: ${(c.fired || []).join(', ')}` : 'no mistake') +
        (c.answer ? `\nanswer: ${c.answer}` : '');
      lane.appendChild(makeSegment(ws, we, '', '#5b6472', tip,
        'vlm-window' + (fired ? ' fired' : ''), p2));               // faint window band
      lane.appendChild(makeSegment(c.t, c.t, '', fired ? '#e5484d' : '#8a93a3', tip,
        'vlm-moment' + (fired ? ' fired' : ''), p2));               // crisp call-moment tick
    }
    els.t2Tracks.appendChild(track);
  }

  renderReminderLists(faEvents, arm?.events || [], !!arm);
  updateT1T2Playhead();
}

// The "time + content" side lists for T2: GT reminders (CC4D-derived) and the
// model's predicted reminders.
function renderReminderLists(gt, pred, hasArm) {
  els.t2GtList.innerHTML = gt.length
    ? gt.map(ev => {
        const col = reminderColor(ev.subtype);
        const when = ev.start_s === ev.end_s ? `@${fmtTime(ev.t ?? ev.start_s)}` : `${fmtTime(ev.start_s)} – ${fmtTime(ev.end_s)}`;
        return `<div class="rem-item" style="border-left-color:${col}">
          <div class="rem-head"><span class="rem-tag" style="background:${col}">${escapeHtml(reminderLabel(ev))}</span>
          <span class="muted">${when}</span></div>
          <div>${escapeHtml(reminderContent(ev))}</div></div>`;
      }).join('')
    : '<div class="muted">no GT reminders for this recording</div>';
  els.t2PredList.innerHTML = pred.length
    ? pred.map(ev => {
        const col = reminderColor(ev.subtype);
        return `<div class="rem-item" style="border-left-color:${col}">
          <div class="rem-head"><span class="rem-tag" style="background:${col}">${escapeHtml(ev.subtype || ev.class || '?')}</span>
          <span class="muted">@${fmtTime(ev.t)}</span></div>
          <div>${escapeHtml(ev.message || '')}</div></div>`;
      }).join('')
    : `<div class="muted">${hasArm
        ? 'this arm predicted no reminders for this recording'
        : 'no qualcomm_run result for this recording'}</div>`;
}

function updateT1T2Playhead() {
  if (!els.t1t2View) return;
  const t = els.player2?.currentTime || 0;
  for (const ph of els.t1t2View.querySelectorAll('.playhead')) {
    ph.style.left = pct(t) + '%';
  }
}

function renderAxis() {
  renderAxisInto(els.axis);
}

function renderAxisInto(target) {
  if (!target) return;
  target.innerHTML = '';
  if (state.duration <= 0) return;
  const targetTicks = 10;
  const raw = state.duration / targetTicks;
  const steps = [5, 10, 15, 30, 60, 120, 300];
  const step = steps.find(s => s >= raw) || 600;
  for (let t = 0; t <= state.duration; t += step) {
    const tick = document.createElement('div');
    tick.className = 'axis-tick';
    tick.style.left = pct(t) + '%';
    tick.textContent = fmtTime(t);
    target.appendChild(tick);
  }
}

// ---------------------------------------------------------------------------
// Side panel
// ---------------------------------------------------------------------------

function renderSidePanel() {
  const tl = state.timeline;
  const task = tl?.task;
  const meta = tl?.meta;
  els.taskSteps.innerHTML = '';
  els.taskReminders.innerHTML = '';

  els.taskTitle.textContent = meta?.activity_name || task?.title || tl?.video_id || 'No annotation';

  // Recording metadata line.
  els.recMeta.innerHTML = '';
  if (meta) {
    const g = tl.task_graph;
    els.recMeta.innerHTML =
      `<span class="meta-chip">activity ${meta.activity_id}</span>` +
      `<span class="meta-chip">person ${meta.person}</span>` +
      `<span class="meta-chip">env ${meta.environment}</span>` +
      (meta.is_error
        ? `<span class="meta-chip err">error recording</span>`
        : `<span class="meta-chip ok">normal recording</span>`) +
      (g ? `<span class="meta-chip">graph: ${g.steps.length} steps, ${g.edges.length} edges</span>` : '');
  }

  // Links to source JSON files in tasks/ (open the Task JSON view).
  els.jsonLinks.innerHTML = '';
  for (const f of [task?._file, tl?.annotation?._file].filter(Boolean)) {
    const a = document.createElement('span');
    a.className = 'json-link';
    a.textContent = `{ } tasks/${f}`;
    a.addEventListener('click', () => openTaskFile(f));
    els.jsonLinks.appendChild(a);
  }

  // Steps panel = the performed GT step segments (the annotations), with their
  // descriptions, times, and any error tags inline. Falls back to a
  // hand-written task's step list for recordings that have no CC4D GT (the
  // HD-EPIC clip).
  const gtSteps = tl?.gt?.steps || null;
  if (gtSteps && gtSteps.length) {
    els.stepsHeader.textContent = `Steps (${gtSteps.length} performed)`;
    gtSteps.forEach((s, i) => {
      const key = String(s.step_id);
      const skipped = s.start_time < 0;
      const div = document.createElement('div');
      div.className = 'step-item' + (s.errors.length ? ' has-error' : '');
      div.dataset.stepId = key;
      div.style.borderLeftColor = colorFor(key);
      const timeStr = skipped ? '<span class="muted">not performed</span>'
        : `${fmtTime(s.start_time)} – ${fmtTime(s.end_time)}`;
      const errHtml = s.errors.map(e =>
        `<div class="step-err">⚠ ${e.tag}: ${escapeHtml(e.description || '')}</div>`).join('');
      div.innerHTML = `
        <div class="step-order">${i + 1}</div>
        <div class="step-body">
          <div>${escapeHtml(cleanDesc(s.description))}</div>
          <div class="step-duration">${timeStr}</div>
          ${s.modified_description ? `<div class="step-modified">actual: ${escapeHtml(s.modified_description)}</div>` : ''}
          ${errHtml}
        </div>`;
      if (!skipped) div.addEventListener('click', () => { els.player.currentTime = s.start_time; els.player.play(); });
      els.taskSteps.appendChild(div);
    });
  } else if (task?.steps) {
    els.stepsHeader.textContent = 'Steps';
    for (const s of task.steps) {
      const dur = s.expected_duration_s;
      const div = document.createElement('div');
      div.className = 'step-item';
      div.dataset.stepId = s.step_id;
      div.style.borderLeftColor = colorFor(s.step_id);
      div.innerHTML = `
        <div class="step-order">${s.order}</div>
        <div class="step-body">
          <div>${escapeHtml(s.instruction)}</div>
          <div class="step-id">${s.step_id}</div>
          ${dur ? `<div class="step-duration">expected ${dur.min}–${dur.max}s (typ. ${dur.typical}s)</div>` : ''}
        </div>`;
      els.taskSteps.appendChild(div);
    }
  } else {
    els.taskSteps.innerHTML = '<div class="muted">—</div>';
  }

  // Reminders only exist on hand-written tasks; hide the section otherwise.
  const reminders = task?.reminders || [];
  els.remindersHeader.hidden = reminders.length === 0;
  for (const r of reminders) {
    const div = document.createElement('div');
    div.className = 'reminder-item' + (r.type === 'warning' ? ' warning' : '');
    div.innerHTML = `<div><b>${r.reminder_id}</b> <span class="muted">(${r.type}, step: ${r.step_id})</span></div>
      <div>“${escapeHtml(r.message)}”</div>
      <div class="trigger">trigger: ${escapeHtml(r.trigger)}</div>`;
    els.taskReminders.appendChild(div);
  }
  updatePlayhead();
}

// CC4D descriptions are "Verb-Verb the rest"; collapse the doubled verb prefix.
function cleanDesc(d) {
  return String(d || '').replace(/^([A-Za-z ]+?)-\1?/, (m, v) => v.trim() + ' ')
    .replace(/^\s*-\s*/, '').replace(/\s+/g, ' ').trim() || d;
}

// Active step/event display + playhead position, driven by timeupdate.
function updatePlayhead() {
  const t = currentT();
  for (const ph of els.videoView.querySelectorAll('.playhead')) {
    ph.style.left = pct(t) + '%';
  }

  const tl = state.timeline;
  const events = [];
  let activeStepIds = new Set();

  if (tl?.annotation) {
    for (const seg of tl.annotation.stage_segments || []) {
      if (t >= seg.start_s && t < seg.end_s) activeStepIds.add(seg.step_id);
    }
    for (const w of tl.annotation.reminder_windows || []) {
      if (t >= w.start_s && t < w.end_s) {
        events.push({ tag: 'reminder window', cls: 'reminder', text: `${w.reminder_id} → ${w.expected_action}` });
      }
    }
    for (const m of tl.annotation.mistake_events || []) {
      if (t >= m.start_s && t < m.end_s) {
        events.push({ tag: m.type, cls: '', text: `${m.event_id} → ${m.expected_action}` });
      }
    }
  }
  if (tl?.gt) {
    for (const s of tl.gt.steps || []) {
      if (s.start_time >= 0 && t >= s.start_time && t < s.end_time) {
        activeStepIds.add(String(s.step_id));
        for (const e of s.errors || []) {
          events.push({ tag: e.tag, cls: '', text: e.description });
        }
      }
    }
  }
  if (tl?.qualcomm) {
    // Surface a Qualcomm mistake when the playhead is within 2s of its
    // visibility timestamp (the events are points, not spans).
    for (const m of tl.qualcomm.mistakes || []) {
      if (Math.abs(t - m.t) <= 2) {
        events.push({ tag: `qc ${m.class}`, cls: '', text: m.text });
      }
    }
  }

  for (const div of els.taskSteps.querySelectorAll('.step-item')) {
    div.classList.toggle('active', activeStepIds.has(div.dataset.stepId));
  }

  els.activeEvents.innerHTML = events.length
    ? events.map(e => `<div class="event-item"><span class="tag ${e.cls}">${e.tag}</span>${e.text}</div>`).join('')
    : '<div class="muted">none</div>';

  updateVlmContext(t);
  updateMonitorContext(t);
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function showTooltip(e, text) {
  els.tooltip.textContent = text;
  els.tooltip.style.display = 'block';
  const pad = 12;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = els.tooltip.getBoundingClientRect();
  if (x + r.width > window.innerWidth) x = e.clientX - r.width - pad;
  if (y + r.height > window.innerHeight) y = e.clientY - r.height - pad;
  els.tooltip.style.left = x + 'px';
  els.tooltip.style.top = y + 'px';
}

function hideTooltip() {
  els.tooltip.style.display = 'none';
}

els.player.addEventListener('timeupdate', updatePlayhead);
els.player.addEventListener('play', () => { if (state.replaying) stopReplay(); });
if (els.replayBtn) els.replayBtn.addEventListener('click', toggleReplay);
els.player.addEventListener('loadedmetadata', () => {
  // Prefer the browser-reported duration over ffprobe if they differ.
  if (els.player.duration && isFinite(els.player.duration)) {
    state.duration = els.player.duration;
    renderTimeline();
    updatePlayhead();
  }
});

// T1/T2 view: its own player drives its own playheads (and refines duration).
els.player2.addEventListener('timeupdate', updateT1T2Playhead);
els.player2.addEventListener('loadedmetadata', () => {
  if (els.player2.duration && isFinite(els.player2.duration)) {
    state.duration = els.player2.duration;
    renderT1T2();
  }
});

init();
