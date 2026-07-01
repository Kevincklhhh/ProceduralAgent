#!/usr/bin/env node
// Video + data server for ProceduralAgent task visualization.
// Mirrors self_data/annotator/video-server.js: serves videos with range
// support plus JSON APIs for tasks/annotations/ground truth. Port 4002.
// Dependency-free (plain node:http) so no npm install is required.

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PORT = parseInt(process.env.PORT || '4002', 10);

const ROOT = path.resolve(__dirname, '..');
const TASKS_DIR = path.join(ROOT, 'tasks');
const DATA_DIR = path.join(ROOT, 'data');
const CC4D_ANN = path.join(DATA_DIR, 'cc4d', 'annotations');
const EXP_DIR = path.join(ROOT, 'experiments', 't1_baseline');   // baseline arms + traces (baseline_t1_step.py)
const MON_DIR = path.join(ROOT, 'experiments', 'proposed_system'); // procedure-monitor arms
const QRUN_DIR = path.join(ROOT, 'experiments', 'qualcomm_run');  // turn/stream Qualcomm runs (T1 stages + T2 events)
const PROACTIVE_DIR = path.join(DATA_DIR, 'cc4d_proactive');       // execution-mistake reminder GT (eval/gt_build_proactive.py): {t, content, subtype}
const PROACTIVE_OM_DIR = path.join(DATA_DIR, 'cc4d_proactive_om'); // order + missing-step reminder GT (eval/gt_build_om.py)

// Video sources in priority order: when a recording exists in more than one
// (e.g. the six activity-8 recordings are in both videos_480p and videos_360p),
// the higher-priority source wins and the duplicate is dropped.
const VIDEO_SOURCES = ['videos_480p', 'clips', 'videos_360p'];

const durationCache = {};

function ffprobeDuration(filepath) {
  if (filepath in durationCache) return durationCache[filepath];
  let dur = 0;
  try {
    dur = parseFloat(execSync(
      `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${filepath}"`,
      { encoding: 'utf-8' }
    ).trim());
  } catch { /* leave 0 */ }
  durationCache[filepath] = dur;
  return dur;
}

function readJson(filepath) {
  try {
    return JSON.parse(fs.readFileSync(filepath, 'utf-8'));
  } catch {
    return null;
  }
}

// Memoized reader for the static CC4D annotation files (they never change
// during a session, and a couple are MB-sized — don't re-parse per request).
const _jsonMemo = {};
function cachedJson(filepath) {
  if (!(filepath in _jsonMemo)) _jsonMemo[filepath] = readJson(filepath);
  return _jsonMemo[filepath];
}

// recipe display name ("Spiced Hot Chocolate") -> task-graph filename stem
// ("spicedhotchocolate").
function normalizeRecipe(name) {
  return String(name || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

// ---------------------------------------------------------------------------
// Data loading (re-read per request so edits to tasks/*.json show on reload)
// ---------------------------------------------------------------------------

function loadTasks() {
  const tasks = {};
  if (!fs.existsSync(TASKS_DIR)) return tasks;
  for (const f of fs.readdirSync(TASKS_DIR).sort()) {
    if (!f.startsWith('task_') || !f.endsWith('.json')) continue;
    const data = readJson(path.join(TASKS_DIR, f));
    if (data && data.task_id) tasks[data.task_id] = { ...data, _file: f };
  }
  return tasks;
}

function loadAnnotations() {
  const annotations = [];
  if (!fs.existsSync(TASKS_DIR)) return annotations;
  for (const f of fs.readdirSync(TASKS_DIR).sort()) {
    if (!f.startsWith('annotation_') || !f.endsWith('.json')) continue;
    const data = readJson(path.join(TASKS_DIR, f));
    // Skip the template (unfilled video_id)
    if (!data || !data.video_id || data.video_id.startsWith('FILL')) continue;
    annotations.push({ ...data, _file: f });
  }
  return annotations;
}

// Unified CC4D ground-truth index, keyed by recording_id. Joins the temporal
// step segments (complete_step_annotations.json) with the per-step error tags
// (error_annotations.json). Covers all 384 recordings, not just activity 8.
function loadCC4DGt() {
  const memoKey = '__cc4d_gt__';
  if (_jsonMemo[memoKey]) return _jsonMemo[memoKey];

  const steps = cachedJson(path.join(CC4D_ANN, 'annotation_json', 'complete_step_annotations.json')) || {};
  const errors = cachedJson(path.join(CC4D_ANN, 'annotation_json', 'error_annotations.json')) || [];

  // error_annotations.json is a list; index it by recording_id, and within
  // each recording index error step_annotations by step_id.
  const errByRec = {};
  for (const rec of errors) {
    const byStep = {};
    for (const sa of rec.step_annotations || []) {
      if (sa.errors && sa.errors.length) byStep[sa.step_id] = sa;
    }
    errByRec[rec.recording_id] = { is_error: rec.is_error, byStep };
  }

  const index = {};
  for (const [recId, rec] of Object.entries(steps)) {
    const errInfo = errByRec[recId] || { is_error: false, byStep: {} };
    index[recId] = {
      recording_id: recId,
      activity_id: rec.activity_id,
      activity_name: rec.activity_name,
      person: rec.person_id ?? rec.person ?? null,
      environment: rec.environment ?? null,
      is_error: errInfo.is_error ?? false,
      steps: (rec.steps || []).map(s => ({
        step_id: s.step_id,
        start_time: s.start_time,
        end_time: s.end_time,
        description: s.description,
        has_errors: s.has_errors,
        errors: errInfo.byStep[s.step_id]?.errors || [],
        modified_description: errInfo.byStep[s.step_id]?.modified_description || null,
      })),
    };
  }
  _jsonMemo[memoKey] = index;
  return index;
}

// The Qualcomm Interactive Cooking layer, pre-flattened from parquet to JSON by
// scripts/export_qualcomm_timeline.py (the server is dependency-free and can't
// read parquet). Keyed by video_id == recording_id. Each entry has instructions
// (step-start guidance), derived completions, success confirmations, and typed
// mistake events (timestamp = when the mistake first becomes visible).
function loadQualcomm() {
  const memoKey = '__qualcomm__';
  if (memoKey in _jsonMemo) return _jsonMemo[memoKey];
  const fp = path.join(DATA_DIR, 'qualcomm_interactive_cooking', 'qualcomm_timeline.json');
  _jsonMemo[memoKey] = readJson(fp) || {};
  return _jsonMemo[memoKey];
}

// Durations for all CC4D recordings straight from the metadata CSV — instant,
// no ffprobe over 384 files.
function loadDurations() {
  const memoKey = '__durations__';
  if (_jsonMemo[memoKey]) return _jsonMemo[memoKey];
  const map = {};
  const csv = path.join(CC4D_ANN, 'metadata', 'video_information.csv');
  try {
    const lines = fs.readFileSync(csv, 'utf-8').trim().split('\n');
    const header = lines[0].split(',');
    const idIdx = header.indexOf('recording_id');
    const durIdx = header.indexOf('duration(sec)');
    for (const line of lines.slice(1)) {
      const cols = line.split(',');
      map[cols[idIdx]] = parseFloat(cols[durIdx]) || 0;
    }
  } catch { /* leave empty */ }
  _jsonMemo[memoKey] = map;
  return map;
}

// Load the recipe task-graph DAG for an activity name, if present.
function loadTaskGraph(activityName) {
  const stem = normalizeRecipe(activityName);
  if (!stem) return null;
  const fp = path.join(CC4D_ANN, 'task_graphs', `${stem}.json`);
  const g = cachedJson(fp);
  if (!g) return null;
  // Drop the synthetic START/END nodes for display.
  const steps = Object.entries(g.steps || {})
    .filter(([, label]) => label !== 'START' && label !== 'END')
    .map(([id, label]) => ({ id: Number(id), label }));
  return { steps, edges: g.edges || [], _file: `${stem}.json` };
}

// Map a video id to its CC4D recording id (the HoloLens-only fallback file is
// named "12_6_hololens_pv" but its GT lives under "12_6").
function recordingIdOf(videoId) {
  return videoId.replace(/_hololens_pv$/, '');
}

// Map CC4D numeric step ids -> {task_id, step_id} using the cc4d_step_ids
// block in hand-written task definitions, so GT steps can be linked to the
// richer hand-written task steps (only activity 8 has this today).
function cc4dStepMap(tasks) {
  const map = {};
  for (const [taskId, task] of Object.entries(tasks)) {
    for (const [stepId, num] of Object.entries(task.cc4d_step_ids || {})) {
      map[num] = { task_id: taskId, step_id: stepId };
    }
  }
  return map;
}

function listVideos() {
  const annotations = loadAnnotations();
  const gt = loadCC4DGt();
  const durations = loadDurations();
  const qualcomm = loadQualcomm();

  const seen = new Set();
  const videos = [];
  for (const source of VIDEO_SOURCES) {
    const dir = path.join(DATA_DIR, source);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.mp4')).sort()) {
      const id = f.replace(/\.mp4$/, '');
      if (seen.has(id)) continue;       // duplicate across sources — keep first
      seen.add(id);
      const recId = recordingIdOf(id);
      const gtEntry = gt[recId] || null;
      const qcEntry = qualcomm[recId] || null;
      const annotation = annotations.find(a => a.video_id === id) || null;
      videos.push({
        id,
        recording_id: recId,
        source,
        filename: f,
        duration: durations[recId] || (annotation ? ffprobeDuration(path.join(dir, f)) : 0),
        activity_id: gtEntry?.activity_id ?? null,
        activity_name: gtEntry?.activity_name ?? (annotation ? annotation.task_id : null),
        person: gtEntry?.person ?? null,
        environment: gtEntry?.environment ?? null,
        has_gt: !!gtEntry,
        has_annotation: !!annotation,
        has_qualcomm: !!qcEntry,
        num_qc_mistakes: qcEntry ? qcEntry.mistakes.length : 0,
        is_error: gtEntry?.is_error ?? false,
        num_error_steps: gtEntry ? gtEntry.steps.filter(s => s.errors.length).length : 0,
      });
    }
  }
  return videos;
}

// Bundle everything the frontend needs for one video.
function buildTimeline(videoId) {
  const tasks = loadTasks();
  const annotations = loadAnnotations();
  const gt = loadCC4DGt();
  const durations = loadDurations();
  const qualcomm = loadQualcomm();
  const stepMap = cc4dStepMap(tasks);

  const recId = recordingIdOf(videoId);
  const annotation = annotations.find(a => a.video_id === videoId) || null;
  const gtEntry = gt[recId] || null;
  const qcEntry = qualcomm[recId] || null;
  if (!annotation && !gtEntry && !qcEntry) return null;

  // Prefer a hand-written task (richer: reminders, expected durations); else
  // fall back to the CC4D recipe task graph for the recording's activity.
  let task = null;
  if (annotation && annotation.task_id) task = tasks[annotation.task_id] || null;
  if (!task && gtEntry) {
    const viaStep = (gtEntry.steps || []).map(s => stepMap[s.step_id]).find(Boolean);
    if (viaStep) task = tasks[viaStep.task_id] || null;
  }
  const taskGraph = gtEntry ? loadTaskGraph(gtEntry.activity_name) : null;

  // GT steps already carry their joined errors from loadCC4DGt; just attach the
  // hand-written task step id where one exists.
  let gtSteps = null;
  if (gtEntry) {
    gtSteps = gtEntry.steps.map(s => ({
      ...s,
      task_step_id: stepMap[s.step_id]?.step_id || null,
    }));
  }

  // Proactive-reminder GT: point reminders {t, content, subtype} from TWO dirs — execution
  // mistakes (cc4d_proactive) + order/missing (cc4d_proactive_om) — merged into one T2 GT
  // track. Reminders are points (no window); start_s==end_s==t. `cls` drives nothing scored
  // here, just a label group. Event-detection scheme: no decision points / silent labels.
  let familyA = null;
  {
    const byStep = {};
    for (const s of (gtSteps || [])) byStep[s.step_id] = s;
    const mapRem = (r, cls) => {
      const step = byStep[r.anchor_step];
      return {
        ...r, cls,
        start_s: Array.isArray(r.window) ? r.window[0] : r.t,
        end_s: Array.isArray(r.window) ? r.window[1] : r.t,
        anchor_desc: step ? step.description : null,
      };
    };
    const fa = readJson(path.join(PROACTIVE_DIR, `${recId}.json`));
    const om = readJson(path.join(PROACTIVE_OM_DIR, `${recId}.json`));
    const events = [];
    if (fa && Array.isArray(fa.reminders))
      for (const r of fa.reminders) events.push(mapRem(r, r.subtype === 'timing' ? 'parameter' : 'execution'));
    if (om && Array.isArray(om.reminders))
      for (const r of om.reminders) events.push(mapRem(r, 'precondition'));
    if (fa || om) {
      events.sort((a, b) => a.start_s - b.start_s);
      familyA = { recording_id: recId, is_error: (fa?.is_error ?? om?.is_error ?? null), events };
    }
  }

  // Duration: CSV first, then the browser fills the precise value client-side.
  const duration = durations[recId]
    || (gtSteps ? Math.max(0, ...gtSteps.map(s => s.end_time || 0)) : 0);

  return {
    video_id: videoId,
    recording_id: recId,
    duration,
    meta: gtEntry ? {
      activity_id: gtEntry.activity_id,
      activity_name: gtEntry.activity_name,
      person: gtEntry.person,
      environment: gtEntry.environment,
      is_error: gtEntry.is_error,
    } : null,
    task,
    task_graph: taskGraph,
    annotation,
    gt: gtEntry ? { steps: gtSteps, is_error: gtEntry.is_error } : null,
    qualcomm: qcEntry,
    family_a: familyA,
  };
}

// ---------------------------------------------------------------------------
// Baseline predictions + per-call VLM context (experiments/t1_baseline/<arm>/...)
// ---------------------------------------------------------------------------

// All arms that have a result file for this recording, each with its predicted
// stage_intervals, cost, and the per-call trace (frames + prompt + prediction).
function buildBaseline(rid) {
  const arms = [];

  if (fs.existsSync(EXP_DIR)) {
    for (const arm of fs.readdirSync(EXP_DIR).sort()) {
      const resPath = path.join(EXP_DIR, arm, `${rid}.json`);
      if (!fs.existsSync(resPath)) continue;
      const res = readJson(resPath) || {};
      // per-call trace (optional — present only when run with --trace)
      let calls = [];
      let systemPrompt = null;
      const trPath = path.join(EXP_DIR, arm, 'traces', `${rid}.jsonl`);
      if (fs.existsSync(trPath)) {
        for (const line of fs.readFileSync(trPath, 'utf-8').split('\n')) {
          if (!line.trim()) continue;
          const c = JSON.parse(line);
          if (systemPrompt === null) systemPrompt = c.system_prompt || null;
          calls.push({
            t: c.t, start_s: c.start_s, end_s: c.end_s,
            frame_urls: (c.frame_files || []).map(
              f => `/baseline_frames/${encodeURIComponent(arm)}/${encodeURIComponent(rid)}/${encodeURIComponent(f)}`),
            user_prompt: c.user_prompt, prev_responses: c.prev_responses || c.completed_step_ids || [],
            pred_step: c.pred_step, pred_status: c.pred_status,
            pred_evidence: c.pred_evidence, raw: c.raw, latency_s: c.latency_s,
          });
        }
      }
      arms.push({
        arm, stage_intervals: res.stage_intervals || [], cost: res.cost || {},
        meta: res._meta || {}, system_prompt: systemPrompt, calls,
      });
    }
  }

  return { arms };
}

// ---------------------------------------------------------------------------
// Procedure-monitor runs (experiments/proposed_system/<arm>/...)
// ---------------------------------------------------------------------------

// Flatten the compiled plan's steps + step_blocks into a uniform unit list keyed
// by id (stage names in the run line up with these ids).
function planUnits(plan) {
  if (!plan || !plan.graph) return [];
  const mon = u => ({ primitives: (u.monitor || {}).primitives || [],
                      sleep: (u.monitor || {}).sleep || [] });
  const out = [];
  for (const s of plan.graph.steps || []) {
    out.push({ id: s.step_id, kind: 'step', order: s.order || 0,
      requires: s.requires || [], produces: s.produces || [],
      sensing_role: s.sensing_role || null, instruction: s.instruction || '',
      members: [], ...mon(s) });
  }
  for (const b of plan.graph.step_blocks || []) {
    out.push({ id: b.block_id, kind: 'block', order: b.order || 0,
      requires: b.requires || [], produces: b.produces || [],
      sensing_role: b.sensing_role || null, instruction: '',
      members: (b.members || []).map(m => ({ id: m.step_id, cc4d: m.cc4d_step_id,
                                             instruction: m.instruction })),
      ...mon(b) });
  }
  out.sort((a, b) => a.order - b.order);
  return out;
}

// All monitor arms (subdirs of proposed_system/ with a <rid>.json), each with its
// stage_intervals, transition_trace, sensor_events, cost, and per-block VLM polls.
function buildMonitor(rid) {
  if (!fs.existsSync(MON_DIR)) return { plan: null, arms: [] };
  const arms = [];
  let plan = null;
  for (const arm of fs.readdirSync(MON_DIR).sort()) {
    if (arm === 'results') continue;
    const resPath = path.join(MON_DIR, arm, `${rid}.json`);
    if (!fs.existsSync(resPath)) continue;
    const res = readJson(resPath) || {};
    if (!plan) {
      // Prefer the COMPILED plan the runtime emitted for this arm (graph form — its
      // step/block ids line up with the run's stage_intervals). Fall back to the
      // hand-written executable monitor.json (byte-identical graph), then to the
      // nodes-form sensorplan.json (panel will be sparse: no graph to flatten).
      plan = readJson(path.join(MON_DIR, arm, '_compiled_plan.json'))
          || (res.task_id && cachedJson(path.join(TASKS_DIR, 'cc4d', `${res.task_id}.monitor.json`)))
          || (res.task_id && cachedJson(path.join(TASKS_DIR, 'cc4d', `${res.task_id}.sensorplan.json`)));
    }
    // per-block VLM poll traces (optional — present only for the real-VLM arm)
    let polls = [];
    const trDir = path.join(MON_DIR, arm, 'traces');
    if (fs.existsSync(trDir)) {
      for (const f of fs.readdirSync(trDir)) {
        if (!f.endsWith('.jsonl')) continue;
        for (const line of fs.readFileSync(path.join(trDir, f), 'utf-8').split('\n')) {
          if (!line.trim()) continue;
          try {
            const obj = JSON.parse(line);
            if (obj.recording && obj.recording !== rid) continue;  // shared per-block file, tagged by recording
            polls.push(obj);
          } catch { /* skip bad line */ }
        }
      }
    }
    polls.sort((a, b) => (a.t || 0) - (b.t || 0));
    arms.push({
      arm, stage_intervals: res.stage_intervals || [],
      transition_trace: res.transition_trace || [], sensor_events: res.sensor_events || [],
      cost: res.cost || {}, polls,
    });
  }
  return {
    plan: plan ? { units: planUnits(plan), vlm_policy: plan.vlm_policy || {},
                   task: plan.task || {} } : null,
    arms,
  };
}

// ---------------------------------------------------------------------------
// Turn/stream Qualcomm runs (experiments/qualcomm_run/<arm>/<rid>.json)
// ---------------------------------------------------------------------------

// Each arm is a subdir with a unified <rid>.json: T1 stage_intervals + T2 events
// (typed mistake/reminder predictions). The dir also holds *.log files — skip
// non-directories.
function buildQualcrun(rid) {
  const arms = [];
  if (fs.existsSync(QRUN_DIR)) {
    for (const arm of fs.readdirSync(QRUN_DIR).sort()) {
      const armDir = path.join(QRUN_DIR, arm);
      let st;
      try { st = fs.statSync(armDir); } catch { continue; }
      if (!st.isDirectory()) continue;
      const resPath = path.join(armDir, `${rid}.json`);
      if (!fs.existsSync(resPath)) continue;
      const res = readJson(resPath) || {};
      arms.push({
        arm,
        stage_intervals: res.stage_intervals || [],
        events: res.events || [],
        calls: res.calls || [],            // per-call VLM trace (t, window, frames, fired)
        cost: res.cost || {},
        meta: res._meta || {},
      });
    }
  }
  return { arms };
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

function sendJson(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(JSON.stringify(data));
}

function streamVideo(req, res, filepath) {
  if (!fs.existsSync(filepath)) {
    res.writeHead(404, { 'Access-Control-Allow-Origin': '*' });
    return res.end('Not found');
  }
  const stat = fs.statSync(filepath);
  const fileSize = stat.size;
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': end - start + 1,
      'Content-Type': 'video/mp4',
      'Access-Control-Allow-Origin': '*',
    });
    fs.createReadStream(filepath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
      'Accept-Ranges': 'bytes',
      'Access-Control-Allow-Origin': '*',
    });
    fs.createReadStream(filepath).pipe(res);
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const parts = url.pathname.split('/').filter(Boolean);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': '*',
    });
    return res.end();
  }

  try {
    // GET /api/videos
    if (parts[0] === 'api' && parts[1] === 'videos' && parts.length === 2) {
      return sendJson(res, 200, listVideos());
    }
    // GET /api/tasks
    if (parts[0] === 'api' && parts[1] === 'tasks' && parts.length === 2) {
      return sendJson(res, 200, loadTasks());
    }
    // GET /api/taskfiles — list every JSON file in tasks/ with light metadata
    if (parts[0] === 'api' && parts[1] === 'taskfiles' && parts.length === 2) {
      const files = fs.existsSync(TASKS_DIR)
        ? fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.json')).sort()
        : [];
      return sendJson(res, 200, files.map(f => {
        const data = readJson(path.join(TASKS_DIR, f)) || {};
        const isTemplate = typeof data.video_id === 'string' && data.video_id.startsWith('FILL');
        const kind = f.startsWith('task_') ? 'task'
          : isTemplate ? 'template'
          : f.startsWith('annotation_') ? 'annotation' : 'other';
        return {
          filename: f,
          kind,
          task_id: data.task_id || null,
          video_id: data.video_id || null,
          title: data.title || null,
        };
      }));
    }
    // GET /api/taskfiles/:filename — raw parsed JSON of one tasks/ file
    if (parts[0] === 'api' && parts[1] === 'taskfiles' && parts.length === 3) {
      const name = decodeURIComponent(parts[2]);
      if (!name.endsWith('.json') || name.includes('/') || name.includes('..')) {
        return sendJson(res, 400, { error: 'Bad filename' });
      }
      const filepath = path.join(TASKS_DIR, name);
      if (!fs.existsSync(filepath)) return sendJson(res, 404, { error: 'Not found' });
      const data = readJson(filepath);
      if (data === null) return sendJson(res, 500, { error: 'Invalid JSON on disk' });
      return sendJson(res, 200, data);
    }
    // GET /api/timeline/:videoId
    if (parts[0] === 'api' && parts[1] === 'timeline' && parts.length === 3) {
      const timeline = buildTimeline(decodeURIComponent(parts[2]));
      if (!timeline) return sendJson(res, 404, { error: 'No annotation or GT for this video' });
      return sendJson(res, 200, timeline);
    }
    // GET /api/baseline/:videoId — baseline arms (predicted steps + per-call VLM context)
    if (parts[0] === 'api' && parts[1] === 'baseline' && parts.length === 3) {
      return sendJson(res, 200, buildBaseline(decodeURIComponent(parts[2])));
    }
    // GET /api/monitor/:videoId — procedure-monitor arms (plan + run + VLM polls)
    if (parts[0] === 'api' && parts[1] === 'monitor' && parts.length === 3) {
      return sendJson(res, 200, buildMonitor(decodeURIComponent(parts[2])));
    }
    // GET /api/qualcrun/:videoId — turn/stream Qualcomm runs (T1 stages + T2 reminder events)
    if (parts[0] === 'api' && parts[1] === 'qualcrun' && parts.length === 3) {
      return sendJson(res, 200, buildQualcrun(recordingIdOf(decodeURIComponent(parts[2]))));
    }
    // GET /api/stepdescriptions — global CC4D step_idx -> description (for unified labels)
    if (parts[0] === 'api' && parts[1] === 'stepdescriptions' && parts.length === 2) {
      return sendJson(res, 200,
        cachedJson(path.join(CC4D_ANN, 'annotation_json', 'step_idx_description.json')) || {});
    }
    // GET /baseline_frames/:arm/:rid/:file — a single frame sent to the VLM
    if (parts[0] === 'baseline_frames' && parts.length === 4) {
      const [arm, rid, file] = parts.slice(1).map(decodeURIComponent);
      if ([arm, rid, file].some(s => s.includes('..') || s.includes('/')) || !file.endsWith('.jpg')) {
        res.writeHead(400, { 'Access-Control-Allow-Origin': '*' }); return res.end('Bad path');
      }
      const fp = path.join(EXP_DIR, arm, 'frames', rid, file);
      if (!fs.existsSync(fp)) { res.writeHead(404, { 'Access-Control-Allow-Origin': '*' }); return res.end('Not found'); }
      res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Access-Control-Allow-Origin': '*' });
      return fs.createReadStream(fp).pipe(res);
    }
    // GET /videos/:source/:filename
    if (parts[0] === 'videos' && parts.length === 3) {
      const source = decodeURIComponent(parts[1]);
      const filename = decodeURIComponent(parts[2]);
      if (!VIDEO_SOURCES.includes(source) || filename.includes('..')) {
        res.writeHead(400, { 'Access-Control-Allow-Origin': '*' });
        return res.end('Bad path');
      }
      return streamVideo(req, res, path.join(DATA_DIR, source, filename));
    }

    sendJson(res, 404, { error: 'Unknown route' });
  } catch (e) {
    sendJson(res, 500, { error: String(e) });
  }
});

server.listen(PORT, () => {
  console.log(`Video/data server listening on http://localhost:${PORT}`);
  console.log(`  tasks dir: ${TASKS_DIR}`);
  console.log(`  data dir:  ${DATA_DIR}`);
});
