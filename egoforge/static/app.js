'use strict';

const $ = id => document.getElementById(id);
const LABELS = {quiescent:'Quiescent',left_dominant:'Left dominant',right_dominant:'Right dominant',bilateral_coordinated:'Bilateral coordinated',bilateral_independent:'Bilateral independent',unknown:'Unknown / abstained'};
const COLORS = {quiescent:'#bcc2c7',left_dominant:'#688a9c',right_dominant:'#aa8971',bilateral_coordinated:'#748c77',bilateral_independent:'#9993aa',unknown:'#bd8c85'};
const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const data = {summary:{},episodes:[],segments:[],queue:[],reviews:[],frames:[],episode:null,segment:null,camera:'main_stereo_left',time:0,overlay:true,queueOnly:true,playing:false,lastTick:0,loadSerial:0};
const video = $('video');
const num = (value,digits=2) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toFixed(digits) : '—';
const text = (id,value) => { $(id).textContent = value; };
const arr = (value,key) => Array.isArray(value) ? value : value && Array.isArray(value[key]) ? value[key] : [];
const eid = ep => Number(ep.episode_id ?? ep.episode_index ?? ep.index);
const sid = seg => String(seg.segment_id);
const start = seg => Number(seg.start_time_s ?? seg.start_timestamp_s ?? seg.start_s ?? 0);
const end = seg => Number(seg.end_time_s ?? seg.end_timestamp_s ?? seg.end_s ?? start(seg));
const stamp = seconds => { const s = Math.max(0,Number(seconds)||0); return `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(2).padStart(5,'0')}`; };
const episodeDuration = () => Number(data.episode?.duration_s ?? data.episode?.duration ?? (data.episode?.frame_count / 30)) || Math.max(...episodeSegments().map(end),1);
const episodeSegments = () => data.segments.filter(s => Number(s.episode_id) === eid(data.episode || {}));
const latestReview = id => [...data.reviews].reverse().find(r => String(r.segment_id) === String(id));
const hasVideo = ep => ep.video_available ?? (Array.isArray(ep.available_cameras) ? ep.available_cameras.length > 0 : eid(ep) === 4);

async function api(path,options) {
  const response = await fetch(path,options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
function elem(tag,className,value) { const el=document.createElement(tag); if(className)el.className=className;if(value!==undefined)el.textContent=value;return el; }
function error(message) { text('load-error',message);$('load-error').hidden=false; }
function metricFrames() {return data.summary.total_frames ?? data.summary.frames_processed ?? data.summary.frame_count ?? data.episodes.reduce((sum,ep)=>sum+Number(ep.frame_count ?? ep.frames ?? ep.length ?? 0),0);}

function renderMetrics() {
  text('metric-frames',Number(metricFrames()).toLocaleString());
  text('metric-episodes',`${data.episodes.length} episodes · ${num(data.episodes.reduce((s,e)=>s+Number(e.duration_s??e.duration??0),0)/60,1)} minutes of motion`);
  text('metric-segments',data.segments.length.toLocaleString());text('metric-queue',data.queue.length.toLocaleString());
  text('metric-reviewed',data.reviews.filter(r=>r.actor_type==='human').length.toLocaleString());text('episode-count',data.episodes.length);text('sampling-note','Sampled display; full data in exports.');
}

function renderEpisodes() {
  const select = $('episode-select');
  select.replaceChildren();
  for (const ep of data.episodes) {
    const label = `${String(eid(ep)).padStart(2, '0')} · ${ep.task ?? ep.task_name ?? 'Episode'}${hasVideo(ep) ? ' · 4 cameras' : ''}`;
    const option = elem('option', '', label);
    option.value = eid(ep);
    option.selected = data.episode && eid(ep) === eid(data.episode);
    select.append(option);
  }
}

async function selectEpisode(id,seekTo=0) {
  const serial=++data.loadSerial;
  data.episode=data.episodes.find(ep=>eid(ep)===Number(id));if(!data.episode)return;
  pause();data.frames=[];data.time=seekTo;data.segment=null;renderEpisodes();
  text('episode-title',`${String(id).padStart(2,'0')} / ${data.episode.task ?? data.episode.task_name ?? 'Motion sequence'}`);
  $('seek').max=episodeDuration();$('seek').value=seekTo;text('axis-end',`${num(episodeDuration(),1)}s`);
  await loadCamera();
  const result=await api(`/api/frames?episode=${Number(id)}`);
  if(serial!==data.loadSerial)return;
  data.frames=arr(result,'frames').sort((a,b)=>Number(a.timestamp_s??a.timestamp??0)-Number(b.timestamp_s??b.timestamp??0));
  seek(seekTo);renderQueue();draw();
}

async function loadCamera() {
  const available=hasVideo(data.episode);
  video.pause();video.hidden=!available;$('video-empty').hidden=available;
  document.querySelectorAll('[data-camera]').forEach(button=>{button.classList.toggle('selected',button.dataset.camera===data.camera);button.disabled=!available;});
  text('camera-label',({main_stereo_left:'Head / left',main_stereo_right:'Head / right',aux_left:'Wrist / left',aux_right:'Wrist / right'})[data.camera]);
  const url=`/media/episode_${String(eid(data.episode)).padStart(6,'0')}/${data.camera}.mp4`;
  if(available){video.src=url;video.load();}else{video.removeAttribute('src');video.load();}
  $('overlay-toggle').disabled=data.camera!=='main_stereo_left';
  text('overlay-toggle',data.camera==='main_stereo_left'?(data.overlay?'Skeleton on':'Skeleton off'):'No 2D overlay');
}

function nearestFrame() {
  if(!data.frames.length)return null;
  let lo=0,hi=data.frames.length-1;
  while(lo<hi){let mid=Math.floor((lo+hi)/2);if(Number(data.frames[mid].timestamp_s??data.frames[mid].timestamp??0)<data.time)lo=mid+1;else hi=mid;}
  const a=data.frames[lo],b=data.frames[Math.max(0,lo-1)];
  return Math.abs(Number(a.timestamp_s??a.timestamp??0)-data.time)<Math.abs(Number(b.timestamp_s??b.timestamp??0)-data.time)?a:b;
}
function seek(value) {
  data.time=Math.max(0,Math.min(Number(value)||0,episodeDuration()));
  if(hasVideo(data.episode) && video.readyState>=1 && Math.abs(video.currentTime-data.time)>.045)video.currentTime=data.time;
  updateCurrent();draw();
}
function updateCurrent() {
  $('seek').value=data.time;text('timecode',`${stamp(data.time)} / ${stamp(episodeDuration())}`);
  const segment=episodeSegments().find(s=>start(s)<=data.time && end(s)>data.time) || (data.time>=episodeDuration()-.02?episodeSegments().at(-1):null);
  const changed=sid(segment||{})!==sid(data.segment||{});data.segment=segment;
  const frame=nearestFrame();text('frame-stamp',`Frame ${frame?.frame_index ?? Math.round(data.time*30)}${frame?' · sampled':''}`);
  text('phase-current',LABELS[segment?.state??frame?.state]??'Unknown / abstained');
  renderEvidence(frame,changed);
  if(changed)renderQueue();
}
function renderEvidence(frame,changed) {
  const s=data.segment,e=s?.evidence||{};
  text('segment-state',s?LABELS[s.state]??s.state:'No segment');text('segment-range',s?`${stamp(start(s))} → ${stamp(end(s))}`:'—');text('segment-id',s?sid(s):'—');
  const review=s?latestReview(sid(s)):null;text('review-status',review?`${review.actor_type} · ${review.decision}`:'Unreviewed');
  text('quality-value',num(frame?.quality_score??e.quality_score));text('uncertainty-value',num(frame?.uncertainty??s?.uncertainty));
  text('left-speed',`${num(frame?.left_speed_mps??e.left_speed_mps)} m/s`);text('right-speed',`${num(frame?.right_speed_mps??e.right_speed_mps)} m/s`);
  text('wrist-distance',`${num(frame?.wrist_distance_m??e.wrist_distance_m)} m`);text('priority-value',num(s?.review_priority));
  const flags=[...(frame?.qa_flags??[]),...(s?.qa_flags??[])];
  if(frame?.usable===false)flags.push('Frame excluded from temporal inference');
  if(s?.state==='unknown')flags.push('No phase assigned: insufficient usable data');
  const unique=[...new Set(flags.map(f=>typeof f==='string'?f:(f.message??f.rule??JSON.stringify(f))))];
  $('qa-findings').replaceChildren(...(unique.length?unique.map(f=>elem('li','',f.replaceAll('_',' '))):[elem('li','clear','No flags at this sampled frame.')]));
  if(changed){$('correction').value='';$('review-note').value='';text('review-message','');}
  ['approve','reject','correct'].forEach(id=>$(id).disabled=!s);
}

function setupCanvas(canvas,height) {
  const ratio=window.devicePixelRatio||1,w=Math.max(canvas.clientWidth,1),h=height??canvas.clientHeight;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);return{ctx,w,h};
}
function drawOverlay() {
  const {ctx,w,h}=setupCanvas($('overlay'));
  if(!data.overlay||data.camera!=='main_stereo_left'||!hasVideo(data.episode))return;
  const frame=nearestFrame();if(!frame)return;
  const iw=video.videoWidth||1920,ih=video.videoHeight||1200,scale=Math.min(w/iw,h/ih),dx=(w-iw*scale)/2,dy=(h-ih*scale)/2;
  for(const side of ['left','right']) {
    let points=frame.keypoints_2d?.[side]??frame[`keypoints_2d_${side}`]??frame[`keypoints_${side}`]??frame[`observation.keypoints_2d.${side}`];
    if(!Array.isArray(points)||!points.length)continue;
    if(typeof points[0]==='number')points=Array.from({length:Math.floor(points.length/2)},(_,i)=>points.slice(i*2,i*2+2));
    const valid=p=>Array.isArray(p)&&p.length>=2&&p.every(Number.isFinite)&&(p[0]!==0||p[1]!==0);
    ctx.strokeStyle=side==='left'?'#7cdbb3':'#bd9cf1';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1.7;ctx.shadowColor='#000';ctx.shadowBlur=2;
    for(const[a,b]of EDGES){if(!valid(points[a])||!valid(points[b]))continue;ctx.beginPath();ctx.moveTo(dx+points[a][0]*scale,dy+points[a][1]*scale);ctx.lineTo(dx+points[b][0]*scale,dy+points[b][1]*scale);ctx.stroke();}
    for(const p of points){if(valid(p)){ctx.beginPath();ctx.arc(dx+p[0]*scale,dy+p[1]*scale,2.3,0,Math.PI*2);ctx.fill();}}
  }
}
function drawPhases() {
  const {ctx,w,h}=setupCanvas($('phase-timeline'),27),duration=episodeDuration();ctx.fillStyle='#f4f5f6';ctx.fillRect(0,0,w,h);
  for(const s of episodeSegments()){ctx.fillStyle=COLORS[s.state]??COLORS.unknown;ctx.globalAlpha=data.segment && sid(s)===sid(data.segment)?1:.68;ctx.fillRect(start(s)/duration*w,2,Math.max((end(s)-start(s))/duration*w-1,1),h-4);}
  ctx.globalAlpha=1;ctx.strokeStyle='#202124';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(data.time/duration*w,0);ctx.lineTo(data.time/duration*w,h);ctx.stroke();
}
function drawSignals() {
  const {ctx,w,h}=setupCanvas($('signal-chart'),85),duration=episodeDuration(),pad=4;
  const vals=data.frames.flatMap(f=>[f.left_speed_mps,f.right_speed_mps]).filter(v=>Number.isFinite(v));
  vals.sort((a,b)=>a-b);const vmax=Math.max(vals[Math.floor(vals.length*.98)]||.3,.15);text('signal-unit','m/s · P98 cap');
  ctx.strokeStyle='#e3e5e7';ctx.lineWidth=.7;
  for(let i=1;i<=3;i++){let y=i/4*(h-pad*2)+pad;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  ctx.font='10px ui-monospace,monospace';ctx.fillStyle='#72767c';ctx.fillText(`${vmax.toFixed(2)}`,2,10);
  for(const[side,color]of[['left','#537e8b'],['right','#927153']]){ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();let begun=false;
    for(const f of data.frames){const v=f[`${side}_speed_mps`],t=Number(f.timestamp_s??f.timestamp??0);if(!Number.isFinite(v)){begun=false;continue;}const x=t/duration*w,y=h-pad-Math.min(v/vmax,1)*(h-pad*2);if(begun)ctx.lineTo(x,y);else{ctx.moveTo(x,y);begun=true;}}
    ctx.stroke();}
  ctx.strokeStyle='#555b62';ctx.globalAlpha=.55;ctx.beginPath();ctx.moveTo(data.time/duration*w,0);ctx.lineTo(data.time/duration*w,h);ctx.stroke();ctx.globalAlpha=1;
}
function draw() {drawOverlay();drawPhases();drawSignals();}

function renderQueue() {
  const source=data.queueOnly?data.queue:data.segments;
  const rows=source.map(item=>typeof item==='string'?data.segments.find(s=>sid(s)===item):item).filter(Boolean);
  text('queue-count',`${rows.length} segments`);$('queue-body').replaceChildren();
  for(const s of rows){const tr=elem('tr',data.segment&&sid(s)===sid(data.segment)?'selected':'');tr.tabIndex=0;tr.setAttribute('aria-label',`Inspect segment ${sid(s)}`);
    const phase=elem('span','phase-pill');const dot=elem('i');dot.style.background=COLORS[s.state]??COLORS.unknown;phase.append(dot,document.createTextNode(LABELS[s.state]??s.state));
    const review=latestReview(sid(s));const phaseCell=elem('td');phaseCell.append(phase);
    tr.append(elem('td','',sid(s)),elem('td','mono',String(s.episode_id).padStart(2,'0')),phaseCell,elem('td','mono',`${stamp(start(s))} – ${stamp(end(s))}`),elem('td','mono',num(s.review_priority)),elem('td','review-tag '+(review?.decision??''),review?`${review.actor_type} · ${review.decision}`:'Unreviewed'),elem('td','','↗'));
    const pick=()=>{if(eid(data.episode||{})!==Number(s.episode_id))selectEpisode(Number(s.episode_id),start(s)).catch(e=>error(e.message));else seek(start(s)+.001);};tr.onclick=pick;tr.onkeydown=e=>{if(e.key==='Enter'){pick();}};$('queue-body').append(tr);}
}

function pause(){data.playing=false;video.pause();text('play','▶');}
async function togglePlay(){if(data.playing){pause();return;}if(data.time>=episodeDuration()-.03)seek(0);data.playing=true;data.lastTick=performance.now();text('play','Ⅱ');if(hasVideo(data.episode)){try{await video.play();}catch(e){pause();error(`Video playback unavailable: ${e.message}`);}}}
function tick(now){if(data.playing){if(hasVideo(data.episode)){data.time=video.currentTime;}else{data.time+=Math.min((now-data.lastTick)/1000,.1)*Number($('speed').value);}data.lastTick=now;if(data.time>=episodeDuration()){data.time=episodeDuration();pause();}updateCurrent();draw();}requestAnimationFrame(tick);}
function moveSegment(direction){const segs=episodeSegments(),current=segs.findIndex(s=>data.segment&&sid(s)===sid(data.segment));if(segs.length){const i=Math.max(0,Math.min(segs.length-1,current+direction));seek(start(segs[i])+.001);}}

async function review(decision){
  if(!data.segment)return;
  const actor=$('reviewer').value.trim(),note=$('review-note').value.trim(),corrected=$('correction').value;
  $('review-message').classList.remove('error');
  if(!actor){$('reviewer').focus();$('review-message').classList.add('error');text('review-message','Enter a reviewer identifier before recording a decision.');return;}
  if(decision==='correct'&&!corrected){$('correction').focus();$('review-message').classList.add('error');text('review-message','Select the corrected phase.');return;}
  if(decision!=='approve'&&!note){$('review-note').focus();$('review-message').classList.add('error');text('review-message','Describe the evidence for this rejection or correction.');return;}
  const payload={actor,actor_type:'human',decision,segment_id:sid(data.segment),note,request_id:crypto.randomUUID(),run_id:data.segment._review?.run_id??data.summary.run_id,proposal_sha256:data.segment._review?.proposal_sha256};if(decision==='correct')payload.corrected_state=corrected;
  ['approve','reject','correct'].forEach(id=>$(id).disabled=true);
  try{const event=await api('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});data.reviews.push(event);localStorage.setItem('egoforge.reviewer',actor);renderMetrics();renderEvidence(nearestFrame(),false);renderQueue();text('review-message',`Saved ${decision}. Journal ${event.event_hash.slice(0,12)}…`);}catch(e){$('review-message').classList.add('error');text('review-message',e.message);}finally{['approve','reject','correct'].forEach(id=>$(id).disabled=!data.segment);}
}

async function initialize(){
  const [summary,episodes,segments,queue,reviews]=await Promise.all(['/api/summary','/api/episodes','/api/segments','/api/review-queue','/api/reviews'].map(p=>api(p)));
  data.summary=summary;data.episodes=arr(episodes,'episodes');data.segments=arr(segments,'segments');data.queue=arr(queue,'segments');data.reviews=arr(reviews,'events');
  if(!data.episodes.length)throw new Error('No run artifacts found. Run the EgoForge pipeline, then reload this page.');
  for(const[state,label]of Object.entries(LABELS)){const item=elem('span');const dot=elem('i');dot.style.background=COLORS[state];item.append(dot,document.createTextNode(label));$('phase-legend').append(item);const option=elem('option','',label);option.value=state;$('correction').append(option);}
  try{$('reviewer').value=localStorage.getItem('egoforge.reviewer')||'';}catch{}
  renderMetrics();await selectEpisode(eid(data.episodes.find(ep=>hasVideo(ep))??data.episodes[0]));requestAnimationFrame(tick);
  if (window.location.hash === '#benchmarks') showView('benchmarks');
}
$('episode-select').onchange=e=>selectEpisode(Number(e.target.value)).catch(e=>error(e.message));
$('play').onclick=togglePlay;$('seek').oninput=e=>seek(e.target.value);$('speed').onchange=e=>video.playbackRate=Number(e.target.value);
$('overlay-toggle').onclick=()=>{data.overlay=!data.overlay;text('overlay-toggle',data.overlay?'Skeleton on':'Skeleton off');$('overlay-toggle').setAttribute('aria-pressed',String(data.overlay));drawOverlay();};
document.querySelectorAll('[data-camera]').forEach(button=>button.onclick=async()=>{pause();data.camera=button.dataset.camera;await loadCamera();drawOverlay();});
video.onloadedmetadata=()=>{video.currentTime=Math.min(data.time,video.duration||data.time);video.playbackRate=Number($('speed').value);draw();};video.onseeked=()=>{updateCurrent();draw();};video.onended=pause;
video.onerror=()=>{if(hasVideo(data.episode)&&video.getAttribute('src')){error('This camera file is unavailable or cannot be decoded. Other evidence remains available.');}};
for(const id of ['phase-timeline','signal-chart'])$(id).onclick=e=>{const r=e.currentTarget.getBoundingClientRect();seek((e.clientX-r.left)/r.width*episodeDuration());};
$('queue-only').onclick=()=>{data.queueOnly=true;$('queue-only').classList.add('selected');$('all-segments').classList.remove('selected');renderQueue();};$('all-segments').onclick=()=>{data.queueOnly=false;$('all-segments').classList.add('selected');$('queue-only').classList.remove('selected');renderQueue();};
for(const decision of ['approve','reject','correct'])$(decision).onclick=()=>review(decision);
$('review-form').onsubmit=e=>e.preventDefault();
document.addEventListener('keydown',e=>{if(!$('benchmarks-view').hidden||['INPUT','TEXTAREA','SELECT','BUTTON'].includes(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key==='ArrowLeft'){e.preventDefault();moveSegment(-1);}if(e.key==='ArrowRight'){e.preventDefault();moveSegment(1);}});
window.addEventListener('resize',draw);initialize().catch(e=>error(e.message));


function showView(view) {
  const benchmark = view === 'benchmarks';
  $('review-view').hidden = benchmark;
  $('benchmarks-view').hidden = !benchmark;
  $('tab-review').setAttribute('aria-selected', String(!benchmark));
  $('tab-benchmarks').setAttribute('aria-selected', String(benchmark));
  history.replaceState(null, '', benchmark ? '#benchmarks' : window.location.pathname);
  if (benchmark) { pause(); loadBenchmarks(); } else { draw(); }
}

function count(value) {
  return Number.isFinite(Number(value)) && value !== null && value !== undefined
    ? Number(value).toLocaleString() : '—';
}

function firstNumber(object, keys) {
  for (const key of keys) if (Number.isFinite(object?.[key])) return object[key];
  return null;
}

function benchmarkFact(label, value) {
  if (value === null) return;
  const item = elem('div');
  item.append(elem('dt', '', label), elem('dd', '', count(value)));
  $('benchmark-ablation').append(item);
}

function renderBenchmarks(result) {
  const summary = result.summary || {};
  text('benchmark-scope', `${count(summary.episodes)} episodes · ${count(summary.frames)} frames · ${count(summary.imu_samples)} IMU samples · ${count(summary.repeats)} timing runs`);
  $('benchmark-performance').replaceChildren();
  const stageNames = {load:'Load source data',loading:'Load source data',qa:'Quality checks',qc:'Quality checks',quality:'Quality checks',temporal:'Temporal decoding',recurrence:'Recurrence detection',total:'Total',pipeline:'Total'};
  for (const row of result.performance || []) {
    const tr = elem('tr');
    tr.append(elem('td', '', stageNames[row.stage] || String(row.stage).replaceAll('_', ' ')), elem('td', 'mono', `${num(row.median_s, 3)} s`), elem('td', 'mono', `${num(row.p95_s, 3)} s`), elem('td', 'mono', count(Math.round(row.fps))));
    $('benchmark-performance').append(tr);
  }
  text('benchmark-memory', `Median total: ${num(summary.total_median_s, 2)} s · Throughput: ${count(Math.round(summary.fps))} frames/s · Peak process memory: ${num(summary.peak_rss_mb, 1)} MB. Video decoding is excluded from these timings.`);
  const ablation = result.ablation?.summary || result.ablation || {};
  $('benchmark-ablation').replaceChildren();
  benchmarkFact('Raw phase switches', firstNumber(ablation, ['raw_transitions', 'raw_switches', 'total_raw_transitions']));
  benchmarkFact('Decoded phase switches', firstNumber(ablation, ['temporal_transitions', 'decoded_switches', 'total_temporal_transitions']));
  benchmarkFact('Unknown frames', firstNumber(ablation, ['unknown_frames', 'abstained_frames', 'total_unknown_frames']));
  benchmarkFact('Short raw runs', firstNumber(ablation, ['raw_short_interior_runs']));
  benchmarkFact('Short decoded runs', firstNumber(ablation, ['temporal_short_interior_runs', 'dwell_violations', 'minimum_dwell_violations', 'decoded_dwell_violations']));
  if (!$('benchmark-ablation').children.length) {
    for (const [key, value] of Object.entries(ablation)) if (typeof value === 'number') benchmarkFact(key.replaceAll('_', ' '), value);
  }
  text('benchmark-fault-summary', `Severe corruption cases detected: ${count(summary.synthetic_cases_passed)} / ${count(summary.synthetic_cases_total)}. The table varies fault magnitude and combines eligible targets across the tested episodes.`);
  const groups = new Map();
  for (const row of result.robustness || []) {
    const key = JSON.stringify([row.family, row.magnitude, row.unit]);
    const item = groups.get(key) || {family:row.family, magnitude:row.magnitude, unit:row.unit, detected:0, eligible:0};
    item.detected += Number(row.newly_detected_targets) || 0;
    item.eligible += Number(row.eligible_targets) || 0;
    groups.set(key, item);
  }
  $('benchmark-robustness').replaceChildren();
  for (const row of groups.values()) {
    const tr = elem('tr');
    const sensitivity = row.eligible ? `${num(row.detected / row.eligible * 100, 1)}%` : '—';
    tr.append(elem('td', '', String(row.family).replaceAll('_', ' ')), elem('td', 'mono', `${row.magnitude} ${row.unit || ''}`.trim()), elem('td', 'mono', `${count(row.detected)} / ${count(row.eligible)}`), elem('td', 'mono', sensitivity));
    $('benchmark-robustness').append(tr);
  }
  text('benchmark-environment', JSON.stringify({environment:result.environment, provenance:result.provenance}, null, 2));
  $('benchmark-limitations').replaceChildren(...(result.limitations || []).map(value=>elem('li', '', typeof value === 'string' ? value : JSON.stringify(value))));
  $('benchmark-content').hidden = false;
  text('benchmark-status', '');
}

async function loadBenchmarks() {
  text('benchmark-status', 'Loading measured results…');
  $('refresh-benchmarks').disabled = true;
  try {
    const result = await api('/api/benchmarks');
    if (!result.summary || !result.performance?.length) throw new Error('Benchmark results are not available yet.');
    renderBenchmarks(result);
  } catch (e) {
    $('benchmark-content').hidden = true;
    text('benchmark-status', 'Benchmark results are not available yet. Refresh when the run finishes.');
  } finally {
    $('refresh-benchmarks').disabled = false;
  }
}
$('tab-review').onclick = () => showView('review');
$('tab-benchmarks').onclick = () => showView('benchmarks');
$('refresh-benchmarks').onclick = loadBenchmarks;
