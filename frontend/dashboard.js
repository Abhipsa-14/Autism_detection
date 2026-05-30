// AQSense — dashboard logic
const API = '';
const token = localStorage.getItem('token');
if (!token) window.location.href = '/';
const user = JSON.parse(localStorage.getItem('user') || '{}');

const AQ_QUESTIONS = [
  "I often notice small sounds when others do not.",
  "I usually concentrate more on the whole picture, rather than the small details.",
  "I find it easy to do more than one thing at once.",
  "If there is an interruption, I can switch back to what I was doing very quickly.",
  "I find it easy to read between the lines when someone is talking to me.",
  "I know how to tell if someone listening to me is getting bored.",
  "When I'm reading a story, I find it difficult to work out the characters' intentions.",
  "I like to collect information about categories of things (e.g. types of cars, birds, trains).",
  "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
  "I find it difficult to work out people's intentions.",
];

let lastResult = null;
let modelInfo = null;

// ── boot ─────────────────────────────────────────────────────
document.getElementById('sb-name').textContent  = user.full_name || 'User';
document.getElementById('sb-email').textContent = user.email || '';
document.getElementById('avatar').textContent = (user.full_name || 'U').trim().charAt(0).toUpperCase();

// ── Theme ────────────────────────────────────────────────────
function syncThemeIcon() {
  const dark = document.documentElement.classList.contains('dark');
  const sun = document.getElementById('icon-sun');
  const moon = document.getElementById('icon-moon');
  if (sun) sun.classList.toggle('hidden', !dark);
  if (moon) moon.classList.toggle('hidden', dark);
}
function toggleTheme() {
  const dark = document.documentElement.classList.toggle('dark');
  try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch(e) {}
  syncThemeIcon();
}
syncThemeIcon();

// ── Sidebar drawer (mobile / tablet) ─────────────────────────
function openSidebar() {
  document.getElementById('sidebar').classList.add('open');
  const s = document.getElementById('scrim');
  s.classList.remove('opacity-0','pointer-events-none');
  s.classList.add('opacity-100');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  const s = document.getElementById('scrim');
  s.classList.add('opacity-0','pointer-events-none');
  s.classList.remove('opacity-100');
}

const VIEW_TITLES = { screen:'New screening', history:'History', how:'How it works', limits:'Limitations' };

function showView(name) {
  ['screen','history','how','limits'].forEach(v => {
    document.getElementById('view-' + v).classList.toggle('active', v === name);
    const nav = document.getElementById('nav-' + v);
    if (nav) nav.classList.toggle('active', v === name);
  });
  document.getElementById('page-title').textContent = VIEW_TITLES[name] || '';
  document.querySelector('main').scrollTop = 0;
  closeSidebar();
  if (name === 'history') loadHistory();
  if (name === 'how') loadModelInfo();
}

function showToast(msg, type) {
  type = type || 'error';
  const t = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  document.getElementById('toast-icon').innerHTML = type === 'success'
    ? '<svg class="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>'
    : '<svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
  t.classList.add('show');
  setTimeout(function(){ t.classList.remove('show'); }, 3500);
}

function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = '/';
}

// ── SCREENING VIEW ───────────────────────────────────────────
function renderScreenView() {
  document.getElementById('view-screen').innerHTML =
    '<div class="mb-7">' +
      '<h2 class="font-display text-2xl text-ink-900">AQ-10 screening</h2>' +
      '<p class="text-ink-500 text-sm mt-1">A short, evidence-based questionnaire. Takes about two minutes.</p>' +
    '</div>' +
    '<ol class="flex items-center gap-3 mb-7 text-sm">' +
      '<li id="stp-1" class="flex items-center gap-2 font-medium text-teal-700">' +
        '<span class="w-6 h-6 rounded-full bg-teal-600 text-white text-xs flex items-center justify-center">1</span> About you</li>' +
      '<li class="flex-1 h-px bg-line"></li>' +
      '<li id="stp-2" class="flex items-center gap-2 font-medium text-ink-400">' +
        '<span id="stp-2-dot" class="w-6 h-6 rounded-full bg-slate-200 text-ink-500 text-xs flex items-center justify-center">2</span> Questionnaire</li>' +
    '</ol>' +
    '<section id="step-1" class="bg-white rounded-2xl border border-line shadow-card p-6 sm:p-7">' +
      '<h3 class="font-semibold text-ink-900 mb-1">A few details first</h3>' +
      '<p class="text-sm text-ink-500 mb-5">These help interpret the questionnaire. Nothing here is shared without you.</p>' +
      '<div class="grid sm:grid-cols-2 gap-4">' +
        '<div><label class="block text-xs font-medium text-ink-700 mb-1.5">Age</label>' +
          '<input id="age" type="number" min="1" max="120" placeholder="e.g. 24" class="field w-full rounded-lg border border-line bg-white px-3.5 py-2.5 text-sm placeholder:text-ink-400" /></div>' +
        '<div><label class="block text-xs font-medium text-ink-700 mb-1.5">Gender</label>' +
          '<select id="gender" class="field w-full rounded-lg border border-line bg-white px-3.5 py-2.5 text-sm"><option value="">Select…</option><option value="m">Male</option><option value="f">Female</option></select></div>' +
        '<div><label class="block text-xs font-medium text-ink-700 mb-1.5">Jaundice at birth</label>' +
          '<select id="jaundice" class="field w-full rounded-lg border border-line bg-white px-3.5 py-2.5 text-sm"><option value="">Select…</option><option value="true">Yes</option><option value="false">No</option></select></div>' +
        '<div><label class="block text-xs font-medium text-ink-700 mb-1.5">Immediate family member with autism</label>' +
          '<select id="family_autism" class="field w-full rounded-lg border border-line bg-white px-3.5 py-2.5 text-sm"><option value="">Select…</option><option value="true">Yes</option><option value="false">No</option></select></div>' +
      '</div>' +
      '<div class="mt-3 text-xs text-ink-400 flex items-start gap-1.5">' +
        '<svg class="w-4 h-4 flex-shrink-0 mt-px" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>' +
        '<span>For children, a parent or caregiver should answer on the child\'s behalf.</span></div>' +
      '<button onclick="goStep2()" class="btn btn-primary mt-6 w-full sm:w-auto px-6 py-2.5 rounded-lg bg-teal-600 text-white font-semibold text-sm">Continue to questionnaire</button>' +
    '</section>' +
    '<section id="step-2" class="hidden">' +
      '<div class="bg-white rounded-2xl border border-line shadow-card p-6 sm:p-7">' +
        '<div class="flex items-center justify-between mb-1"><h3 class="font-semibold text-ink-900">The questionnaire</h3>' +
          '<span id="answered-count" class="text-xs font-medium text-ink-400">0 / 10</span></div>' +
        '<p class="text-sm text-ink-500 mb-5">Choose the response that best fits. There are no right or wrong answers.</p>' +
        '<div id="questions" class="divide-y divide-line"></div></div>' +
      '<div class="flex flex-col-reverse sm:flex-row gap-3 mt-5">' +
        '<button onclick="goStep1()" class="px-5 py-2.5 rounded-lg border border-line bg-white text-sm font-medium text-ink-700 hover:bg-slate-50 transition-colors">Back</button>' +
        '<button onclick="submitScreening()" id="btn-submit" class="btn btn-primary flex-1 px-6 py-2.5 rounded-lg bg-teal-600 text-white font-semibold text-sm flex items-center justify-center gap-2">Generate result</button></div>' +
    '</section>' +
    '<section id="result" class="hidden"></section>';

  document.getElementById('questions').innerHTML = AQ_QUESTIONS.map(function(text, i) {
    return '<div class="py-4 first:pt-0 last:pb-0"><div class="flex gap-3">' +
      '<span class="text-sm font-semibold text-teal-700 w-5 flex-shrink-0">' + (i+1) + '</span>' +
      '<div class="flex-1"><p class="text-sm text-ink-800 mb-3">' + text + '</p>' +
      '<div class="grid grid-cols-2 gap-2.5 max-w-xs choice">' +
        '<div><input type="radio" id="q' + i + '-y" name="q' + i + '" value="1" onchange="updateCount()" />' +
          '<label for="q' + i + '-y" class="block text-center text-sm font-medium text-ink-600 border border-line rounded-lg py-2">Agree</label></div>' +
        '<div><input type="radio" id="q' + i + '-n" name="q' + i + '" value="0" onchange="updateCount()" />' +
          '<label for="q' + i + '-n" class="block text-center text-sm font-medium text-ink-600 border border-line rounded-lg py-2">Disagree</label></div>' +
      '</div></div></div></div>';
  }).join('');
}

function updateCount() {
  let n = 0;
  for (let i=0;i<10;i++) if (document.querySelector('input[name="q' + i + '"]:checked')) n++;
  document.getElementById('answered-count').textContent = n + ' / 10';
}

function goStep2() {
  const age = document.getElementById('age').value;
  const gender = document.getElementById('gender').value;
  const jaundice = document.getElementById('jaundice').value;
  const family = document.getElementById('family_autism').value;
  if (!age || !gender || !jaundice || !family) { showToast('Please complete all four fields.'); return; }
  document.getElementById('step-1').classList.add('hidden');
  document.getElementById('step-2').classList.remove('hidden');
  document.getElementById('result').classList.add('hidden');
  document.getElementById('stp-2').classList.remove('text-ink-400');
  document.getElementById('stp-2').classList.add('text-teal-700');
  document.getElementById('stp-2-dot').classList.remove('bg-slate-200','text-ink-500');
  document.getElementById('stp-2-dot').classList.add('bg-teal-600','text-white');
}
function goStep1() {
  document.getElementById('step-1').classList.remove('hidden');
  document.getElementById('step-2').classList.add('hidden');
}

// ── submit & result ─────────────────────────────────────────
async function submitScreening() {
  const answers = [];
  for (let i=0;i<10;i++) {
    const sel = document.querySelector('input[name="q' + i + '"]:checked');
    if (!sel) { showToast('Please answer question ' + (i+1) + '.'); return; }
    answers.push(parseInt(sel.value));
  }
  const payload = {
    a1:answers[0],a2:answers[1],a3:answers[2],a4:answers[3],a5:answers[4],
    a6:answers[5],a7:answers[6],a8:answers[7],a9:answers[8],a10:answers[9],
    age: parseFloat(document.getElementById('age').value),
    gender: document.getElementById('gender').value,
    jaundice: document.getElementById('jaundice').value === 'true',
    family_autism: document.getElementById('family_autism').value === 'true',
  };
  const btn = document.getElementById('btn-submit');
  btn.disabled = true; btn.classList.add('opacity-70');
  btn.innerHTML = '<svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg> Analysing';
  try {
    const res = await fetch(API + '/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json','Authorization':'Bearer ' + token},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');
    data._input = payload;
    lastResult = data;
    renderResult(data);
  } catch(e) { showToast(e.message); }
  finally { btn.disabled=false; btn.classList.remove('opacity-70'); btn.innerHTML='Generate result'; }
}

function riskTheme(risk) {
  if (risk === 'High Risk')     return { txt:'text-rose-700', bg:'bg-rose-50', ring:'ring-rose-100', dot:'bg-rose-500', stroke:'#e11d48' };
  if (risk === 'Moderate Risk') return { txt:'text-amber-700', bg:'bg-amber-50', ring:'ring-amber-100', dot:'bg-amber-500', stroke:'#d97706' };
  return { txt:'text-teal-700', bg:'bg-teal-50', ring:'ring-teal-100', dot:'bg-teal-500', stroke:'#0d9488' };
}

function renderResult(d) {
  const pct = Math.round(d.confidence * 100);
  const t = riskTheme(d.risk_level);
  const positive = d.prediction === 1;
  const circ = 2 * Math.PI * 52;
  const date = new Date(d.created_at || Date.now()).toLocaleString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'});

  const factors = (d.explanation || []).map(function(e) {
    const up = e.direction === 'increases';
    const w = Math.min(100, Math.round(Math.abs(e.impact) * 38) + 12);
    return '<div class="flex items-center gap-3 py-2">' +
      '<span class="flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center ' + (up?'bg-rose-50 text-rose-600':'bg-teal-50 text-teal-600') + '">' +
        '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.4" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="' + (up?'M5 10l7-7m0 0l7 7m-7-7v18':'M19 14l-7 7m0 0l-7-7m7 7V3') + '"/></svg></span>' +
      '<span class="text-sm text-ink-700 flex-1 truncate">' + e.label + '</span>' +
      '<span class="hidden sm:block w-28 h-1.5 rounded-full bg-slate-100 overflow-hidden">' +
        '<span class="block h-full bar-grow ' + (up?'bg-rose-400':'bg-teal-400') + '" style="width:0%" data-w="' + w + '"></span></span>' +
    '</div>';
  }).join('');

  document.getElementById('step-2').classList.add('hidden');
  const el = document.getElementById('result');
  el.classList.remove('hidden');
  el.innerHTML =
    '<div class="bg-white rounded-2xl border border-line shadow-card overflow-hidden">' +
      '<div class="px-6 sm:px-8 pt-7 pb-6 border-b border-line flex flex-col sm:flex-row sm:items-center gap-6">' +
        '<div class="relative flex-shrink-0 mx-auto sm:mx-0">' +
          '<svg class="w-32 h-32" viewBox="0 0 120 120">' +
            '<circle cx="60" cy="60" r="52" fill="none" stroke="#eef2f6" stroke-width="9" class="gauge-bg"/>' +
            '<circle id="arc" cx="60" cy="60" r="52" fill="none" stroke="' + t.stroke + '" stroke-width="9" stroke-linecap="round" stroke-dasharray="' + circ + '" stroke-dashoffset="' + circ + '" transform="rotate(-90 60 60)" class="gauge-ring"/></svg>' +
          '<div class="absolute inset-0 flex flex-col items-center justify-center">' +
            '<span class="text-2xl font-semibold text-ink-900">' + pct + '%</span>' +
            '<span class="text-[11px] text-ink-400">model confidence</span></div></div>' +
        '<div class="text-center sm:text-left flex-1">' +
          '<span class="inline-flex items-center gap-1.5 text-xs font-medium ' + t.txt + ' ' + t.bg + ' ring-1 ' + t.ring + ' rounded-full px-2.5 py-1 mb-3">' +
            '<span class="w-1.5 h-1.5 rounded-full ' + t.dot + '"></span> ' + d.risk_level + '</span>' +
          '<h3 class="font-display text-2xl text-ink-900">' + (positive?'Autism-associated traits indicated':'Few autism-associated traits') + '</h3>' +
          '<p class="text-sm text-ink-500 mt-2 max-w-md">' + (positive
            ? 'Your responses align with patterns associated with autism. This is a prompt to seek a formal evaluation — not a diagnosis.'
            : 'Your responses show few of the patterns associated with autism at this time. If you have ongoing concerns, speak with a professional.') + '</p></div></div>' +
      '<div class="grid grid-cols-3 divide-x divide-line border-b border-line">' +
        '<div class="px-4 py-4 text-center"><p class="text-xl font-semibold text-ink-900">' + d.aq_score + '<span class="text-ink-400 text-sm font-normal">/10</span></p><p class="text-xs text-ink-400 mt-0.5">AQ-10 score</p></div>' +
        '<div class="px-4 py-4 text-center"><p class="text-xl font-semibold text-ink-900">' + pct + '%</p><p class="text-xs text-ink-400 mt-0.5">Confidence</p></div>' +
        '<div class="px-4 py-4 text-center"><p class="text-xl font-semibold ' + t.txt + '">' + (positive?'Positive':'Negative') + '</p><p class="text-xs text-ink-400 mt-0.5">Screening flag</p></div></div>' +
      (factors ?
        '<div class="px-6 sm:px-8 py-6 border-b border-line">' +
          '<div class="flex items-center justify-between mb-2"><h4 class="text-sm font-semibold text-ink-900">What influenced this result</h4>' +
            '<span class="text-xs text-ink-400">top contributing answers</span></div>' +
          '<p class="text-xs text-ink-500 mb-3">Teal lowered the estimated likelihood; rose raised it.</p>' +
          '<div class="divide-y divide-line/70">' + factors + '</div></div>' : '') +
      '<div class="px-6 sm:px-8 py-5 flex flex-col sm:flex-row items-center gap-3 bg-slate-50/60">' +
        '<button onclick="exportPDF()" class="btn btn-primary w-full sm:w-auto px-5 py-2.5 rounded-lg bg-teal-600 text-white font-semibold text-sm flex items-center justify-center gap-2">' +
          '<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg> Download PDF for clinician</button>' +
        '<button onclick="resetForm()" class="w-full sm:w-auto px-5 py-2.5 rounded-lg border border-line bg-white text-sm font-medium text-ink-700 hover:bg-slate-50 transition-colors">Take another screening</button>' +
        '<span class="sm:ml-auto text-xs text-ink-400">' + date + '</span></div>' +
    '</div>' +
    '<div class="mt-4 flex items-start gap-2 text-xs text-ink-400 px-1">' +
      '<svg class="w-4 h-4 flex-shrink-0 mt-px" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>' +
      '<span>This summary is screening support only and is not a clinical diagnosis. Please review it with a qualified healthcare professional.</span></div>';

  requestAnimationFrame(function() {
    setTimeout(function() {
      document.getElementById('arc').style.strokeDashoffset = circ - (circ * d.confidence);
      el.querySelectorAll('.bar-grow').forEach(function(b){ b.style.width = b.dataset.w + '%'; });
    }, 80);
  });
  el.scrollIntoView({ behavior:'smooth', block:'start' });
}

function resetForm() {
  renderScreenView();
}

// ── PDF export (clinician-ready report) ──────────────────────
function exportPDF() {
  if (!lastResult) { showToast('No result to export yet.'); return; }
  const d = lastResult;
  const jsPDF = window.jspdf.jsPDF;
  const doc = new jsPDF({ unit:'pt', format:'a4' });
  const W = doc.internal.pageSize.getWidth();
  const M = 48;
  let y = 0;

  const teal = [13,148,136], ink = [15,23,42], grey = [100,116,139], line = [226,232,240];
  const pct = Math.round(d.confidence * 100);
  const positive = d.prediction === 1;
  const now = new Date(d.created_at || Date.now());
  const inp = d._input || {};
  const rc = positive ? [225,29,72] : [13,148,136];

  // header band
  doc.setFillColor(ink[0],ink[1],ink[2]); doc.rect(0, 0, W, 92, 'F');
  doc.setFillColor(teal[0],teal[1],teal[2]); doc.roundedRect(M, 30, 26, 26, 5, 5, 'F');
  doc.setDrawColor(255,255,255); doc.setLineWidth(2.2);
  doc.line(M+6, 43, M+11, 48); doc.line(M+11, 48, M+20, 38);
  doc.setTextColor(255,255,255);
  doc.setFont('helvetica','bold'); doc.setFontSize(16); doc.text('AQSense', M+38, 44);
  doc.setFont('helvetica','normal'); doc.setFontSize(9); doc.setTextColor(180,200,210);
  doc.text('AQ-10 Autism Screening Summary', M+38, 58);
  doc.setFontSize(8); doc.setTextColor(150,170,185);
  doc.text('Generated ' + now.toLocaleString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'}), W-M, 44, {align:'right'});
  doc.text('Model ' + (d.model_version!=null?'v'+d.model_version:'-'), W-M, 58, {align:'right'});
  y = 124;

  // disclaimer banner
  doc.setFillColor(255,247,237); doc.setDrawColor(251,211,141); doc.setLineWidth(0.8);
  doc.roundedRect(M, y, W-2*M, 42, 4, 4, 'FD');
  doc.setTextColor(146,64,14); doc.setFont('helvetica','bold'); doc.setFontSize(8.5);
  doc.text('SCREENING SUPPORT ONLY - NOT A CLINICAL DIAGNOSIS', M+12, y+16);
  doc.setFont('helvetica','normal'); doc.setTextColor(120,80,30); doc.setFontSize(8);
  doc.text(doc.splitTextToSize('This document summarises a self-administered AQ-10 questionnaire processed by a machine-learning model. It is intended to support, not replace, evaluation by a qualified clinician.', W-2*M-24), M+12, y+28);
  y += 62;

  // outcome row
  const colW = (W - 2*M - 16) / 2;
  doc.setDrawColor(line[0],line[1],line[2]); doc.setLineWidth(0.8);
  doc.roundedRect(M, y, colW, 96, 5, 5, 'S');
  doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','bold'); doc.setFontSize(8);
  doc.text('SCREENING OUTCOME', M+14, y+20);
  doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold'); doc.setFontSize(13);
  doc.text(positive ? 'Traits indicated' : 'Few traits', M+14, y+42);
  doc.setFillColor(rc[0],rc[1],rc[2]); doc.circle(M+18, y+58, 3, 'F');
  doc.setTextColor(rc[0],rc[1],rc[2]); doc.setFont('helvetica','bold'); doc.setFontSize(9);
  doc.text(d.risk_level, M+26, y+61);
  doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','normal'); doc.setFontSize(8.5);
  doc.text('AQ-10 score: ' + d.aq_score + ' / 10', M+14, y+82);

  const rx = M + colW + 16;
  doc.roundedRect(rx, y, colW, 96, 5, 5, 'S');
  doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','bold'); doc.setFontSize(8);
  doc.text('MODEL CONFIDENCE', rx+14, y+20);
  doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold'); doc.setFontSize(26);
  doc.text(pct + '%', rx+14, y+52);
  doc.setFillColor(238,242,246); doc.roundedRect(rx+14, y+64, colW-28, 8, 4, 4, 'F');
  doc.setFillColor(rc[0],rc[1],rc[2]); doc.roundedRect(rx+14, y+64, (colW-28)*d.confidence, 8, 4, 4, 'F');
  doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','normal'); doc.setFontSize(8);
  doc.text('Calibrated probability of autism-associated traits', rx+14, y+86);
  y += 120;

  // respondent details
  doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold'); doc.setFontSize(10);
  doc.text('Respondent details', M, y); y += 8;
  doc.setDrawColor(line[0],line[1],line[2]); doc.line(M, y, W-M, y); y += 18;
  const details = [
    ['Name', user.full_name || '-'],
    ['Age', inp.age!=null ? String(inp.age) : '-'],
    ['Gender', inp.gender==='m'?'Male':inp.gender==='f'?'Female':'-'],
    ['Jaundice at birth', inp.jaundice ? 'Yes' : 'No'],
    ['Family history of autism', inp.family_autism ? 'Yes' : 'No'],
  ];
  doc.setFontSize(9);
  let ry = y;
  details.forEach(function(row, i) {
    const col = i % 2, cx = M + col * (colW + 16);
    if (col === 0 && i > 0) ry += 22;
    doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','normal');
    doc.text(row[0], cx, ry);
    doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold');
    doc.text(row[1], cx + 130, ry);
  });
  y = ry + 28;

  // contributing factors
  if (Array.isArray(d.explanation) && d.explanation.length) {
    doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold'); doc.setFontSize(10);
    doc.text('Key contributing answers', M, y); y += 8;
    doc.setDrawColor(line[0],line[1],line[2]); doc.line(M, y, W-M, y); y += 6;
    doc.setFont('helvetica','normal'); doc.setFontSize(8); doc.setTextColor(grey[0],grey[1],grey[2]);
    doc.text('Ranked by influence on this result. Arrows show direction of effect.', M, y+10); y += 24;
    const maxImp = Math.max.apply(null, d.explanation.map(function(e){ return Math.abs(e.impact); }).concat([1]));
    d.explanation.forEach(function(e) {
      const up = e.direction === 'increases';
      const bc = up ? [225,29,72] : [13,148,136];
      doc.setTextColor(bc[0],bc[1],bc[2]); doc.setFont('helvetica','bold'); doc.setFontSize(11);
      doc.text(up ? '\u2191' : '\u2193', M, y+2);
      doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','normal'); doc.setFontSize(9);
      doc.text(doc.splitTextToSize(e.label, 300)[0], M+16, y+2);
      const bw = 150, bx = W - M - bw;
      doc.setFillColor(238,242,246); doc.roundedRect(bx, y-5, bw, 7, 3.5, 3.5, 'F');
      doc.setFillColor(bc[0],bc[1],bc[2]); doc.roundedRect(bx, y-5, bw*(Math.abs(e.impact)/maxImp), 7, 3.5, 3.5, 'F');
      y += 20;
    });
    y += 6;
  }

  // next steps
  if (y > 650) { doc.addPage(); y = 60; }
  doc.setTextColor(ink[0],ink[1],ink[2]); doc.setFont('helvetica','bold'); doc.setFontSize(10);
  doc.text('Suggested next steps', M, y); y += 8;
  doc.setDrawColor(line[0],line[1],line[2]); doc.line(M, y, W-M, y); y += 18;
  const steps = [
    'Share this summary with a GP, paediatrician, or psychologist.',
    'A formal diagnosis requires a comprehensive in-person clinical assessment.',
    'Bring any observations about communication, routines, and sensory preferences.',
  ];
  doc.setFont('helvetica','normal'); doc.setFontSize(9); doc.setTextColor(ink[0],ink[1],ink[2]);
  steps.forEach(function(s) {
    doc.setFillColor(teal[0],teal[1],teal[2]); doc.circle(M+2, y-3, 1.8, 'F');
    doc.text(doc.splitTextToSize(s, W-2*M-16), M+12, y); y += 18;
  });

  // footer
  const fy = doc.internal.pageSize.getHeight() - 36;
  doc.setDrawColor(line[0],line[1],line[2]); doc.line(M, fy, W-M, fy);
  doc.setTextColor(grey[0],grey[1],grey[2]); doc.setFont('helvetica','normal'); doc.setFontSize(7.5);
  doc.text('AQSense AQ-10 Screening Platform - The AQ-10 is a recognised screening instrument; outcomes do not constitute a diagnosis.', M, fy+14);

  doc.save('AQSense-Screening-' + now.toISOString().slice(0,10) + '.pdf');
  showToast('PDF downloaded.', 'success');
}

// ── HISTORY VIEW ─────────────────────────────────────────────
async function loadHistory() {
  const v = document.getElementById('view-history');
  v.innerHTML =
    '<div class="mb-7"><h2 class="font-display text-2xl text-ink-900">Screening history</h2>' +
    '<p class="text-ink-500 text-sm mt-1">Every screening saved to your account.</p></div>' +
    '<div id="history-body"><div class="text-center text-ink-400 text-sm py-16">Loading…</div></div>';
  const body = document.getElementById('history-body');
  try {
    const res = await fetch(API + '/api/history', { headers:{'Authorization':'Bearer ' + token} });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not load history');
    if (!data.total) {
      body.innerHTML =
        '<div class="bg-white rounded-2xl border border-line shadow-card py-16 text-center">' +
          '<div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4">' +
            '<svg class="w-6 h-6 text-ink-400" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg></div>' +
          '<p class="text-ink-700 font-medium">No screenings yet</p>' +
          '<p class="text-ink-400 text-sm mt-1">Your completed screenings will appear here.</p>' +
          '<button onclick="showView(\'screen\')" class="btn btn-primary mt-5 px-5 py-2.5 rounded-lg bg-teal-600 text-white text-sm font-semibold">Start a screening</button></div>';
      return;
    }
    body.innerHTML = '<div class="space-y-3">' + data.results.map(function(r) {
      const pct = Math.round(r.confidence*100);
      const t = riskTheme(r.risk_level);
      const date = new Date(r.created_at).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'});
      return '<div class="bg-white rounded-xl border border-line shadow-card px-5 py-4 flex items-center gap-4">' +
        '<div class="w-10 h-10 rounded-full ' + t.bg + ' ring-1 ' + t.ring + ' flex items-center justify-center flex-shrink-0">' +
          '<span class="w-2 h-2 rounded-full ' + t.dot + '"></span></div>' +
        '<div class="flex-1 min-w-0"><div class="flex items-center gap-2">' +
          '<span class="text-sm font-medium text-ink-900">' + (r.prediction===1?'Traits indicated':'Few traits') + '</span>' +
          '<span class="text-xs ' + t.txt + ' font-medium">· ' + r.risk_level + '</span></div>' +
          '<p class="text-xs text-ink-400 mt-0.5">' + date + ' · AQ ' + r.aq_score + '/10 · Age ' + r.age + ' · ' + (r.gender==='m'?'Male':'Female') + '</p></div>' +
        '<div class="text-right flex-shrink-0"><p class="text-lg font-semibold text-ink-900">' + pct + '%</p>' +
          '<p class="text-[11px] text-ink-400">confidence</p></div></div>';
    }).join('') + '</div>';
  } catch(e) {
    body.innerHTML = '<div class="text-center text-red-500 text-sm py-16">' + e.message + '</div>';
  }
}

// ── HOW IT WORKS VIEW ────────────────────────────────────────
async function loadModelInfo() {
  const v = document.getElementById('view-how');
  if (!modelInfo) {
    try {
      const res = await fetch(API + '/api/model-info');
      modelInfo = await res.json();
      if (modelInfo.version != null) document.getElementById('badge-version').textContent = 'v' + modelInfo.version;
    } catch(e) { modelInfo = {}; }
  }
  const m = modelInfo.metrics || {};
  const fmtPct = function(x){ return x!=null ? Math.round(x*100)+'%' : '-'; };
  const metricCards = [
    ['Sensitivity', fmtPct(m.sensitivity_recall), 'Correctly flags true cases'],
    ['Specificity', fmtPct(m.specificity), 'Correctly clears non-cases'],
    ['Precision', fmtPct(m.precision), 'Of those flagged, share correct'],
    ['ROC-AUC', m.roc_auc!=null?m.roc_auc.toFixed(3):'-', 'Overall ranking quality'],
  ];
  const steps = [
    ['Answer the AQ-10', 'You respond to ten short statements about attention, communication, and social preference, plus a few background details.'],
    ['The model reviews your pattern', 'A calibrated gradient-boosted model, trained on 1,100 screening records across children, adolescents, and adults, scores your responses.'],
    ['You get an explained result', 'The result shows a calibrated confidence and the specific answers that influenced it the most.'],
    ['Take it further', 'Download a PDF summary and share it with a qualified clinician for a formal assessment.'],
  ];
  v.innerHTML =
    '<div class="mb-7"><h2 class="font-display text-2xl text-ink-900">How AQSense works</h2>' +
    '<p class="text-ink-500 text-sm mt-1">Transparent by design. Here\'s exactly what happens behind a screening.</p></div>' +
    '<div class="bg-white rounded-2xl border border-line shadow-card p-6 sm:p-7 mb-5"><ol class="space-y-5">' +
      steps.map(function(s,i){ return '<li class="flex gap-4">' +
        '<span class="flex-shrink-0 w-7 h-7 rounded-full bg-teal-50 text-teal-700 text-sm font-semibold flex items-center justify-center ring-1 ring-teal-100">' + (i+1) + '</span>' +
        '<div><p class="text-sm font-semibold text-ink-900">' + s[0] + '</p>' +
        '<p class="text-sm text-ink-500 mt-0.5">' + s[1] + '</p></div></li>'; }).join('') +
    '</ol></div>' +
    '<div class="mb-3 flex items-center justify-between"><h3 class="text-sm font-semibold text-ink-900">Measured performance</h3>' +
      '<span class="text-xs text-ink-400">held-out test set · ' + (modelInfo.n_samples||'-') + ' records</span></div>' +
    '<div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-2">' +
      metricCards.map(function(c){ return '<div class="bg-white rounded-xl border border-line shadow-card p-4">' +
        '<p class="text-2xl font-semibold text-ink-900">' + c[1] + '</p>' +
        '<p class="text-xs font-medium text-ink-700 mt-1">' + c[0] + '</p>' +
        '<p class="text-[11px] text-ink-400 mt-0.5 leading-snug">' + c[2] + '</p></div>'; }).join('') +
    '</div>' +
    '<p class="text-xs text-ink-400 px-1 mb-6">Model: ' + (modelInfo.model_type||'-') + ' · Source: ' + (modelInfo.data_source||'-') + '. Metrics are recomputed on each training run.</p>' +
    '<div class="bg-teal-50/60 rounded-2xl border border-teal-100 p-6">' +
      '<h3 class="text-sm font-semibold text-teal-800 mb-1">Why we show the "why"</h3>' +
      '<p class="text-sm text-teal-900/70 leading-relaxed">A confidence number on its own isn\'t trustworthy. Every AQSense result lists the individual answers that moved the estimate up or down, so you and your clinician can see the reasoning rather than taking a black box at its word.</p></div>';
}

// ── LIMITATIONS VIEW ─────────────────────────────────────────
function renderLimits() {
  const items = [
    ['Not a diagnosis', 'AQSense is a screening aid. Only a qualified clinician can diagnose autism, through a comprehensive in-person assessment.', 'M9 12l2 2 4-4'],
    ['The label problem', 'In the public dataset used for training, the autism label is derived from the AQ-10 score itself rather than an independent clinical diagnosis. A simple "score of 6 or more" rule already matches the label about 88% of the time, so high accuracy partly reflects the model re-learning the questionnaire\'s own scoring rule.', 'M12 9v3.75m0 3.75h.01'],
    ['Self-report bias', 'Answers are self-reported (or given by a caregiver for children). Responses can be affected by mood, insight, and interpretation, which the model cannot detect.', 'M16 7a4 4 0 11-8 0 4 4 0 018 0z'],
    ['Population limits', 'The training data covers specific cohorts and may not represent every age, culture, or background equally. Results may be less reliable outside those groups.', 'M3 6l3 1m0 0l-3 9a5 5 0 006 0'],
    ['One questionnaire, one moment', 'A single AQ-10 captures one snapshot in time. Autism presentation is complex and varies across contexts and over time.', 'M12 8v4l3 3'],
  ];
  document.getElementById('view-limits').innerHTML =
    '<div class="mb-7"><h2 class="font-display text-2xl text-ink-900">Limitations & honesty</h2>' +
    '<p class="text-ink-500 text-sm mt-1">What this tool can and can\'t tell you. Please read before relying on a result.</p></div>' +
    '<div class="bg-amber-50 border border-amber-200 rounded-2xl p-5 mb-5 flex gap-3">' +
      '<svg class="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m0 3.75h.01M10.34 3.94l-7.5 12.99A1.5 1.5 0 004.16 19.5h15.68a1.5 1.5 0 001.32-2.57l-7.5-12.99a1.5 1.5 0 00-2.62 0z"/></svg>' +
      '<p class="text-sm text-amber-900/80 leading-relaxed"><span class="font-semibold text-amber-900">AQSense does not diagnose autism.</span> It is an educational screening aid. If a result concerns you, or even if it doesn\'t but you have ongoing concerns, please consult a healthcare professional.</p></div>' +
    '<div class="space-y-3">' +
      items.map(function(it){ return '<div class="bg-white rounded-xl border border-line shadow-card p-5 flex gap-4">' +
        '<div class="w-9 h-9 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">' +
          '<svg class="w-5 h-5 text-ink-500" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="' + it[2] + '"/></svg></div>' +
        '<div><p class="text-sm font-semibold text-ink-900">' + it[0] + '</p>' +
        '<p class="text-sm text-ink-500 mt-1 leading-relaxed">' + it[1] + '</p></div></div>'; }).join('') +
    '</div>' +
    '<p class="text-xs text-ink-400 text-center mt-8">If you are in crisis or need urgent help, contact your local emergency services.</p>';
}

// ── init ─────────────────────────────────────────────────────
renderScreenView();
renderLimits();
loadModelInfo();
