const D = window.DATA;
const $ = (id) => document.getElementById(id);
const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
const REF = D.table.find((m) => m.name === 'Reference CNN');
const SINGLE = D.table.find((m) => m.name === 'Single-image CNN');

/* ---------- progress and XP (per-viewer convenience only) ---------- */
const store = {
  read() { try { return JSON.parse(localStorage.getItem('forgerylab.v1')) || {}; } catch { return {}; } },
  write(v) { try { localStorage.setItem('forgerylab.v1', JSON.stringify(v)); } catch { /* private mode */ } },
};
const state = { xp: 0, done: {}, ...store.read() };
$('xp').textContent = state.xp;

function award(key, points) {
  if (state.done[key]) return;
  state.done[key] = true;
  addXp(points);
}
function addXp(points) {
  state.xp += points;
  store.write(state);
  $('xp').textContent = state.xp;
  const pill = $('xp').parentElement;
  pill.classList.remove('bump');
  void pill.offsetWidth;
  pill.classList.add('bump');
}

/* ---------- tooltip ---------- */
const tip = $('tip');
function showTip(event, text) {
  tip.textContent = text;
  tip.classList.add('on');
  const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${event.clientY - 38}px`;
}
const hideTip = () => tip.classList.remove('on');

/* ---------- hero ---------- */
$('marquee').innerHTML = Array(10).fill('<span>A PERFECT SCORE CAN BE A BROKEN EXPERIMENT ✳</span>').join('');
$('lede').textContent = `A CNN on raw scans hit ${pct(D.control_acc, 0)}, and so did a one-line brightness rule, because genuine and forged signatures were scanned differently. Remove that shortcut, test on strangers, and the honest answer is ${pct(REF.Accuracy)}. Here is how to tell the difference.`;
$('facts').innerHTML = [
  ['Signatures', D.n_images.toLocaleString('en-IN')],
  ['Signers', D.n_writers],
  ['Honest accuracy', pct(REF.Accuracy)],
].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');

/* ---------- lesson 01: the trap ---------- */
const W = 640, H = 240, PAD = { l: 42, r: 8, t: 10, b: 30 };
const bins = D.background.bins;
const maxCount = Math.max(...D.background.genuine, ...D.background.forged);
const bx = (v) => PAD.l + ((v - bins[0]) / (bins[bins.length - 1] + 1 - bins[0])) * (W - PAD.l - PAD.r);
const by = (v) => H - PAD.b - (v / maxCount) * (H - PAD.t - PAD.b);

function trapChart() {
  const bw = bx(bins[0] + 1) - bx(bins[0]) - 2;
  const bars = bins.map((b, i) => [['genuine', D.background.genuine[i]], ['forged', D.background.forged[i]]]
    .filter(([, n]) => n)
    .map(([c, n]) => `<rect class="hit" data-t="${c === 'genuine' ? 'Genuine' : 'Forged'}, grey level ${b}: ${n} images" x="${bx(b) + 1}" y="${by(n)}" width="${bw}" height="${H - PAD.b - by(n)}" rx="2" fill="var(--${c})"/>`).join('')).join('');
  const ticks = [235, 240, 245, 250, 255].map((t) => `<text x="${bx(t + .5)}" y="${H - 10}" text-anchor="middle">${t}</text>`).join('');
  const yTicks = [0, 250, 500].filter((t) => t <= maxCount).map((t) => `<text x="${PAD.l - 8}" y="${by(t) + 4}" text-anchor="end">${t}</text><line x1="${PAD.l}" x2="${W - PAD.r}" y1="${by(t)}" y2="${by(t)}" stroke="var(--line)"/>`).join('');
  $('trap-chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of background brightness: genuine images cluster near 240, forged images near 254, with no overlap">${yTicks}${bars}${ticks}<line id="cut-line" y1="${PAD.t}" y2="${H - PAD.b}" stroke="var(--paper)" stroke-width="2"/></svg>`;
}
function trapUpdate() {
  const t = Number($('trap-range').value);
  let correct = 0, total = 0;
  bins.forEach((b, i) => {
    correct += (b < t ? D.background.genuine[i] : D.background.forged[i]);
    total += D.background.genuine[i] + D.background.forged[i];
  });
  const acc = correct / total;
  $('trap-t').textContent = t;
  $('trap-acc').textContent = pct(acc);
  const line = document.getElementById('cut-line');
  line.setAttribute('x1', bx(t)); line.setAttribute('x2', bx(t));
  if (acc === 1) {
    $('trap-note').textContent = `100%. You just matched the raw-image CNN, which scored ${pct(D.control_acc, 0)}, without reading a single stroke. Any model given these images can do the same, so its score proves nothing.`;
    award('trap', 20);
  } else {
    $('trap-note').textContent = acc > .75 ? 'Close. Keep dragging towards the gap between the two groups.' : 'Drag the line into the gap between the two groups.';
  }
}
trapChart();
$('trap-range').addEventListener('input', trapUpdate);
trapUpdate();

/* ---------- lesson 02: clean ---------- */
$('clean-steps').innerHTML = D.clean_demo.map(([src, title, text]) => `
  <figure><div class="pic"><img src="${src}" alt="${title} of a signature" loading="lazy" /></div>
  <figcaption><b>${title}</b>${text}</figcaption></figure>`).join('');

/* ---------- lesson 03: split ---------- */
function setSplit(kind) {
  const random = kind === 'random';
  $('split-random').classList.toggle('on', random);
  $('split-signer').classList.toggle('on', !random);
  $('split-acc').textContent = pct(random ? D.leak_acc : D.single_acc);
  $('split-note').textContent = random
    ? `Looks better, but ${((D.leak_acc - D.single_acc) * 100).toFixed(1)} points of that come from having met the same people in training.`
    : 'This is the fair figure: every test signer is a stranger to the model.';
}
$('split-random').addEventListener('click', () => { setSplit('random'); award('split', 10); });
$('split-signer').addEventListener('click', () => setSplit('signer'));
setSplit('signer');

/* ---------- lesson 04: learning curves ---------- */
(function curves() {
  const series = [['Reference CNN', D.history.reference, 'var(--genuine)'], ['Single-image CNN', D.history.single, '#a58f6f']];
  const n = D.history.reference.length;
  const h = 220, p = { l: 42, r: 10, t: 10, b: 28 };
  const x = (i) => p.l + (i / (n - 1)) * (W - p.l - p.r);
  const y = (v) => h - p.b - ((v - .4) / .6) * (h - p.t - p.b);
  const grid = [.4, .6, .8, 1].map((v) => `<line x1="${p.l}" x2="${W - p.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${p.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join('');
  const lines = series.map(([, s, c]) => `<polyline fill="none" stroke="${c}" stroke-width="2.5" stroke-linejoin="round" points="${s.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>`).join('');
  const hits = D.history.reference.map((v, i) => `<rect class="hit" data-t="Epoch ${i + 1}: reference ${pct(v)}, single ${pct(D.history.single[i])}" x="${x(i) - (W / n) / 2}" y="0" width="${W / n}" height="${h}" fill="transparent"/>`).join('');
  const ticks = [0, Math.floor((n - 1) / 2), n - 1].map((i) => `<text x="${x(i)}" y="${h - 8}" text-anchor="middle">epoch ${i + 1}</text>`).join('');
  $('curve-chart').innerHTML = `<svg viewBox="0 0 ${W} ${h}" role="img" aria-label="Validation accuracy per epoch for both CNNs">${grid}${lines}${ticks}${hits}</svg>`;
})();

/* ---------- lesson 05: results ---------- */
$('results-title').textContent = D.ref_beats_80
  ? `${pct(REF.Accuracy)} on strangers, without the shortcut.`
  : `${pct(REF.Accuracy)}: below 80%, and honest about why.`;
$('n-test').textContent = D.test_queries;
$('kpis').innerHTML = [
  ['Accuracy', pct(REF.Accuracy), 'Reference CNN, unseen signers'],
  ['ROC AUC', REF['ROC AUC'].toFixed(3), 'Ranking quality across all cut-offs'],
  ['Forgeries accepted', pct(REF.FAR), 'False acceptance rate at 0.50'],
  ['Stability', `±${(D.stability_std * 100).toFixed(1)}`, `Mean ${pct(D.stability_mean)} over 3 signer splits`],
].map(([k, v, s]) => `<div class="kpi"><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join('');

document.querySelector('#model-table tbody').innerHTML = D.table.map((m) => {
  const cls = m.name === 'Reference CNN' ? 'star' : m.name.startsWith('Raw') ? 'void' : '';
  return `<tr class="${cls}"><td>${m.name.replace(' (invalid)', '')}</td>
    <td><div class="bar"><i style="width:${m.Accuracy * 80}%"></i>${m.Accuracy.toFixed(3)}</div></td>
    <td>${m.F1.toFixed(3)}</td><td>${m['ROC AUC'].toFixed(3)}</td><td>${pct(m.FAR)}</td><td>${pct(m.FRR)}</td></tr>`;
}).join('');

function cutUpdate() {
  const cut = Number($('cut-range').value);
  const c = { tn: 0, fp: 0, fn: 0, tp: 0 };
  D.scores.p.forEach((p, i) => {
    const forged = p > cut, truth = D.scores.y[i] === 1;
    c[truth ? (forged ? 'tp' : 'fn') : (forged ? 'fp' : 'tn')] += 1;
  });
  const total = c.tn + c.fp + c.fn + c.tp, max = Math.max(...Object.values(c));
  $('cut-v').textContent = cut.toFixed(2);
  $('cut-stats').innerHTML = [
    ['Accuracy', (c.tn + c.tp) / total],
    ['Forgeries accepted', c.fn / (c.fn + c.tp)],
    ['Genuine rejected', c.fp / (c.fp + c.tn)],
  ].map(([k, v]) => `<div><b>${pct(v)}</b>${k}</div>`).join('');
  const cell = (n, good) => {
    const a = .12 + .78 * (n / max), rgb = good ? '37,106,191' : '198,67,46';
    return `<div class="cell" style="background:rgba(${rgb},${a});color:${a > .5 ? '#fff' : 'var(--paper)'}"><b>${n}</b></div>`;
  };
  $('matrix').innerHTML = `<div></div><div class="h">Called genuine</div><div class="h">Called forged</div>
    <div class="h">Actually genuine</div>${cell(c.tn, true)}${cell(c.fp, false)}
    <div class="h">Actually forged</div>${cell(c.fn, false)}${cell(c.tp, true)}`;
}
$('cut-range').addEventListener('input', () => { cutUpdate(); award('cut', 10); });
cutUpdate();

/* ---------- final mission ---------- */
const game = { i: 0, you: 0, cnn: 0, answered: false };
function renderRound() {
  const round = D.game[game.i];
  game.answered = false;
  $('game').innerHTML = `
    <div class="game-top"><span>Round ${game.i + 1} of ${D.game.length}</span><span>You ${game.you} · CNN ${game.cnn}</span></div>
    <div class="pair">
      <figure><figcaption>On file: genuine</figcaption><div class="pic"><img src="${round.ref}" alt="Genuine reference signature" /></div></figure>
      <figure><figcaption>Presented: genuine or forged?</figcaption><div class="pic"><img src="${round.q}" alt="Signature to check" /></div></figure>
    </div>
    <div class="choices" id="choices">
      <button class="btn alt" type="button" data-a="0">Genuine</button>
      <button class="btn" type="button" data-a="1">Forged</button>
    </div>
    <div id="verdict"></div>`;
}
function answer(a) {
  if (game.answered) return;
  game.answered = true;
  const round = D.game[game.i];
  const right = a === round.truth, cnnCall = round.p > .5 ? 1 : 0, cnnRight = cnnCall === round.truth;
  if (right) { game.you += 1; addXp(5); }
  if (cnnRight) game.cnn += 1;
  const name = (v) => (v ? 'forged' : 'genuine');
  $('verdict').innerHTML = `<p class="verdict">${right ? 'Correct.' : 'Not this time.'} This one is ${name(round.truth)}. The CNN said ${name(cnnCall)} with p(forged) = ${round.p.toFixed(2)}, so it was ${cnnRight ? 'right' : 'wrong'}.</p>`;
  $('choices').innerHTML = `<button class="btn ghosty" type="button" data-next="1">${game.i + 1 < D.game.length ? 'Next round →' : 'See the score →'}</button>`;
}
function renderEnd() {
  const n = D.game.length;
  const line = game.you > game.cnn ? 'You beat the CNN.' : game.you === game.cnn ? 'A dead heat.' : 'The CNN wins this one.';
  $('game').innerHTML = `<div class="game-end"><h3>${line}</h3>
    <p>You got ${game.you} of ${n}. The CNN got ${game.cnn} of ${n}. On the full test set it scores ${pct(REF.Accuracy)}. These are real test signatures from people it never saw in training.</p>
    <button class="btn" type="button" data-again="1">Play again</button></div>`;
}
$('game').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.a !== undefined) answer(Number(b.dataset.a));
  if (b.dataset.next) {
    game.i += 1;
    if (game.i < D.game.length) renderRound(); else { award('game', 30); renderEnd(); }
  }
  if (b.dataset.again) {
    Object.assign(game, { i: 0, you: 0, cnn: 0 });
    D.game.sort(() => Math.random() - .5);
    renderRound();
  }
});
renderRound();

/* ---------- ambience ---------- */
document.addEventListener('pointermove', (e) => {
  const t = e.target.closest && e.target.closest('.hit');
  if (t) showTip(e, t.dataset.t); else hideTip();
});
const bar = $('progress');
const onScroll = () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  bar.style.transform = `scaleX(${max > 0 ? window.scrollY / max : 0})`;
};
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

const io = new IntersectionObserver((entries) => {
  entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
}, { rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('.lesson-head, .split-2, .panel, .steps, .flow, .kpis, .table-wrap, .game').forEach((el) => {
  el.classList.add('reveal');
  io.observe(el);
});
