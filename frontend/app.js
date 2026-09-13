// RoomSpec AI web client: form handling, async spec requests and BOM/elevation rendering.
(() => {
  'use strict';

  const API = '/api/v1';
  const $ = (id) => document.getElementById(id);
  const state = { file: null, sampleUrl: null, lastResult: null, stageTimer: null };

  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const usd = (n) => '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const cm = (n) => `${Number(n).toLocaleString('en-US', { maximumFractionDigits: 1 })} cm`;

  // ------------------------------------------------------------------ bootstrap
  async function init() {
    loadHealth();
    loadFacets();
    loadSamples();
    bindUpload();
    $('spec-form').addEventListener('submit', onSubmit);
  }

  async function loadHealth() {
    const el = $('health');
    try {
      const h = await (await fetch(`${API}/health`)).json();
      const ok = h.status === 'ok';
      el.innerHTML = `<span class="h-1.5 w-1.5 rounded-full ${ok ? 'bg-pass' : 'bg-warn'}"></span>` +
        `${h.catalog_rows} parts · ${esc(h.embedding_backend)} · llm:${esc(h.llm_provider)}`;
      el.title = `database: ${h.database}\nvector store: ${h.vector_store} (${h.vector_points} points)`;
    } catch {
      el.innerHTML = '<span class="h-1.5 w-1.5 rounded-full bg-fail"></span>API offline';
    }
  }

  async function loadFacets() {
    try {
      const f = await (await fetch(`${API}/catalog/facets`)).json();
      const sel = $('finish_style');
      for (const name of f.cabinet_finishes) sel.add(new Option(name, name));
    } catch { /* select keeps its default option */ }
  }

  async function loadSamples() {
    try {
      const samples = await (await fetch(`${API}/samples`)).json();
      $('samples').innerHTML = samples
        .map((s) => `<button type="button" data-url="${esc(s.url)}" title="${esc(s.name)}"
            class="sample aspect-square overflow-hidden rounded border-2 border-transparent hover:border-ink-faint focus:outline-none focus-visible:border-clay">
            <img src="${esc(s.url)}" alt="${esc(s.name)}" class="h-full w-full object-cover" loading="lazy" /></button>`)
        .join('');
      $('samples').querySelectorAll('.sample').forEach((btn) =>
        btn.addEventListener('click', () => selectSample(btn.dataset.url, btn)),
      );
    } catch { /* samples are optional */ }
  }

  // ------------------------------------------------------------------ image input
  function bindUpload() {
    const dz = $('dropzone');
    const input = $('image-input');
    input.addEventListener('change', () => input.files[0] && setFile(input.files[0]));
    ['dragenter', 'dragover'].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }),
    );
    ['dragleave', 'drop'].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }),
    );
    dz.addEventListener('drop', (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
    $('clear-image').addEventListener('click', (e) => { e.preventDefault(); clearImage(); });
  }

  function setFile(file) {
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) return showError('Please choose a JPEG, PNG or WebP image.');
    if (file.size > 8 * 1024 * 1024) return showError('Image is larger than 8 MB.');
    showError('');
    state.file = file;
    state.sampleUrl = null;
    markSample(null);
    showPreview(URL.createObjectURL(file));
  }

  async function selectSample(url, btn) {
    const blob = await (await fetch(url)).blob();
    state.file = new File([blob], url.split('/').pop(), { type: blob.type || 'image/jpeg' });
    state.sampleUrl = url;
    markSample(btn);
    showPreview(url);
  }

  function markSample(btn) {
    document.querySelectorAll('.sample').forEach((b) => b.classList.toggle('!border-clay', b === btn));
  }

  function showPreview(src) {
    $('preview-img').src = src;
    $('preview-overlay').innerHTML = '';
    $('drop-empty').classList.add('hidden');
    $('drop-preview').classList.remove('hidden');
  }

  function clearImage() {
    state.file = null;
    state.sampleUrl = null;
    $('image-input').value = '';
    markSample(null);
    $('drop-preview').classList.add('hidden');
    $('drop-empty').classList.remove('hidden');
  }

  function drawOverlay(a) {
    if (!a) return;
    const parts = [];
    if (a.floor_line_y_ratio != null) {
      const y = a.floor_line_y_ratio * 100;
      parts.push(`<line x1="0" y1="${y}" x2="100" y2="${y}" stroke="#b4532a" stroke-width="0.8" stroke-dasharray="2 1.2" vector-effect="non-scaling-stroke" style="stroke-width:2px"/>`);
    }
    if (a.corner_x_ratio != null) {
      const x = a.corner_x_ratio * 100;
      parts.push(`<line x1="${x}" y1="0" x2="${x}" y2="100" stroke="#fbfaf7" stroke-dasharray="2 1.2" vector-effect="non-scaling-stroke" style="stroke-width:2px"/>`);
    }
    $('preview-overlay').innerHTML = parts.join('');
  }

  // ------------------------------------------------------------------ submit
  function readConstraints() {
    const num = (id) => { const v = $(id).value.trim(); return v === '' ? null : Number(v); };
    const c = {
      max_width_cm: num('max_width_cm'),
      budget_usd: num('budget_usd'),
      finish_style: $('finish_style').value || null,
      style_prompt: $('style_prompt').value.trim() || null,
      max_depth_cm: num('max_depth_cm'),
      ceiling_height_cm: num('ceiling_height_cm'),
      front_clearance_cm: num('front_clearance_cm'),
      include_wall_cabinets: $('include_wall_cabinets').checked,
      include_countertop: $('include_countertop').checked,
      include_tall_units: $('include_tall_units').checked,
    };
    Object.keys(c).forEach((k) => c[k] === null && delete c[k]);
    return c;
  }

  function validate(c) {
    if (!(c.max_width_cm > 20 && c.max_width_cm <= 1200)) return 'Wall run width must be between 21 and 1200 cm.';
    if (!(c.budget_usd > 0)) return 'Budget must be greater than 0.';
    if (c.ceiling_height_cm != null && c.ceiling_height_cm <= 150) return 'Ceiling height must be over 150 cm.';
    return '';
  }

  async function onSubmit(e) {
    e.preventDefault();
    const constraints = readConstraints();
    const err = validate(constraints);
    if (err) return showError(err);
    showError('');

    const body = new FormData();
    body.append('constraints', JSON.stringify(constraints));
    if (state.file) body.append('image', state.file);

    setLoading(true);
    try {
      const res = await fetch(`${API}/spec`, { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data));
      state.lastResult = data;
      render(data);
    } catch (ex) {
      showError(ex.message || 'Request failed.');
      $('empty-state').classList.toggle('hidden', !!state.lastResult);
      $('output').classList.toggle('hidden', !state.lastResult);
    } finally {
      setLoading(false);
    }
  }

  function formatApiError(data) {
    const d = data && data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x) => `${(x.loc || []).slice(-1)[0] ?? 'input'}: ${x.msg}`).join('; ');
    return 'The server could not produce a specification.';
  }

  function showError(msg) {
    const el = $('form-error');
    el.textContent = msg;
    el.classList.toggle('hidden', !msg);
  }

  function setLoading(on) {
    $('submit').disabled = on;
    $('submit').textContent = on ? 'Generating…' : 'Generate specification';
    const stages = [...document.querySelectorAll('#stages .stage')];
    clearInterval(state.stageTimer);
    if (on) {
      $('empty-state').classList.add('hidden');
      $('output').classList.add('hidden');
      $('loading').classList.remove('hidden');
      let i = state.file ? 0 : 1;
      stages.forEach((s, j) => s.className = `stage flex items-center gap-2 ${j < i ? 'done text-ink-faint line-through' : ''}`);
      const tick = () => {
        stages.forEach((s, j) => {
          s.classList.toggle('active', j === i);
          s.classList.toggle('done', j < i);
        });
        if (i < stages.length - 1) i++;
      };
      tick();
      state.stageTimer = setInterval(tick, 180);
    } else {
      $('loading').classList.add('hidden');
    }
  }

  // ------------------------------------------------------------------ render
  function render(r) {
    drawOverlay(r.image_analysis);
    const out = $('output');
    out.innerHTML = [
      renderHeader(r),
      renderElevation(r),
      renderBOM(r),
      `<div class="grid gap-5 xl:grid-cols-2">${renderNotes(r)}${renderCompliance(r)}</div>`,
      `<div class="grid gap-5 xl:grid-cols-2">${renderAnalysis(r)}${renderTimings(r)}</div>`,
    ].join('');
    out.classList.remove('hidden');
    $('export-csv')?.addEventListener('click', () => exportCSV(r));
    $('copy-json')?.addEventListener('click', (e) => copyJSON(r, e.currentTarget));
    if (window.matchMedia('(max-width: 1023px)').matches) out.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const STATUS = {
    ok: ['Fits', 'bg-pass', 'All constraints satisfied'],
    partial: ['Partial fit', 'bg-warn', 'Some elements were limited by budget, stock or site constraints'],
    infeasible: ['No fit', 'bg-fail', 'No compliant layout exists for these constraints'],
  };

  function renderHeader(r) {
    const [label, bg, sub] = STATUS[r.status];
    const c = r.constraints;
    const stat = (k, v, extra = '') => `<div class="min-w-0"><dt class="text-xs text-ink-soft">${k}</dt><dd class="mt-0.5 font-mono num text-lg ${extra}">${v}</dd></div>`;
    const pct = Math.min(100, (r.total_usd / c.budget_usd) * 100);
    return `
    <section class="rounded-lg border border-rule bg-white p-5">
      <div class="flex flex-wrap items-start gap-3">
        <span class="inline-flex items-center rounded ${bg} px-2 py-0.5 text-xs font-medium text-white">${label}</span>
        <div class="min-w-0 flex-1">
          <p class="text-sm text-ink-soft">${esc(sub)}</p>
          <p class="mt-1 text-[15px] leading-relaxed">${esc(r.summary)}</p>
        </div>
      </div>
      <dl class="mt-5 grid grid-cols-2 md:grid-cols-4 gap-4 border-t border-rule pt-4">
        ${stat('Total', usd(r.total_usd))}
        ${stat('Run width', `${cm(r.run_width_cm)} <span class="text-ink-faint text-sm">/ ${cm(c.max_width_cm)}</span>`)}
        ${stat('Open gap', cm(r.wall_gap_cm), r.wall_gap_cm > 20 ? 'text-warn' : '')}
        ${stat('Finish', esc(r.finish_style_resolved || '—'), 'font-sans text-base truncate')}
      </dl>
      <div class="mt-4">
        <div class="flex justify-between text-xs text-ink-soft"><span>Budget used</span><span class="font-mono num">${usd(r.total_usd)} of ${usd(c.budget_usd)}</span></div>
        <div class="mt-1 h-1.5 rounded bg-rule overflow-hidden"><div class="h-full bg-ink" style="width:${pct}%"></div></div>
      </div>
    </section>`;
  }

  function renderElevation(r) {
    if (!r.layout.length) return '';
    const c = r.constraints;
    const W = c.max_width_cm;
    const topCm = Math.max(c.ceiling_height_cm || 0, ...r.layout.map((p) => p.bottom_cm + p.height_cm)) + 10;
    const padL = 46, padR = 20, padT = 34, padB = 58;
    const scale = Math.min(900 / W, 420 / topCm);
    const vbW = padL + W * scale + padR;
    const vbH = padT + topCm * scale + padB;
    const X = (x) => padL + x * scale;
    const Y = (h) => padT + (topCm - h) * scale;
    const light = (hex) => {
      const n = parseInt(hex.slice(1), 16);
      return ((n >> 16) * 0.299 + ((n >> 8) & 255) * 0.587 + (n & 255) * 0.114) > 140;
    };
    const fs = Math.max(9, Math.min(13, 11 * (vbW / 1000)));
    const out = [];

    // wall + floor
    out.push(`<rect x="${X(0)}" y="${Y(topCm - 10)}" width="${W * scale}" height="${(topCm - 10) * scale}" fill="#f4efe7"/>`);
    if (r.wall_gap_cm > 0) {
      out.push(`<rect x="${X(r.run_width_cm)}" y="${Y(topCm - 10)}" width="${r.wall_gap_cm * scale}" height="${(topCm - 10) * scale}" fill="url(#hatch)"/>`);
    }
    if (c.ceiling_height_cm) {
      out.push(`<line x1="${X(0)}" x2="${X(W)}" y1="${Y(c.ceiling_height_cm)}" y2="${Y(c.ceiling_height_cm)}" stroke="#57534e" stroke-dasharray="6 4"/>`);
      out.push(`<text x="${X(W) - 4}" y="${Y(c.ceiling_height_cm) - 5}" text-anchor="end" font-size="${fs}" fill="#57534e">ceiling ${c.ceiling_height_cm}</text>`);
    }

    const order = { tall: 0, base: 1, filler: 1, countertop: 2, wall: 3 };
    for (const p of [...r.layout].sort((a, b) => order[a.zone] - order[b.zone])) {
      const x = X(p.x_cm), w = p.width_cm * scale, y = Y(p.bottom_cm + p.height_cm), h = p.height_cm * scale;
      const stroke = '#1c1917';
      const detail = light(p.swatch_hex) ? '#57534e' : '#d6d3d1';
      const title = `<title>${esc(p.part_id)} · ${cm(p.width_cm)} × ${cm(p.height_cm)} × ${cm(p.depth_cm)}</title>`;
      out.push(`<g>${title}`);
      if (p.zone === 'base' || p.zone === 'tall' || p.zone === 'filler') {
        out.push(`<rect x="${x}" y="${Y(p.bottom_cm)}" width="${w}" height="${p.bottom_cm * scale}" fill="#44403c"/>`);
      }
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${p.swatch_hex}" stroke="${stroke}" stroke-width="1.2"/>`);
      if (p.zone === 'base' || p.zone === 'wall' || p.zone === 'tall') {
        const isDrawer = /-DB-/.test(p.part_id);
        if (isDrawer) {
          for (let i = 1; i < 3; i++) out.push(`<line x1="${x}" x2="${x + w}" y1="${y + (h * i) / 3}" y2="${y + (h * i) / 3}" stroke="${detail}"/>`);
          for (let i = 0; i < 3; i++) out.push(`<rect x="${x + w / 2 - 6 * scale}" y="${y + (h * i) / 3 + 5 * scale}" width="${12 * scale}" height="${1.5 * scale}" fill="${detail}"/>`);
        } else {
          const doors = p.width_cm >= 60 ? 2 : 1;
          for (let d = 0; d < doors; d++) {
            const dx = x + (w * d) / doors;
            out.push(`<rect x="${dx + 1.5 * scale}" y="${y + 1.5 * scale}" width="${w / doors - 3 * scale}" height="${h - 3 * scale}" fill="none" stroke="${detail}" stroke-width="0.8"/>`);
            const hx = doors === 2 ? (d === 0 ? dx + w / doors - 5 * scale : dx + 3.5 * scale) : x + w - 5 * scale;
            const hy = p.zone === 'wall' ? y + h - 14 * scale : y + 6 * scale;
            out.push(`<rect x="${hx}" y="${hy}" width="${1.5 * scale}" height="${8 * scale}" fill="${detail}"/>`);
          }
        }
      }
      if (w > 34 && p.zone !== 'countertop' && p.zone !== 'filler') {
        const label = p.part_id.replace(/^RS-/, '');
        out.push(`<text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle" font-size="${fs * 0.9}" font-family="IBM Plex Mono, monospace" fill="${light(p.swatch_hex) ? '#1c1917' : '#fafaf9'}" opacity="0.85">${esc(label)}</text>`);
      }
      out.push('</g>');
    }
    // floor
    out.push(`<line x1="${padL - 10}" x2="${X(W) + 10}" y1="${Y(0)}" y2="${Y(0)}" stroke="#1c1917" stroke-width="2"/>`);

    // dimension chain along floor units
    const dimY = Y(0) + 22;
    const floorUnits = r.layout.filter((p) => p.zone === 'base' || p.zone === 'tall' || p.zone === 'filler').sort((a, b) => a.x_cm - b.x_cm);
    const tick = (x, y) => `<line x1="${x - 4}" y1="${y + 4}" x2="${x + 4}" y2="${y - 4}" stroke="#1c1917" stroke-width="1.2"/>`;
    out.push(`<line x1="${X(0)}" x2="${X(r.run_width_cm)}" y1="${dimY}" y2="${dimY}" stroke="#1c1917" stroke-width="0.8"/>`);
    for (const p of floorUnits) {
      out.push(tick(X(p.x_cm), dimY));
      if (p.width_cm * scale > 26) out.push(`<text x="${X(p.x_cm + p.width_cm / 2)}" y="${dimY - 5}" text-anchor="middle" font-size="${fs}" font-family="IBM Plex Mono, monospace" fill="#1c1917">${p.width_cm}</text>`);
    }
    out.push(tick(X(r.run_width_cm), dimY));
    // overall wall width
    const dimY2 = dimY + 24;
    out.push(`<line x1="${X(0)}" x2="${X(W)}" y1="${dimY2}" y2="${dimY2}" stroke="#b4532a" stroke-width="0.8"/>`);
    out.push(tick(X(0), dimY2).replace('#1c1917', '#b4532a'), tick(X(W), dimY2).replace('#1c1917', '#b4532a'));
    out.push(`<text x="${X(W / 2)}" y="${dimY2 - 5}" text-anchor="middle" font-size="${fs}" font-family="IBM Plex Mono, monospace" fill="#b4532a">wall ${W}</text>`);
    // height marks
    const marks = [];
    for (const v of [...new Set(r.layout.map((p) => Math.round((p.bottom_cm + p.height_cm) * 10) / 10))].sort((a, b) => b - a)) {
      if (!marks.length || (marks[marks.length - 1] - v) * scale > fs * 1.1) marks.push(v); // skip labels that would collide
    }
    for (const m of marks) {
      out.push(`<line x1="${padL - 8}" x2="${padL - 2}" y1="${Y(m)}" y2="${Y(m)}" stroke="#57534e"/>`);
      out.push(`<text x="${padL - 11}" y="${Y(m) + 3.5}" text-anchor="end" font-size="${fs * 0.85}" font-family="IBM Plex Mono, monospace" fill="#57534e">${m}</text>`);
    }

    return `
    <section class="rounded-lg border border-rule bg-white">
      <div class="flex items-center justify-between px-5 pt-4">
        <h2 class="text-sm font-semibold">Front elevation</h2>
        <span class="text-xs text-ink-soft">to scale · cm${r.wall_gap_cm > 0 ? ' · hatched = open wall' : ''}</span>
      </div>
      <div class="overflow-x-auto px-3 pb-3">
        <svg viewBox="0 0 ${vbW} ${vbH}" class="w-full min-w-[640px] h-auto max-h-[520px]" role="img" aria-label="Front elevation of the proposed cabinet run">
          <defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="#d6cfc4" stroke-width="2"/></pattern></defs>
          ${out.join('')}
        </svg>
      </div>
    </section>`;
  }

  function renderBOM(r) {
    if (!r.bom.length) {
      return `<section class="rounded-lg border border-rule bg-white p-5"><h2 class="text-sm font-semibold">Bill of materials</h2><p class="mt-2 text-sm text-ink-soft">No parts selected.</p></section>`;
    }
    const rows = r.bom.map((l) => `
      <tr class="border-t border-rule align-top">
        <td class="py-2.5 pl-5 pr-2 font-mono text-xs text-ink-faint num">${l.line}</td>
        <td class="py-2 pr-3"><img src="${API}/catalog/${encodeURIComponent(l.part_id)}/thumbnail.svg" alt="" class="h-10 w-10 rounded bg-paper"/></td>
        <td class="py-2.5 pr-3 min-w-[220px]">
          <div class="font-mono text-[13px]">${esc(l.part_id)}</div>
          <div class="text-ink-soft text-xs mt-0.5">${esc(l.part_name)}</div>
          ${l.note ? `<div class="text-xs text-clay mt-0.5">${esc(l.note)}</div>` : ''}
        </td>
        <td class="py-2.5 pr-3 text-xs text-ink-soft whitespace-nowrap">${esc(l.category)}</td>
        <td class="py-2.5 pr-3 font-mono text-xs num whitespace-nowrap">${l.width_cm} × ${l.height_cm} × ${l.depth_cm}</td>
        <td class="py-2.5 pr-3 font-mono num text-right">${l.quantity}</td>
        <td class="py-2.5 pr-3 font-mono num text-right whitespace-nowrap">${usd(l.unit_price_usd)}</td>
        <td class="py-2.5 pr-5 font-mono num text-right whitespace-nowrap">${usd(l.line_total_usd)}</td>
      </tr>`).join('');
    return `
    <section class="rounded-lg border border-rule bg-white">
      <div class="flex flex-wrap items-center gap-2 px-5 pt-4 pb-3">
        <h2 class="text-sm font-semibold">Bill of materials</h2>
        <span class="text-xs text-ink-soft">${r.bom.reduce((s, l) => s + l.quantity, 0)} units · ${r.candidates_compliant} compliant of ${r.candidates_considered} retrieved</span>
        <div class="ml-auto flex gap-2">
          <button id="copy-json" type="button" class="rounded-md border border-rule px-2.5 py-1 text-xs hover:border-ink">Copy JSON</button>
          <button id="export-csv" type="button" class="rounded-md border border-ink px-2.5 py-1 text-xs font-medium hover:bg-ink hover:text-paper">Export CSV</button>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-xs text-ink-soft">
            <tr>
              <th class="py-2 pl-5 pr-2 font-medium">#</th><th class="py-2 pr-3"><span class="sr-only">Preview</span></th>
              <th class="py-2 pr-3 font-medium">Part</th><th class="py-2 pr-3 font-medium">Category</th>
              <th class="py-2 pr-3 font-medium whitespace-nowrap">W × H × D (cm)</th><th class="py-2 pr-3 font-medium text-right">Qty</th>
              <th class="py-2 pr-3 font-medium text-right">Unit</th><th class="py-2 pr-5 font-medium text-right">Total</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
          <tfoot>
            <tr class="border-t-2 border-ink">
              <td colspan="7" class="py-3 pl-5 text-right text-sm font-medium">Total</td>
              <td class="py-3 pr-5 text-right font-mono num font-medium">${usd(r.total_usd)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>`;
  }

  function renderCompliance(r) {
    const items = r.compliance.map((ch) => {
      const [bg, icon, label] = ch.passed ? ['bg-pass', '✓', 'passed'] : ch.kind === 'completeness' ? ['bg-warn', '!', 'incomplete'] : ['bg-fail', '✕', 'failed'];
      return `
      <li class="flex gap-3 py-2 border-t border-rule first:border-t-0">
        <span class="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full ${bg} text-white text-[11px]" aria-label="${label}">${icon}</span>
        <div class="min-w-0"><p class="text-sm font-medium">${esc(ch.name.replace(/_/g, ' '))}</p><p class="text-xs text-ink-soft font-mono break-words">${esc(ch.detail)}</p></div>
      </li>`;
    }).join('');
    const h = r.hallucination_check;
    const hallu = `
      <div class="mt-3 rounded-md ${h.passed ? 'bg-paper' : 'bg-clay-tint'} p-3 text-xs">
        <p class="font-medium">${h.passed ? 'Grounding check passed' : 'Grounding check flagged output'}</p>
        <p class="mt-1 text-ink-soft">${h.llm_used ? `Notes written by ${esc(h.provider)} and verified against the catalog rows.` : esc(h.fallback_reason || 'Deterministic template used.')}</p>
        ${h.unknown_part_ids.length ? `<p class="mt-1 font-mono text-fail">unknown parts: ${esc(h.unknown_part_ids.join(', '))}</p>` : ''}
        ${h.unverified_numbers.length ? `<p class="mt-1 font-mono text-fail">unverified values: ${esc(h.unverified_numbers.join(', '))}</p>` : ''}
      </div>`;
    return `
    <section class="rounded-lg border border-rule bg-white p-5">
      <h2 class="text-sm font-semibold">Compliance checks</h2>
      <ul class="mt-2">${items || '<li class="text-sm text-ink-soft">No parts to check.</li>'}</ul>
      ${hallu}
    </section>`;
  }

  function renderNotes(r) {
    const notes = r.installation_notes.map((n) => `<li class="pl-1">${esc(n)}</li>`).join('');
    return `
    <section class="rounded-lg border border-rule bg-white p-5">
      <h2 class="text-sm font-semibold">Installation notes</h2>
      <ol class="mt-3 list-decimal list-outside ml-5 space-y-2 text-sm leading-relaxed marker:font-mono marker:text-ink-faint">${notes || '<li>No installation steps.</li>'}</ol>
    </section>`;
  }

  function renderAnalysis(r) {
    const a = r.image_analysis;
    if (!a) {
      return `<section class="rounded-lg border border-rule bg-white p-5"><h2 class="text-sm font-semibold">Photo analysis</h2>
        <p class="mt-2 text-sm text-ink-soft">No photo supplied. Style came from your brief:</p>
        <p class="mt-2 font-mono text-xs bg-paper rounded p-2">${esc(r.query_text)}</p></section>`;
    }
    const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
    const row = (k, v) => `<div class="flex justify-between gap-3 py-1.5 border-t border-rule first:border-t-0"><dt class="text-ink-soft">${k}</dt><dd class="font-mono num text-right">${v}</dd></div>`;
    return `
    <section class="rounded-lg border border-rule bg-white p-5">
      <h2 class="text-sm font-semibold">Photo analysis</h2>
      <div class="mt-3 flex gap-1.5">${a.dominant_colors.map((c) => `<span class="h-8 flex-1 rounded border border-rule" style="background:${esc(c)}" title="${esc(c)}"></span>`).join('')}</div>
      <p class="mt-3 text-xs text-ink-soft">Closest finishes</p>
      <div class="mt-1 flex flex-wrap gap-1.5">${a.suggested_finishes.map((f, i) => `<span class="rounded-full border ${i === 0 ? 'border-clay text-clay' : 'border-rule text-ink-soft'} px-2.5 py-0.5 text-xs">${esc(f)}</span>`).join('')}</div>
      <dl class="mt-4 text-xs">
        ${row('Floor line', pct(a.floor_line_y_ratio) + ' from top')}
        ${row('Wall corner', pct(a.corner_x_ratio) + ' from left')}
        ${row('Camera tilt corrected', `${a.perspective_tilt_deg}°`)}
        ${row('Contrast (σ)', `${a.contrast_before} → ${a.contrast_after}`)}
        ${row('Resolution', `${a.width_px} × ${a.height_px}`)}
      </dl>
      ${a.warnings.map((w) => `<p class="mt-2 text-xs text-warn">⚠ ${esc(w)}</p>`).join('')}
    </section>`;
  }

  function renderTimings(r) {
    const t = { ...r.timings_ms };
    const total = t.total || 1;
    delete t.total;
    const labels = { cv_preprocess: 'OpenCV preprocess', embedding: 'Embedding', vector_search: 'Qdrant search', sql_filter: 'SQL filter', layout_solver: 'Layout solver', bom_generation: 'BOM + notes' };
    const bars = Object.entries(t).map(([k, v]) => `
      <div class="grid grid-cols-[120px_1fr_64px] items-center gap-3 py-1">
        <span class="text-xs text-ink-soft">${esc(labels[k] || k)}</span>
        <div class="h-2 rounded bg-paper overflow-hidden"><div class="h-full bg-ink" style="width:${Math.max(1, (v / total) * 100)}%"></div></div>
        <span class="font-mono num text-xs text-right">${v.toFixed(1)} ms</span>
      </div>`).join('');
    return `
    <section class="rounded-lg border border-rule bg-white p-5">
      <div class="flex items-baseline justify-between"><h2 class="text-sm font-semibold">Pipeline timing</h2><span class="font-mono num text-xs">${total.toFixed(1)} ms server</span></div>
      <div class="mt-3">${bars}</div>
      <p class="mt-3 text-xs text-ink-faint font-mono">request ${esc(r.request_id)}</p>
    </section>`;
  }

  // ------------------------------------------------------------------ export
  function exportCSV(r) {
    const head = ['line', 'part_id', 'part_name', 'category', 'finish_style', 'quantity', 'unit_price_usd', 'line_total_usd', 'width_cm', 'height_cm', 'depth_cm', 'note'];
    const q = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const lines = [head.join(','), ...r.bom.map((l) => head.map((k) => q(l[k])).join(','))];
    lines.push(['', '', '', '', '', '', 'TOTAL', r.total_usd].map(q).join(','));
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `roomspec-bom-${r.request_id}.csv` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function copyJSON(r, btn) {
    try {
      await navigator.clipboard.writeText(JSON.stringify(r, null, 2));
      btn.textContent = 'Copied';
    } catch {
      btn.textContent = 'Copy failed';
    }
    setTimeout(() => (btn.textContent = 'Copy JSON'), 1500);
  }

  init();
})();
