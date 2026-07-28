/*
 * mset_visualizer.c
 * ==================
 *
 * C (C99) port of mset_visualizer.py's MSETAnomalyVisualizer.
 *
 * A headless container has no display to show a live GUI window, so the true
 * equivalent of an interactive matplotlib figure here is a single
 * self-contained HTML file: it embeds the computed data as JSON and a small
 * hand-written vanilla-JS renderer (Canvas 2D, no external libraries, no
 * network access required) that reproduces the same interactions as the
 * Python version:
 *
 *   top chart  - 3D scatter of input states, MSET estimates, residual lines
 *                and SPRT-flagged anomalies, all in one graph. Rotate by
 *                dragging, zoom with the scroll wheel, click a point to
 *                inspect it.
 *   bottom chart - SPRT module: LLR+ / LLR- trajectories against the Wald
 *                decision boundaries, fault alarms, and shaded anomaly
 *                bands, linked to the 3D chart through a shared time cursor.
 *   controls   - a time slider, layer checkboxes (training data, MSET
 *                estimates, residual lines, anomalies, trajectory), radio
 *                buttons to switch the SPRT channel between parameters and
 *                the ||R|| norm, and a monospace info panel.
 *
 * Usage:
 *     #include "mset_rul.h"
 *     #include "mset_visualizer.h"
 *
 *     MSET *model = mset_create(kernel_inverse_distance, 0.30, 1e-8, 1);
 *     mset_fit(model, training, l, n, 14);
 *
 *     MSETVisualizerOptions opt;
 *     mset_visualizer_default_options(&opt);
 *     mset_visualize_write_html("out.html", model, observations, times, k,
 *                               param_names, &opt);
 *
 * Build (as part of a program, linked against mset_rul.c):
 *     gcc -std=c99 -O2 -Wall -Wextra -c mset_rul.c -DMSET_RUL_NO_MAIN -o mset_rul.o
 *     gcc -std=c99 -O2 -Wall -Wextra -c mset_visualizer.c -o mset_visualizer.o
 *     gcc mset_rul.o mset_visualizer.o your_main.o -lm -o your_program
 *
 * Then open the generated HTML file in any browser.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mset_rul.h"
#include "mset_visualizer.h"

/* ======================================================================== */
/* 1. Full-trajectory two-sided SPRT (for plotting - mirrors Python's        */
/*    SPRTTrace, distinct from mset_rul.c's reset-on-decision SPRT)         */
/* ======================================================================== */

typedef struct {
    double mu0, sigma, shift, upper, lower;
    double *llr_pos, *llr_neg;   /* length k, post-update value each sample */
    int *fault_idx;              /* length k (upper bound), filled to count */
    int fault_count;
} VizSprt;

static void vizsprt_run(VizSprt *t, double mean, double sigma, double alpha,
                        double beta, double disturbance,
                        const double *series, int k)
{
    t->mu0 = mean;
    t->sigma = (sigma > 1e-12) ? sigma : 1e-12;
    t->shift = disturbance * t->sigma;
    t->upper = log((1.0 - beta) / alpha);
    t->lower = log(beta / (1.0 - alpha));
    t->llr_pos = malloc(sizeof(double) * (size_t)k);
    t->llr_neg = malloc(sizeof(double) * (size_t)k);
    t->fault_idx = malloc(sizeof(int) * (size_t)k);
    t->fault_count = 0;

    double pos = 0.0, neg = 0.0;
    double gain = t->shift / (t->sigma * t->sigma);
    for (int i = 0; i < k; ++i) {
        double dev = series[i] - t->mu0;
        pos += gain * (dev - t->shift / 2.0);
        neg += gain * (-dev - t->shift / 2.0);

        int fault = (pos >= t->upper || neg >= t->upper);
        int healthy = (!fault) && (pos <= t->lower && neg <= t->lower);

        t->llr_pos[i] = pos;
        t->llr_neg[i] = neg;

        if (fault) {
            t->fault_idx[t->fault_count++] = i;
            pos = neg = 0.0;
        } else if (healthy) {
            pos = neg = 0.0;
        } else {
            if (pos < t->lower) pos = t->lower;
            if (neg < t->lower) neg = t->lower;
        }
    }
}

static void vizsprt_free(VizSprt *t)
{
    free(t->llr_pos);
    free(t->llr_neg);
    free(t->fault_idx);
}

/* ======================================================================== */
/* 2. Minimal JSON writer (doubles, ints-as-bool, strings, arrays)          */
/* ======================================================================== */

static void jw_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        switch (*p) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f);  break;
            case '\r': fputs("\\r", f);  break;
            case '\t': fputs("\\t", f);  break;
            default:
                if (*p < 0x20) fprintf(f, "\\u%04x", *p);
                else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void jw_double_array(FILE *f, const char *key, const double *arr, int n)
{
    fprintf(f, "  \"%s\": [", key);
    for (int i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        double v = arr[i];
        if (isnan(v) || isinf(v)) fputs("0", f);   /* JSON has no NaN/Inf */
        else fprintf(f, "%.6g", v);
    }
    fputs("]", f);
}

static void jw_bool_array(FILE *f, const char *key, const int *arr, int n)
{
    fprintf(f, "  \"%s\": [", key);
    for (int i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        fputs(arr[i] ? "true" : "false", f);
    }
    fputs("]", f);
}

static void jw_int_array(FILE *f, const char *key, const int *arr, int n)
{
    fprintf(f, "\"%s\": [", key);
    for (int i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        fprintf(f, "%d", arr[i]);
    }
    fputs("]", f);
}

static void jw_string_array(FILE *f, const char *key,
                            const char * const *arr, int n)
{
    fprintf(f, "  \"%s\": [", key);
    for (int i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        jw_string(f, arr[i]);
    }
    fputs("]", f);
}

/* ======================================================================== */
/* 3. Static HTML/CSS/JS template pieces                                    */
/* ======================================================================== */

static const char *HTML_TOP =
"<!DOCTYPE html>\n"
"<html lang=\"en\">\n"
"<head>\n"
"<meta charset=\"utf-8\">\n"
"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
"<title>MSET input &amp; anomalies (3D) + SPRT fault detection</title>\n"
"<style>\n"
"  body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;\n"
"       margin:18px;color:#222;background:#fff;}\n"
"  h1{font-size:16px;font-weight:600;margin:0 0 4px 0;}\n"
"  .legend{font-size:12px;color:#555;margin-bottom:10px;}\n"
"  .hint{font-size:11px;color:#888;margin-top:4px;}\n"
"  .container{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;}\n"
"  .left{flex:0 0 auto;}\n"
"  .right{flex:0 0 260px;display:flex;flex-direction:column;gap:12px;}\n"
"  canvas{border:1px solid #ddd;border-radius:6px;background:#fff;\n"
"         display:block;cursor:grab;}\n"
"  canvas:active{cursor:grabbing;}\n"
"  #chartSprt{cursor:pointer;margin-top:12px;}\n"
"  .box{border:1px solid #ccc;border-radius:6px;padding:10px 12px;font-size:13px;}\n"
"  .box-title{font-weight:600;font-size:12px;color:#444;margin-bottom:6px;\n"
"             text-transform:uppercase;letter-spacing:.02em;}\n"
"  .box label{display:block;padding:2px 0;cursor:pointer;font-size:13px;}\n"
"  #infoPanel{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;\n"
"             white-space:pre-wrap;background:#f7f7f7;border:1px solid #ccc;\n"
"             border-radius:6px;padding:10px;min-height:230px;line-height:1.45;}\n"
"  #sprtTitle{font-size:13px;font-weight:600;margin:14px 0 4px 2px;}\n"
"  input[type=range]{width:100%;}\n"
"  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;\n"
"          margin-right:5px;vertical-align:middle;}\n"
"</style>\n"
"</head>\n"
"<body>\n"
"<h1>Input states, MSET estimates and detected anomalies</h1>\n"
"<div class=\"legend\">\n"
"  <span class=\"swatch\" style=\"background:#9ecae1\"></span>healthy training data &nbsp;\n"
"  <span class=\"swatch\" style=\"background:#31688e\"></span>input (colour = ||residual||) &nbsp;\n"
"  <span class=\"swatch\" style=\"background:#7f7f7f\"></span>MSET estimate (triangle) &nbsp;\n"
"  <span class=\"swatch\" style=\"background:#d62728\"></span>anomaly (SPRT fault, X) &nbsp;\n"
"  <span class=\"swatch\" style=\"background:#ff7f0e\"></span>current time (ring)\n"
"</div>\n"
"<div class=\"container\">\n"
"  <div class=\"left\">\n"
"    <canvas id=\"chart3d\" width=\"900\" height=\"540\"></canvas>\n"
"    <div class=\"hint\">rotate: drag &middot; zoom: scroll &middot; click a point to inspect</div>\n"
"    <div id=\"sprtTitle\"></div>\n"
"    <canvas id=\"chartSprt\" width=\"900\" height=\"220\"></canvas>\n"
"    <div style=\"margin-top:10px;\">\n"
"      <label for=\"timeSlider\" style=\"font-size:12px;color:#444;\">time index</label>\n"
"      <input type=\"range\" id=\"timeSlider\" min=\"0\" max=\"1\" step=\"1\" value=\"0\">\n"
"    </div>\n"
"  </div>\n"
"  <div class=\"right\">\n"
"    <div class=\"box\">\n"
"      <div class=\"box-title\">layers</div>\n"
"      <label><input type=\"checkbox\" id=\"chk_training\" checked> training data</label>\n"
"      <label><input type=\"checkbox\" id=\"chk_estimates\" checked> MSET estimates</label>\n"
"      <label><input type=\"checkbox\" id=\"chk_residuals\" checked> residual lines</label>\n"
"      <label><input type=\"checkbox\" id=\"chk_anomalies\" checked> anomalies</label>\n"
"      <label><input type=\"checkbox\" id=\"chk_trajectory\" checked> trajectory</label>\n"
"    </div>\n"
"    <div class=\"box\">\n"
"      <div class=\"box-title\">SPRT channel</div>\n"
"      <div id=\"channelRadios\"></div>\n"
"    </div>\n"
"    <div class=\"box\">\n"
"      <div class=\"box-title\">sample info</div>\n"
"      <div id=\"infoPanel\"></div>\n"
"    </div>\n"
"  </div>\n"
"</div>\n";

static const char *HTML_BEFORE_SCRIPT =
"\n";

/* The renderer: vanilla Canvas 2D, no external dependencies. References the
 * `DATA` object written immediately before this template in the same
 * <script> tag. Adjacent C string literals concatenate automatically. */
static const char *JS_TEMPLATE =
"(function(){\n"
"  const state = {\n"
"    rotX: -0.32, rotY: 0.55, zoom: 1.0,\n"
"    cursor: DATA.k - 1,\n"
"    channel: DATA.channels.length - 1,\n"
"    layers: {training:true, estimates:true, residuals:true, anomalies:true, trajectory:true},\n"
"    dragging:false, moved:false, lastX:0, lastY:0, downX:0, downY:0\n"
"  };\n"
"\n"
"  const c3d = document.getElementById('chart3d');\n"
"  const ctx3d = c3d.getContext('2d');\n"
"  const cSprt = document.getElementById('chartSprt');\n"
"  const ctxSprt = cSprt.getContext('2d');\n"
"  const slider = document.getElementById('timeSlider');\n"
"  const info = document.getElementById('infoPanel');\n"
"  const sprtTitle = document.getElementById('sprtTitle');\n"
"\n"
"  slider.min = 0; slider.max = DATA.k - 1; slider.value = state.cursor;\n"
"\n"
"  /* ---- build the SPRT channel radio buttons (count is data-dependent) --- */\n"
"  const radioHost = document.getElementById('channelRadios');\n"
"  DATA.channels.forEach((name, i) => {\n"
"    const label = document.createElement('label');\n"
"    const input = document.createElement('input');\n"
"    input.type = 'radio'; input.name = 'sprtChannel'; input.id = 'chan_' + i;\n"
"    if (i === state.channel) input.checked = true;\n"
"    input.addEventListener('change', () => { state.channel = i; redrawSPRT(); });\n"
"    label.appendChild(input);\n"
"    label.appendChild(document.createTextNode(' ' + name));\n"
"    radioHost.appendChild(label);\n"
"  });\n"
"\n"
"  /* ---- bounds + normalisation into a [-1,1]^3 cube ---- */\n"
"  const allX = DATA.objX.concat(DATA.trainX, DATA.estX);\n"
"  const allY = DATA.objY.concat(DATA.trainY, DATA.estY);\n"
"  const allZ = DATA.objZ.concat(DATA.trainZ, DATA.estZ);\n"
"  const lo = [Math.min(...allX), Math.min(...allY), Math.min(...allZ)];\n"
"  const hi = [Math.max(...allX), Math.max(...allY), Math.max(...allZ)];\n"
"  function norm3(x, y, z) {\n"
"    return [\n"
"      2*(x-lo[0])/((hi[0]-lo[0])||1) - 1,\n"
"      2*(y-lo[1])/((hi[1]-lo[1])||1) - 1,\n"
"      2*(z-lo[2])/((hi[2]-lo[2])||1) - 1\n"
"    ];\n"
"  }\n"
"\n"
"  function project(p) {\n"
"    const cy = Math.cos(state.rotY), sy = Math.sin(state.rotY);\n"
"    const cx = Math.cos(state.rotX), sx = Math.sin(state.rotX);\n"
"    let x = p[0]*cy - p[2]*sy;\n"
"    let z = p[0]*sy + p[2]*cy;\n"
"    let y = p[1]*cx - z*sx;\n"
"    z    = p[1]*sx + z*cx;\n"
"    const scale = 150 * state.zoom;\n"
"    const persp = 1 / (1 + (z + 2) * 0.12);\n"
"    return {\n"
"      x: c3d.width/2 + x*scale*persp,\n"
"      y: c3d.height/2 + 30 - y*scale*persp,\n"
"      depth: z\n"
"    };\n"
"  }\n"
"\n"
"  const VIRIDIS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];\n"
"  function colorRamp(v, vmin, vmax) {\n"
"    let t = (v - vmin) / ((vmax - vmin) || 1);\n"
"    t = Math.max(0, Math.min(1, t));\n"
"    const seg = t * (VIRIDIS.length - 1);\n"
"    const i = Math.min(VIRIDIS.length - 2, Math.floor(seg));\n"
"    const f = seg - i, a = VIRIDIS[i], b = VIRIDIS[i+1];\n"
"    const r = Math.round(a[0]+(b[0]-a[0])*f);\n"
"    const g = Math.round(a[1]+(b[1]-a[1])*f);\n"
"    const bl= Math.round(a[2]+(b[2]-a[2])*f);\n"
"    return `rgb(${r},${g},${bl})`;\n"
"  }\n"
"\n"
"  function drawAxes() {\n"
"    const O = project(norm3(lo[0], lo[1], lo[2]));\n"
"    const X = project(norm3(hi[0], lo[1], lo[2]));\n"
"    const Y = project(norm3(lo[0], hi[1], lo[2]));\n"
"    const Z = project(norm3(lo[0], lo[1], hi[2]));\n"
"    ctx3d.strokeStyle = '#aaa'; ctx3d.lineWidth = 1;\n"
"    ctx3d.font = '11px sans-serif'; ctx3d.fillStyle = '#444';\n"
"    [[O,X,DATA.paramNames[0]],[O,Y,DATA.paramNames[1]],[O,Z,DATA.paramNames[2]]].forEach(([a,b,label]) => {\n"
"      ctx3d.beginPath(); ctx3d.moveTo(a.x,a.y); ctx3d.lineTo(b.x,b.y); ctx3d.stroke();\n"
"      ctx3d.fillText(label, b.x+4, b.y+4);\n"
"    });\n"
"  }\n"
"\n"
"  function redraw3D() {\n"
"    ctx3d.clearRect(0, 0, c3d.width, c3d.height);\n"
"    drawAxes();\n"
"\n"
"    const items = [];\n"
"\n"
"    if (state.layers.training) {\n"
"      for (let i = 0; i < DATA.trainX.length; ++i) {\n"
"        const P = project(norm3(DATA.trainX[i], DATA.trainY[i], DATA.trainZ[i]));\n"
"        items.push({type:'dot', p:P, r:2.4, color:'rgba(158,202,225,0.55)', depth:P.depth});\n"
"      }\n"
"    }\n"
"\n"
"    if (state.layers.trajectory) {\n"
"      ctx3d.strokeStyle = 'rgba(120,120,120,0.55)'; ctx3d.lineWidth = 1;\n"
"      ctx3d.beginPath();\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));\n"
"        if (i === 0) ctx3d.moveTo(P.x, P.y); else ctx3d.lineTo(P.x, P.y);\n"
"      }\n"
"      ctx3d.stroke();\n"
"    }\n"
"\n"
"    if (state.layers.residuals) {\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        const A = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));\n"
"        const B = project(norm3(DATA.estX[i], DATA.estY[i], DATA.estZ[i]));\n"
"        items.push({type:'line', a:A, b:B, color:'rgba(127,127,127,0.55)', depth:(A.depth+B.depth)/2});\n"
"      }\n"
"    }\n"
"\n"
"    if (state.layers.estimates) {\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        const P = project(norm3(DATA.estX[i], DATA.estY[i], DATA.estZ[i]));\n"
"        items.push({type:'tri', p:P, color:'#7f7f7f', depth:P.depth});\n"
"      }\n"
"    }\n"
"\n"
"    const vmin = Math.min(...DATA.residualNorm), vmax = Math.max(...DATA.residualNorm);\n"
"    for (let i = 0; i < DATA.k; ++i) {\n"
"      if (DATA.anomalyMask[i]) continue;\n"
"      const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));\n"
"      items.push({type:'dot', p:P, r:5, color:colorRamp(DATA.residualNorm[i], vmin, vmax), depth:P.depth});\n"
"    }\n"
"\n"
"    if (state.layers.anomalies) {\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        if (!DATA.anomalyMask[i]) continue;\n"
"        const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));\n"
"        items.push({type:'x', p:P, color:'#d62728', depth:P.depth});\n"
"      }\n"
"    }\n"
"\n"
"    items.sort((a, b) => a.depth - b.depth);\n"
"    items.forEach(it => {\n"
"      if (it.type === 'dot') {\n"
"        ctx3d.beginPath(); ctx3d.arc(it.p.x, it.p.y, it.r, 0, 2*Math.PI);\n"
"        ctx3d.fillStyle = it.color; ctx3d.fill();\n"
"      } else if (it.type === 'tri') {\n"
"        ctx3d.beginPath();\n"
"        ctx3d.moveTo(it.p.x, it.p.y-5); ctx3d.lineTo(it.p.x-5, it.p.y+4); ctx3d.lineTo(it.p.x+5, it.p.y+4);\n"
"        ctx3d.closePath(); ctx3d.fillStyle = it.color; ctx3d.fill();\n"
"      } else if (it.type === 'line') {\n"
"        ctx3d.strokeStyle = it.color; ctx3d.lineWidth = 1;\n"
"        ctx3d.beginPath(); ctx3d.moveTo(it.a.x, it.a.y); ctx3d.lineTo(it.b.x, it.b.y); ctx3d.stroke();\n"
"      } else if (it.type === 'x') {\n"
"        ctx3d.strokeStyle = it.color; ctx3d.lineWidth = 2.4;\n"
"        ctx3d.beginPath();\n"
"        ctx3d.moveTo(it.p.x-6, it.p.y-6); ctx3d.lineTo(it.p.x+6, it.p.y+6);\n"
"        ctx3d.moveTo(it.p.x+6, it.p.y-6); ctx3d.lineTo(it.p.x-6, it.p.y+6);\n"
"        ctx3d.stroke();\n"
"      }\n"
"    });\n"
"\n"
"    const ci = state.cursor;\n"
"    const CP = project(norm3(DATA.objX[ci], DATA.objY[ci], DATA.objZ[ci]));\n"
"    ctx3d.beginPath(); ctx3d.arc(CP.x, CP.y, 11, 0, 2*Math.PI);\n"
"    ctx3d.strokeStyle = '#ff7f0e'; ctx3d.lineWidth = 2.5; ctx3d.stroke();\n"
"\n"
"    updateInfo();\n"
"  }\n"
"\n"
"  function pickIndex(mx, my) {\n"
"    let best = -1, bestD = 1e18;\n"
"    for (let i = 0; i < DATA.k; ++i) {\n"
"      const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));\n"
"      const d = Math.hypot(P.x - mx, P.y - my);\n"
"      if (d < bestD) { bestD = d; best = i; }\n"
"    }\n"
"    return bestD < 16 ? best : -1;\n"
"  }\n"
"\n"
"  function setCursor(i) {\n"
"    state.cursor = Math.max(0, Math.min(DATA.k - 1, i));\n"
"    slider.value = state.cursor;\n"
"    redraw3D();\n"
"    redrawSPRT();\n"
"  }\n"
"\n"
"  c3d.addEventListener('mousedown', e => {\n"
"    const rect = c3d.getBoundingClientRect();\n"
"    state.dragging = true; state.moved = false;\n"
"    state.downX = e.clientX - rect.left; state.downY = e.clientY - rect.top;\n"
"    state.lastX = state.downX; state.lastY = state.downY;\n"
"  });\n"
"  window.addEventListener('mousemove', e => {\n"
"    if (!state.dragging) return;\n"
"    const rect = c3d.getBoundingClientRect();\n"
"    const mx = e.clientX - rect.left, my = e.clientY - rect.top;\n"
"    if (Math.hypot(mx - state.downX, my - state.downY) > 3) state.moved = true;\n"
"    const dx = mx - state.lastX, dy = my - state.lastY;\n"
"    state.rotY += dx * 0.008; state.rotX += dy * 0.008;\n"
"    state.lastX = mx; state.lastY = my;\n"
"    redraw3D();\n"
"  });\n"
"  window.addEventListener('mouseup', e => {\n"
"    if (state.dragging && !state.moved) {\n"
"      const rect = c3d.getBoundingClientRect();\n"
"      const idx = pickIndex(e.clientX - rect.left, e.clientY - rect.top);\n"
"      if (idx >= 0) setCursor(idx);\n"
"    }\n"
"    state.dragging = false;\n"
"  });\n"
"  c3d.addEventListener('wheel', e => {\n"
"    e.preventDefault();\n"
"    state.zoom *= (e.deltaY < 0 ? 1.08 : 0.93);\n"
"    state.zoom = Math.max(0.3, Math.min(3.5, state.zoom));\n"
"    redraw3D();\n"
"  }, {passive:false});\n"
"\n"
"  const PAD_L = 46, PAD_R = 12, PAD_T = 24, PAD_B = 28;\n"
"  function redrawSPRT() {\n"
"    ctxSprt.clearRect(0, 0, cSprt.width, cSprt.height);\n"
"    const ch = DATA.sprt[state.channel];\n"
"    const W = cSprt.width, H = cSprt.height;\n"
"    const tmin = DATA.times[0], tmax = DATA.times[DATA.k - 1];\n"
"    const allVals = ch.llrPos.concat(ch.llrNeg, [ch.upper, ch.lower, 0]);\n"
"    const vmin = Math.min(...allVals), vmax = Math.max(...allVals);\n"
"    const X = t => PAD_L + (t - tmin) / ((tmax - tmin) || 1) * (W - PAD_L - PAD_R);\n"
"    const Y = v => H - PAD_B - (v - vmin) / ((vmax - vmin) || 1) * (H - PAD_T - PAD_B);\n"
"\n"
"    if (state.layers.anomalies) {\n"
"      ctxSprt.fillStyle = 'rgba(214,39,40,0.09)';\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        if (!DATA.anomalyMask[i]) continue;\n"
"        const x = X(DATA.times[i]);\n"
"        ctxSprt.fillRect(x - 3, PAD_T, 6, H - PAD_T - PAD_B);\n"
"      }\n"
"    }\n"
"\n"
"    ctxSprt.strokeStyle = '#333'; ctxSprt.lineWidth = 1;\n"
"    ctxSprt.strokeRect(PAD_L, PAD_T, W - PAD_L - PAD_R, H - PAD_T - PAD_B);\n"
"    ctxSprt.fillStyle = '#333'; ctxSprt.font = '10px monospace';\n"
"    ctxSprt.fillText(vmax.toFixed(1), 4, Y(vmax) + 3);\n"
"    ctxSprt.fillText(vmin.toFixed(1), 4, Y(vmin) + 3);\n"
"    ctxSprt.fillText(String(Math.round(tmin)), PAD_L - 6, H - 8);\n"
"    ctxSprt.fillText(String(Math.round(tmax)), W - PAD_R - 22, H - 8);\n"
"    ctxSprt.fillText(DATA.timeLabel, (W + PAD_L - PAD_R) / 2 - 18, H - 6);\n"
"\n"
"    ctxSprt.setLineDash([5, 4]);\n"
"    ctxSprt.strokeStyle = '#d62728';\n"
"    ctxSprt.beginPath(); ctxSprt.moveTo(PAD_L, Y(ch.upper)); ctxSprt.lineTo(W - PAD_R, Y(ch.upper)); ctxSprt.stroke();\n"
"    ctxSprt.strokeStyle = '#666';\n"
"    ctxSprt.beginPath(); ctxSprt.moveTo(PAD_L, Y(ch.lower)); ctxSprt.lineTo(W - PAD_R, Y(ch.lower)); ctxSprt.stroke();\n"
"    ctxSprt.setLineDash([]);\n"
"\n"
"    function poly(vals, color) {\n"
"      ctxSprt.strokeStyle = color; ctxSprt.lineWidth = 1.6;\n"
"      ctxSprt.beginPath();\n"
"      for (let i = 0; i < DATA.k; ++i) {\n"
"        const x = X(DATA.times[i]), y = Y(vals[i]);\n"
"        if (i === 0) ctxSprt.moveTo(x, y); else ctxSprt.lineTo(x, y);\n"
"      }\n"
"      ctxSprt.stroke();\n"
"    }\n"
"    poly(ch.llrPos, '#1f77b4');\n"
"    poly(ch.llrNeg, '#2ca02c');\n"
"\n"
"    ctxSprt.strokeStyle = '#d62728'; ctxSprt.lineWidth = 2.2;\n"
"    ch.faultIndices.forEach(i => {\n"
"      const x = X(DATA.times[i]), y = Y(Math.max(ch.llrPos[i], ch.llrNeg[i]));\n"
"      ctxSprt.beginPath();\n"
"      ctxSprt.moveTo(x-5,y-5); ctxSprt.lineTo(x+5,y+5);\n"
"      ctxSprt.moveTo(x+5,y-5); ctxSprt.lineTo(x-5,y+5);\n"
"      ctxSprt.stroke();\n"
"    });\n"
"\n"
"    const cx = X(DATA.times[state.cursor]);\n"
"    ctxSprt.strokeStyle = '#ff7f0e'; ctxSprt.lineWidth = 2;\n"
"    ctxSprt.beginPath(); ctxSprt.moveTo(cx, PAD_T); ctxSprt.lineTo(cx, H - PAD_B); ctxSprt.stroke();\n"
"\n"
"    sprtTitle.textContent = 'SPRT fault detection - channel: ' + DATA.channels[state.channel] +\n"
"      '   (mu0=' + ch.mu0.toFixed(4) + ', sigma=' + ch.sigma.toFixed(4) +\n"
"      ', alarms=' + ch.faultIndices.length + ')';\n"
"  }\n"
"\n"
"  cSprt.addEventListener('click', e => {\n"
"    const rect = cSprt.getBoundingClientRect();\n"
"    const mx = e.clientX - rect.left;\n"
"    const tmin = DATA.times[0], tmax = DATA.times[DATA.k - 1];\n"
"    const t = tmin + (mx - PAD_L) / (cSprt.width - PAD_L - PAD_R) * (tmax - tmin);\n"
"    let best = 0, bestD = Infinity;\n"
"    for (let i = 0; i < DATA.k; ++i) {\n"
"      const d = Math.abs(DATA.times[i] - t);\n"
"      if (d < bestD) { bestD = d; best = i; }\n"
"    }\n"
"    setCursor(best);\n"
"  });\n"
"\n"
"  slider.addEventListener('input', () => setCursor(parseInt(slider.value, 10)));\n"
"\n"
"  ['training','estimates','residuals','anomalies','trajectory'].forEach(name => {\n"
"    document.getElementById('chk_' + name).addEventListener('change', e => {\n"
"      state.layers[name] = e.target.checked;\n"
"      redraw3D(); redrawSPRT();\n"
"    });\n"
"  });\n"
"\n"
"  function updateInfo() {\n"
"    const i = state.cursor;\n"
"    const ch = DATA.sprt[state.channel];\n"
"    const names = [DATA.paramNames[0], DATA.paramNames[1], DATA.paramNames[2]];\n"
"    const obs = [DATA.objX[i], DATA.objY[i], DATA.objZ[i]];\n"
"    const est = [DATA.estX[i], DATA.estY[i], DATA.estZ[i]];\n"
"    const lines = [];\n"
"    lines.push('sample  #' + (i+1) + '/' + DATA.k);\n"
"    lines.push('time    ' + DATA.times[i]);\n"
"    lines.push('status  ' + (DATA.anomalyMask[i] ? 'ANOMALY' : 'healthy'));\n"
"    lines.push('||R||   ' + DATA.residualNorm[i].toFixed(4));\n"
"    lines.push('');\n"
"    for (let j = 0; j < 3; ++j) {\n"
"      lines.push(names[j].slice(0,12).padEnd(12) + ' obs ' + obs[j].toFixed(3));\n"
"      lines.push(''.padEnd(12) + ' est ' + est[j].toFixed(3));\n"
"    }\n"
"    lines.push('');\n"
"    lines.push('SPRT [' + DATA.channels[state.channel].slice(0,14) + ']');\n"
"    lines.push('  LLR+  ' + ch.llrPos[i].toFixed(2));\n"
"    lines.push('  LLR-  ' + ch.llrNeg[i].toFixed(2));\n"
"    info.textContent = lines.join('\\n');\n"
"  }\n"
"\n"
"  redraw3D();\n"
"  redrawSPRT();\n"
"})();\n";

/* ======================================================================== */
/* 4. Public API                                                            */
/* ======================================================================== */

void mset_visualizer_default_options(MSETVisualizerOptions *opt)
{
    if (!opt) return;
    opt->alpha = 0.01;
    opt->beta = 0.01;
    opt->disturbance = 3.0;
    opt->title = NULL;
    opt->time_label = NULL;
}

int mset_visualize_write_html(const char *path,
                              const MSET *model,
                              const double *observations,
                              const double *times,
                              int k,
                              const char * const *param_names,
                              const MSETVisualizerOptions *opt)
{
    if (!path || !model || !model->fitted || !observations || k < 2) return -1;
    if (model->n_params < 3) return -1;
    int n = model->n_params;
    int rc = 0;

    MSETVisualizerOptions default_opt;
    if (!opt) { mset_visualizer_default_options(&default_opt); opt = &default_opt; }

    /* every heap pointer used below is declared here and NULL-initialised,
     * so the single `cleanup:` label at the bottom can unconditionally
     * free() everything (free(NULL) is a no-op) regardless of where an
     * error path breaks out. */
    double *own_times = NULL;
    char **own_names = NULL;
    double *est = NULL, *res = NULL, *resnorm = NULL;
    double *mu = NULL, *sd = NULL;
    VizSprt *traces = NULL;
    int n_channels = n + 1;
    double *series_buf = NULL;
    int *anomaly = NULL;
    double *trainX = NULL, *trainY = NULL, *trainZ = NULL;
    double *objX = NULL, *objY = NULL, *objZ = NULL;
    double *estX = NULL, *estY = NULL, *estZ = NULL;
    char **channel_names = NULL;
    FILE *f = NULL;

    /* --- fill in optional times / names -------------------------------- */
    const double *tm = times;
    if (!tm) {
        own_times = malloc(sizeof(double) * (size_t)k);
        if (!own_times) { rc = -1; goto cleanup; }
        for (int i = 0; i < k; ++i) own_times[i] = i + 1;
        tm = own_times;
    }

    const char * const *pnames = param_names;
    if (!pnames) {
        own_names = calloc((size_t)n, sizeof(char *));
        if (!own_names) { rc = -1; goto cleanup; }
        for (int j = 0; j < n; ++j) {
            own_names[j] = malloc(16);
            if (own_names[j]) snprintf(own_names[j], 16, "p%d", j + 1);
        }
        pnames = (const char * const *)own_names;
    }

    /* --- MSET estimates / residuals / residual norms -------------------- */
    est = malloc(sizeof(double) * (size_t)k * n);
    res = malloc(sizeof(double) * (size_t)k * n);   /* scaled */
    resnorm = malloc(sizeof(double) * (size_t)k);
    if (!est || !res || !resnorm) { rc = -1; goto cleanup; }

    for (int i = 0; i < k; ++i) {
        if (mset_estimate(model, &observations[(size_t)i * n], &est[(size_t)i * n]) != 0 ||
            mset_residual(model, &observations[(size_t)i * n], &res[(size_t)i * n], 1) != 0) {
            rc = -1; goto cleanup;
        }
        resnorm[i] = euclidean_norm(&res[(size_t)i * n], n);
    }

    /* --- healthy statistics (per parameter + for the ||R|| norm) -------- */
    mu = malloc(sizeof(double) * (size_t)n);
    sd = malloc(sizeof(double) * (size_t)n);
    if (!mu || !sd) { rc = -1; goto cleanup; }
    mset_healthy_statistics(model, mu, sd);

    double normmu = 0.0, normsd = 1e-6;
    if (model->n_remaining > 0) {
        int nrem = model->n_remaining;
        double *healthy_res = malloc(sizeof(double) * (size_t)nrem * n);
        double *hn = malloc(sizeof(double) * (size_t)nrem);
        if (!healthy_res || !hn) { free(healthy_res); free(hn); rc = -1; goto cleanup; }
        if (mset_healthy_residuals(model, healthy_res) == 0) {
            double s = 0.0;
            for (int i = 0; i < nrem; ++i) {
                hn[i] = euclidean_norm(&healthy_res[(size_t)i * n], n);
                s += hn[i];
            }
            normmu = s / nrem;
            double var = 0.0;
            if (nrem > 1) {
                for (int i = 0; i < nrem; ++i) {
                    double d = hn[i] - normmu;
                    var += d * d;
                }
                var /= (nrem - 1);
            }
            normsd = sqrt(var);
            if (normsd < 1e-9) normsd = 1e-9;
        }
        free(healthy_res);
        free(hn);
    }

    /* --- SPRT traces: one per parameter, plus the ||R|| norm channel ---- */
    traces = calloc((size_t)n_channels, sizeof(VizSprt));
    series_buf = malloc(sizeof(double) * (size_t)k);
    if (!traces || !series_buf) { rc = -1; goto cleanup; }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) series_buf[i] = res[(size_t)i * n + j];
        vizsprt_run(&traces[j], mu[j], sd[j], opt->alpha, opt->beta,
                   opt->disturbance, series_buf, k);
    }
    vizsprt_run(&traces[n], normmu, normsd, opt->alpha, opt->beta,
               opt->disturbance, resnorm, k);
    free(series_buf);
    series_buf = NULL;

    anomaly = calloc((size_t)k, sizeof(int));
    if (!anomaly) { rc = -1; goto cleanup; }
    for (int c = 0; c < n_channels; ++c)
        for (int fidx = 0; fidx < traces[c].fault_count; ++fidx)
            anomaly[traces[c].fault_idx[fidx]] = 1;

    /* --- training cloud: unscale D_states + L_states, first 3 dims ------ */
    {
        int ntrain = model->n_memory + model->n_remaining;
        int alloc_n = ntrain > 0 ? ntrain : 1;
        trainX = malloc(sizeof(double) * (size_t)alloc_n);
        trainY = malloc(sizeof(double) * (size_t)alloc_n);
        trainZ = malloc(sizeof(double) * (size_t)alloc_n);
        if (!trainX || !trainY || !trainZ) { rc = -1; goto cleanup; }

        int ti = 0;
        for (int i = 0; i < model->n_memory; ++i, ++ti) {
            const double *s = &model->D_states[(size_t)i * n];
            trainX[ti] = model->normalize ? s[0] * model->span[0] + model->lo[0] : s[0];
            trainY[ti] = model->normalize ? s[1] * model->span[1] + model->lo[1] : s[1];
            trainZ[ti] = model->normalize ? s[2] * model->span[2] + model->lo[2] : s[2];
        }
        for (int i = 0; i < model->n_remaining; ++i, ++ti) {
            const double *s = &model->L_states[(size_t)i * n];
            trainX[ti] = model->normalize ? s[0] * model->span[0] + model->lo[0] : s[0];
            trainY[ti] = model->normalize ? s[1] * model->span[1] + model->lo[1] : s[1];
            trainZ[ti] = model->normalize ? s[2] * model->span[2] + model->lo[2] : s[2];
        }

        /* --- first-3-column views of obs / estimates --------------------*/
        objX = malloc(sizeof(double) * (size_t)k);
        objY = malloc(sizeof(double) * (size_t)k);
        objZ = malloc(sizeof(double) * (size_t)k);
        estX = malloc(sizeof(double) * (size_t)k);
        estY = malloc(sizeof(double) * (size_t)k);
        estZ = malloc(sizeof(double) * (size_t)k);
        if (!objX || !objY || !objZ || !estX || !estY || !estZ) { rc = -1; goto cleanup; }
        for (int i = 0; i < k; ++i) {
            objX[i] = observations[(size_t)i * n + 0];
            objY[i] = observations[(size_t)i * n + 1];
            objZ[i] = observations[(size_t)i * n + 2];
            estX[i] = est[(size_t)i * n + 0];
            estY[i] = est[(size_t)i * n + 1];
            estZ[i] = est[(size_t)i * n + 2];
        }

        /* --- channel display names -------------------------------------*/
        channel_names = calloc((size_t)n_channels, sizeof(char *));
        if (!channel_names) { rc = -1; goto cleanup; }
        for (int c = 0; c < n_channels; ++c) {
            if (c < n) {
                channel_names[c] = malloc(strlen(pnames[c]) + 1);
                if (channel_names[c]) strcpy(channel_names[c], pnames[c]);
            } else {
                channel_names[c] = malloc(16);
                if (channel_names[c]) strcpy(channel_names[c], "||R|| norm");
            }
        }

        /* --- write the file ---------------------------------------------*/
        f = fopen(path, "w");
        if (!f) { rc = -1; goto cleanup; }

        fputs(HTML_TOP, f);
        fputs(HTML_BEFORE_SCRIPT, f);
        fputs("<script>\nconst DATA = {\n", f);

        fputs("  \"title\": ", f);
        jw_string(f, opt->title ? opt->title
                                : "MSET input & anomalies (3D) + SPRT fault detection");
        fputs(",\n", f);

        fputs("  \"timeLabel\": ", f);
        jw_string(f, opt->time_label ? opt->time_label : "Time");
        fputs(",\n", f);

        jw_string_array(f, "paramNames", pnames, n);
        fputs(",\n", f);

        fprintf(f, "  \"k\": %d,\n", k);

        jw_double_array(f, "times", tm, k);        fputs(",\n", f);
        jw_double_array(f, "objX", objX, k);       fputs(",\n", f);
        jw_double_array(f, "objY", objY, k);       fputs(",\n", f);
        jw_double_array(f, "objZ", objZ, k);       fputs(",\n", f);
        jw_double_array(f, "estX", estX, k);       fputs(",\n", f);
        jw_double_array(f, "estY", estY, k);       fputs(",\n", f);
        jw_double_array(f, "estZ", estZ, k);       fputs(",\n", f);
        jw_double_array(f, "residualNorm", resnorm, k); fputs(",\n", f);
        jw_bool_array(f, "anomalyMask", anomaly, k);    fputs(",\n", f);
        jw_double_array(f, "trainX", trainX, ntrain);   fputs(",\n", f);
        jw_double_array(f, "trainY", trainY, ntrain);   fputs(",\n", f);
        jw_double_array(f, "trainZ", trainZ, ntrain);   fputs(",\n", f);
        jw_string_array(f, "channels", (const char * const *)channel_names, n_channels);
        fputs(",\n", f);

        fputs("  \"sprt\": [\n", f);
        for (int c = 0; c < n_channels; ++c) {
            VizSprt *t = &traces[c];
            fputs("    {", f);
            fprintf(f, "\"mu0\":%.6g,\"sigma\":%.6g,\"upper\":%.6g,\"lower\":%.6g,",
                   t->mu0, t->sigma, t->upper, t->lower);
            fputs("\"llrPos\":[", f);
            for (int i = 0; i < k; ++i) { if (i) fputc(',', f); fprintf(f, "%.6g", t->llr_pos[i]); }
            fputs("],\"llrNeg\":[", f);
            for (int i = 0; i < k; ++i) { if (i) fputc(',', f); fprintf(f, "%.6g", t->llr_neg[i]); }
            fputs("],", f);
            jw_int_array(f, "faultIndices", t->fault_idx, t->fault_count);
            fputs(c < n_channels - 1 ? "},\n" : "}\n", f);
        }
        fputs("  ]\n};\n", f);

        fputs(JS_TEMPLATE, f);
        fputs("</script>\n</body>\n</html>\n", f);
    }

cleanup:
    if (f) fclose(f);
    if (channel_names) {
        for (int c = 0; c < n_channels; ++c) free(channel_names[c]);
        free(channel_names);
    }
    free(objX); free(objY); free(objZ);
    free(estX); free(estY); free(estZ);
    free(trainX); free(trainY); free(trainZ);
    free(anomaly);
    free(series_buf);
    if (traces) {
        for (int c = 0; c < n_channels; ++c) vizsprt_free(&traces[c]);
        free(traces);
    }
    free(mu); free(sd);
    free(est); free(res); free(resnorm);
    free(own_times);
    if (own_names) {
        for (int j = 0; j < n; ++j) free(own_names[j]);
        free(own_names);
    }
    return rc;
}
