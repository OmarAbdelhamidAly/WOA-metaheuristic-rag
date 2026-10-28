/**
 * ==============================================================================
 * WHALE OPTIMIZATION ALGORITHM (WOA) & RAG SIMULATOR - CLIENT LOGIC
 * ==============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSwarmSimulation();
    initRAGTuner();
    initBenchmark();
    initExamSolver();
});

/* ==============================================================================
   1. TAB SWITCHING SYSTEM
   ============================================================================== */
function initTabs() {
    const tabButtons = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTabId = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetTabId);
            if (targetEl) targetEl.classList.add('active');

            // Trigger redraws if canvas is in newly activated tab
            if (targetTabId === 'tab-swarm') {
                window.dispatchEvent(new Event('resize'));
            }
        });
    });
}

/* ==============================================================================
   2. 2D SWARM VISUAL SIMULATION (CANVAS)
   ============================================================================== */
function initSwarmSimulation() {
    const canvas = document.getElementById('swarm-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Controls
    const funcSelect = document.getElementById('sim-function');
    const popSlider = document.getElementById('sim-pop');
    const popVal = document.getElementById('sim-pop-val');
    const epochSlider = document.getElementById('sim-epochs');
    const epochVal = document.getElementById('sim-epochs-val');
    const speedSlider = document.getElementById('sim-speed');
    const speedVal = document.getElementById('sim-speed-val');

    const btnRun = document.getElementById('btn-sim-run');
    const btnPause = document.getElementById('btn-sim-pause');
    const btnStep = document.getElementById('btn-sim-step');
    const btnReset = document.getElementById('btn-sim-reset');

    // Telemetry Elements
    const badgeEpoch = document.getElementById('canvas-epoch-badge');
    const badgePhase = document.getElementById('canvas-phase-badge');
    const telemetryA = document.getElementById('telemetry-a');
    const telemetryVectorA = document.getElementById('telemetry-A');
    const telemetryVectorAStatus = document.getElementById('telemetry-A-status');
    const telemetryP = document.getElementById('telemetry-p');
    const telemetryFitness = document.getElementById('telemetry-fitness');
    const telemetryCoords = document.getElementById('telemetry-coords');
    const barA = document.getElementById('bar-a');
    const barVectorA = document.getElementById('bar-A');
    const barP = document.getElementById('bar-p');

    // State
    let simulationData = null;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let animationInterval = null;
    let animSpeed = 1.0;

    // Slider Listeners
    popSlider.addEventListener('input', () => popVal.textContent = popSlider.value);
    epochSlider.addEventListener('input', () => epochVal.textContent = epochSlider.value);
    speedSlider.addEventListener('input', () => {
        animSpeed = parseFloat(speedSlider.value);
        speedVal.textContent = animSpeed.toFixed(1) + 'x';
        if (isPlaying) {
            restartTimer();
        }
    });

    // Run / Fetch Simulation
    btnRun.addEventListener('click', async () => {
        btnRun.disabled = true;
        btnRun.textContent = 'Simulating...';

        try {
            const resp = await fetch('/api/simulation/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    function_name: funcSelect.value,
                    pop_size: parseInt(popSlider.value),
                    max_iter: parseInt(epochSlider.value),
                    seed: 42
                })
            });
            simulationData = await resp.json();
            currentFrameIdx = 0;
            isPlaying = true;
            btnPause.disabled = false;
            btnPause.innerHTML = '<span class="btn-icon">⏸</span> Pause';
            restartTimer();
        } catch (err) {
            alert('Simulation failed: ' + err.message);
        } finally {
            btnRun.disabled = false;
            btnRun.innerHTML = '<span class="btn-icon">▶</span> Re-Run';
        }
    });

    btnPause.addEventListener('click', () => {
        if (isPlaying) {
            pauseAnimation();
        } else {
            resumeAnimation();
        }
    });

    btnStep.addEventListener('click', () => {
        pauseAnimation();
        if (simulationData && currentFrameIdx < simulationData.frames.length - 1) {
            currentFrameIdx++;
            renderFrame(currentFrameIdx);
        }
    });

    btnReset.addEventListener('click', () => {
        pauseAnimation();
        currentFrameIdx = 0;
        if (simulationData) {
            renderFrame(0);
        } else {
            drawEmptyCanvas();
        }
    });

    function pauseAnimation() {
        isPlaying = false;
        clearInterval(animationInterval);
        btnPause.innerHTML = '<span class="btn-icon">▶</span> Resume';
    }

    function resumeAnimation() {
        if (!simulationData) return;
        isPlaying = true;
        btnPause.innerHTML = '<span class="btn-icon">⏸</span> Pause';
        restartTimer();
    }

    function restartTimer() {
        clearInterval(animationInterval);
        const delay = Math.max(30, 280 / animSpeed);
        animationInterval = setInterval(() => {
            if (!simulationData) return;
            if (currentFrameIdx >= simulationData.frames.length - 1) {
                pauseAnimation();
                return;
            }
            currentFrameIdx++;
            renderFrame(currentFrameIdx);
        }, delay);
    }

    // Canvas Coordinate Mapping
    function mapCoords(val, minVal, maxVal, canvasDim) {
        return ((val - minVal) / (maxVal - minVal)) * (canvasDim - 80) + 40;
    }

    function renderFrame(frameIdx) {
        if (!simulationData || !simulationData.frames[frameIdx]) return;
        const frame = simulationData.frames[frameIdx];
        const bounds = simulationData.bounds;
        const minVal = bounds[0];
        const maxVal = bounds[1];

        const w = canvas.width;
        const h = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#030712';
        ctx.fillRect(0, 0, w, h);

        // Draw coordinate grid
        drawGrid(minVal, maxVal);

        // Draw spiral effect if phase is spiral attack
        const leaderX = mapCoords(frame.leader[0], minVal, maxVal, w);
        const leaderY = mapCoords(frame.leader[1], minVal, maxVal, h);

        if (frame.active_phase.includes('Spiral')) {
            drawSpiralAura(leaderX, leaderY);
        }

        // Draw whales
        frame.whales.forEach((whale) => {
            const wx = mapCoords(whale[0], minVal, maxVal, w);
            const wy = mapCoords(whale[1], minVal, maxVal, h);

            // Path to leader
            ctx.beginPath();
            ctx.moveTo(wx, wy);
            ctx.lineTo(leaderX, leaderY);
            if (frame.active_phase.includes('Spiral')) {
                ctx.strokeStyle = 'rgba(192, 132, 252, 0.25)';
            } else if (frame.active_phase.includes('Global')) {
                ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
            } else {
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
            }
            ctx.lineWidth = 1;
            ctx.stroke();

            // Whale dot
            ctx.beginPath();
            ctx.arc(wx, wy, 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = '#38BDF8';
            ctx.shadowColor = '#00F0FF';
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.shadowBlur = 0;
        });

        // Draw Leader (Golden Crown/Star)
        ctx.beginPath();
        ctx.arc(leaderX, leaderY, 8.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#F59E0B';
        ctx.shadowColor = '#F59E0B';
        ctx.shadowBlur = 18;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Update Telemetry Elements
        badgeEpoch.textContent = `Epoch: ${frame.epoch} / ${simulationData.total_epochs}`;
        badgePhase.textContent = frame.active_phase;

        if (frame.active_phase.includes('Spiral')) {
            badgePhase.className = 'badge badge-purple';
        } else if (frame.active_phase.includes('Global')) {
            badgePhase.className = 'badge';
            badgePhase.style.background = 'rgba(244, 63, 94, 0.2)';
            badgePhase.style.borderColor = 'rgba(244, 63, 94, 0.5)';
            badgePhase.style.color = '#FB7185';
        } else {
            badgePhase.className = 'badge';
            badgePhase.style.background = 'rgba(56, 189, 248, 0.15)';
            badgePhase.style.borderColor = 'rgba(56, 189, 248, 0.3)';
            badgePhase.style.color = '#38BDF8';
        }

        telemetryA.textContent = frame.a.toFixed(3);
        barA.style.width = `${(frame.a / 2.0) * 100}%`;

        telemetryVectorA.textContent = frame.avg_abs_A.toFixed(3);
        if (frame.avg_abs_A >= 1.0) {
            telemetryVectorAStatus.textContent = 'Exploration (|A| \u2265 1)';
            telemetryVectorAStatus.className = 'metric-sub text-amber';
            barVectorA.className = 'progress-bar-fill fill-amber';
        } else {
            telemetryVectorAStatus.textContent = 'Exploitation (|A| < 1)';
            telemetryVectorAStatus.className = 'metric-sub text-cyan';
            barVectorA.className = 'progress-bar-fill fill-cyan';
        }
        barVectorA.style.width = `${Math.min(100, (frame.avg_abs_A / 2.0) * 100)}%`;

        telemetryP.textContent = frame.p.toFixed(3);
        barP.style.width = `${frame.p * 100}%`;

        telemetryFitness.textContent = frame.best_fitness < 1e-4
            ? frame.best_fitness.toExponential(4)
            : frame.best_fitness.toFixed(6);

        telemetryCoords.textContent = `X*: (${frame.leader[0].toFixed(3)}, ${frame.leader[1].toFixed(3)})`;
    }

    function drawGrid(minVal, maxVal) {
        const w = canvas.width;
        const h = canvas.height;

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;

        const steps = 6;
        for (let i = 0; i <= steps; i++) {
            const x = 40 + (i / steps) * (w - 80);
            const y = 40 + (i / steps) * (h - 80);

            ctx.beginPath();
            ctx.moveTo(x, 40);
            ctx.lineTo(x, h - 40);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(40, y);
            ctx.lineTo(w - 40, y);
            ctx.stroke();
        }

        // Center crosshair
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
        ctx.beginPath();
        ctx.moveTo(w / 2, 40);
        ctx.lineTo(w / 2, h - 40);
        ctx.moveTo(40, h / 2);
        ctx.lineTo(w - 40, h / 2);
        ctx.stroke();
    }

    function drawSpiralAura(cx, cy) {
        ctx.strokeStyle = 'rgba(192, 132, 252, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let theta = 0; theta < 4 * Math.PI; theta += 0.1) {
            const r = 4.0 * Math.exp(0.18 * theta);
            const x = cx + r * Math.cos(theta);
            const y = cy + r * Math.sin(theta);
            if (theta === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function drawEmptyCanvas() {
        ctx.fillStyle = '#030712';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawGrid(-10, 10);
        ctx.fillStyle = '#64748B';
        ctx.font = '14px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click "Run Simulation" to visualize Whale Pod dynamics', canvas.width / 2, canvas.height / 2);
    }

    drawEmptyCanvas();
}

/* ==============================================================================
   3. RAG AUTO-TUNER SYSTEM
   ============================================================================== */
function initRAGTuner() {
    const sliderChunk = document.getElementById('rag-slider-chunk');
    const valChunk = document.getElementById('rag-val-chunk');
    const sliderOverlap = document.getElementById('rag-slider-overlap');
    const valOverlap = document.getElementById('rag-val-overlap');
    const sliderTopK = document.getElementById('rag-slider-topk');
    const valTopK = document.getElementById('rag-val-topk');
    const sliderAlpha = document.getElementById('rag-slider-alpha');
    const valAlpha = document.getElementById('rag-val-alpha');
    const sliderRerank = document.getElementById('rag-slider-rerank');
    const valRerank = document.getElementById('rag-val-rerank');

    const btnRunOpt = document.getElementById('btn-run-rag-opt');
    const spinner = document.getElementById('rag-opt-spinner');

    const elPrec = document.getElementById('metric-precision');
    const elRec = document.getElementById('metric-recall');
    const elF1 = document.getElementById('metric-f1');
    const elLoss = document.getElementById('metric-loss');
    const elGain = document.getElementById('metric-gain');

    // Slider Listeners
    sliderChunk.addEventListener('input', () => {
        valChunk.textContent = `${sliderChunk.value} tokens`;
        sliderOverlap.max = Math.floor(sliderChunk.value * 0.40);
        if (parseInt(sliderOverlap.value) > parseInt(sliderOverlap.max)) {
            sliderOverlap.value = sliderOverlap.max;
            valOverlap.textContent = `${sliderOverlap.value} tokens`;
        }
    });

    sliderOverlap.addEventListener('input', () => valOverlap.textContent = `${sliderOverlap.value} tokens`);
    sliderTopK.addEventListener('input', () => valTopK.textContent = sliderTopK.value);
    sliderAlpha.addEventListener('input', () => {
        const val = parseFloat(sliderAlpha.value);
        valAlpha.textContent = `${val.toFixed(2)} (${Math.round(val * 100)}% Dense / ${Math.round((1 - val) * 100)}% BM25)`;
    });
    sliderRerank.addEventListener('input', () => valRerank.textContent = parseFloat(sliderRerank.value).toFixed(2));

    // Run WOA on RAG Pipeline
    btnRunOpt.addEventListener('click', async () => {
        btnRunOpt.disabled = true;
        spinner.classList.remove('hidden');

        try {
            const resp = await fetch('/api/rag/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pop_size: 20, max_iter: 25, seed: 42 })
            });
            const data = await resp.json();

            // Update sliders with optimal values
            const hp = data.optimized_hyperparameters;
            sliderChunk.value = hp.chunk_size;
            valChunk.textContent = `${hp.chunk_size} tokens`;
            sliderOverlap.value = hp.chunk_overlap;
            valOverlap.textContent = `${hp.chunk_overlap} tokens`;
            sliderTopK.value = hp.top_k;
            valTopK.textContent = hp.top_k;
            sliderAlpha.value = hp.hybrid_alpha;
            valAlpha.textContent = hp.hybrid_alpha.toFixed(2);
            sliderRerank.value = hp.rerank_threshold;
            valRerank.textContent = hp.rerank_threshold.toFixed(2);

            // Update scorecards
            const m = data.optimized_metrics;
            elPrec.textContent = `${m.precision_pct.toFixed(1)}%`;
            elRec.textContent = `${m.recall_pct.toFixed(1)}%`;
            elF1.textContent = `${m.f1_pct.toFixed(1)}%`;
            elLoss.textContent = m.composite_loss.toFixed(4);
            elGain.textContent = `+${data.gain_vs_baseline_pct.toFixed(1)}% Gain vs Default`;

            // Draw Charts
            drawConvergenceChart(data.convergence_curve);
            drawRadarChart(hp);

        } catch (err) {
            alert('Optimization error: ' + err.message);
        } finally {
            btnRunOpt.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    function drawConvergenceChart(curve) {
        const cvs = document.getElementById('rag-convergence-chart');
        if (!cvs) return;
        const c = cvs.getContext('2d');
        const w = cvs.width;
        const h = cvs.height;

        c.fillStyle = '#070D19';
        c.fillRect(0, 0, w, h);

        const pad = 35;
        const minL = Math.min(...curve);
        const maxL = Math.max(...curve);

        // Draw axes
        c.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(pad, pad);
        c.lineTo(pad, h - pad);
        c.lineTo(w - pad, h - pad);
        c.stroke();

        // Draw line
        c.strokeStyle = '#00F0FF';
        c.lineWidth = 2.5;
        c.beginPath();
        curve.forEach((val, i) => {
            const x = pad + (i / (curve.length - 1)) * (w - 2 * pad);
            const y = (h - pad) - ((val - minL) / (maxL - minL || 1)) * (h - 2 * pad);
            if (i === 0) c.moveTo(x, y);
            else c.lineTo(x, y);
        });
        c.stroke();

        // Fill glow gradient under curve
        c.lineTo(w - pad, h - pad);
        c.lineTo(pad, h - pad);
        c.fillStyle = 'rgba(0, 240, 255, 0.08)';
        c.fill();
    }

    function drawRadarChart(hp) {
        const cvs = document.getElementById('rag-radar-chart');
        if (!cvs) return;
        const c = cvs.getContext('2d');
        const w = cvs.width;
        const h = cvs.height;
        const cx = w / 2;
        const cy = h / 2 + 5;
        const radius = Math.min(w, h) / 2 - 35;

        c.fillStyle = '#070D19';
        c.fillRect(0, 0, w, h);

        const labels = ['Chunk', 'Overlap', 'Top-K', 'Alpha', 'Rerank'];
        const values = [
            (hp.chunk_size - 128) / (1024 - 128),
            hp.chunk_overlap / 256.0,
            (hp.top_k - 1) / 19.0,
            hp.hybrid_alpha,
            hp.rerank_threshold / 0.90
        ];

        const numVars = labels.length;

        // Draw web rings
        c.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        c.lineWidth = 1;
        for (let r = 0.25; r <= 1.0; r += 0.25) {
            c.beginPath();
            for (let i = 0; i < numVars; i++) {
                const angle = (i / numVars) * 2 * Math.PI - Math.PI / 2;
                const x = cx + radius * r * Math.cos(angle);
                const y = cy + radius * r * Math.sin(angle);
                if (i === 0) c.moveTo(x, y);
                else c.lineTo(x, y);
            }
            c.closePath();
            c.stroke();
        }

        // Draw axes and labels
        c.fillStyle = '#94A3B8';
        c.font = '11px Outfit, sans-serif';
        c.textAlign = 'center';
        for (let i = 0; i < numVars; i++) {
            const angle = (i / numVars) * 2 * Math.PI - Math.PI / 2;
            const x = cx + (radius + 18) * Math.cos(angle);
            const y = cy + (radius + 18) * Math.sin(angle);
            c.fillText(labels[i], x, y + 4);
        }

        // Draw polygon
        c.beginPath();
        for (let i = 0; i < numVars; i++) {
            const angle = (i / numVars) * 2 * Math.PI - Math.PI / 2;
            const r = Math.max(0.1, values[i]) * radius;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            if (i === 0) c.moveTo(x, y);
            else c.lineTo(x, y);
        }
        c.closePath();
        c.fillStyle = 'rgba(0, 240, 255, 0.25)';
        c.fill();
        c.strokeStyle = '#00F0FF';
        c.lineWidth = 2;
        c.stroke();
    }
}

/* ==============================================================================
   4. 5-ALGORITHM BENCHMARK
   ============================================================================== */
function initBenchmark() {
    const btnRun = document.getElementById('btn-run-benchmark');
    const spinner = document.getElementById('benchmark-spinner');
    const tbody = document.getElementById('benchmark-tbody');

    btnRun.addEventListener('click', async () => {
        btnRun.disabled = true;
        spinner.classList.remove('hidden');

        try {
            const resp = await fetch('/api/benchmark/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pop_size: 15, max_iter: 20, seed: 42 })
            });
            const data = await resp.json();
            const b = data.benchmark;

            tbody.innerHTML = '';
            for (const [name, row] of Object.entries(b)) {
                const tr = document.createElement('tr');
                if (name.includes('WOA') || name.includes('Whale')) {
                    tr.className = 'highlight-row';
                }
                tr.innerHTML = `
                    <td><strong>${name}</strong></td>
                    <td>${row.loss.toFixed(4)}</td>
                    <td>${row.precision.toFixed(1)}%</td>
                    <td>${row.recall.toFixed(1)}%</td>
                    <td>${row.f1.toFixed(1)}%</td>
                    <td>${row.latency.toFixed(3)}</td>
                    <td class="${row.gain_vs_grid_pct > 0 ? 'text-emerald' : ''}">${row.gain_vs_grid_pct > 0 ? '+' : ''}${row.gain_vs_grid_pct}%</td>
                    <td>${row.runtime_sec.toFixed(2)}s</td>
                `;
                tbody.appendChild(tr);
            }

            drawBenchmarkBars(b);

        } catch (err) {
            alert('Benchmark failed: ' + err.message);
        } finally {
            btnRun.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    function drawBenchmarkBars(bData) {
        const cvs = document.getElementById('benchmark-bar-chart');
        if (!cvs) return;
        const c = cvs.getContext('2d');
        const w = cvs.width;
        const h = cvs.height;

        c.fillStyle = '#070D19';
        c.fillRect(0, 0, w, h);

        const keys = Object.keys(bData);
        const barWidth = 60;
        const gap = (w - keys.length * barWidth) / (keys.length + 1);

        keys.forEach((k, i) => {
            const d = bData[k];
            const x = gap + i * (barWidth + gap);
            // Lower loss is better; invert height or draw direct loss
            const barHeight = (d.loss / 0.035) * (h - 80);
            const y = h - 45 - barHeight;

            c.fillStyle = k.includes('Whale') ? '#00F0FF' : '#3B82F6';
            c.fillRect(x, y, barWidth, barHeight);

            // Label
            c.fillStyle = '#FFFFFF';
            c.font = '10px Outfit, sans-serif';
            c.textAlign = 'center';
            c.fillText(d.loss.toFixed(4), x + barWidth / 2, y - 8);

            c.fillStyle = '#94A3B8';
            c.fillText(k.split(' ')[0], x + barWidth / 2, h - 25);
        });
    }
}

/* ==============================================================================
   5. EXAM PROBLEM SOLVER
   ============================================================================== */
function initExamSolver() {
    const btns = document.querySelectorAll('.exam-btn');
    const titleEl = document.getElementById('exam-title');
    const givenEl = document.getElementById('exam-given-card');
    const stepsEl = document.getElementById('exam-steps-container');
    const btnRecalc = document.getElementById('btn-recalc-exam');

    let currentProblemId = 1;

    btns.forEach(b => {
        b.addEventListener('click', () => {
            btns.forEach(x => x.classList.remove('active'));
            b.classList.add('active');
            currentProblemId = parseInt(b.getAttribute('data-problem'));
            fetchExamSolution(currentProblemId);
        });
    });

    btnRecalc.addEventListener('click', () => fetchExamSolution(currentProblemId));

    async function fetchExamSolution(pId) {
        try {
            const resp = await fetch('/api/exam/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem_id: pId })
            });
            const data = await resp.json();

            titleEl.textContent = data.title;

            // Render Given
            let givenHtml = `<strong>Given Parameters:</strong><br/>`;
            for (const [k, v] of Object.entries(data.given)) {
                if (typeof v === 'object') {
                    givenHtml += `&bull; ${k}: ${JSON.stringify(v)}<br/>`;
                } else {
                    givenHtml += `&bull; ${k}: ${v}<br/>`;
                }
            }
            givenEl.innerHTML = givenHtml;

            // Render Step Cards
            stepsEl.innerHTML = '';
            data.steps.forEach(s => {
                const card = document.createElement('div');
                card.className = 'step-card';
                card.innerHTML = `
                    <div class="step-badge-text">Step ${s.step} &bull; ${s.title}</div>
                    <div class="step-formula">${s.formula}</div>
                    <div class="step-calc-line">${s.values}</div>
                    <div class="step-result-callout">&rarr; ${s.result}</div>
                `;
                stepsEl.appendChild(card);
            });

        } catch (err) {
            alert('Exam solver failed: ' + err.message);
        }
    }

    // Initial load
    fetchExamSolution(1);
}
