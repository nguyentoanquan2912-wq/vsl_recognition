// VN Check - Train
const socket = io();
let isTraining = false;

document.addEventListener('DOMContentLoaded', async () => {
    await loadStats();
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) trainBtn.addEventListener('click', startTraining);
});

async function loadStats() {
    try {
        const data = await apiFetch('/api/dataset/stats');
        const el = document.getElementById('train-samples');
        const cl = document.getElementById('train-classes');
        const msg = document.getElementById('data-status-msg');
        const btn = document.getElementById('train-btn');

        if (el) el.textContent = data.total_samples || 0;
        if (cl) cl.textContent = data.total_classes || 0;

        if (data.total_classes >= 2 && data.total_samples >= 5) {
            if (msg) { msg.textContent = 'Du lieu du de huan luyen'; msg.style.color = 'var(--success)'; }
            if (btn) btn.disabled = false;
        } else {
            if (msg) { msg.textContent = 'Can it nhat 2 lop va 5 mau'; msg.style.color = 'var(--danger)'; }
            if (btn) btn.disabled = true;
        }
    } catch (e) {
        const msg = document.getElementById('data-status-msg');
        if (msg) { msg.textContent = 'Loi tai du lieu'; msg.style.color = 'var(--danger)'; }
    }
}

function setStep(stepId) {
    document.querySelectorAll('.p-step').forEach(s => s.classList.remove('active', 'done'));
    const steps = ['step-data', 'step-train', 'step-eval', 'step-done'];
    const idx = steps.indexOf(stepId);
    steps.forEach((s, i) => {
        const el = document.getElementById(s);
        if (i < idx) el.classList.add('done');
        if (i === idx) el.classList.add('active');
    });
}

async function startTraining() {
    if (isTraining) return;
    isTraining = true;

    const model = document.getElementById('model-type').value;
    const trainBtn = document.getElementById('train-btn');
    const idleDiv = document.getElementById('status-idle');
    const trainDiv = document.getElementById('status-training');
    const resultDiv = document.getElementById('status-result');
    const logEl = document.getElementById('training-log');

    if (trainBtn) trainBtn.disabled = true;
    if (idleDiv) idleDiv.style.display = 'none';
    if (trainDiv) trainDiv.style.display = 'flex';
    if (resultDiv) resultDiv.style.display = 'none';

    setStep('step-data');
    if (logEl) logEl.textContent = 'Dang tai du lieu...';

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: model })
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Loi huan luyen');
        }

        showToast('Da bat dau huan luyen', 'info');
    } catch (e) {
        showToast(e.message, 'error');
        if (logEl) logEl.textContent = 'Loi: ' + e.message;
        isTraining = false;
        if (trainBtn) trainBtn.disabled = false;
    }
}

socket.on('training_progress', (data) => {
    const logEl = document.getElementById('training-log');
    if (logEl) logEl.textContent = data.message || data.step || '...';

    if (data.step === 'loading' || data.step === 'augmenting') setStep('step-data');
    else if (data.step === 'training') setStep('step-train');
    else if (data.step === 'evaluating') setStep('step-eval');
});

socket.on('training_complete', (data) => {
    isTraining = false;
    setStep('step-done');

    const trainDiv = document.getElementById('status-training');
    const resultDiv = document.getElementById('status-result');
    const trainBtn = document.getElementById('train-btn');

    if (trainDiv) trainDiv.style.display = 'none';
    if (resultDiv) resultDiv.style.display = 'block';
    if (trainBtn) trainBtn.disabled = false;

    if (data.metrics) {
        const m = data.metrics;
        const acc = document.getElementById('res-accuracy');
        const prec = document.getElementById('res-precision');
        const rec = document.getElementById('res-recall');
        const f1 = document.getElementById('res-f1');

        if (acc) acc.textContent = (m.accuracy * 100).toFixed(1) + '%';
        if (prec) prec.textContent = (m.precision * 100).toFixed(1) + '%';
        if (rec) rec.textContent = (m.recall * 100).toFixed(1) + '%';
        if (f1) f1.textContent = (m.f1 * 100).toFixed(1) + '%';
    }

    showToast('Huan luyen hoan tat!', 'success');
});

socket.on('training_error', (data) => {
    isTraining = false;
    const logEl = document.getElementById('training-log');
    const trainBtn = document.getElementById('train-btn');
    if (logEl) logEl.textContent = 'Loi: ' + (data.error || 'Unknown');
    if (trainBtn) trainBtn.disabled = false;
    showToast('Loi huan luyen: ' + (data.error || ''), 'error');
});
