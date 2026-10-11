// VN Check Dashboard
Chart.defaults.color = '#8888aa';
Chart.defaults.font.family = "'Be Vietnam Pro', sans-serif";
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.04)';

let metricsData = null;
let statsData = null;

async function loadData() {
    try {
        const [metricsRes, statsRes] = await Promise.all([
            fetch('/api/models/metrics'),
            fetch('/api/dataset/stats')
        ]);
        const metricsJson = await metricsRes.json();
        const statsJson = await statsRes.json();

        if (!metricsRes.ok || !metricsJson.metrics) {
            document.getElementById('no-metrics').style.display = 'block';
            return;
        }

        metricsData = metricsJson.metrics;
        statsData = statsJson.stats || {};
        document.getElementById('dashboard-content').style.display = 'block';

        renderKPIs();
        renderClassChart();
        renderRadar();
        renderConfusion();
        renderDistribution();
    } catch (e) {
        console.error(e);
        showToast("Loi tai du lieu dashboard", "error");
    }
}

function renderKPIs() {
    document.getElementById('kpi-accuracy').textContent = `${(metricsData.accuracy * 100).toFixed(1)}%`;
    document.getElementById('kpi-precision').textContent = `${(metricsData.precision * 100).toFixed(1)}%`;
    document.getElementById('kpi-recall').textContent = `${(metricsData.recall * 100).toFixed(1)}%`;
    document.getElementById('kpi-f1').textContent = `${(metricsData.f1 * 100).toFixed(1)}%`;

    const badge = document.getElementById('model-badge');
    if (badge && metricsData.model_type) badge.textContent = metricsData.model_type.toUpperCase();
}

function renderClassChart() {
    const ctx = document.getElementById('class-accuracy-chart').getContext('2d');
    const report = metricsData.classification_report;
    if (!report) return;

    const labels = [], f1s = [], precs = [], recs = [];
    const skip = ['accuracy', 'macro avg', 'weighted avg'];
    for (const k in report) {
        if (!skip.includes(k) && typeof report[k] === 'object') {
            labels.push(k);
            f1s.push(report[k]['f1-score'] * 100);
            precs.push(report[k]['precision'] * 100);
            recs.push(report[k]['recall'] * 100);
        }
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'F1 (%)', data: f1s, backgroundColor: 'rgba(255,107,107,0.6)', borderColor: '#ff6b6b', borderWidth: 1, borderRadius: 4 },
                { label: 'Precision (%)', data: precs, backgroundColor: 'rgba(255,217,61,0.4)', borderColor: '#ffd93d', borderWidth: 1, borderRadius: 4 },
                { label: 'Recall (%)', data: recs, backgroundColor: 'rgba(107,203,119,0.3)', borderColor: '#6bcb77', borderWidth: 1, borderRadius: 4 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.03)' } }, x: { grid: { display: false } } },
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } } }
        }
    });
}

function renderRadar() {
    const ctx = document.getElementById('radar-chart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Performance',
                data: [metricsData.accuracy*100, metricsData.precision*100, metricsData.recall*100, metricsData.f1*100],
                backgroundColor: 'rgba(255,107,107,0.12)',
                borderColor: '#ff6b6b', borderWidth: 2,
                pointBackgroundColor: '#ff6b6b', pointBorderColor: '#0c0c1d', pointBorderWidth: 2, pointRadius: 5
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { r: {
                angleLines: { color: 'rgba(255,255,255,0.05)' },
                grid: { color: 'rgba(255,255,255,0.05)' },
                pointLabels: { color: '#f0f0f5', font: { size: 12, weight: '600' } },
                ticks: { color: '#8888aa', backdropColor: 'transparent', beginAtZero: true, max: 100, stepSize: 20, font: { size: 9 } }
            }},
            plugins: { legend: { display: false } }
        }
    });
}

function renderConfusion() {
    const table = document.getElementById('confusion-matrix');
    const matrix = metricsData.confusion_matrix;
    const labels = metricsData.labels || [];
    if (!matrix || !labels.length) { table.innerHTML = '<tr><td>Khong co du lieu</td></tr>'; return; }

    let maxVal = 0;
    matrix.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

    let h = '<thead><tr><th></th>';
    labels.forEach(l => { h += `<th style="writing-mode:vertical-rl;transform:rotate(180deg);height:45px;font-size:0.65rem;">${l}</th>`; });
    h += '</tr></thead><tbody>';

    matrix.forEach((row, i) => {
        h += `<tr><th style="text-align:right;padding-right:6px;font-size:0.7rem;">${labels[i]}</th>`;
        row.forEach((v, j) => {
            let bg = 'transparent', col = 'var(--text)';
            if (v > 0) {
                const t = Math.max(0.1, v / maxVal);
                if (i === j) { bg = `rgba(255,107,107,${t})`; if (t > 0.5) col = '#000'; }
                else { bg = `rgba(255,71,87,${t * 0.5})`; }
            }
            h += `<td style="background:${bg};color:${col};border:1px solid var(--border);font-size:0.75rem;padding:0.35rem;">${v}</td>`;
        });
        h += '</tr>';
    });
    table.innerHTML = h + '</tbody>';
}

function renderDistribution() {
    const ctx = document.getElementById('distribution-chart').getContext('2d');
    const labels = Object.keys(statsData);
    const data = Object.values(statsData);
    if (!labels.length) return;

    const colors = labels.map((_, i) => {
        const hue = (i * 360 / labels.length + 0) % 360;
        return `hsla(${hue}, 65%, 60%, 0.6)`;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ label: 'So mau', data, backgroundColor: colors, borderColor: colors.map(c => c.replace('0.6', '1')), borderWidth: 1, borderRadius: 4 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' } }, x: { grid: { display: false } } },
            plugins: { legend: { display: false } }
        }
    });
}

document.addEventListener('DOMContentLoaded', loadData);
