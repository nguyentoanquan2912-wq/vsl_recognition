let HandLandmarker, FilesetResolver, DrawingUtils;
let handLandmarker;
let webcamRunning = false;
let autoRecordInterval = null;
let currentLandmarks = null;
let datasetStats = {};

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const labelSelect = document.getElementById("label-select");
const recordBtn = document.getElementById("record-btn");
const autoRecordBtn = document.getElementById("auto-record-btn");
const currentLabelCount = document.getElementById("current-label-count");
const detectionStatus = document.getElementById("detection-status");
const toggleCameraBtn = document.getElementById("toggle-camera-btn");
const recIndicator = document.getElementById("rec-indicator");
const sampleProgress = document.getElementById("sample-progress");
const sampleTargetText = document.getElementById("sample-target-text");

const SAMPLE_TARGET = 100;
const LANDMARKS_PER_HAND = 21;

const HAND_COLORS = [
    { conn: "#00e5bf", lm: "#ffffff" },
    { conn: "#f472b6", lm: "#fef08a" }
];

async function initMediaPipe() {
    try {
        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs');
        HandLandmarker = vision.HandLandmarker;
        FilesetResolver = vision.FilesetResolver;
        DrawingUtils = vision.DrawingUtils;

        const visionWasm = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
        );

        handLandmarker = await HandLandmarker.createFromOptions(visionWasm, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });

        loadingOverlay.style.display = "none";
        const camStatus = document.getElementById("camera-status");
        if (camStatus) camStatus.textContent = "San sang";
        startWebcam();
    } catch (error) {
        console.error(error);
        if (loadingOverlay) loadingOverlay.innerHTML = '<p style="color:var(--danger)">Loi khoi tao MediaPipe</p>';
        showToast("Loi khoi tao MediaPipe", "error");
    }
}

async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showToast("Trinh duyet khong ho tro webcam", "error");
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
        if (toggleCameraBtn) toggleCameraBtn.textContent = "Tat Camera";
    } catch (err) {
        showToast("Loi truy cap webcam", "error");
    }
}

function stopWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    webcamRunning = false;
    if (toggleCameraBtn) toggleCameraBtn.textContent = "Bat Camera";
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

if (toggleCameraBtn) {
    toggleCameraBtn.addEventListener("click", () => {
        if (webcamRunning) stopWebcam();
        else startWebcam();
    });
}

function buildLandmarksPayload(results) {
    const zeroHand = Array.from({ length: LANDMARKS_PER_HAND }, () => [0, 0, 0]);
    let leftHand = zeroHand, rightHand = zeroHand;

    if (results.landmarks && results.handednesses) {
        for (let i = 0; i < results.landmarks.length; i++) {
            const handedness = results.handednesses[i]?.[0]?.categoryName;
            const lm = results.landmarks[i].map(l => [l.x, l.y, l.z]);
            if (handedness === 'Left') {
                rightHand = lm;
            } else {
                leftHand = lm;
            }
        }
    }

    return leftHand.concat(rightHand);
}

let lastVideoTime = -1;
function predictWebcam() {
    if (!webcamRunning) return;

    if (video.videoWidth > 0 && video.videoHeight > 0 && canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime && handLandmarker) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        const numHands = results.landmarks ? results.landmarks.length : 0;

        if (numHands > 0) {
            currentLandmarks = buildLandmarksPayload(results);
            detectionStatus.textContent = numHands === 2 ? "2 tay" : "1 tay";
            detectionStatus.style.color = numHands === 2 ? "var(--accent)" : "var(--success)";

            if (labelSelect.value) {
                recordBtn.disabled = false;
                if (!autoRecordInterval) autoRecordBtn.disabled = false;
            }

            const drawingUtils = new DrawingUtils(canvasCtx);
            for (let i = 0; i < numHands; i++) {
                const color = HAND_COLORS[i] || HAND_COLORS[0];
                drawingUtils.drawConnectors(results.landmarks[i], HandLandmarker.HAND_CONNECTIONS, {
                    color: color.conn, lineWidth: 3
                });
                drawingUtils.drawLandmarks(results.landmarks[i], {
                    color: color.lm, lineWidth: 2, radius: 3
                });
            }
        } else {
            currentLandmarks = null;
            detectionStatus.textContent = "Khong thay tay";
            detectionStatus.style.color = "var(--danger)";
            recordBtn.disabled = true;
            if (!autoRecordInterval) autoRecordBtn.disabled = true;
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}

async function loadLabels() {
    try {
        const data = await apiFetch('/api/labels');
        labelSelect.innerHTML = '<option value="">-- Chon nhan --</option>';

        const optgroupAlpha = document.createElement('optgroup');
        optgroupAlpha.label = "Bang chu cai";
        data.alphabet.forEach(l => {
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = data.display_names[l] || l;
            optgroupAlpha.appendChild(opt);
        });
        labelSelect.appendChild(optgroupAlpha);

        const optgroupWord = document.createElement('optgroup');
        optgroupWord.label = "Tu vung";
        data.words.forEach(w => {
            const opt = document.createElement('option');
            opt.value = w;
            opt.textContent = data.display_names[w] || w;
            optgroupWord.appendChild(opt);
        });
        labelSelect.appendChild(optgroupWord);
    } catch (e) {
        labelSelect.innerHTML = '<option value="">Loi tai danh sach</option>';
    }
}

async function loadDatasetStats() {
    try {
        const data = await apiFetch('/api/dataset/stats');
        datasetStats = data.stats || {};
        renderDatasetTable();
        updateCurrentLabelCount();
    } catch (e) {
        const tbody = document.querySelector('#dataset-table tbody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger">Loi tai du lieu</td></tr>';
    }
}

function renderDatasetTable() {
    const tbody = document.querySelector('#dataset-table tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (Object.keys(datasetStats).length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="text-center text-secondary">Chua co du lieu</td></tr>';
        return;
    }

    for (const [label, count] of Object.entries(datasetStats)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:500;">${label}</td>
            <td>
                <span style="font-weight:600;">${count}</span>
                <span class="text-secondary" style="font-size:0.75rem;margin-left:0.35rem;">/ ${SAMPLE_TARGET}</span>
            </td>
            <td class="text-center">
                <button class="btn btn-secondary" style="padding:0.2rem 0.5rem;font-size:0.75rem;border-color:rgba(244,63,94,0.2);color:var(--danger);" onclick="deleteClass('${label}')">Xoa</button>
            </td>
        `;
        tbody.appendChild(tr);
    }
}

function updateCurrentLabelCount() {
    const label = labelSelect.value;
    const count = label ? (datasetStats[label] || 0) : 0;
    currentLabelCount.textContent = count;

    if (sampleProgress) {
        const pct = Math.min(100, (count / SAMPLE_TARGET) * 100);
        sampleProgress.style.width = `${pct}%`;
    }
    if (sampleTargetText) {
        sampleTargetText.textContent = `${count} / ${SAMPLE_TARGET}`;
    }
}

async function saveSample() {
    const label = labelSelect.value;
    if (!label || !currentLandmarks) return;

    try {
        await apiFetch('/api/collect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label, landmarks: currentLandmarks })
        });

        datasetStats[label] = (datasetStats[label] || 0) + 1;
        updateCurrentLabelCount();
        renderDatasetTable();

        canvasElement.style.boxShadow = "0 0 20px rgba(16,185,129,0.5)";
        setTimeout(() => canvasElement.style.boxShadow = "none", 200);
    } catch (e) {
        console.error("Save error:", e);
    }
}

window.deleteClass = async function(label) {
    if (!confirm(`Ban co chac muon xoa toan bo du lieu cua nhan "${label}"?`)) return;

    try {
        await apiFetch(`/api/dataset/delete/${label}`, { method: 'DELETE' });
        showToast(`Da xoa nhan ${label}`, 'success');
        delete datasetStats[label];
        renderDatasetTable();
        updateCurrentLabelCount();
    } catch (e) {
        showToast("Loi khi xoa", "error");
    }
};

labelSelect.addEventListener('change', () => {
    updateCurrentLabelCount();
    if (labelSelect.value && currentLandmarks) {
        recordBtn.disabled = false;
        autoRecordBtn.disabled = false;
    } else {
        recordBtn.disabled = true;
        autoRecordBtn.disabled = true;
    }
});

recordBtn.addEventListener('click', saveSample);

autoRecordBtn.addEventListener('click', () => {
    if (autoRecordInterval) {
        clearInterval(autoRecordInterval);
        autoRecordInterval = null;
        autoRecordBtn.textContent = 'Tu dong Luu';
        autoRecordBtn.classList.remove('btn-danger');
        autoRecordBtn.classList.add('btn-secondary');
        recordBtn.disabled = false;
        if (recIndicator) recIndicator.classList.remove('on');
    } else {
        if (!labelSelect.value) return;
        autoRecordInterval = setInterval(() => {
            if (currentLandmarks) saveSample();
        }, 500);

        autoRecordBtn.textContent = 'Dung tu dong';
        autoRecordBtn.classList.remove('btn-secondary');
        autoRecordBtn.classList.add('btn-danger');
        recordBtn.disabled = true;
        if (recIndicator) recIndicator.classList.add('on');
    }
});

initMediaPipe();
loadLabels();
loadDatasetStats();
