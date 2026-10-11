let HandLandmarker, FaceLandmarker, FilesetResolver, DrawingUtils;
let handLandmarker, faceLandmarker;
let webcamRunning = false;
let socket;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.querySelector("#loading-overlay p");

const currentPredictionEl = document.getElementById("current-prediction");
const confidenceTextEl = document.getElementById("confidence-text");
const confidenceBarEl = document.getElementById("confidence-bar");
const topPredictionsEl = document.getElementById("top-predictions");
const socketStatusDot = document.getElementById("socket-status-dot");
const socketStatusText = document.getElementById("socket-status-text");
const fpsCounter = document.getElementById("fps-counter");
const connectionPill = document.getElementById("connection-pill");
const handsCounter = document.getElementById("hands-counter");

const wordDisplay = document.getElementById("word-display");
const speakBtn = document.getElementById("speak-btn");
const spaceBtn = document.getElementById("space-btn");
const clearBtn = document.getElementById("clear-btn");
const historyBadges = document.getElementById("history-badges");

// Face expression elements
const faceExprEl = document.getElementById("face-expression");
const faceConfEl = document.getElementById("face-confidence");
const faceBarEl = document.getElementById("face-bar");
const faceIconEl = document.getElementById("face-icon");
const faceStatusEl = document.getElementById("face-status");

let currentWord = "";
let stablePrediction = "";
let stableCount = 0;
let STABLE_THRESHOLD = 15;
let CONF_THRESHOLD = 0.6;
let predictionHistory = [];

const HAND_COLORS = [
    { conn: "#00e5bf", lm: "#ffffff" },
    { conn: "#f472b6", lm: "#fef08a" }
];

// Expression mapping
const EXPRESSIONS = {
    happy: { vi: "Vui", icon: "^_^", color: "#10b981" },
    sad: { vi: "Buon", icon: "T_T", color: "#6366f1" },
    surprised: { vi: "Ngac nhien", icon: "O_O", color: "#f59e0b" },
    angry: { vi: "Tuc gian", icon: ">_<", color: "#ef4444" },
    fear: { vi: "So hai", icon: "o_o", color: "#8b5cf6" },
    neutral: { vi: "Binh thuong", icon: "-_-", color: "#64748b" }
};

socket = io();

socket.on('connect', () => {
    if (socketStatusDot) socketStatusDot.className = "status-indicator connected";
    if (socketStatusText) socketStatusText.textContent = "Da ket noi";
    if (connectionPill) connectionPill.className = "status-pill online";
});

socket.on('disconnect', () => {
    if (socketStatusDot) socketStatusDot.className = "status-indicator";
    if (socketStatusText) socketStatusText.textContent = "Mat ket noi";
    if (connectionPill) connectionPill.className = "status-pill offline";
});

const SMOOTHING_WINDOW = 5;
let predictionBuffer = [];

function smoothPrediction(data) {
    predictionBuffer.push(data);
    if (predictionBuffer.length > SMOOTHING_WINDOW) predictionBuffer.shift();
    
    // Find most common label in buffer
    const counts = {};
    let bestLabel = data.label, bestCount = 0;
    for (const p of predictionBuffer) {
        counts[p.label] = (counts[p.label] || 0) + 1;
        if (counts[p.label] > bestCount) {
            bestCount = counts[p.label];
            bestLabel = p.label;
        }
    }
    
    // Average confidence for the best label
    const relevantPreds = predictionBuffer.filter(p => p.label === bestLabel);
    const avgConf = relevantPreds.reduce((s, p) => s + p.confidence, 0) / relevantPreds.length;
    
    return { ...data, label: bestLabel, confidence: avgConf, raw_label: data.label };
}

socket.on('prediction_result', (data) => {
    if (data.label && data.label !== 'unknown') {
        const smoothed = smoothPrediction(data);
        updatePredictionUI(smoothed);
        handleWordBuilder(smoothed);
    } else {
        // Handle unknown/low confidence
        updatePredictionUI({ label: '?', confidence: 0, top_5: data.top_5 || [] });
    }
});

async function initMediaPipe() {
    try {
        if (loadingText) loadingText.textContent = "Dang tai MediaPipe Vision...";
        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs');
        HandLandmarker = vision.HandLandmarker;
        FaceLandmarker = vision.FaceLandmarker;
        FilesetResolver = vision.FilesetResolver;
        DrawingUtils = vision.DrawingUtils;

        const visionWasm = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
        );

        if (loadingText) loadingText.textContent = "Dang tai Hand Landmarker...";
        handLandmarker = await HandLandmarker.createFromOptions(visionWasm, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });

        if (loadingText) loadingText.textContent = "Dang tai Face Landmarker...";
        faceLandmarker = await FaceLandmarker.createFromOptions(visionWasm, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true
        });

        if (loadingOverlay) loadingOverlay.style.display = "none";
        startWebcam();
    } catch (error) {
        console.error(error);
        if (loadingText) loadingText.textContent = "Loi khoi tao: " + error.message;
        showToast("Loi khoi tao MediaPipe", "error");
    }
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
    } catch (err) {
        showToast("Loi truy cap webcam", "error");
    }
}

let lastVideoTime = -1;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastEmitTime = 0;
const EMIT_INTERVAL = 100; // ms, ~10 FPS

function buildLandmarksPayload(handResults) {
    const LANDMARKS_PER_HAND = 21;
    const zeroHand = Array.from({ length: LANDMARKS_PER_HAND }, () => [0, 0, 0]);
    let leftHand = zeroHand, rightHand = zeroHand;

    if (handResults.landmarks && handResults.handednesses) {
        for (let i = 0; i < handResults.landmarks.length; i++) {
            const handedness = handResults.handednesses[i]?.[0]?.categoryName;
            const lm = handResults.landmarks[i].map(l => [l.x, l.y, l.z]);
            // MediaPipe reports mirrored handedness, so "Left" in output = user's right hand
            if (handedness === 'Left') {
                rightHand = lm;
            } else {
                leftHand = lm;
            }
        }
    }
    return leftHand.concat(rightHand);
}

// ── Face Expression Analysis using Blendshapes ──
function analyzeFaceExpression(faceResults) {
    if (!faceResults || !faceResults.faceBlendshapes || faceResults.faceBlendshapes.length === 0) {
        return { expression: "neutral", confidence: 0, detected: false };
    }

    const bs = {};
    faceResults.faceBlendshapes[0].categories.forEach(c => {
        bs[c.categoryName] = c.score;
    });

    // Score each expression
    const scores = {
        happy: (bs.mouthSmileLeft || 0) * 0.4 + (bs.mouthSmileRight || 0) * 0.4 + (bs.cheekSquintLeft || 0) * 0.1 + (bs.cheekSquintRight || 0) * 0.1,
        surprised: (bs.jawOpen || 0) * 0.3 + (bs.eyeWideLeft || 0) * 0.2 + (bs.eyeWideRight || 0) * 0.2 + (bs.browOuterUpLeft || 0) * 0.15 + (bs.browOuterUpRight || 0) * 0.15,
        angry: (bs.browDownLeft || 0) * 0.3 + (bs.browDownRight || 0) * 0.3 + (bs.mouthFrownLeft || 0) * 0.15 + (bs.mouthFrownRight || 0) * 0.15 + (bs.noseSneerLeft || 0) * 0.05 + (bs.noseSneerRight || 0) * 0.05,
        sad: (bs.mouthFrownLeft || 0) * 0.3 + (bs.mouthFrownRight || 0) * 0.3 + (bs.browInnerUp || 0) * 0.2 + (bs.mouthPucker || 0) * 0.2,
        fear: (bs.eyeWideLeft || 0) * 0.25 + (bs.eyeWideRight || 0) * 0.25 + (bs.browInnerUp || 0) * 0.2 + (bs.jawOpen || 0) * 0.15 + (bs.mouthFrownLeft || 0) * 0.075 + (bs.mouthFrownRight || 0) * 0.075
    };

    let bestExpr = "neutral";
    let bestScore = 0.15; // Minimum threshold

    for (const [expr, score] of Object.entries(scores)) {
        if (score > bestScore) {
            bestScore = score;
            bestExpr = expr;
        }
    }

    return { expression: bestExpr, confidence: Math.min(bestScore * 2, 1), detected: true };
}

function drawFaceMesh(faceResults) {
    if (!faceResults || !faceResults.faceLandmarks || faceResults.faceLandmarks.length === 0) return;

    const landmarks = faceResults.faceLandmarks[0];
    const drawingUtils = new DrawingUtils(canvasCtx);

    // Draw face oval
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
        color: "rgba(0, 229, 191, 0.3)", lineWidth: 1
    });
    // Draw lips
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
        color: "rgba(244, 114, 182, 0.5)", lineWidth: 1
    });
    // Draw eyes
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
        color: "rgba(56, 189, 248, 0.4)", lineWidth: 1
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
        color: "rgba(56, 189, 248, 0.4)", lineWidth: 1
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
        color: "rgba(245, 158, 11, 0.4)", lineWidth: 1
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
        color: "rgba(245, 158, 11, 0.4)", lineWidth: 1
    });
}

function updateFaceUI(faceData) {
    if (!faceExprEl) return;

    if (!faceData.detected) {
        faceExprEl.textContent = "--";
        faceExprEl.style.color = "var(--text-muted)";
        if (faceConfEl) faceConfEl.textContent = "0%";
        if (faceBarEl) faceBarEl.style.width = "0%";
        if (faceIconEl) faceIconEl.textContent = "-_-";
        if (faceStatusEl) faceStatusEl.textContent = "Khong thay mat";
        if (faceStatusEl) faceStatusEl.style.color = "var(--danger)";
        return;
    }

    const expr = EXPRESSIONS[faceData.expression] || EXPRESSIONS.neutral;
    faceExprEl.textContent = expr.vi;
    faceExprEl.style.color = expr.color;

    if (faceIconEl) {
        faceIconEl.textContent = expr.icon;
        faceIconEl.style.color = expr.color;
    }
    if (faceStatusEl) {
        faceStatusEl.textContent = "Dang phan tich";
        faceStatusEl.style.color = "var(--success)";
    }

    const confPct = (faceData.confidence * 100).toFixed(0);
    if (faceConfEl) faceConfEl.textContent = confPct + "%";
    if (faceBarEl) {
        faceBarEl.style.width = confPct + "%";
        faceBarEl.style.background = `linear-gradient(90deg, ${expr.color}, ${expr.color}88)`;
    }
}

function predictWebcam() {
    if (!webcamRunning) return;

    if (video.videoWidth > 0 && canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let now = performance.now();
    frameCount++;
    if (now - lastFpsTime >= 1000) {
        if (fpsCounter) fpsCounter.textContent = `${frameCount}`;
        frameCount = 0;
        lastFpsTime = now;
    }

    if (lastVideoTime !== video.currentTime && handLandmarker) {
        lastVideoTime = video.currentTime;

        // Hand detection
        const handResults = handLandmarker.detectForVideo(video, now);
        // Face detection
        let faceResults = null;
        if (faceLandmarker) {
            faceResults = faceLandmarker.detectForVideo(video, now);
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Draw face mesh first (behind hands)
        if (faceResults) {
            drawFaceMesh(faceResults);
            const faceData = analyzeFaceExpression(faceResults);
            updateFaceUI(faceData);
        }

        // Draw hands
        const numHands = handResults.landmarks ? handResults.landmarks.length : 0;
        if (handsCounter) handsCounter.textContent = `${numHands}`;

        if (numHands > 0) {
            const drawingUtils = new DrawingUtils(canvasCtx);
            for (let i = 0; i < numHands; i++) {
                const color = HAND_COLORS[i] || HAND_COLORS[0];
                drawingUtils.drawConnectors(handResults.landmarks[i], HandLandmarker.HAND_CONNECTIONS, { color: color.conn, lineWidth: 3 });
                drawingUtils.drawLandmarks(handResults.landmarks[i], { color: color.lm, lineWidth: 2, radius: 3 });
            }

            const payload = buildLandmarksPayload(handResults);
            if (socket.connected) {
                if (now - lastEmitTime >= EMIT_INTERVAL) {
                    socket.emit('predict_frame', { landmarks: payload });
                    lastEmitTime = now;
                }
            }
        } else {
            currentPredictionEl.textContent = "\u2014";
            currentPredictionEl.classList.add('empty');
            confidenceTextEl.textContent = "0%";
            confidenceBarEl.style.width = "0%";
            if (topPredictionsEl) topPredictionsEl.innerHTML = '<div style="text-align:center;padding:0.5rem 0;"><span class="text-3" style="font-size:0.75rem;">Dua tay vao khung hinh</span></div>';
            stablePrediction = "";
            stableCount = 0;
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}

function updatePredictionUI(data) {
    if (!data.label) return;

    currentPredictionEl.textContent = data.label;
    currentPredictionEl.classList.remove('empty');
    const conf = (data.confidence * 100).toFixed(1);
    confidenceTextEl.textContent = `${conf}%`;
    confidenceBarEl.style.width = `${conf}%`;

    if (data.confidence > 0.8) {
        confidenceBarEl.style.background = 'linear-gradient(90deg, var(--success), #34d399)';
    } else if (data.confidence > 0.5) {
        confidenceBarEl.style.background = 'linear-gradient(90deg, var(--p), var(--s))';
    } else {
        confidenceBarEl.style.background = 'linear-gradient(90deg, var(--warning), #fbbf24)';
    }

    if (data.top_5 && topPredictionsEl) {
        topPredictionsEl.innerHTML = '';
        data.top_5.forEach(pred => {
            const pConf = (pred.confidence * 100).toFixed(1);
            const barWidth = (pred.confidence * 100).toFixed(0);
            const div = document.createElement('div');
            div.className = 'top-item';
            div.innerHTML = `
                <span class="top-lbl">${pred.label}</span>
                <div class="top-bg"><div class="top-fill" style="width:${barWidth}%"></div></div>
                <span class="top-pct">${pConf}%</span>
            `;
            topPredictionsEl.appendChild(div);
        });
    }
}

function handleWordBuilder(data) {
    // Read thresholds from sliders if available
    const confSlider = document.getElementById('conf-threshold');
    const stableSlider = document.getElementById('stable-threshold');
    if (confSlider) CONF_THRESHOLD = parseInt(confSlider.value) / 100;
    if (stableSlider) STABLE_THRESHOLD = parseInt(stableSlider.value);

    if (data.confidence > CONF_THRESHOLD) {
        if (data.label === stablePrediction) {
            stableCount++;
            if (stableCount === STABLE_THRESHOLD) {
                appendChar(data.label);
                addToHistory(data.label);
                currentPredictionEl.style.transform = "scale(1.15)";
                setTimeout(() => currentPredictionEl.style.transform = "scale(1)", 200);
            }
        } else {
            stablePrediction = data.label;
            stableCount = 1;
        }
    } else {
        stableCount = 0;
    }
}

function addToHistory(char) {
    predictionHistory.push(char);
    if (predictionHistory.length > 20) predictionHistory.shift();
    renderHistory();
}

function renderHistory() {
    if (!historyBadges) return;
    if (predictionHistory.length === 0) {
        historyBadges.innerHTML = '<span class="text-3" style="font-size:0.7rem;">\u2014</span>';
        return;
    }
    historyBadges.innerHTML = predictionHistory.map(c =>
        `<span class="hist-i">${c}</span>`
    ).join('');
}

function appendChar(char) {
    currentWord += char;
    updateWordDisplay();
}

function updateWordDisplay() {
    if (wordDisplay) wordDisplay.textContent = currentWord;
    if (speakBtn) speakBtn.disabled = currentWord.trim().length === 0;
    const cc = document.getElementById('word-char-count');
    if (cc) cc.textContent = currentWord.length + ' ky tu';
}

if (spaceBtn) spaceBtn.addEventListener('click', () => appendChar(' '));
if (clearBtn) clearBtn.addEventListener('click', () => { currentWord = ""; updateWordDisplay(); });

if (speakBtn) speakBtn.addEventListener('click', async () => {
    if (!currentWord.trim()) return;
    speakBtn.disabled = true;
    const origText = speakBtn.textContent;
    speakBtn.textContent = 'Dang xu ly...';

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentWord })
        });
        if (!response.ok) throw new Error("TTS failed");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();
    } catch (error) {
        showToast("Loi chuyen doi giong noi", "error");
    } finally {
        speakBtn.disabled = false;
        speakBtn.textContent = origText;
    }
});

initMediaPipe();
