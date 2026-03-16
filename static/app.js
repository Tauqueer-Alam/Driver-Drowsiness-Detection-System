const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const alarmDelaySlider = document.getElementById('delay-slider');
const alarmDelayVal = document.getElementById('delay-val');
const earThreshSlider = document.getElementById('ear-slider');
const earThreshVal = document.getElementById('ear-val');
const modeSelect = document.getElementById('mode-select');

const cnnStatus = document.getElementById('cnn-status');
const earValue = document.getElementById('ear-value');
const frameCountEl = document.getElementById('frame-count');
const statusOverlay = document.getElementById('status-overlay');
const alarmSound = document.getElementById('alarm-sound');

let isRunning = false;
let frameCount = 0;
let alarmDelay = 10;
let earThresh = 0.25;
let isAlarmOn = false;

alarmDelaySlider.addEventListener('input', (e) => {
    alarmDelay = parseInt(e.target.value);
    alarmDelayVal.innerText = alarmDelay;
});

earThreshSlider.addEventListener('input', (e) => {
    earThresh = parseFloat(e.target.value);
    earThreshVal.innerText = earThresh.toFixed(2);
});

async function startCamera() {
    // Unlock Audio Context on user interaction
    alarmSound.play().then(() => {
        alarmSound.pause();
        alarmSound.currentTime = 0;
    }).catch(e => console.log("Audio unlock failed (user may not have interacted yet)", e));

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = stream;
        isRunning = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            statusOverlay.innerText = "MONITORING";
            statusOverlay.className = "status-overlay monitoring";
            processFrame();
        };
    } catch (err) {
        alert("Camera access denied or unavailable.");
    }
}

function stopCamera() {
    isRunning = false;
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stopAlarm();
    statusOverlay.innerText = "SYSTEM IDLE";
    statusOverlay.className = "status-overlay idle";

    cnnStatus.innerText = "N/A";
    earValue.innerText = "0.00";
    frameCount = 0;
    frameCountEl.innerText = "0";
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

async function processFrame() {
    if (!isRunning) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.8);

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Image,
                mode: modeSelect.value,
                ear_thresh: earThresh
            })
        });

        const data = await response.json();

        if (data.image) {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = data.image;
        }

        if (data.status) {
            cnnStatus.innerText = data.status;
            earValue.innerText = data.ear.toFixed(2);

            if (data.trigger) {
                frameCount++;
            } else {
                frameCount = Math.max(0, frameCount - 2);
            }

            frameCountEl.innerText = frameCount;

            if (frameCount >= alarmDelay && !isAlarmOn) {
                triggerAlarm();
            } else if (frameCount <= 0 && isAlarmOn) {
                stopAlarm();
            }
        }
    } catch (err) {
        console.error("Error processing frame", err);
    }

    // throttle FPS request to avoid overwhelming backend on older phones
    if (isRunning) {
        setTimeout(processFrame, 50);
    }
}

function triggerAlarm() {
    isAlarmOn = true;
    statusOverlay.innerText = "DROWSY!";
    statusOverlay.className = "status-overlay danger";
    alarmSound.play().catch(e => console.log("Audio play blocked by browser policies", e));
}

function stopAlarm() {
    if (isAlarmOn) {
        isAlarmOn = false;
        statusOverlay.innerText = "MONITORING";
        statusOverlay.className = "status-overlay monitoring";
        alarmSound.pause();
        alarmSound.currentTime = 0;
    }
}
