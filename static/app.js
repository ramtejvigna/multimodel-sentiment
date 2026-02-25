// Multimodal Sentiment Analysis - Frontend JavaScript

// Global variables
let webcamStream = null;
let lastCapturedImage = null;

// DOM Elements
const textInput = document.getElementById('textInput');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startWebcamBtn = document.getElementById('startWebcamBtn');
const captureBtn = document.getElementById('captureBtn');
const analyzeTextBtn = document.getElementById('analyzeTextBtn');
const analyzeMultimodalBtn = document.getElementById('analyzeMultimodalBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    startWebcamBtn.addEventListener('click', toggleWebcam);
    captureBtn.addEventListener('click', captureAndAnalyze);
    analyzeTextBtn.addEventListener('click', analyzeText);
    analyzeMultimodalBtn.addEventListener('click', analyzeMultimodal);
    
    textInput.addEventListener('input', updateMultimodalButton);
}

// Check server status
async function checkServerStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'ready') {
            showStatus('System ready!', 'success');
        } else {
            showStatus('Models not initialized. Please train models first.', 'error');
        }
    } catch (error) {
        showStatus('Cannot connect to server', 'error');
    }
}

// Webcam Functions
async function toggleWebcam() {
    if (webcamStream) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        webcam.srcObject = webcamStream;
        
        startWebcamBtn.textContent = 'Stop Webcam';
        startWebcamBtn.classList.remove('btn-secondary');
        startWebcamBtn.classList.add('btn-primary');
        captureBtn.disabled = false;
        
        updateMultimodalButton();
    } catch (error) {
        showStatus('Error accessing webcam: ' + error.message, 'error');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        webcamStream = null;
        
        startWebcamBtn.textContent = 'Start Webcam';
        startWebcamBtn.classList.remove('btn-primary');
        startWebcamBtn.classList.add('btn-secondary');
        captureBtn.disabled = true;
        
        updateMultimodalButton();
    }
}

function captureFrame() {
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);
    return canvas.toDataURL('image/jpeg');
}

// Analysis Functions
async function analyzeText() {
    const text = textInput.value.trim();
    
    if (!text) {
        showStatus('Please enter some text', 'error');
        return;
    }
    
    showStatus('Analyzing text...', 'info');
    
    try {
        const response = await fetch('/api/analyze/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showStatus('Error: ' + result.error, 'error');
        } else {
            displayTextResults(result);
            showStatus('Text analysis complete!', 'success');
        }
    } catch (error) {
        showStatus('Error analyzing text: ' + error.message, 'error');
    }
}

async function captureAndAnalyze() {
    const imageData = captureFrame();
    lastCapturedImage = imageData;
    
    showStatus('Analyzing facial emotion...', 'info');
    
    try {
        const response = await fetch('/api/analyze/face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.error) {
            showStatus('Error: ' + result.error, 'error');
        } else {
            displayFaceResults(result);
            showStatus('Face analysis complete!', 'success');
        }
        
        updateMultimodalButton();
    } catch (error) {
        showStatus('Error analyzing face: ' + error.message, 'error');
    }
}

async function analyzeMultimodal() {
    const text = textInput.value.trim();
    
    if (!text || !lastCapturedImage) {
        showStatus('Please provide both text and capture your face', 'error');
        return;
    }
    
    showStatus('Performing multimodal analysis...', 'info');
    
    try {
        // Convert base64 to blob
        const blob = await (await fetch(lastCapturedImage)).blob();
        
        const formData = new FormData();
        formData.append('text', text);
        formData.append('image', blob, 'capture.jpg');
        
        const response = await fetch('/api/analyze/multimodal', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            showStatus('Error: ' + result.error, 'error');
        } else {
            displayMultimodalResults(result);
            showStatus('Multimodal analysis complete!', 'success');
        }
    } catch (error) {
        showStatus('Error analyzing multimodal: ' + error.message, 'error');
    }
}

// Display Functions
function displayTextResults(result) {
    const container = document.getElementById('textResults');
    const prediction = document.getElementById('textPrediction');
    const confidence = document.getElementById('textConfidence');
    const confidenceBar = document.getElementById('textConfidenceBar');
    const probabilities = document.getElementById('textProbabilities');
    
    prediction.textContent = result.prediction;
    prediction.className = 'prediction-value ' + result.prediction;
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidence.textContent = confidencePercent + '%';
    confidenceBar.style.width = confidencePercent + '%';
    
    probabilities.innerHTML = '';
    for (const [label, prob] of Object.entries(result.probabilities)) {
        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <span class="probability-label">${label}</span>
            <span class="probability-value">${(prob * 100).toFixed(1)}%</span>
        `;
        probabilities.appendChild(item);
    }
    
    container.style.display = 'block';
}

function displayFaceResults(result) {
    const container = document.getElementById('faceResults');
    const prediction = document.getElementById('facePrediction');
    const confidence = document.getElementById('faceConfidence');
    const confidenceBar = document.getElementById('faceConfidenceBar');
    const probabilities = document.getElementById('faceProbabilities');
    
    if (result.error) {
        prediction.textContent = 'Error: ' + result.error;
        prediction.className = 'prediction-value neutral';
        confidence.textContent = '0%';
        confidenceBar.style.width = '0%';
        probabilities.innerHTML = '';
    } else {
        prediction.textContent = result.prediction;
        prediction.className = 'prediction-value ' + getSentimentClass(result.prediction);
        
        const confidencePercent = (result.confidence * 100).toFixed(1);
        confidence.textContent = confidencePercent + '%';
        confidenceBar.style.width = confidencePercent + '%';
        
        probabilities.innerHTML = '';
        for (const [label, prob] of Object.entries(result.probabilities)) {
            const item = document.createElement('div');
            item.className = 'probability-item';
            item.innerHTML = `
                <span class="probability-label">${label}</span>
                <span class="probability-value">${(prob * 100).toFixed(1)}%</span>
            `;
            probabilities.appendChild(item);
        }
    }
    
    container.style.display = 'block';
}

function displayMultimodalResults(result) {
    const container = document.getElementById('multimodalResults');
    const prediction = document.getElementById('multimodalPrediction');
    const confidence = document.getElementById('multimodalConfidence');
    const confidenceBar = document.getElementById('multimodalConfidenceBar');
    const agreement = document.getElementById('agreementStatus');
    
    prediction.textContent = result.multimodal_prediction;
    prediction.className = 'prediction-value ' + getSentimentClass(result.multimodal_prediction);
    
    const confidencePercent = (result.multimodal_confidence * 100).toFixed(1);
    confidence.textContent = confidencePercent + '%';
    confidenceBar.style.width = confidencePercent + '%';
    
    if (result.agreement) {
        agreement.textContent = '✓ Text and face predictions agree';
        agreement.className = 'agreement agree';
    } else {
        agreement.textContent = '⚠ Text and face predictions differ';
        agreement.className = 'agreement disagree';
    }
    
    // Also display individual results
    if (result.text_analysis) {
        displayTextResults(result.text_analysis);
    }
    if (result.face_analysis) {
        displayFaceResults(result.face_analysis);
    }
    
    container.style.display = 'block';
}

// Utility Functions
function getSentimentClass(emotion) {
    const positiveEmotions = ['happy', 'surprise', 'positive'];
    const negativeEmotions = ['angry', 'sad', 'fear', 'disgust', 'negative'];
    
    if (positiveEmotions.includes(emotion.toLowerCase())) {
        return 'positive';
    } else if (negativeEmotions.includes(emotion.toLowerCase())) {
        return 'negative';
    }
    return 'neutral';
}

function updateMultimodalButton() {
    const hasText = textInput.value.trim().length > 0;
    const hasImage = lastCapturedImage !== null;
    analyzeMultimodalBtn.disabled = !(hasText && hasImage);
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.className = 'status-message ' + type;
    statusDiv.style.display = 'block';
    
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 5000);
}
