document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = key === 'oldpeak' ? parseFloat(value) : parseInt(value);
    }
    
    showLoading();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResult(result);
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
});

function showLoading() {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('results').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function displayResult(result) {
    const resultsDiv = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    
    const isDiseaseDetected = result.prediction === 1;
    const riskClass = getRiskClass(result.risk_level);
    
    resultContent.innerHTML = `
        <div class="result-box ${isDiseaseDetected ? 'disease' : 'no-disease'}">
            <div class="result-icon">${isDiseaseDetected ? '⚠️' : '✅'}</div>
            <div class="result-title">
                ${isDiseaseDetected ? 'Heart Disease Detected' : 'No Heart Disease Detected'}
            </div>
            <div class="result-subtitle">
                ${result.interpretation}
            </div>
        </div>
        
        <div class="result-details">
            <div class="detail-item">
                <span class="detail-label">Prediction</span>
                <span class="detail-value">${isDiseaseDetected ? 'Disease' : 'No Disease'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Confidence</span>
                <span class="detail-value">${(result.probability * 100).toFixed(1)}%</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Risk Level</span>
                <span class="risk-badge ${riskClass}">${result.risk_level} Risk</span>
            </div>
        </div>
        
        <div class="info-box" style="margin-top: 20px;">
            <h3>📋 Recommendation</h3>
            <p>${getRecommendation(result.risk_level, isDiseaseDetected)}</p>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function getRiskClass(riskLevel) {
    const levels = {
        'Low': 'risk-low',
        'Moderate': 'risk-moderate',
        'High': 'risk-high',
        'Very High': 'risk-very-high'
    };
    return levels[riskLevel] || 'risk-moderate';
}

function getRecommendation(riskLevel, isDiseaseDetected) {
    if (!isDiseaseDetected && riskLevel === 'Low') {
        return 'Your results look good! Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.';
    } else if (riskLevel === 'Moderate') {
        return 'Consider scheduling a check-up with your doctor. Maintain a heart-healthy lifestyle and monitor your symptoms.';
    } else {
        return 'Please consult a cardiologist as soon as possible for a comprehensive evaluation. This prediction should not replace professional medical advice.';
    }
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').style.display = 'none';
}
