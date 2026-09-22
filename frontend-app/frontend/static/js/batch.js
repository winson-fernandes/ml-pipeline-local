let currentResults = null;

document.getElementById('csvFile').addEventListener('change', (e) => {
    const fileName = e.target.files[0]?.name || '';
    document.getElementById('fileName').textContent = fileName ? `Selected: ${fileName}` : '';
});

document.getElementById('csvForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    
    try {
        const response = await fetch('/api/predict/csv', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        currentResults = result;
        displayResults(result);
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

function displayResults(result) {
    const summaryDiv = document.getElementById('summary');
    const resultsBody = document.getElementById('resultsBody');
    
    // Display summary
    summaryDiv.innerHTML = `
        <div class="summary-card">
            <h3>${result.total_patients}</h3>
            <p>Total Patients</p>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
            <h3>${result.summary.disease_detected}</h3>
            <p>Disease Detected</p>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h3>${result.summary.no_disease}</h3>
            <p>No Disease</p>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);">
            <h3>${result.summary.high_risk}</h3>
            <p>High Risk</p>
        </div>
    `;
    
    // Display table
    resultsBody.innerHTML = '';
    result.predictions.forEach(pred => {
        const row = document.createElement('tr');
        const riskClass = getRiskClass(pred.risk_level);
        
        row.innerHTML = `
            <td>${pred.row + 1}</td>
            <td>${pred.patient_data.age}</td>
            <td>${pred.patient_data.sex === 1 ? 'M' : 'F'}</td>
            <td>${pred.prediction === 1 ? '⚠️ Disease' : '✅ No Disease'}</td>
            <td>${(pred.probability * 100).toFixed(1)}%</td>
            <td><span class="risk-badge ${riskClass}">${pred.risk_level}</span></td>
            <td style="max-width: 300px;">${pred.interpretation}</td>
        `;
        resultsBody.appendChild(row);
    });
    
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
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

function resetUpload() {
    document.getElementById('csvForm').reset();
    document.getElementById('fileName').textContent = '';
    document.getElementById('results').style.display = 'none';
    currentResults = null;
}

function downloadSample() {
    const sampleData = `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,3,145,233,1,0,150,0,2.3,0,0,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2
41,0,1,130,204,0,0,172,0,1.4,2,0,2
56,1,1,120,236,0,1,178,0,0.8,2,0,2
57,0,0,120,354,0,1,163,1,0.6,2,0,2`;
    
    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_heart_disease.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function downloadResults() {
    if (!currentResults) {
        alert('No results to download');
        return;
    }
    
    // Create CSV header
    let csv = 'Row,Age,Sex,Prediction,Probability,Risk Level,Interpretation\n';
    
    // Add data rows
    currentResults.predictions.forEach(pred => {
        csv += `${pred.row + 1},`;
        csv += `${pred.patient_data.age},`;
        csv += `${pred.patient_data.sex === 1 ? 'Male' : 'Female'},`;
        csv += `${pred.prediction === 1 ? 'Disease' : 'No Disease'},`;
        csv += `${(pred.probability * 100).toFixed(1)}%,`;
        csv += `${pred.risk_level},`;
        csv += `"${pred.interpretation}"\n`;
    });
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `heart_disease_predictions_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
