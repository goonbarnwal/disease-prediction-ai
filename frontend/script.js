// ============================================
// DISEASE PREDICTION AI - FRONTEND LOGIC
// Save as: Frontend/script.js
// ============================================

// Global variables
let selectedSymptoms = [];
const BACKEND_URL = 'https://disease-prediction-ai-backend.onrender.com';
// DOM Elements
const symptomInput = document.getElementById('symptomInput');
const selectedSymptomsDiv = document.getElementById('selectedSymptoms');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingDiv = document.getElementById('loading');
const resultsCard = document.getElementById('resultsCard');
const predictionResult = document.getElementById('predictionResult');
const alternativesSection = document.getElementById('alternativesSection');
const alternativesList = document.getElementById('alternativesList');

// ============================================
// SYMPTOMS MANAGEMENT
// ============================================

// Add symptom from input
function addSymptom() {
    const symptom = symptomInput.value.trim();
    if (!symptom) return;
    
    processSymptom(symptom);
    symptomInput.value = '';
    symptomInput.focus();
}

// Add quick symptom
function addQuickSymptom(symptom) {
    processSymptom(symptom);
}

// Process and add symptom
function processSymptom(symptom) {
    // Convert to lowercase and replace spaces with underscores
    const formattedSymptom = symptom.toLowerCase().replace(/\s+/g, '_');
    
    // Check if already added
    if (selectedSymptoms.includes(formattedSymptom)) {
        showNotification('Symptom already added', 'info');
        return;
    }
    
    // Add to array
    selectedSymptoms.push(formattedSymptom);
    
    // Update display
    updateSelectedSymptomsDisplay();
    
    // Enable analyze button if enough symptoms
    updateAnalyzeButton();
}

// Remove symptom
function removeSymptom(symptom) {
    selectedSymptoms = selectedSymptoms.filter(s => s !== symptom);
    updateSelectedSymptomsDisplay();
    updateAnalyzeButton();
}

// Update selected symptoms display
function updateSelectedSymptomsDisplay() {
    selectedSymptomsDiv.innerHTML = '';
    
    if (selectedSymptoms.length === 0) {
        selectedSymptomsDiv.innerHTML = '<span class="placeholder">No symptoms selected yet</span>';
        return;
    }
    
    selectedSymptoms.forEach(symptom => {
        const tag = document.createElement('div');
        tag.className = 'symptom-tag';
        tag.innerHTML = `
            ${symptom.replace(/_/g, ' ')}
            <span onclick="removeSymptom('${symptom}')" class="remove-tag">×</span>
        `;
        selectedSymptomsDiv.appendChild(tag);
    });
}

// Update analyze button state
function updateAnalyzeButton() {
    analyzeBtn.disabled = selectedSymptoms.length === 0;
}

// Handle Enter key in input
symptomInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        addSymptom();
    }
});

// ============================================
// API CALLS
// ============================================

// Main analysis function
async function analyzeSymptoms() {
    if (selectedSymptoms.length === 0) {
        showNotification('Please add at least one symptom', 'error');
        return;
    }
    
    // Show loading
    loadingDiv.style.display = 'flex';
    analyzeBtn.disabled = true;
    resultsCard.style.display = 'none';
    
    try {
        // Call backend API
        const response = await fetch(`${BACKEND_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symptoms: selectedSymptoms
            })
        });
        
        const data = await response.json();
        
        // Hide loading
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
        
        if (data.status === 'success') {
            displayResults(data);
            showNotification('Analysis complete!', 'success');
        } else {
            showNotification(data.error || 'Analysis failed', 'error');
            console.error('API Error:', data);
        }
        
    } catch (error) {
        console.error('Network Error:', error);
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
        showNotification('Cannot connect to server. Please check backend.', 'error');
    }
}

// ============================================
// RESULTS DISPLAY
// ============================================

// Display results
function displayResults(data) {
    const prediction = data.prediction;
    
    // Format disease name for display
    const diseaseName = prediction.disease
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
    
    // Create main result HTML
    predictionResult.innerHTML = `
        <div class="disease-name">
            <i class="fas fa-disease"></i>
            ${diseaseName}
        </div>
        
        <div class="prediction-meta">
            <div class="confidence-badge">
                <i class="fas fa-chart-line"></i>
                ${prediction.confidence}% Confidence
            </div>
            <div class="risk-badge ${prediction.risk_level.toLowerCase()}">
                <i class="fas fa-exclamation-triangle"></i>
                ${prediction.risk_level} Risk
            </div>
        </div>
        
        <div class="description-box">
            <h3><i class="fas fa-file-medical-alt"></i> Description</h3>
            <p>${prediction.description}</p>
        </div>
        
        <div class="precautions-box">
            <h3><i class="fas fa-shield-alt"></i> Precautions & Recommendations</h3>
            <ul class="precautions-list">
                ${prediction.precautions.map(p => `
                    <li><i class="fas fa-check-circle"></i> ${p}</li>
                `).join('')}
                <li><i class="fas fa-user-md"></i> ${prediction.recommendation}</li>
            </ul>
        </div>
    `;
    
    // Display alternative predictions if available
    if (data.alternatives && data.alternatives.length > 0) {
        displayAlternatives(data.alternatives);
        alternativesSection.style.display = 'block';
    } else {
        alternativesSection.style.display = 'none';
    }
    
    // Show results section
    resultsCard.style.display = 'block';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display alternative predictions
function displayAlternatives(alternatives) {
    alternativesList.innerHTML = '';
    
    alternatives.forEach(alt => {
        const altName = alt.disease.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        const altItem = document.createElement('div');
        altItem.className = 'alternative-item';
        altItem.innerHTML = `
            <h4>${altName}</h4>
            <div class="alt-probability">
                <i class="fas fa-percentage"></i> ${alt.probability}% Probability
            </div>
        `;
        
        alternativesList.appendChild(altItem);
    });
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notification
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Reset analysis
function resetAnalysis() {
    selectedSymptoms = [];
    updateSelectedSymptomsDisplay();
    updateAnalyzeButton();
    resultsCard.style.display = 'none';
    symptomInput.value = '';
    symptomInput.focus();
    showNotification('Ready for new analysis', 'info');
}

// ============================================
// INITIALIZATION
// ============================================

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set focus to input
    symptomInput.focus();
    
    // Check backend connection
    checkBackendConnection();
    
    // Welcome message
    setTimeout(() => {
        showNotification('Welcome! Add symptoms to get AI-powered diagnosis', 'info');
    }, 1000);
});

// Check backend connection
async function checkBackendConnection() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('✅ Backend connected:', data);
            showNotification('AI system ready', 'success');
        }
    } catch (error) {
        console.warn('⚠️ Backend not connected:', error);
        showNotification('Backend not connected. Please start the server.', 'error');
    }
}

// Add notification styles dynamically
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 1000;
        transform: translateX(150%);
        transition: transform 0.3s ease-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 350px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification.success {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        border-left: 4px solid #2E7D32;
    }
    
    .notification.error {
        background: linear-gradient(135deg, #f44336 0%, #c62828 100%);
        border-left: 4px solid #c62828;
    }
    
    .notification.info {
        background: linear-gradient(135deg, #2196F3 0%, #0d47a1 100%);
        border-left: 4px solid #0d47a1;
    }
    
    .notification i {
        font-size: 1.2rem;
    }
`;
document.head.appendChild(notificationStyles);