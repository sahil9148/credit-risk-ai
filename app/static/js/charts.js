// Credit Risk AI - Chart Rendering

// Human-readable feature names
const FEATURE_LABELS = {
    'age': 'Age',
    'annual_income': 'Annual Income',
    'employment_length': 'Employment Length',
    'employment_type': 'Employment Type',
    'home_ownership': 'Home Ownership',
    'existing_debt': 'Existing Debt',
    'num_credit_lines': 'Credit Lines',
    'credit_history_length': 'Credit History',
    'late_payments': 'Late Payments',
    'bankruptcies': 'Bankruptcies',
    'loan_amount': 'Loan Amount',
    'loan_purpose': 'Loan Purpose',
    'loan_tenure': 'Loan Tenure',
    'debt_to_income_ratio': 'Debt-to-Income Ratio'
};

// Render SHAP Waterfall Chart
function renderShapChart(canvasId, shapData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !shapData) return;
    
    // Sort by absolute SHAP value
    const entries = Object.entries(shapData)
        .map(([key, value]) => ({ feature: FEATURE_LABELS[key] || key, value: parseFloat(value) }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 12); // Top 12 features
    
    const labels = entries.map(e => e.feature);
    const values = entries.map(e => e.value);
    const colors = values.map(v => v > 0 ? 'rgba(220, 38, 38, 0.8)' : 'rgba(5, 150, 105, 0.8)');
    const borderColors = values.map(v => v > 0 ? 'rgba(220, 38, 38, 1)' : 'rgba(5, 150, 105, 1)');
    
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Impact on Risk Score',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.parsed.x;
                            const direction = val > 0 ? '↑ Increases Risk' : '↓ Decreases Risk';
                            return `${direction}: ${Math.abs(val).toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: '← Reduces Risk          Increases Risk →', font: { size: 12 } },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// Render Risk Distribution Doughnut Chart (Admin)
function renderRiskDistribution(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    
    new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
            datasets: [{
                data: [data.low || 0, data.medium || 0, data.high || 0, data.very_high || 0],
                backgroundColor: ['#059669', '#d97706', '#ea580c', '#dc2626'],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 20, usePointStyle: true }
                }
            },
            cutout: '65%'
        }
    });
}

// Render Application Trend Line Chart (Admin)
function renderApplicationTrend(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: data.labels || [],
            datasets: [{
                label: 'Applications',
                data: data.values || [],
                borderColor: '#1e40af',
                backgroundColor: 'rgba(30, 64, 175, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#1e40af',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { stepSize: 1 },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

// Render Feature Importance Bar Chart
function renderFeatureImportance(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    
    const entries = Object.entries(data)
        .map(([key, value]) => ({ feature: FEATURE_LABELS[key] || key, value: Math.abs(parseFloat(value)) }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10);
    
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: entries.map(e => e.feature),
            datasets: [{
                label: 'Feature Importance',
                data: entries.map(e => e.value),
                backgroundColor: 'rgba(30, 64, 175, 0.7)',
                borderColor: 'rgba(30, 64, 175, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: 'Absolute SHAP Value' },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                y: { grid: { display: false } }
            }
        }
    });
}
