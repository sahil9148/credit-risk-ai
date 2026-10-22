// Credit Risk AI - Main Application JS

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(amount);
}

// Format number with commas
function formatNumber(num) {
    return new Intl.NumberFormat('en-IN').format(num);
}

// Animate number counting up
function animateCounter(element, target, duration = 1500) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Show toast notification
function showToast(message, type = 'info') {
    const colors = {
        'info': 'bg-blue-500',
        'success': 'bg-emerald-500',
        'error': 'bg-red-500',
        'warning': 'bg-amber-500'
    };
    
    const toast = document.createElement('div');
    toast.className = `fixed top-4 right-4 z-50 px-6 py-3 rounded-lg text-white shadow-lg ${colors[type]} transform transition-all duration-300 translate-x-full`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.classList.remove('translate-x-full'), 10);
    setTimeout(() => {
        toast.classList.add('translate-x-full');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Credit Score Gauge Animation
function initScoreGauge(score, maxScore = 1000) {
    const gauge = document.getElementById('scoreGauge');
    if (!gauge) return;
    
    const circle = gauge.querySelector('.gauge-fill');
    const scoreText = gauge.querySelector('.score-number');
    if (!circle || !scoreText) return;
    
    const radius = 90;
    const circumference = 2 * Math.PI * radius;
    const percentage = score / maxScore;
    const offset = circumference * (1 - percentage);
    
    circle.style.strokeDasharray = circumference;
    circle.style.strokeDashoffset = circumference;
    
    // Determine color class
    let colorClass = 'gauge-very-high';
    let textColorClass = 'score-very-high';
    if (score >= 750) { colorClass = 'gauge-low'; textColorClass = 'score-low'; }
    else if (score >= 600) { colorClass = 'gauge-medium'; textColorClass = 'score-medium'; }
    else if (score >= 400) { colorClass = 'gauge-high'; textColorClass = 'score-high'; }
    
    circle.classList.add(colorClass);
    scoreText.classList.add(textColorClass);
    
    // Animate
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
        animateCounter(scoreText, score);
    }, 300);
}

// Navbar scroll effect
function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('bg-white', 'shadow-md');
            navbar.classList.remove('bg-transparent');
        } else {
            navbar.classList.remove('bg-white', 'shadow-md');
            navbar.classList.add('bg-transparent');
        }
    });
}

// What-If Analysis
async function runWhatIfAnalysis(formData) {
    try {
        const response = await fetch('/api/assessments/whatif', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        const result = await response.json();
        return result;
    } catch (error) {
        showToast('Analysis failed. Please try again.', 'error');
        return null;
    }
}

// Admin: Update application status
async function updateApplicationStatus(appId, status) {
    try {
        const response = await fetch(`/api/admin/applications/${appId}/status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: status })
        });
        
        if (response.ok) {
            showToast(`Application ${status} successfully`, 'success');
            setTimeout(() => location.reload(), 1000);
        } else {
            showToast('Failed to update status', 'error');
        }
    } catch (error) {
        showToast('Error updating status', 'error');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initNavbarScroll();
    
    // Auto-animate counters
    document.querySelectorAll('[data-counter]').forEach(el => {
        const target = parseInt(el.dataset.counter);
        if (!isNaN(target)) animateCounter(el, target);
    });
});
