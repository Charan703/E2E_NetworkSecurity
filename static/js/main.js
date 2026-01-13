// File upload functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');

    if (uploadArea && fileInput) {
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = '#764ba2';
            uploadArea.style.background = 'linear-gradient(45deg, #e8f0ff, #f0f8ff)';
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = 'linear-gradient(45deg, #f8f9ff, #e8f0ff)';
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });

        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
            showNotification('Please select a CSV file', 'error');
            return;
        }

        const fileName = document.getElementById('fileName');
        if (fileName) {
            fileName.textContent = `Selected: ${file.name}`;
            fileName.style.color = '#667eea';
        }

        // Auto-submit form
        if (uploadForm) {
            submitForm();
        }
    }

    function submitForm() {
        if (loading) {
            loading.style.display = 'block';
        }
        
        const formData = new FormData(uploadForm);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            document.body.innerHTML = html;
            initializeResultsPage();
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error processing file', 'error');
        })
        .finally(() => {
            if (loading) {
                loading.style.display = 'none';
            }
        });
    }

    // Initialize results page functionality
    function initializeResultsPage() {
        addTableInteractivity();
        calculateStats();
        animateElements();
    }

    function addTableInteractivity() {
        const table = document.querySelector('.result-table table');
        if (table) {
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(row => {
                const predictionCell = row.cells[row.cells.length - 1];
                if (predictionCell) {
                    const prediction = parseFloat(predictionCell.textContent);
                    if (prediction === 1.0) {
                        row.classList.add('threat-high');
                        predictionCell.innerHTML = '🚨 PHISHING DETECTED';
                    } else {
                        row.classList.add('threat-low');
                        predictionCell.innerHTML = '✅ SAFE';
                    }
                }
            });
        }
    }

    function calculateStats() {
        const table = document.querySelector('.result-table table');
        if (table) {
            const rows = table.querySelectorAll('tbody tr');
            let totalSites = rows.length;
            let phishingSites = 0;
            let safeSites = 0;

            rows.forEach(row => {
                const predictionCell = row.cells[row.cells.length - 1];
                if (predictionCell && predictionCell.textContent.includes('PHISHING')) {
                    phishingSites++;
                } else {
                    safeSites++;
                }
            });

            updateStatsDisplay(totalSites, phishingSites, safeSites);
        }
    }

    function updateStatsDisplay(total, phishing, safe) {
        const statsContainer = document.querySelector('.stats-grid');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <div class="stat-card">
                    <div class="stat-number">${total}</div>
                    <div>Total Sites</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${phishing}</div>
                    <div>🚨 Phishing Detected</div>
                </div>
                <div class="stat-card safe">
                    <div class="stat-number">${safe}</div>
                    <div>✅ Safe Sites</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-number">${((phishing/total)*100).toFixed(1)}%</div>
                    <div>Threat Level</div>
                </div>
            `;
        }
    }

    function animateElements() {
        const cards = document.querySelectorAll('.card, .stat-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            setTimeout(() => {
                card.style.transition = 'all 0.5s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            background: ${type === 'error' ? '#ff6b6b' : '#00d2d3'};
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // Add CSS animation for notifications
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
});

// Export functionality
function exportResults(format) {
    const table = document.querySelector('.result-table table');
    if (!table) return;

    if (format === 'csv') {
        exportToCSV(table);
    } else if (format === 'json') {
        exportToJSON(table);
    }
}

function exportToCSV(table) {
    const rows = Array.from(table.querySelectorAll('tr'));
    const csv = rows.map(row => {
        const cells = Array.from(row.querySelectorAll('th, td'));
        return cells.map(cell => `"${cell.textContent}"`).join(',');
    }).join('\n');

    downloadFile(csv, 'phishing_results.csv', 'text/csv');
}

function exportToJSON(table) {
    const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent);
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    
    const data = rows.map(row => {
        const cells = Array.from(row.querySelectorAll('td'));
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = cells[index] ? cells[index].textContent : '';
        });
        return obj;
    });

    downloadFile(JSON.stringify(data, null, 2), 'phishing_results.json', 'application/json');
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}