// File for handling dataset upload, preview, and labeling
document.addEventListener('DOMContentLoaded', function() {
    const previewButton = document.getElementById('previewButton');
    const labelButton = document.getElementById('labelButton');
    
    if (previewButton) {
        previewButton.addEventListener('click', previewDataset);
    }
    
    // Initialize file input validation
    const datasetFileInput = document.getElementById('datasetFile');
    if (datasetFileInput) {
        datasetFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                if (!file.name.endsWith('.csv')) {
                    alert('Hanya file CSV yang diperbolehkan');
                    this.value = '';
                } else if (file.size > 10 * 1024 * 1024) { // 10MB limit
                    alert('Ukuran file tidak boleh lebih dari 10MB');
                    this.value = '';
                }
            }
        });
    }
});

// Preview dataset before labeling
function previewDataset() {
    const fileInput = document.getElementById('datasetFile');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('Silakan pilih file CSV terlebih dahulu');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    const previewButton = document.getElementById('previewButton');
    previewButton.disabled = true;
    previewButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Memproses...';
    
    // Upload and preview
    fetch('/api/preview-dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        previewButton.disabled = false;
        previewButton.innerHTML = '<i class="fas fa-eye mr-1"></i> Preview & Pelabelan';
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Close upload modal and open preview modal
        closeModal('uploadModal');
        renderPreview(data);
        openModal('previewModal');
    })
    .catch(error => {
        previewButton.disabled = false;
        previewButton.innerHTML = '<i class="fas fa-eye mr-1"></i> Preview & Pelabelan';
        alert('Error: ' + error);
    });
}

// Render preview table
function renderPreview(data) {
    // Display preview data
    const previewTable = document.getElementById('previewTable');
    const headerRow = document.getElementById('previewTableHeader');
    const tableBody = document.getElementById('previewTableBody');
    
    // Clear existing data
    headerRow.innerHTML = '';
    tableBody.innerHTML = '';
    
    // Set counters
    document.getElementById('previewRowCount').textContent = data.preview_data.length;
    document.getElementById('totalRowCount').textContent = data.total_rows;
    
    // Check for missing or empty columns
    const missingColumnsWarning = document.getElementById('missingColumnsWarning');
    const missingColumnsList = document.getElementById('missingColumnsList');
    missingColumnsList.innerHTML = '';
    
    let hasMissingColumns = false;
    if (data.missing_columns && data.missing_columns.length > 0) {
        data.missing_columns.forEach(col => {
            const li = document.createElement('li');
            li.textContent = col;
            missingColumnsList.appendChild(li);
        });
        hasMissingColumns = true;
    }
    
    missingColumnsWarning.style.display = hasMissingColumns ? 'block' : 'none';
    
    // Add table headers
    if (data.columns && data.columns.length > 0) {
        data.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
    }
    
    // Add data rows
    if (data.preview_data && data.preview_data.length > 0) {
        data.preview_data.forEach(row => {
            const tr = document.createElement('tr');
            data.columns.forEach(col => {
                const td = document.createElement('td');
                td.textContent = row[col] !== null && row[col] !== undefined ? row[col] : '';
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    }
    
    // Reset labeling state
    document.getElementById('labelingLog').classList.add('hidden');
    document.getElementById('progressBar').classList.add('hidden');
    document.getElementById('labelingStatus').textContent = '';
    document.getElementById('labelButton').disabled = false;
}

// Start the labeling process
function startLabeling() {
    const labelButton = document.getElementById('labelButton');
    labelButton.disabled = true;
    labelButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Processing...';
    
    // Show progress elements
    document.getElementById('labelingLog').classList.remove('hidden');
    document.getElementById('progressBar').classList.remove('hidden');
    document.getElementById('labelingStatus').textContent = 'Memulai proses...';
    document.getElementById('logContent').innerHTML = '<div class="text-blue-600">Memulai proses pelabelan...</div>';
    
    // Start the labeling process
    fetch('/api/process-dataset', {
        method: 'POST'
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let progress = 0;
        let logContent = document.getElementById('logContent');
        
        function processStream({ done, value }) {
            if (done) {
                // Processing complete
                document.getElementById('labelingStatus').textContent = 'Selesai!';
                document.getElementById('progressBarFill').style.width = '100%';
                
                // Enable reload button
                labelButton.disabled = false;
                labelButton.innerHTML = '<i class="fas fa-check mr-1"></i> Selesai, Muat Ulang';
                labelButton.onclick = function() {
                    window.location.reload();
                };
                
                return;
            }
            
            // Process the current chunk
            const chunk = decoder.decode(value, { stream: true });
            try {
                const data = JSON.parse(chunk);
                
                if (data.progress) {
                    progress = data.progress;
                    document.getElementById('progressBarFill').style.width = `${progress}%`;
                    document.getElementById('labelingStatus').textContent = `${progress}% selesai`;
                }
                
                if (data.log) {
                    const logDiv = document.createElement('div');
                    if (data.type === 'error') {
                        logDiv.className = 'text-red-600';
                    } else if (data.type === 'success') {
                        logDiv.className = 'text-green-600';
                    } else {
                        logDiv.className = 'text-gray-700';
                    }
                    logDiv.textContent = data.log;
                    logContent.appendChild(logDiv);
                    
                    // Auto-scroll to bottom
                    const logContainer = document.getElementById('labelingLog');
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            } catch (e) {
                // Not JSON, might be heartbeat or other data
                console.log('Non-JSON chunk:', chunk);
            }
            
            // Read the next chunk
            return reader.read().then(processStream);
        }
        
        // Start reading the stream
        return reader.read().then(processStream);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('labelingStatus').textContent = 'Error!';
        document.getElementById('logContent').innerHTML += `<div class="text-red-600">Error: ${error}</div>`;
        
        // Re-enable button
        labelButton.disabled = false;
        labelButton.innerHTML = '<i class="fas fa-redo mr-1"></i> Coba Lagi';
    });
}
