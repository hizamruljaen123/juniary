// Toggle advanced parameters
$('#toggleAdvanced').click(function() {
    $('#advancedParams').toggleClass('hidden');
    if ($('#advancedParams').hasClass('hidden')) {
        $('#toggleText').text('Tampilkan');
        $('#toggleIcon').removeClass('fa-chevron-up').addClass('fa-chevron-down');
    } else {
        $('#toggleText').text('Sembunyikan');
        $('#toggleIcon').removeClass('fa-chevron-down').addClass('fa-chevron-up');
    }
});

// Enhanced Training button click
$('#trainButton').click(function() {
    $(this).prop('disabled', true).text('Melatih...');
    
    // Basic parameters
    const splitPercent = $('#splitSlider').val();
    
    // Advanced parameters
    const removeStopwords = $('#removeStopwords').is(':checked');
    const stemming = $('#stemming').is(':checked');
    const lemmatization = $('#lemmatization').is(':checked');
    
    const vectorizationMethod = $('#vectorizationMethod').val();
    const maxFeatures = $('#maxFeatures').val();
    
    const maxDepth = $('#maxDepth').val();
    const minSamplesSplit = $('#minSamplesSplit').val();
    const minSamplesLeaf = $('#minSamplesLeaf').val();
    const criterion = $('#criterion').val();

    $.ajax({
        url: '/api/train-model',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 
            split_percent: splitPercent,
            tokenization: {
                remove_stopwords: removeStopwords,
                use_stemming: stemming,
                use_lemmatization: lemmatization
            },
            vectorization: {
                method: vectorizationMethod,
                max_features: maxFeatures
            },
            decision_tree: {
                max_depth: maxDepth,
                min_samples_split: minSamplesSplit,
                min_samples_leaf: minSamplesLeaf,
                criterion: criterion
            }
        }),        success: function(response) {
            $('#trainResult').removeClass('hidden').addClass('block');                    
            
            // Format training logs
            let logsHtml = '';
            if (response.training_logs && response.training_logs.length > 0) {
                logsHtml = '<div class="mt-3 p-3 bg-gray-100 rounded-lg text-sm font-mono max-h-60 overflow-y-auto">';
                logsHtml += '<div class="text-gray-700 font-semibold mb-2">Log Tahapan Pelatihan:</div>';
                logsHtml += '<ol class="list-decimal pl-5 space-y-1">';
                response.training_logs.forEach(log => {
                    logsHtml += `<li class="text-gray-600">${log}</li>`;
                });
                logsHtml += '</ol></div>';
            }
            
            $('#trainResult').html(`
                <div class="text-green-700 mb-2">Model berhasil dilatih!</div>
                <div class="text-sm text-gray-600 mb-2">
                    Split: ${splitPercent}% data latih / ${100-splitPercent}% data uji
                </div>
                <div class="text-sm text-gray-600 mb-3">
                    Akurasi: ${response.accuracy ? (response.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                </div>                ${logsHtml}
                <div class="mt-3 text-right">
                    <button id="expandLogsBtn" class="text-blue-600 hover:text-blue-800 text-sm underline">Lihat Detail Log</button>
                </div>
            `);
            
            // Auto-reload page setelah 5 detik
            let countdown = 5;
            const countdownTimer = setInterval(() => {
                countdown--;
                $('#reloadCountdown').text(countdown);
                if (countdown <= 0) {
                    clearInterval(countdownTimer);
                    location.reload();
                }
            }, 1000);
            
            // Add event handler for expand logs button
            $('#expandLogsBtn').click(function(e) {
                e.preventDefault();
                clearInterval(countdownTimer); // Cancel auto-reload
                
                $('#trainingLogsModal').removeClass('hidden').addClass('flex');
                
                let fullLogsHtml = '<ol class="list-decimal pl-5 space-y-2">';
                response.training_logs.forEach(log => {
                    fullLogsHtml += `<li class="text-gray-600">${log}</li>`;
                });
                fullLogsHtml += '</ol>';
                
                $('#trainingLogsContent').html(fullLogsHtml);
            });
            
            // Update charts immediately
            loadTestDataDistribution();
            loadDataDistribution();
        },
        error: function(xhr) {
            alert('Terjadi kesalahan saat melatih model: ' + (xhr.responseJSON?.error || 'Unknown error'));
        },
        complete: function() {
            $('#trainButton').prop('disabled', false).text('Mulai Training');
        }
    });
});
