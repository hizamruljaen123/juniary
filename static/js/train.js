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

                // Display training logs
                let fullLogsHtml = '<ol class="list-decimal pl-5 space-y-2">';
                response.training_logs.forEach(log => {
                    fullLogsHtml += `<li class="text-gray-600">${log}</li>`;
                });
                fullLogsHtml += '</ol>';

                $('#trainingLogsContent').html(fullLogsHtml);

                // Add confusion matrix visualization
                if (response.eval_result && response.eval_result.confusion_matrix && response.eval_result.labels) {
                    // Create a new div for the confusion matrix if it doesn't exist
                    if ($('#confusionMatrixContainer').length === 0) {
                        $('.overflow-y-auto.flex-grow.p-6.pt-0').prepend(`
                            <div class="mb-6">
                                <h3 class="text-md font-semibold mb-2 text-gray-700">Confusion Matrix</h3>
                                <div id="confusionMatrixContainer" style="width:100%; height:400px;"></div>
                            </div>
                        `);
                    }

                    // Plot the confusion matrix
                    const z = response.eval_result.confusion_matrix;
                    const x = response.eval_result.labels;
                    const y = response.eval_result.labels;
                    // Convert z to string for annotation
                    const text = z.map(row => row.map(val => val.toString()));
                    const data = [{
                        z: z,
                        x: x,
                        y: y,
                        type: 'heatmap',
                        colorscale: 'Blues',
                        showscale: true,
                        text: text,
                        texttemplate: "%{text}",
                        textfont: { color: "black", size: 16 }
                    }];
                    const layout = {
                        title: 'Confusion Matrix',
                        xaxis: { title: 'Predicted Label' },
                        yaxis: { title: 'True Label' }
                    };
                    Plotly.newPlot('confusionMatrixContainer', data, layout);
                }

                // Add classification report
                if (response.eval_result && response.eval_result.classification_report) {
                    // Create a new div for the classification report if it doesn't exist
                    if ($('#classificationReportContainer').length === 0) {
                        $('.overflow-y-auto.flex-grow.p-6.pt-0').prepend(`
                            <div class="mb-6">
                                <h3 class="text-md font-semibold mb-2 text-gray-700">Classification Report</h3>
                                <div id="classificationReportContainer" class="overflow-x-auto">
                                    <table class="min-w-full bg-white border border-gray-200 rounded-lg">
                                        <thead>
                                            <tr class="bg-gray-50">
                                                <th class="py-2 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Class</th>
                                                <th class="py-2 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Precision</th>
                                                <th class="py-2 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Recall</th>
                                                <th class="py-2 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider border-b">F1-Score</th>
                                                <th class="py-2 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider border-b">Support</th>
                                            </tr>
                                        </thead>
                                        <tbody id="modalClassificationReportBody">
                                            <!-- Will be populated by JavaScript -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        `);
                    }

                    // Populate the classification report table
                    const reportBody = $('#modalClassificationReportBody');
                    reportBody.empty();

                    const report = response.eval_result.classification_report;

                    // Add rows for each class
                    for (const className of response.eval_result.labels) {
                        if (report[className]) {
                            const classData = report[className];
                            reportBody.append(`
                                <tr>
                                    <td class="py-2 px-3">${className}</td>
                                    <td class="py-2 px-3">${(classData.precision * 100).toFixed(2)}%</td>
                                    <td class="py-2 px-3">${(classData.recall * 100).toFixed(2)}%</td>
                                    <td class="py-2 px-3">${(classData['f1-score'] * 100).toFixed(2)}%</td>
                                    <td class="py-2 px-3">${classData.support}</td>
                                </tr>
                            `);
                        }
                    }

                    // Add row for accuracy
                    reportBody.append(`
                        <tr class="bg-gray-50 font-medium">
                            <td class="py-2 px-3">Accuracy</td>
                            <td class="py-2 px-3" colspan="3">${(report.accuracy * 100).toFixed(2)}%</td>
                            <td class="py-2 px-3">${report['weighted avg'].support}</td>
                        </tr>
                    `);
                }
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
