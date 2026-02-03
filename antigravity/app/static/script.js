/**
 * Emotion Recognition - Frontend Logic
 * Handles file upload, progress tracking, and result display
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const audioInput = document.getElementById('audio-input');
    const uploadLabel = document.getElementById('upload-label');
    const uploadText = document.getElementById('upload-text');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultSection = document.getElementById('result-section');
    const resultEmotion = document.getElementById('result-emotion');
    const resultConfidence = document.getElementById('result-confidence');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');

    // State
    let isUploading = false;
    let currentLang = 'en';
    let lastResult = null;

    // Translations
    const translations = {
        en: {
            subtitle: 'Detect emotions from your voice',
            upload_text: 'Upload Audio',
            result_label: 'Detected Emotion',
            footer: 'Created by Dănilă Laurențiu',
            confidence: 'Confidence'
        },
        ro: {
            subtitle: 'Detectează emoțiile din vocea ta',
            upload_text: 'Încarcă Audio',
            result_label: 'Emoție Detectată',
            footer: 'Creat de Dănilă Laurențiu',
            confidence: 'Încredere'
        }
    };

    // Language Toggle
    const btnEn = document.getElementById('btn-en');
    const btnRo = document.getElementById('btn-ro');

    btnEn.addEventListener('click', () => {
        if (currentLang !== 'en') {
            currentLang = 'en';
            updateLanguage();
        }
    });

    btnRo.addEventListener('click', () => {
        if (currentLang !== 'ro') {
            currentLang = 'ro';
            updateLanguage();
        }
    });

    function updateLanguage() {
        const t = translations[currentLang];

        // Update Text Elements
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            if (t[key]) {
                element.textContent = t[key];
            }
        });

        // Update Toggle Buttons
        if (currentLang === 'en') {
            btnEn.classList.add('active');
            btnRo.classList.remove('active');
        } else {
            btnEn.classList.remove('active');
            btnRo.classList.add('active');
        }

        // Update Upload Text if no file selected
        if (!isUploading && !uploadLabel.classList.contains('has-file')) {
            uploadText.textContent = t.upload_text;
        }

        // Update Result if visible
        if (lastResult && resultSection.classList.contains('visible')) {
            const label = t.confidence;
            if (lastResult.confidence) {
                resultConfidence.textContent = `${label}: ${lastResult.confidence}%`;
            }
        }
    }

    // File input change handler
    audioInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Drag and drop handlers
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!isUploading) {
            uploadLabel.classList.add('dragging');
        }
    });

    uploadLabel.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadLabel.classList.remove('dragging');
    });

    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.classList.remove('dragging');

        if (isUploading) return;

        const file = e.dataTransfer.files[0];
        if (file) {
            // Validate extension
            const ext = file.name.split('.').pop().toLowerCase();
            if (['wav', 'mp3', 'm4a'].includes(ext)) {
                handleFileUpload(file);
            } else {
                showError('Please upload a .wav, .mp3, or .m4a file');
            }
        }
    });

    /**
     * Handle file upload and prediction
     */
    async function handleFileUpload(file) {
        isUploading = true;

        // Update UI
        hideError();
        hideResult();
        uploadLabel.classList.add('has-file');
        uploadText.textContent = file.name;
        showProgress();
        updateProgress(0);

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Use XMLHttpRequest for upload progress
            const result = await uploadWithProgress('/predict', formData);

            // Show result
            showResult(result.emotion, result.confidence);

        } catch (error) {
            showError(error.message || 'Failed to process audio');
        } finally {
            isUploading = false;
            resetUploadLabel();
        }
    }

    /**
     * Upload file with progress tracking
     */
    function uploadWithProgress(url, formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Upload progress (0-70%)
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 70);
                    updateProgress(percent);
                }
            });

            // Upload complete, now processing (70-95%)
            xhr.upload.addEventListener('load', () => {
                updateProgress(75);
                simulateProcessing();
            });

            // Response received
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    updateProgress(100);
                    try {
                        const response = JSON.parse(xhr.responseText);
                        setTimeout(() => resolve(response), 200);
                    } catch (e) {
                        reject(new Error('Invalid response from server'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.detail || 'Server error'));
                    } catch (e) {
                        reject(new Error(`Server error: ${xhr.status}`));
                    }
                }
            });

            // Error handling
            xhr.addEventListener('error', () => {
                reject(new Error('Network error - please check your connection'));
            });

            xhr.addEventListener('abort', () => {
                reject(new Error('Upload cancelled'));
            });

            // Send request
            xhr.open('POST', url);
            xhr.send(formData);
        });
    }

    /**
     * Simulate processing progress (70-95%)
     */
    function simulateProcessing() {
        let progress = 75;
        const interval = setInterval(() => {
            progress += Math.random() * 3;
            if (progress >= 95) {
                progress = 95;
                clearInterval(interval);
            }
            updateProgress(Math.round(progress));
        }, 150);
    }

    /**
     * Update progress bar
     */
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
        progressText.textContent = `${percent}%`;
    }

    /**
     * Show progress section
     */
    function showProgress() {
        progressSection.classList.add('visible');
    }

    /**
     * Hide progress section
     */
    function hideProgress() {
        progressSection.classList.remove('visible');
        updateProgress(0);
    }

    /**
     * Show result
     */
    function showResult(emotion, confidence) {
        lastResult = { emotion, confidence };
        setTimeout(() => {
            hideProgress();
            resultEmotion.textContent = emotion;
            if (confidence) {
                const label = translations[currentLang].confidence;
                resultConfidence.textContent = `${label}: ${confidence}%`;
            } else {
                resultConfidence.textContent = '';
            }
            resultSection.classList.add('visible');
        }, 300);
    }

    /**
     * Hide result
     */
    function hideResult() {
        resultSection.classList.remove('visible');
    }

    /**
     * Show error
     */
    function showError(message) {
        hideProgress();
        errorMessage.textContent = message;
        errorSection.classList.add('visible');
    }

    /**
     * Hide error
     */
    function hideError() {
        errorSection.classList.remove('visible');
    }

    /**
     * Reset upload label
     */
    function resetUploadLabel() {
        uploadLabel.classList.remove('has-file');
        uploadText.textContent = translations[currentLang].upload_text;
        audioInput.value = '';
    }
});