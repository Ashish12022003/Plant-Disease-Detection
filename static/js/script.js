document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const resultSection = document.getElementById('resultSection');
    const previewImage = document.getElementById('previewImage');
    const predictionStatus = document.getElementById('predictionStatus');
    const confidence = document.getElementById('confidence');

    // Handle drag and drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2980b9';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle file upload and prediction
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            resultSection.style.display = 'flex';
        };
        reader.readAsDataURL(file);

        // Upload and get prediction
        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Update prediction results
            predictionStatus.textContent = data.class.charAt(0).toUpperCase() + data.class.slice(1);
            confidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;

            // Update status color
            predictionStatus.style.color = data.class === 'healthy' ? '#27ae60' : '#e74c3c';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
        });
    }
}); 