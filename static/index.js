// Scroll to the results section if there's a prediction
window.onload = function () {
    const predictionSection = document.getElementById('prediction-section');
    if (predictionSection && predictionSection.innerText.trim()) {
        predictionSection.scrollIntoView({ behavior: 'smooth' });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    let isAnimating = false;

    function toggleMenu() {
        if (isAnimating) return;
        isAnimating = true;

        navToggle.classList.toggle('active');
        navLinks.classList.toggle('active');
        
        // Update ARIA attributes
        const isExpanded = navToggle.classList.contains('active');
        navToggle.setAttribute('aria-expanded', isExpanded);

        // Reset animation lock after transition
        setTimeout(() => {
            isAnimating = false;
        }, 300);
    }

    // Toggle menu on button click
    navToggle.addEventListener('click', toggleMenu);

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        const isClickInside = navLinks.contains(e.target) || 
                            navToggle.contains(e.target);
        
        if (!isClickInside && navLinks.classList.contains('active')) {
            toggleMenu();
        }
    });

    // Close menu when clicking links
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            if (navLinks.classList.contains('active')) {
                toggleMenu();
            }
        });
    });

    // Handle escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && navLinks.classList.contains('active')) {
            toggleMenu();
        }
    });

    // Handle resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            if (window.innerWidth > 768 && navLinks.classList.contains('active')) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
            }
        }, 250);
    });
});

document.getElementById('year').textContent = new Date().getFullYear();
        
// Get DOM elements
const fileUpload = document.getElementById('file-upload');
const previewContainer = document.getElementById('preview');
const previewImage = document.getElementById('preview-image');
const cameraContainer = document.getElementById('camera-container');
const cameraPreview = document.getElementById('camera-preview');
const captureButton = document.getElementById('capture-button');
const cancelCamera = document.getElementById('cancel-camera');
let stream = null;

// Function to handle file preview
function handleFilePreview(file) {
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.hidden = false;
            cameraContainer.hidden = true;
        }
        reader.readAsDataURL(file);
    }
}

// Handle file upload
fileUpload.addEventListener('change', function(e) {
    handleFilePreview(e.target.files[0]);
});

// Start camera stream
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }, 
            audio: false 
        });
        cameraPreview.srcObject = stream;
        previewContainer.hidden = true;
        cameraContainer.hidden = false;
    } catch (err) {
        alert('Unable to access camera: ' + err.message);
    }
}

// Stop camera stream
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraContainer.hidden = true;
}

// Handle camera button click
document.querySelector('.camera-label').addEventListener('click', function(e) {
    e.preventDefault();
    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
        startCamera();
    } else {
        alert('Sorry, your browser doesn\'t support accessing the camera');
    }
});

// Handle capture button click
captureButton.addEventListener('click', function() {
    const canvas = document.createElement('canvas');
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    canvas.getContext('2d').drawImage(cameraPreview, 0, 0);
    
    canvas.toBlob(function(blob) {
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
        
        // Create a new FileList containing the captured image
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileUpload.files = dataTransfer.files;
        
        handleFilePreview(file);
        stopCamera();
    }, 'image/jpeg');
});

// Handle cancel button click
cancelCamera.addEventListener('click', stopCamera);

// Handle drag and drop
const uploadBox = document.querySelector('.upload-box');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFilePreview(file);
        fileUpload.files = e.dataTransfer.files;
    }
});