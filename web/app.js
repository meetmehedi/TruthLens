document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const mediaInput = document.getElementById('mediaInput');
  const filePreviewChip = document.getElementById('filePreviewChip');
  const selectedFileName = document.getElementById('selectedFileName');
  const selectedFileSize = document.getElementById('selectedFileSize');
  const fileTypeIcon = document.getElementById('fileTypeIcon');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const dropContent = document.getElementById('dropContent');
  const analyzeForm = document.getElementById('analyzeForm');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const transcriptText = document.getElementById('transcriptText');
  const loadingState = document.getElementById('loadingState');
  const resultsSection = document.getElementById('resultsSection');
  const scannerStatusText = document.getElementById('scannerStatusText');

  // Preset Buttons
  const presetSyntheticImg = document.getElementById('presetSyntheticImg');
  const presetAuthenticImg = document.getElementById('presetAuthenticImg');
  const presetVoiceClone = document.getElementById('presetVoiceClone');
  const loadDemoBtn = document.getElementById('loadDemoBtn');

  // Results Elements
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictSummary = document.getElementById('verdictSummary');
  const syntheticPercent = document.getElementById('syntheticPercent');
  const syntheticBarFill = document.getElementById('syntheticBarFill');
  const confidencePercent = document.getElementById('confidencePercent');
  const confidenceBarFill = document.getElementById('confidenceBarFill');
  const latencyDisplay = document.getElementById('latencyDisplay');
  const fileHashDisplay = document.getElementById('fileHashDisplay');
  const originalImg = document.getElementById('originalImg');
  const overlayImg = document.getElementById('overlayImg');
  const spectrogramImg = document.getElementById('spectrogramImg');
  const audioMetricsList = document.getElementById('audioMetricsList');
  const reasonsList = document.getElementById('reasonsList');
  const modalityPills = document.getElementById('modalityPills');
  const visualInspectorCard = document.getElementById('visualInspectorCard');
  const audioInspectorCard = document.getElementById('audioInspectorCard');
  const printReportBtn = document.getElementById('printReportBtn');

  // Slider Elements
  const comparisonContainer = document.getElementById('comparisonContainer');
  const clippedForeground = document.getElementById('clippedForeground');
  const sliderDivider = document.getElementById('sliderDivider');

  let currentFile = null;
  let isDraggingSlider = false;

  // Initialize System Health Status
  fetch('/api/health')
    .then(r => r.json())
    .then(data => {
      const statusText = document.querySelector('.status-text');
      if (statusText) statusText.textContent = `${data.device.toUpperCase()} PyTorch Online`;
    })
    .catch(() => {});

  // File Drag & Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  mediaInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    currentFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    
    // Choose icon
    if (file.type.startsWith('image/')) fileTypeIcon.textContent = '🖼️';
    else if (file.type.startsWith('video/')) fileTypeIcon.textContent = '🎬';
    else if (file.type.startsWith('audio/')) fileTypeIcon.textContent = '🎙️';
    else fileTypeIcon.textContent = '📄';

    // If textarea contains leftover preset demo text, clear it for user's fresh file
    const stalePresets = [
      'Statement generated alongside synthetic video broadcast.',
      'Authentic camera capture during press briefing.',
      'Emergency authorization request for immediate fund transfer.'
    ];
    if (transcriptText && stalePresets.includes(transcriptText.value.trim())) {
      transcriptText.value = '';
    }

    dropContent.classList.add('hidden');
    filePreviewChip.classList.remove('hidden');
  }

  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    mediaInput.value = '';
    filePreviewChip.classList.add('hidden');
    dropContent.classList.remove('hidden');
  });

  // Slider Logic (Drag handle)
  function setSliderPosition(percentage) {
    const clamped = Math.max(0, Math.min(100, percentage));
    clippedForeground.style.width = clamped + '%';
    sliderDivider.style.left = clamped + '%';
  }

  function handleSliderMove(e) {
    if (!isDraggingSlider) return;
    const rect = comparisonContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    setSliderPosition(percentage);
  }

  comparisonContainer.addEventListener('mousedown', (e) => {
    isDraggingSlider = true;
    handleSliderMove(e);
  });

  window.addEventListener('mousemove', handleSliderMove);
  window.addEventListener('mouseup', () => { isDraggingSlider = false; });

  // Touch support for mobile
  comparisonContainer.addEventListener('touchstart', (e) => {
    isDraggingSlider = true;
    handleSliderMove(e);
  });
  window.addEventListener('touchmove', handleSliderMove);
  window.addEventListener('touchend', () => { isDraggingSlider = false; });

  // Form Submission
  analyzeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const textVal = transcriptText.value.trim();

    if (!currentFile && !textVal) {
      alert('Please upload an image, video, or audio file, or provide text.');
      return;
    }

    const formData = new FormData();
    if (currentFile) {
      formData.append('file', currentFile);
    }
    if (textVal) {
      formData.append('text_content', textVal);
    }

    // Show loading state
    loadingState.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = true;

    // Loading status text cycle
    const statusCycle = [
      'Extracting spatial landmarks & face crops...',
      'Running CNN backbone feature activations...',
      'Computing 2D Fast Fourier Transform high-frequency spectra...',
      'Evaluating 16 kHz Log-Mel acoustic vocoder harmonics...',
      'Calibrating late-fusion probability thresholds...'
    ];
    let cycleIdx = 0;
    const interval = setInterval(() => {
      cycleIdx = (cycleIdx + 1) % statusCycle.length;
      scannerStatusText.textContent = statusCycle[cycleIdx];
    }, 450);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      alert('Analysis error: ' + err.message);
    } finally {
      clearInterval(interval);
      loadingState.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  });

  function displayResults(data) {
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // 1. Verdict & Badge
    verdictBanner.className = 'card glass-card verdict-banner';
    verdictBadge.className = 'verdict-badge';
    if (data.verdict === 'Likely Synthetic') {
      verdictBanner.classList.add('verdict-synthetic');
      verdictBadge.classList.add('badge-synthetic');
      verdictBadge.textContent = 'Likely Synthetic / Deepfake';
    } else if (data.verdict === 'Authentic') {
      verdictBanner.classList.add('verdict-authentic');
      verdictBadge.classList.add('badge-authentic');
      verdictBadge.textContent = 'Authentic / Genuine';
    } else {
      verdictBanner.classList.add('verdict-uncertain');
      verdictBadge.classList.add('badge-uncertain');
      verdictBadge.textContent = 'Uncertain / Borderline';
    }

    verdictSummary.textContent = data.summary;
    syntheticPercent.textContent = data.synthetic_probability_percent + '%';
    syntheticBarFill.style.width = data.synthetic_probability_percent + '%';
    confidencePercent.textContent = data.confidence_percent + '%';
    confidenceBarFill.style.width = data.confidence_percent + '%';
    latencyDisplay.textContent = data.latency_seconds + 's';
    fileHashDisplay.textContent = 'SHA-256: ' + (data.file_hash_sha256 ? data.file_hash_sha256.substring(0, 16) + '...' : 'N/A');

    // 2. Visual Branch & Grad-CAM
    if (data.visual_branch && data.visual_branch.overlay_image_base64) {
      visualInspectorCard.classList.remove('hidden');
      overlayImg.src = data.visual_branch.overlay_image_base64;
      originalImg.src = data.visual_branch.original_image_base64;
      setSliderPosition(50);
    } else {
      visualInspectorCard.classList.add('hidden');
    }

    // 3. Audio Branch & Spectrogram
    if (data.audio_branch && data.audio_branch.spectrogram_image_base64) {
      audioInspectorCard.classList.remove('hidden');
      spectrogramImg.src = data.audio_branch.spectrogram_image_base64;
      
      audioMetricsList.innerHTML = `
        <div class="audio-metric-item">
          <span>Harmonic Regularity Anomaly:</span>
          <strong>${(data.audio_branch.harmonic_regularity_score * 100).toFixed(1)}%</strong>
        </div>
        <div class="audio-metric-item">
          <span>Spectral Cutoff / Attenuation:</span>
          <strong>${(data.audio_branch.spectral_cutoff_score * 100).toFixed(1)}%</strong>
        </div>
        <div class="audio-metric-item">
          <span>Clip Duration:</span>
          <strong>${data.audio_branch.duration_seconds} sec (16 kHz mono)</strong>
        </div>
      `;
    } else {
      audioInspectorCard.classList.add('hidden');
    }

    // 4. Reasons Behind Flags (Expected Outcome requirement)
    reasonsList.innerHTML = '';
    if (data.reasons && data.reasons.length > 0) {
      data.reasons.forEach(reason => {
        const item = document.createElement('div');
        item.className = 'reason-item';
        item.innerHTML = `
          <span class="reason-icon">🔍</span>
          <span>${reason}</span>
        `;
        reasonsList.appendChild(item);
      });
    }

    // 5. Modality Weight Pills
    modalityPills.innerHTML = '';
    if (data.normalized_weights) {
      for (const [mod, weight] of Object.entries(data.normalized_weights)) {
        const pill = document.createElement('span');
        pill.className = 'modality-pill';
        pill.textContent = `${mod.toUpperCase()} Modality Weight: ${(weight * 100).toFixed(0)}%`;
        modalityPills.appendChild(pill);
      }
    }

    // 6. Setup Printable Certificate
    const certBody = document.getElementById('certBody');
    certBody.innerHTML = `
      <p><strong>Analysis ID:</strong> ${data.analysis_id}</p>
      <p><strong>Target File:</strong> ${data.filename}</p>
      <p><strong>SHA-256 Digest:</strong> ${data.file_hash_sha256}</p>
      <p><strong>Final Forensic Verdict:</strong> ${data.verdict} (${data.synthetic_probability_percent}% synthetic probability)</p>
      <p><strong>Confidence:</strong> ${data.confidence_percent}%</p>
      <p><strong>Timestamp:</strong> ${new Date().toISOString()}</p>
      <h3>Forensic Findings:</h3>
      <ul>${data.reasons.map(r => `<li>${r}</li>`).join('')}</ul>
    `;
  }

  printReportBtn.addEventListener('click', () => {
    window.print();
  });

  // Preset Handlers for Demo Day
  function loadPreset(blob, name, text) {
    const file = new File([blob], name, { type: blob.type });
    handleFileSelected(file);
    if (text) transcriptText.value = text;
    analyzeForm.dispatchEvent(new Event('submit'));
  }

  // Generate synthetic canvas face demo
  function createDemoCanvasImage(isSynthetic) {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    const ctx = canvas.getContext('2d');

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 400, 400);
    bgGrad.addColorStop(0, isSynthetic ? '#1e1b4b' : '#0f172a');
    bgGrad.addColorStop(1, isSynthetic ? '#312e81' : '#1e293b');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, 400, 400);

    // Face oval
    ctx.fillStyle = '#f8d7b8';
    ctx.beginPath();
    ctx.ellipse(200, 200, 110, 140, 0, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = '#222';
    ctx.beginPath();
    ctx.arc(160, 175, 12, 0, Math.PI * 2);
    ctx.arc(240, 175, 12, 0, Math.PI * 2);
    ctx.fill();

    // Mouth
    ctx.beginPath();
    ctx.strokeStyle = '#c2410c';
    ctx.lineWidth = 6;
    if (isSynthetic) {
      // Unnatural sharp mouth seam
      ctx.moveTo(165, 260);
      ctx.lineTo(235, 260);
      ctx.stroke();

      // Add GAN high-frequency noise checkerboard artifact around hair boundary
      for (let i = 0; i < 400; i += 8) {
        for (let j = 0; j < 120; j += 8) {
          if ((i + j) % 16 === 0) {
            ctx.fillStyle = 'rgba(255, 0, 255, 0.25)';
            ctx.fillRect(i, j, 4, 4);
          }
        }
      }
    } else {
      // Natural curved mouth
      ctx.arc(200, 250, 25, 0.2, Math.PI - 0.2);
      ctx.stroke();
    }

    return new Promise(resolve => {
      canvas.toBlob(blob => resolve(blob), 'image/jpeg');
    });
  }

  // Generate synthetic audio tone demo
  function createDemoAudioBlob(isCloned) {
    const sampleRate = 16000;
    const duration = 2.0;
    const numSamples = sampleRate * duration;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    // WAV Header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // Audio samples: Cloned voice has robotic flat harmonics; authentic has modulated micro-tremor
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      let sample = 0;
      if (isCloned) {
        // Rigid harmonics without pitch jitter (hallmark of neural vocoder)
        sample = Math.sin(2 * Math.PI * 220 * t) + 0.5 * Math.sin(2 * Math.PI * 440 * t);
      } else {
        // Natural human vibrato / tremor
        const jitter = Math.sin(2 * Math.PI * 5 * t) * 15;
        sample = Math.sin(2 * Math.PI * (210 + jitter) * t) * 0.8;
      }
      view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, sample)) * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  presetSyntheticImg.addEventListener('click', async () => {
    const blob = await createDemoCanvasImage(true);
    loadPreset(blob, 'deepfake_face_gan.jpg', 'Statement generated alongside synthetic video broadcast.');
  });

  presetAuthenticImg.addEventListener('click', async () => {
    const blob = await createDemoCanvasImage(false);
    loadPreset(blob, 'authentic_photo.jpg', 'Authentic camera capture during press briefing.');
  });

  presetVoiceClone.addEventListener('click', async () => {
    const blob = createDemoAudioBlob(true);
    loadPreset(blob, 'cloned_ceo_voicemail.wav', 'Emergency authorization request for immediate fund transfer.');
  });

  loadDemoBtn.addEventListener('click', () => {
    presetSyntheticImg.click();
  });
});
