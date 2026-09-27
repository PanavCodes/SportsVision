// SportsVision Multi-Sport Frontend Application Logic

let eventSource = null;
let currentVideoStem = "";
let currentResults = null;
let detectedSport = "unknown"; // "basketball" | "cricket" | "unknown"
let isSyncingVideos = false;
let selectedSportOverride = "auto";

// Stage definitions per sport
const BASKETBALL_STAGES = [
  { num: 1, label: "Hardware" },
  { num: 2, label: "AI Models" },
  { num: 3, label: "Video" },
  { num: 4, label: "Tracking" },
  { num: 5, label: "Minimap" },
  { num: 6, label: "Shot Arc" },
  { num: 7, label: "Highlights" },
  { num: 8, label: "Report" }
];

const CRICKET_STAGES = [
  { num: 1, label: "Hardware" },
  { num: 2, label: "AI Models" },
  { num: 3, label: "Video/Audio" },
  { num: 4, label: "Ball Track" },
  { num: 5, label: "Processing" },
  { num: 6, label: "Highlights" },
  { num: 7, label: "Report" }
];

// DOM Elements
const elements = {
  navTabs: document.querySelectorAll('.nav-tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  systemGpuText: document.getElementById('systemGpuText'),
  systemStatusDot: document.getElementById('systemStatusDot'),
  sportBadge: document.getElementById('sportBadge'),
  videoSelect: document.getElementById('videoSelect'),
  refreshVideosBtn: document.getElementById('refreshVideosBtn'),
  btnRunPipeline: document.getElementById('btnRunPipeline'),
  btnStopPipeline: document.getElementById('btnStopPipeline'),
  btnLoadSaved: document.getElementById('btnLoadSaved'),
  hasSavedBadge: document.getElementById('hasSavedBadge'),

  // Upload
  uploadDropZone: document.getElementById('uploadDropZone'),
  uploadInput: document.getElementById('videoUploadInput'),
  uploadProgressWrapper: document.getElementById('uploadProgressWrapper'),
  uploadProgressFill: document.getElementById('uploadProgressFill'),
  uploadProgressText: document.getElementById('uploadProgressText'),

  // Sport radios
  sportRadios: document.querySelectorAll('.sport-radio'),

  // Hardware Profiles
  hardwareCards: document.querySelectorAll('.hardware-card'),
  activeGpuBadge: document.getElementById('activeGpuBadge'),
  inputUseSam2: document.getElementById('inputUseSam2'),
  sam2StatusText: document.getElementById('sam2StatusText'),
  batchSizeHint: document.getElementById('batchSizeHint'),
  frameSkipHint: document.getElementById('frameSkipHint'),

  // Preset Chips
  presetChips: document.querySelectorAll('.preset-chip'),
  inputMaxFrames: document.getElementById('inputMaxFrames'),
  inputFrameSkip: document.getElementById('inputFrameSkip'),
  inputBatchSize: document.getElementById('inputBatchSize'),

  // Stages & Progress
  stageStepsContainer: document.getElementById('stageStepsContainer'),
  progressFill: document.getElementById('progressFill'),
  progressPctText: document.getElementById('progressPctText'),
  progressFramesText: document.getElementById('progressFramesText'),
  metricFps: document.getElementById('metricFps'),
  metricEta: document.getElementById('metricEta'),
  metricVram: document.getElementById('metricVram'),
  metricStageName: document.getElementById('metricStageName'),
  terminalBody: document.getElementById('terminalBody'),
  clearLogsBtn: document.getElementById('clearLogsBtn'),

  // Video Players
  rawVideoPlayer: document.getElementById('rawVideoPlayer'),
  annotatedVideoPlayer: document.getElementById('annotatedVideoPlayer'),
  highlightsVideoPlayer: document.getElementById('highlightsVideoPlayer'),
  btnPlayPauseSync: document.getElementById('btnPlayPauseSync'),
  syncPlayIcon: document.getElementById('syncPlayIcon'),
  syncTimeDisplay: document.getElementById('syncTimeDisplay'),
  playbackSpeedSelect: document.getElementById('playbackSpeedSelect'),
  theaterFeatureCards: document.getElementById('theaterFeatureCards'),

  // Basketball Analytics
  bbSection: document.getElementById('basketball-analytics-section'),
  statPossessionT1: document.getElementById('statPossessionT1'),
  statPossessionT2: document.getElementById('statPossessionT2'),
  possessionFillT1: document.getElementById('possessionFillT1'),
  possessionFillT2: document.getElementById('possessionFillT2'),
  statPassesT1: document.getElementById('statPassesT1'),
  statPassesT2: document.getElementById('statPassesT2'),
  statInterceptT1: document.getElementById('statInterceptT1'),
  statInterceptT2: document.getElementById('statInterceptT2'),
  statShotsDetected: document.getElementById('statShotsDetected'),
  statDuration: document.getElementById('statDuration'),
  statTotalFrames: document.getElementById('statTotalFrames'),
  playerTableBody: document.getElementById('playerTableBody'),

  // Cricket Analytics
  ckSection: document.getElementById('cricket-analytics-section'),
  cricketDeliveries: document.getElementById('cricketDeliveries'),
  cricketOvers: document.getElementById('cricketOvers'),
  cricketWickets: document.getElementById('cricketWickets'),
  cricketFours: document.getElementById('cricketFours'),
  cricketSixes: document.getElementById('cricketSixes'),
  cricketFastestSpeed: document.getElementById('cricketFastestSpeed'),
  cricketAvgSpeed: document.getElementById('cricketAvgSpeed'),
  cricketSlowestSpeed: document.getElementById('cricketSlowestSpeed'),
  cricketDotPct: document.getElementById('cricketDotPct'),
  cricketRuns: document.getElementById('cricketRuns'),
  cricketHighlightCount: document.getElementById('cricketHighlightCount'),
  lineDistContainer: document.getElementById('lineDistContainer'),
  lengthDistContainer: document.getElementById('lengthDistContainer'),
  shotTypeContainer: document.getElementById('shotTypeContainer'),
  drsSection: document.getElementById('drsSection'),
  drsVerdictContainer: document.getElementById('drsVerdictContainer'),
  commentaryLog: document.getElementById('commentaryLog'),
  cricketDeliveriesTableBody: document.getElementById('cricketDeliveriesTableBody'),

  // Heatmaps
  heatmapPlaceholder: document.getElementById('heatmapPlaceholder'),
  bbHeatmapsSection: document.getElementById('basketball-heatmaps-section'),
  ckVisualsSection: document.getElementById('cricket-visuals-section'),
  heatmapTeam1Img: document.getElementById('heatmapTeam1Img'),
  heatmapTeam2Img: document.getElementById('heatmapTeam2Img'),
  cricketPitchMapImg: document.getElementById('cricketPitchMapImg'),
  cricketGroundRadarImg: document.getElementById('cricketGroundRadarImg'),

  // Analytics placeholder
  analyticsPlaceholder: document.getElementById('analyticsPlaceholder'),

  // Highlights
  highlightMethodBadge: document.getElementById('highlightMethodBadge'),
  highlightDescription: document.getElementById('highlightDescription'),
};

// ========================== Initialize ==========================
document.addEventListener('DOMContentLoaded', async () => {
  initTabs();
  initHardwareProfiles();
  initPresets();
  initVideoSync();
  initUploadZone();
  initSportRadios();
  renderStageSteps(BASKETBALL_STAGES); // Default
  await loadSystemInfo();
  await loadVideos();
  checkCurrentPipelineStatus();
});

// ========================== Toast Notifications ==========================
function showToast(type, message) {
  const container = document.getElementById('toastContainer');
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span> ${message}`;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4000);
}

// ========================== Tab Navigation ==========================
function initTabs() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.target;
      elements.navTabs.forEach(t => t.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

function switchTab(tabId) {
  const tabBtn = document.querySelector(`.nav-tab-btn[data-target="${tabId}"]`);
  if (tabBtn) tabBtn.click();
}

// ========================== Hardware Profiles ==========================
function initHardwareProfiles() {
  const profiles = {
    mx450: {
      batchSize: '1',
      frameSkip: '2',
      useSam2: false,
      badgeText: 'MX450 (2GB VRAM)',
      batchHint: 'MX450: Batch 1 (Zero-OOM)',
      skipHint: 'Skip 2 (Reduces VRAM & thermals)',
      sam2Text: 'Disabled for 2GB VRAM (Avoids CUDA OOM)',
      toast: '💻 Applied MX450 2GB profile: Batch 1, Skip 2, SAM2 Disabled (Zero-OOM Safe)'
    },
    rtx3050: {
      batchSize: '6',
      frameSkip: '1',
      useSam2: true,
      badgeText: 'RTX 3050 (6GB VRAM)',
      batchHint: 'RTX 3050: Batch 6',
      skipHint: '1 = Full frame processing',
      sam2Text: 'Player masks enabled (Optimized for 6GB VRAM)',
      toast: '🚀 Applied RTX 3050 6GB profile: Batch 6, Skip 1, SAM2 Enabled'
    },
    rtx4060: {
      batchSize: '16',
      frameSkip: '1',
      useSam2: true,
      badgeText: 'RTX 4060 (8GB+ VRAM)',
      batchHint: 'RTX 4060 8GB+: Batch 16',
      skipHint: '1 = Full frame processing',
      sam2Text: 'Full SAM2 segmentation + ByteTrack',
      toast: '🔥 Applied RTX 4060 profile: Batch 16, Skip 1, SAM2 Enabled'
    }
  };

  elements.hardwareCards.forEach(card => {
    card.addEventListener('click', () => {
      elements.hardwareCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const gpuKey = card.dataset.gpu;
      const conf = profiles[gpuKey];
      if (!conf) return;

      elements.inputBatchSize.value = conf.batchSize;
      elements.inputFrameSkip.value = conf.frameSkip;
      if (elements.inputUseSam2) elements.inputUseSam2.checked = conf.useSam2;
      if (elements.activeGpuBadge) elements.activeGpuBadge.textContent = conf.badgeText;
      if (elements.batchSizeHint) elements.batchSizeHint.textContent = conf.batchHint;
      if (elements.frameSkipHint) elements.frameSkipHint.textContent = conf.skipHint;
      if (elements.sam2StatusText) elements.sam2StatusText.textContent = conf.sam2Text;

      showToast('info', conf.toast);
    });
  });
}

// ========================== Preset Handler ==========================
function initPresets() {
  elements.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const preset = chip.dataset.preset;
      if (preset === 'quick') {
        elements.inputMaxFrames.value = '150';
        elements.inputFrameSkip.value = '2';
      } else if (preset === 'standard') {
        elements.inputMaxFrames.value = '300';
        elements.inputFrameSkip.value = '1';
      } else if (preset === 'full') {
        elements.inputMaxFrames.value = '';
        elements.inputFrameSkip.value = '1';
      }
    });
  });
}

// ========================== Sport Radio ==========================
function initSportRadios() {
  elements.sportRadios.forEach(radio => {
    radio.addEventListener('click', () => {
      elements.sportRadios.forEach(r => r.classList.remove('active'));
      radio.classList.add('active');
      selectedSportOverride = radio.dataset.sport;
      const input = radio.querySelector('input[type="radio"]');
      if (input) input.checked = true;
    });
  });
}

// ========================== Dynamic Stage Stepper ==========================
function renderStageSteps(stages) {
  const container = elements.stageStepsContainer;
  container.innerHTML = '';
  container.style.gridTemplateColumns = `repeat(${stages.length}, 1fr)`;
  stages.forEach(s => {
    const step = document.createElement('div');
    step.className = 'stage-step';
    step.dataset.stage = s.num;
    step.innerHTML = `<span class="stage-num">${String(s.num).padStart(2, '0')}</span><span class="stage-label">${s.label}</span>`;
    container.appendChild(step);
  });
}

function updateStageSteps(currentStage) {
  const steps = elements.stageStepsContainer.querySelectorAll('.stage-step');
  steps.forEach(step => {
    const stageNum = parseInt(step.dataset.stage);
    if (stageNum < currentStage) {
      step.classList.remove('active');
      step.classList.add('completed');
    } else if (stageNum === currentStage) {
      step.classList.add('active');
      step.classList.remove('completed');
    } else {
      step.classList.remove('active', 'completed');
    }
  });
}

function updateStageStepperForSport(sport) {
  const stages = sport === "cricket" ? CRICKET_STAGES : BASKETBALL_STAGES;
  renderStageSteps(stages);
}

// ========================== Upload Zone ==========================
function initUploadZone() {
  const dropZone = elements.uploadDropZone;
  const fileInput = elements.uploadInput;

  ['dragenter', 'dragover'].forEach(e => {
    dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
  });
  ['dragleave', 'drop'].forEach(e => {
    dropZone.addEventListener(e, (ev) => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target.closest('.upload-progress-wrapper')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
    fileInput.value = '';
  });
}

function uploadFile(file) {
  const validExts = ['.mp4', '.mov', '.avi', '.mkv', '.m4v'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) {
    showToast('error', `Unsupported format: ${ext}`);
    return;
  }
  if (file.size > 2 * 1024 * 1024 * 1024) {
    showToast('error', 'File too large (max 2GB)');
    return;
  }

  elements.uploadProgressWrapper.style.display = 'block';
  elements.uploadProgressFill.style.width = '0%';
  elements.uploadProgressText.textContent = 'Preparing upload...';

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      elements.uploadProgressFill.style.width = pct + '%';
      const sizeMB = (e.loaded / (1024 * 1024)).toFixed(1);
      const totalMB = (e.total / (1024 * 1024)).toFixed(1);
      elements.uploadProgressText.textContent = `Uploading... ${pct}% (${sizeMB} / ${totalMB} MB)`;
    }
  };

  xhr.onload = () => {
    setTimeout(() => {
      elements.uploadProgressWrapper.style.display = 'none';
      elements.uploadProgressFill.style.width = '0%';
      elements.uploadProgressText.textContent = '';
    }, 1500);

    if (xhr.status === 200) {
      showToast('success', `"${file.name}" uploaded successfully!`);
      loadVideos(file.name);
    } else {
      try {
        const err = JSON.parse(xhr.responseText);
        showToast('error', err.detail || 'Upload failed');
      } catch {
        showToast('error', 'Upload failed');
      }
    }
  };

  xhr.onerror = () => {
    elements.uploadProgressWrapper.style.display = 'none';
    showToast('error', 'Upload error — network failure');
  };

  xhr.open('POST', '/api/upload');
  xhr.send(formData);
}

// ========================== System Info ==========================
async function loadSystemInfo() {
  try {
    const res = await fetch('/api/system');
    const data = await res.json();
    if (data.cuda_available) {
      elements.systemGpuText.textContent = `${data.gpu_name} (${data.vram_gb} GB VRAM) • CUDA Active`;
      elements.systemStatusDot.classList.remove('inactive');

      // Auto-suggest hardware profile based on detected VRAM / GPU model
      const gpuLower = (data.gpu_name || '').toLowerCase();
      if (data.vram_gb <= 2.5 || gpuLower.includes('mx450') || gpuLower.includes('mx350') || gpuLower.includes('mx')) {
        const mxCard = document.querySelector('.hardware-card[data-gpu="mx450"]');
        if (mxCard) mxCard.click();
      } else if (data.vram_gb <= 6.5 || gpuLower.includes('3050') || gpuLower.includes('1650') || gpuLower.includes('2060')) {
        const rtxCard = document.querySelector('.hardware-card[data-gpu="rtx3050"]');
        if (rtxCard) rtxCard.click();
      }
    } else {
      elements.systemGpuText.textContent = `CPU Mode (${data.gpu_name})`;
      elements.systemStatusDot.classList.add('inactive');
      // For CPU mode, default to MX450 ultra-light mode
      const mxCard = document.querySelector('.hardware-card[data-gpu="mx450"]');
      if (mxCard) mxCard.click();
    }
  } catch (err) {
    console.error('Failed to load system info:', err);
    elements.systemGpuText.textContent = 'SportsVision Ready';
  }
}

// ========================== Load Videos ==========================
async function loadVideos(targetFilename = null) {
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    elements.videoSelect.innerHTML = '';

    if (!data.videos || data.videos.length === 0) {
      elements.videoSelect.innerHTML = '<option value="">No videos found — upload one above</option>';
      return;
    }

    data.videos.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.path;
      opt.dataset.stem = v.stem;
      opt.dataset.filename = v.name;
      opt.dataset.hasResults = v.has_results;
      opt.dataset.sport = v.detected_sport || 'unknown';
      const sportIcon = v.detected_sport === 'basketball' ? '🏀' : (v.detected_sport === 'cricket' ? '🏏' : '🏟️');
      opt.textContent = `${sportIcon} ${v.name} (${v.size_mb} MB)${v.has_results ? ' ★ [Analyzed]' : ''}`;
      elements.videoSelect.appendChild(opt);
    });

    if (targetFilename) {
      for (let i = 0; i < elements.videoSelect.options.length; i++) {
        const opt = elements.videoSelect.options[i];
        if (opt.dataset.filename === targetFilename || opt.dataset.stem === targetFilename) {
          elements.videoSelect.selectedIndex = i;
          break;
        }
      }
    }

    onVideoSelectionChange();
  } catch (err) {
    console.error('Failed to load videos:', err);
  }
}

elements.videoSelect.addEventListener('change', onVideoSelectionChange);
elements.refreshVideosBtn.addEventListener('click', () => loadVideos());

function onVideoSelectionChange() {
  const selectedOpt = elements.videoSelect.selectedOptions[0];
  if (!selectedOpt) return;

  currentVideoStem = selectedOpt.dataset.stem || "";
  const hasResults = selectedOpt.dataset.hasResults === "true";
  const videoSport = selectedOpt.dataset.sport || "unknown";

  if (hasResults) {
    elements.hasSavedBadge.style.display = 'inline-flex';
    elements.btnLoadSaved.style.display = 'inline-flex';
  } else {
    elements.hasSavedBadge.style.display = 'none';
    elements.btnLoadSaved.style.display = 'none';

    // Clear stale outputs from prior video (e.g. FIBA) so unanalyzed video is clean
    elements.annotatedVideoPlayer.removeAttribute('src');
    elements.annotatedVideoPlayer.load();
    elements.highlightsVideoPlayer.removeAttribute('src');
    elements.highlightsVideoPlayer.load();

    elements.bbSection.style.display = 'none';
    elements.ckSection.style.display = 'none';
    elements.analyticsPlaceholder.style.display = 'block';

    elements.bbHeatmapsSection.style.display = 'none';
    elements.ckVisualsSection.style.display = 'none';
    elements.heatmapPlaceholder.style.display = 'block';
  }

  // Update sport badge hint if known
  if (videoSport !== "unknown") {
    updateSportBadge(videoSport);
  } else {
    updateSportBadge(selectedSportOverride);
  }

  // Set raw video source for preview
  const rawFileName = selectedOpt.dataset.filename || (selectedOpt.value ? selectedOpt.value.split(/[\\/]/).pop() : "");
  if (rawFileName) {
    elements.rawVideoPlayer.src = `/api/video/raw/${encodeURIComponent(rawFileName)}`;
    elements.rawVideoPlayer.load();
  }
}

// ========================== Run Pipeline ==========================
elements.btnRunPipeline.addEventListener('click', async () => {
  const videoPath = elements.videoSelect.value;
  if (!videoPath) {
    showToast('error', 'Please select or upload a video first.');
    return;
  }

  const formData = new FormData();
  formData.append('video_path', videoPath);
  formData.append('sport', selectedSportOverride);

  if (elements.inputMaxFrames.value) {
    formData.append('max_frames', elements.inputMaxFrames.value);
  }
  if (elements.inputFrameSkip.value) {
    formData.append('frame_skip', elements.inputFrameSkip.value);
  }
  if (elements.inputBatchSize.value) {
    formData.append('batch_size', elements.inputBatchSize.value);
  }
  if (elements.inputUseSam2) {
    formData.append('use_sam2', elements.inputUseSam2.checked ? 'true' : 'false');
  }

  setPipelineRunningUI(true);

  try {
    const res = await fetch('/api/run', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Could not start pipeline');
    }
    showToast('info', 'Pipeline started! Watch the terminal for live progress.');
    startLogStreaming();
  } catch (err) {
    showToast('error', 'Failed to start: ' + err.message);
    setPipelineRunningUI(false);
  }
});

// ========================== Stop Pipeline ==========================
elements.btnStopPipeline.addEventListener('click', async () => {
  if (!confirm('Are you sure you want to stop the analytics pipeline?')) return;
  try {
    await fetch('/api/stop', { method: 'POST' });
    showToast('info', 'Pipeline aborted.');
  } catch (err) {
    console.error('Stop error:', err);
  }
});

// ========================== Load Saved Results ==========================
elements.btnLoadSaved.addEventListener('click', async () => {
  if (!currentVideoStem) return;
  await fetchAndDisplayResults(currentVideoStem);
  switchTab('theater-pane');
});

// ========================== Pipeline UI State ==========================
function setPipelineRunningUI(isRunning) {
  elements.btnRunPipeline.disabled = isRunning;
  elements.btnRunPipeline.style.display = isRunning ? 'none' : 'inline-flex';
  elements.btnStopPipeline.style.display = isRunning ? 'inline-flex' : 'none';

  if (isRunning) {
    elements.progressFill.style.width = '0%';
    elements.progressPctText.textContent = '0%';
    elements.progressFramesText.textContent = '0 / --';
    elements.metricFps.textContent = '0.0';
    elements.metricEta.textContent = 'Starting...';
    elements.metricVram.textContent = 'Loading...';
    detectedSport = "unknown";
    updateSportBadge("unknown");
  }
}

// ========================== SSE Log Streaming ==========================
function startLogStreaming() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/api/stream-logs');

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);

      if (data.type === 'init') {
        if (data.logs) {
          elements.terminalBody.innerHTML = '';
          data.logs.forEach(appendTerminalLine);
        }
        if (data.status) updateStatusMetrics(data.status);
      } else if (data.type === 'log') {
        appendTerminalLine(data.line);
        if (data.status) updateStatusMetrics(data.status);
      } else if (data.type === 'complete') {
        appendTerminalLine('[✓] Analytics Pipeline Completed Successfully!');
        setPipelineRunningUI(false);
        if (eventSource) eventSource.close();
        loadVideos();
        fetchAndDisplayResults(data.video_name);
        showToast('success', 'Analysis complete! Switching to Video Theater.');
        switchTab('theater-pane');
      } else if (data.type === 'error' || data.type === 'aborted') {
        setPipelineRunningUI(false);
        if (eventSource) eventSource.close();
        if (data.error) showToast('error', data.error);
      }
    } catch (err) {
      console.error('SSE parse error:', err);
    }
  };

  eventSource.onerror = () => {
    console.warn('SSE connection closed or lost.');
    if (eventSource) eventSource.close();
  };
}

function appendTerminalLine(text) {
  const line = document.createElement('div');
  line.className = 'log-line';

  if (/\[\d+\/\d+\]/.test(text)) {
    line.className += ' cyan';
  } else if (text.includes('Dynamic Team Colors') || text.includes('Exported Heatmaps') || text.includes('Pipeline execution complete') || text.includes('[✓]') || text.includes('Analytics Report Saved')) {
    line.className += ' green';
  } else if (text.includes('WARNING') || text.includes('deprecated')) {
    line.className += ' yellow';
  } else if (text.includes('ERROR') || text.includes('CRITICAL')) {
    line.className += ' orange';
  }

  line.textContent = text;
  elements.terminalBody.appendChild(line);
  elements.terminalBody.scrollTop = elements.terminalBody.scrollHeight;
}

elements.clearLogsBtn.addEventListener('click', () => {
  elements.terminalBody.innerHTML = '';
});

function updateStatusMetrics(status) {
  // Sport detection update
  if (status.detected_sport && status.detected_sport !== "unknown" && status.detected_sport !== detectedSport) {
    detectedSport = status.detected_sport;
    updateSportBadge(detectedSport);
    updateStageStepperForSport(detectedSport);
  }

  if (status.stage) updateStageSteps(status.stage);
  if (status.stage_name) {
    elements.metricStageName.textContent = `Stage ${status.stage}: ${status.stage_name}`;
  }
  if (status.progress_pct !== undefined) {
    elements.progressFill.style.width = `${status.progress_pct}%`;
    elements.progressPctText.textContent = `${status.progress_pct}%`;
  }
  if (status.current_frame !== undefined && status.total_frames !== undefined) {
    elements.progressFramesText.textContent = `${status.current_frame} / ${status.total_frames} frames`;
  }
  if (status.fps) elements.metricFps.textContent = status.fps;
  if (status.eta) elements.metricEta.textContent = status.eta;
  if (status.vram) elements.metricVram.textContent = status.vram;
}

function updateSportBadge(sport) {
  const badge = elements.sportBadge;
  badge.className = 'sport-badge';
  if (sport === 'basketball') {
    badge.textContent = '🏀 Basketball';
    badge.classList.add('basketball');
  } else if (sport === 'cricket') {
    badge.textContent = '🏏 Cricket';
    badge.classList.add('cricket');
  } else {
    badge.textContent = '🏟️ Auto-Detect';
  }
}

// ========================== Pipeline Status Check ==========================
async function checkCurrentPipelineStatus() {
  try {
    const res = await fetch('/api/status');
    const status = await res.json();
    if (status.is_running) {
      setPipelineRunningUI(true);
      startLogStreaming();
    }
  } catch (err) {
    console.error('Status check error:', err);
  }
}

// ========================== Results Display ==========================
async function fetchAndDisplayResults(videoStem) {
  try {
    const res = await fetch(`/api/results/${videoStem}`);
    const results = await res.json();
    currentResults = results;
    detectedSport = results.sport || "unknown";
    updateSportBadge(detectedSport);

    // Load annotated video
    if (results.annotated_video) {
      elements.annotatedVideoPlayer.src = `/api/video/output/${results.annotated_video}`;
      elements.annotatedVideoPlayer.load();
    }

    // Load highlights video
    if (results.highlights_video) {
      elements.highlightsVideoPlayer.src = `/api/video/output/${results.highlights_video}`;
      elements.highlightsVideoPlayer.load();
    }

    // Render sport-specific content
    if (detectedSport === 'basketball') {
      renderBasketballResults(results);
    } else if (detectedSport === 'cricket') {
      renderCricketResults(results);
    }

    renderTheaterFeatureCards(detectedSport);
    renderHighlightDescription(detectedSport);

  } catch (err) {
    console.error('Failed to display results:', err);
  }
}

// ========================== Basketball Results ==========================
function renderBasketballResults(results) {
  // Show basketball, hide cricket
  elements.bbSection.style.display = 'block';
  elements.ckSection.style.display = 'none';
  elements.analyticsPlaceholder.style.display = 'none';

  // Heatmaps
  if (results.team1_heatmap && results.team2_heatmap) {
    elements.heatmapPlaceholder.style.display = 'none';
    elements.bbHeatmapsSection.style.display = 'block';
    elements.ckVisualsSection.style.display = 'none';
    elements.heatmapTeam1Img.src = `/api/image/${results.team1_heatmap}?t=${Date.now()}`;
    elements.heatmapTeam2Img.src = `/api/image/${results.team2_heatmap}?t=${Date.now()}`;
  }

  if (results.analytics) {
    populateBasketballAnalyticsUI(results.analytics);
  }
}

function populateBasketballAnalyticsUI(data) {
  elements.statDuration.textContent = `${data.duration_seconds || 0}s`;
  elements.statTotalFrames.textContent = data.total_frames || 0;
  elements.statShotsDetected.textContent = data.shot_events_detected || 0;

  const t1 = data.team_stats ? data.team_stats.team_1 : { possession_pct: 0, total_passes: 0, total_interceptions: 0 };
  const t2 = data.team_stats ? data.team_stats.team_2 : { possession_pct: 0, total_passes: 0, total_interceptions: 0 };

  elements.statPossessionT1.textContent = `${t1.possession_pct || 0}%`;
  elements.statPossessionT2.textContent = `${t2.possession_pct || 0}%`;
  elements.possessionFillT1.style.width = `${t1.possession_pct || 0}%`;
  elements.possessionFillT2.style.width = `${t2.possession_pct || 0}%`;

  elements.statPassesT1.textContent = t1.total_passes || 0;
  elements.statPassesT2.textContent = t2.total_passes || 0;
  elements.statInterceptT1.textContent = t1.total_interceptions || 0;
  elements.statInterceptT2.textContent = t2.total_interceptions || 0;

  // Player Leaderboard
  elements.playerTableBody.innerHTML = '';
  if (data.player_stats) {
    const players = Object.entries(data.player_stats).map(([pid, pdata]) => ({ id: pid, ...pdata }));
    players.sort((a, b) => (b.total_distance_m || 0) - (a.total_distance_m || 0));

    players.forEach(p => {
      const tr = document.createElement('tr');
      const teamClass = p.team === 1 ? 'team1' : (p.team === 2 ? 'team2' : 'ref');
      const teamLabel = p.team === 1 ? 'Team 1' : (p.team === 2 ? 'Team 2' : 'Referee');

      tr.innerHTML = `
        <td style="font-weight: 700; font-family: var(--font-mono);">#${p.id}</td>
        <td><span class="player-team-pill ${teamClass}">${teamLabel}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${p.total_distance_m || 0} m</td>
        <td style="font-family: var(--font-mono);">${p.avg_speed_kmh || 0} km/h</td>
        <td style="font-family: var(--font-mono); color: var(--accent-cyan); font-weight: 700;">${p.max_speed_kmh || 0} km/h</td>
      `;
      elements.playerTableBody.appendChild(tr);
    });
  }
}

// ========================== Cricket Results ==========================
function renderCricketResults(results) {
  // Show cricket, hide basketball
  elements.ckSection.style.display = 'block';
  elements.bbSection.style.display = 'none';
  elements.analyticsPlaceholder.style.display = 'none';

  // Cricket visuals
  elements.heatmapPlaceholder.style.display = 'none';
  elements.bbHeatmapsSection.style.display = 'none';
  elements.ckVisualsSection.style.display = 'block';

  if (results.pitch_map) {
    elements.cricketPitchMapImg.src = `/api/image/${results.pitch_map}?t=${Date.now()}`;
  }
  if (results.ground_radar) {
    elements.cricketGroundRadarImg.src = `/api/image/${results.ground_radar}?t=${Date.now()}`;
  }

  if (results.analytics) {
    populateCricketAnalyticsUI(results.analytics);
  }
}

function populateCricketAnalyticsUI(data) {
  // The cricket JSON has nested structure: metadata, summary, tactical_distributions, etc.
  const summary = data.summary || {};
  const tactical = data.tactical_distributions || {};

  // Match summary cards
  elements.cricketDeliveries.textContent = summary.total_deliveries || 0;
  elements.cricketOvers.textContent = `${summary.overs || '0.0'} Overs`;
  elements.cricketWickets.textContent = summary.wickets || 0;
  elements.cricketFours.textContent = summary.fours || 0;
  elements.cricketSixes.textContent = summary.sixes || 0;

  // Speed gun
  const speeds = summary.speed_analysis_kmh || {};
  elements.cricketFastestSpeed.textContent = speeds.fastest_delivery || '0.0';
  elements.cricketAvgSpeed.textContent = speeds.average_delivery || '0.0';
  elements.cricketSlowestSpeed.textContent = speeds.slowest_delivery || '0.0';

  // Extra stats
  elements.cricketDotPct.textContent = `${summary.dot_ball_pct || 0}%`;
  elements.cricketRuns.textContent = summary.estimated_runs || 0;
  elements.cricketHighlightCount.textContent = summary.audio_visual_highlights_count || 0;

  // Distribution bars
  renderDistributionBars(elements.lineDistContainer, tactical.line || {});
  renderDistributionBars(elements.lengthDistContainer, tactical.length || {});
  renderDistributionBars(elements.shotTypeContainer, tactical.shot_types || {});

  // DRS verdicts from deliveries
  const deliveries = data.deliveries || [];
  renderDRSCards(deliveries);

  // Commentary log
  const commentary = data.commentary_transcript || [];
  renderCommentaryLog(commentary);

  // Deliveries Telemetry Table
  renderCricketDeliveriesTable(deliveries);
}

// ========================== Distribution Bars ==========================
function renderDistributionBars(container, data) {
  container.innerHTML = '';
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  if (total === 0) {
    container.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 16px; font-size: 0.85rem;">No data available</div>';
    return;
  }
  const sorted = Object.entries(data).sort(([, a], [, b]) => b - a);
  sorted.forEach(([label, count]) => {
    const pct = total > 0 ? (count / total * 100) : 0;
    const bar = document.createElement('div');
    bar.className = 'dist-bar-row';
    // Format label: replace underscores, capitalize
    const displayLabel = label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    bar.innerHTML = `
      <span class="dist-label">${displayLabel}</span>
      <div class="dist-bar-track">
        <div class="dist-bar-fill" style="width: ${pct}%;"></div>
      </div>
      <span class="dist-pct">${pct.toFixed(0)}%</span>
    `;
    container.appendChild(bar);
  });
}

// ========================== DRS Verdict Cards ==========================
function renderDRSCards(deliveries) {
  const drsDeliveries = deliveries.filter(d => d.drs_adjudication);
  if (drsDeliveries.length === 0) {
    elements.drsSection.style.display = 'none';
    return;
  }
  elements.drsSection.style.display = 'block';
  elements.drsVerdictContainer.innerHTML = '';

  drsDeliveries.forEach((del, idx) => {
    const drs = del.drs_adjudication;

    const getBoxClass = (val) => {
      if (!val) return '';
      const v = val.toLowerCase();
      if (v.includes('in-line') || v.includes('in_line') || v.includes('hitting')) return 'in-line';
      if (v.includes('missing')) return 'missing';
      if (v.includes("umpire")) return 'umpires-call';
      if (v.includes('outside')) return 'missing';
      return '';
    };

    const card = document.createElement('div');
    card.innerHTML = `
      <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; font-weight: 600;">
        Delivery ${del.delivery_number || (idx + 1)} ${del.over_str ? `(Over ${del.over_str})` : ''}
      </div>
      <div class="drs-verdict-card">
        <div class="drs-box ${getBoxClass(drs.pitching)}">
          <div class="drs-box-title">Pitching</div>
          <div class="drs-box-verdict">${drs.pitching || 'N/A'}</div>
        </div>
        <div class="drs-box ${getBoxClass(drs.impact)}">
          <div class="drs-box-title">Impact</div>
          <div class="drs-box-verdict">${drs.impact || 'N/A'}</div>
        </div>
        <div class="drs-box ${getBoxClass(drs.wickets)}">
          <div class="drs-box-title">Wickets</div>
          <div class="drs-box-verdict">${drs.wickets || 'N/A'}</div>
        </div>
      </div>
      <div class="drs-decision-banner ${drs.final_verdict && drs.final_verdict.toUpperCase().includes('OUT') && !drs.final_verdict.toUpperCase().includes('NOT') ? 'out' : 'not-out'}">
        Decision: ${drs.final_verdict || 'N/A'}
      </div>
    `;
    elements.drsVerdictContainer.appendChild(card);
  });
}

// ========================== Commentary Log ==========================
function renderCommentaryLog(commentary) {
  elements.commentaryLog.innerHTML = '';
  if (!commentary || commentary.length === 0) {
    elements.commentaryLog.innerHTML = '<div class="commentary-empty">No commentary data generated for this video.</div>';
    return;
  }
  commentary.forEach((line, idx) => {
    const entry = document.createElement('div');
    entry.className = 'commentary-entry';
    entry.innerHTML = `<span class="commentary-over-marker">${idx + 1}.</span>${line}`;
    elements.commentaryLog.appendChild(entry);
  });
}

// ========================== Cricket Deliveries Table ==========================
function renderCricketDeliveriesTable(deliveries) {
  const tbody = elements.cricketDeliveriesTableBody;
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!deliveries || deliveries.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 30px;">
          No delivery data available for this match.
        </td>
      </tr>
    `;
    return;
  }

  deliveries.forEach((del, idx) => {
    const tr = document.createElement('tr');
    const delNum = del.delivery_number || (idx + 1);
    const overStr = del.over || del.over_str || `0.${delNum}`;
    const relSpeed = del.release_speed_kmh ? `${del.release_speed_kmh.toFixed(1)}` : '--';
    const bncSpeed = del.bounce_speed_kmh ? `${del.bounce_speed_kmh.toFixed(1)}` : '--';
    const speedDisplay = `${relSpeed} / ${bncSpeed} km/h`;
    const lineStr = del.line || 'Middle';
    const lengthStr = del.length || 'Good Length';
    const shot = del.shot_played ? del.shot_played.replace(/_/g, ' ') : 'N/A';

    let outcomeClass = 'dot';
    let outcomeText = 'Dot Ball';
    const isWicket = del.is_wicket || (del.outcome && del.outcome.toLowerCase().includes('wicket'));
    const runs = del.runs || 0;
    if (isWicket) {
      outcomeClass = 'wicket';
      outcomeText = 'WICKET';
    } else if (runs >= 4) {
      outcomeClass = 'boundary';
      outcomeText = runs === 6 ? 'SIX (6)' : 'FOUR (4)';
    } else if (runs > 0) {
      outcomeClass = 'runs';
      outcomeText = `${runs} Run${runs > 1 ? 's' : ''}`;
    }

    const decel = del.pitch_deceleration_pct !== undefined && del.pitch_deceleration_pct !== null
      ? `${del.pitch_deceleration_pct.toFixed(1)}%`
      : '--';

    const xd = del.expected_dismissal_xd || {};
    const threatLevel = (xd.threat_level || 'LOW').toUpperCase();
    const xdPct = xd.xd_pct !== undefined ? `${xd.xd_pct.toFixed(1)}%` : (xd.xd_score ? `${(xd.xd_score * 100).toFixed(1)}%` : '--');
    const xdClass = threatLevel === 'HIGH' ? 'high' : (threatLevel === 'MEDIUM' ? 'medium' : 'low');

    tr.innerHTML = `
      <td style="font-weight: 700; font-family: var(--font-mono); color: var(--accent-cyan);">#${delNum}</td>
      <td style="font-family: var(--font-mono);">${overStr}</td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${speedDisplay}</td>
      <td>
        <span style="color: var(--text-primary); font-weight: 500;">${lengthStr}</span>
        <span style="color: var(--text-muted); font-size: 0.75rem; display: block;">${lineStr}</span>
      </td>
      <td style="text-transform: capitalize;">${shot}</td>
      <td><span class="outcome-pill ${outcomeClass}">${outcomeText}</span></td>
      <td style="font-family: var(--font-mono);">${decel}</td>
      <td><span class="xd-pill ${xdClass}">${threatLevel} (${xdPct})</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ========================== Theater Feature Cards ==========================
function renderTheaterFeatureCards(sport) {
  const container = elements.theaterFeatureCards;
  if (sport === 'basketball') {
    container.innerHTML = `
      <div class="stat-box cyan">
        <div class="stat-box-title">Player Tracking & SAM2</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          YOLO + BoT-SORT bounding boxes with Meta SAM2 instance segmentation.
        </div>
      </div>
      <div class="stat-box orange">
        <div class="stat-box-title">Ball Path & Glowing Tail</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          Custom YOLOv8 with physics-aware trajectory smoothing & parabolic shot detection.
        </div>
      </div>
      <div class="stat-box green">
        <div class="stat-box-title">Dynamic Jersey Clustering</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          Pixel K-Means automatically separates Team 1 (Red) vs Team 2 (Green) dynamically.
        </div>
      </div>
      <div class="stat-box purple">
        <div class="stat-box-title">2D Tactical Minimap</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          Homography transformation projects court keypoints onto 2D top-down view.
        </div>
      </div>
    `;
  } else if (sport === 'cricket') {
    container.innerHTML = `
      <div class="stat-box cyan">
        <div class="stat-box-title">Ball Flight & Kalman Tracking</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          4-state ballistic Kalman filter with CLAHE-enhanced YOLOv8 ball detection.
        </div>
      </div>
      <div class="stat-box orange">
        <div class="stat-box-title">Hawk-Eye DRS & LBW</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          ICC Law 36 adjudication with 3D trajectory extrapolation to stumps.
        </div>
      </div>
      <div class="stat-box green">
        <div class="stat-box-title">Pitch Map & Line/Length</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          22-yard homography pitch mapping with line and length classification.
        </div>
      </div>
      <div class="stat-box purple">
        <div class="stat-box-title">Speed Gun & Shot Type</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
          3D ballistic kinematics speed calculation with CricShot10 stroke classification.
        </div>
      </div>
    `;
  } else {
    container.innerHTML = `
      <div class="stat-box cyan" style="grid-column: span 4;">
        <div class="stat-box-title">Run a pipeline to see sport-specific feature details</div>
      </div>
    `;
  }
}

// ========================== Highlight Description ==========================
function renderHighlightDescription(sport) {
  if (sport === 'basketball') {
    elements.highlightMethodBadge.textContent = 'Parabolic Shot Detection';
    elements.highlightDescription.innerHTML = `
      <strong>How highlights work:</strong> The geometric shot detection engine isolates high-confidence parabolic shot trajectories
      intersecting the hoop regions (4 seconds before release to 2 seconds after completion) and automatically extracts seamless highlight reels.
    `;
  } else if (sport === 'cricket') {
    elements.highlightMethodBadge.textContent = 'Audio-Visual Fusion';
    elements.highlightDescription.innerHTML = `
      <strong>How highlights work:</strong> Multi-modal audio-visual fusion combining visual event timestamps (wickets, boundaries, dismissals)
      with acoustic energy spike detection (crowd roars, bat impact) to compile broadcast-quality highlight reels.
    `;
  }
}

// ========================== Dual Video Sync ==========================
function initVideoSync() {
  const vRaw = elements.rawVideoPlayer;
  const vAnn = elements.annotatedVideoPlayer;

  elements.btnPlayPauseSync.addEventListener('click', () => {
    if (vAnn.paused) {
      vAnn.play().catch(e => console.log(e));
      vRaw.play().catch(e => console.log(e));
      elements.syncPlayIcon.textContent = '⏸ Pause Synchronized';
    } else {
      vAnn.pause();
      vRaw.pause();
      elements.syncPlayIcon.textContent = '▶ Play Synchronized';
    }
  });

  vAnn.addEventListener('seeking', () => {
    if (!isSyncingVideos) {
      isSyncingVideos = true;
      vRaw.currentTime = vAnn.currentTime;
      isSyncingVideos = false;
    }
  });

  vAnn.addEventListener('timeupdate', () => {
    const cur = formatVideoTime(vAnn.currentTime);
    const dur = formatVideoTime(vAnn.duration || 0);
    elements.syncTimeDisplay.textContent = `${cur} / ${dur}`;
  });

  elements.playbackSpeedSelect.addEventListener('change', (e) => {
    const speed = parseFloat(e.target.value);
    vAnn.playbackRate = speed;
    vRaw.playbackRate = speed;
  });
}

function formatVideoTime(seconds) {
  if (isNaN(seconds)) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
