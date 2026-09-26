// CourtVision Frontend Application Logic

let eventSource = null;
let currentVideoStem = "";
let currentResults = null;
let isSyncingVideos = false;

// DOM Elements
const elements = {
  navTabs: document.querySelectorAll('.nav-tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  systemGpuText: document.getElementById('systemGpuText'),
  systemStatusDot: document.getElementById('systemStatusDot'),
  videoSelect: document.getElementById('videoSelect'),
  refreshVideosBtn: document.getElementById('refreshVideosBtn'),
  btnRunPipeline: document.getElementById('btnRunPipeline'),
  btnStopPipeline: document.getElementById('btnStopPipeline'),
  btnLoadSaved: document.getElementById('btnLoadSaved'),
  hasSavedBadge: document.getElementById('hasSavedBadge'),
  uploadInput: document.getElementById('videoUploadInput'),
  uploadBtn: document.getElementById('uploadBtn'),

  // Preset Chips
  presetChips: document.querySelectorAll('.preset-chip'),
  inputMaxFrames: document.getElementById('inputMaxFrames'),
  inputFrameSkip: document.getElementById('inputFrameSkip'),
  inputBatchSize: document.getElementById('inputBatchSize'),

  // Stages & Progress
  stageSteps: document.querySelectorAll('.stage-step'),
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

  // Analytics Elements
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

  // Heatmaps
  heatmapTeam1Img: document.getElementById('heatmapTeam1Img'),
  heatmapTeam2Img: document.getElementById('heatmapTeam2Img'),
  heatmapPlaceholder: document.getElementById('heatmapPlaceholder'),
  heatmapContainer: document.getElementById('heatmapContainer'),
};

// Initialize App
document.addEventListener('DOMContentLoaded', async () => {
  initTabs();
  initPresets();
  initVideoSync();
  await loadSystemInfo();
  await loadVideos();
  checkCurrentPipelineStatus();
});

// Tab Navigation
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

// Preset Handler
function initPresets() {
  elements.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const preset = chip.dataset.preset;
      if (preset === 'quick') {
        elements.inputMaxFrames.value = '60';
        elements.inputFrameSkip.value = '2';
      } else if (preset === 'standard') {
        elements.inputMaxFrames.value = '180';
        elements.inputFrameSkip.value = '1';
      } else if (preset === 'full') {
        elements.inputMaxFrames.value = '';
        elements.inputFrameSkip.value = '1';
      }
    });
  });
}

// Fetch System Info
async function loadSystemInfo() {
  try {
    const res = await fetch('/api/system');
    const data = await res.json();
    if (data.cuda_available) {
      elements.systemGpuText.textContent = `${data.gpu_name} (${data.vram_gb} GB VRAM) • CUDA Active`;
      elements.systemStatusDot.classList.remove('inactive');
    } else {
      elements.systemGpuText.textContent = `CPU Mode (${data.gpu_name})`;
      elements.systemStatusDot.classList.add('inactive');
    }
  } catch (err) {
    console.error('Failed to load system info:', err);
    elements.systemGpuText.textContent = 'CourtVision Ready';
  }
}

// Load Videos
async function loadVideos() {
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    elements.videoSelect.innerHTML = '';

    if (!data.videos || data.videos.length === 0) {
      elements.videoSelect.innerHTML = '<option value="">No videos found</option>';
      return;
    }

    data.videos.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.path;
      opt.dataset.stem = v.stem;
      opt.dataset.hasResults = v.has_results;
      opt.textContent = `${v.name} (${v.size_mb} MB)${v.has_results ? ' ★ [Analyzed]' : ''}`;
      elements.videoSelect.appendChild(opt);
    });

    onVideoSelectionChange();
  } catch (err) {
    console.error('Failed to load videos:', err);
  }
}

elements.videoSelect.addEventListener('change', onVideoSelectionChange);
elements.refreshVideosBtn.addEventListener('click', loadVideos);

function onVideoSelectionChange() {
  const selectedOpt = elements.videoSelect.selectedOptions[0];
  if (!selectedOpt) return;

  currentVideoStem = selectedOpt.dataset.stem || "";
  const hasResults = selectedOpt.dataset.hasResults === "true";

  if (hasResults) {
    elements.hasSavedBadge.style.display = 'inline-flex';
    elements.btnLoadSaved.style.display = 'inline-flex';
  } else {
    elements.hasSavedBadge.style.display = 'none';
    elements.btnLoadSaved.style.display = 'none';
  }

  // Set raw video source
  const rawFileName = selectedOpt.textContent.split(' ')[0];
  elements.rawVideoPlayer.src = `/api/video/raw/${rawFileName}`;
  elements.rawVideoPlayer.load();
}

// Upload Handling
elements.uploadBtn.addEventListener('click', () => elements.uploadInput.click());
elements.uploadInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  elements.uploadBtn.textContent = 'Uploading...';
  elements.uploadBtn.disabled = true;

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    if (res.ok) {
      await loadVideos();
      alert(`Video "${file.name}" uploaded successfully!`);
    } else {
      alert('Upload failed.');
    }
  } catch (err) {
    alert('Upload error: ' + err.message);
  } finally {
    elements.uploadBtn.textContent = 'Upload Video';
    elements.uploadBtn.disabled = false;
    elements.uploadInput.value = '';
  }
});

// Run Pipeline
elements.btnRunPipeline.addEventListener('click', async () => {
  const videoPath = elements.videoSelect.value;
  if (!videoPath) {
    alert('Please select a video first.');
    return;
  }

  const formData = new FormData();
  formData.append('video_path', videoPath);

  if (elements.inputMaxFrames.value) {
    formData.append('max_frames', elements.inputMaxFrames.value);
  }
  if (elements.inputFrameSkip.value) {
    formData.append('frame_skip', elements.inputFrameSkip.value);
  }
  if (elements.inputBatchSize.value) {
    formData.append('batch_size', elements.inputBatchSize.value);
  }

  setPipelineRunningUI(true);

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Could not start pipeline');
    }
    startLogStreaming();
  } catch (err) {
    alert('Failed to start pipeline: ' + err.message);
    setPipelineRunningUI(false);
  }
});

// Stop Pipeline
elements.btnStopPipeline.addEventListener('click', async () => {
  if (!confirm('Are you sure you want to stop the analytics pipeline?')) return;
  try {
    await fetch('/api/stop', { method: 'POST' });
  } catch (err) {
    console.error('Stop error:', err);
  }
});

// Load Saved Results Button
elements.btnLoadSaved.addEventListener('click', async () => {
  if (!currentVideoStem) return;
  await fetchAndDisplayResults(currentVideoStem);
  switchTab('theater-pane');
});

// Set Running UI State
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
    resetStageSteps();
  }
}

function resetStageSteps() {
  elements.stageSteps.forEach(step => {
    step.classList.remove('active', 'completed');
  });
}

function updateStageSteps(currentStage) {
  elements.stageSteps.forEach(step => {
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

// Log Streaming via SSE
function startLogStreaming() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource('/api/stream-logs');

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);

      if (data.type === 'init') {
        if (data.logs) {
          elements.terminalBody.innerHTML = '';
          data.logs.forEach(appendTerminalLine);
        }
        if (data.status) {
          updateStatusMetrics(data.status);
        }
      } else if (data.type === 'log') {
        appendTerminalLine(data.line);
        if (data.status) {
          updateStatusMetrics(data.status);
        }
      } else if (data.type === 'complete') {
        appendTerminalLine('[✓] Analytics Pipeline Completed Successfully!');
        setPipelineRunningUI(false);
        if (eventSource) eventSource.close();
        loadVideos();
        fetchAndDisplayResults(data.video_name);
        alert('CourtVision Analysis Completed! Switching to Video Theater.');
        switchTab('theater-pane');
      } else if (data.type === 'error' || data.type === 'aborted') {
        setPipelineRunningUI(false);
        if (eventSource) eventSource.close();
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

  if (text.includes('[1/8]') || text.includes('[2/8]') || text.includes('[3/8]') || text.includes('[4/8]') || text.includes('[5/8]') || text.includes('[6/8]') || text.includes('[7/8]') || text.includes('[8/8]')) {
    line.className += ' cyan';
  } else if (text.includes('Dynamic Team Colors') || text.includes('Exported Heatmaps') || text.includes('Pipeline execution complete')) {
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
  if (status.stage) {
    updateStageSteps(status.stage);
  }
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

// Check if pipeline is currently running on page refresh
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

// Results Display
async function fetchAndDisplayResults(videoStem) {
  try {
    const res = await fetch(`/api/results/${videoStem}`);
    const results = await res.json();
    currentResults = results;

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

    // Load heatmaps
    if (results.team1_heatmap && results.team2_heatmap) {
      elements.heatmapPlaceholder.style.display = 'none';
      elements.heatmapContainer.style.display = 'grid';
      elements.heatmapTeam1Img.src = `/api/image/${results.team1_heatmap}?t=${Date.now()}`;
      elements.heatmapTeam2Img.src = `/api/image/${results.team2_heatmap}?t=${Date.now()}`;
    }

    // Load analytics JSON
    if (results.analytics) {
      populateAnalyticsUI(results.analytics);
    }

  } catch (err) {
    console.error('Failed to display results:', err);
  }
}

function populateAnalyticsUI(data) {
  // Duration & Frames
  elements.statDuration.textContent = `${data.duration_seconds || 0}s`;
  elements.statTotalFrames.textContent = data.total_frames || 0;
  elements.statShotsDetected.textContent = data.shot_events_detected || 0;

  // Team Stats
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

  // Player Leaderboard Table
  elements.playerTableBody.innerHTML = '';
  if (data.player_stats) {
    const players = Object.entries(data.player_stats).map(([pid, pdata]) => ({
      id: pid,
      ...pdata
    }));

    // Sort by distance descending
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

// Dual Player Sync Controls
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

  // Sync seeking
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

  // Playback Speed
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
