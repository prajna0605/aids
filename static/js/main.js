/**
 * ECG Insight AI — Frontend Interactions & Telemetry Visualizer (Vanilla JS)
 */

document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  initFileUpload();
  initFormSubmission();
  initTableFilters();
});


/**
 * Dynamic Sliders Live Labels
 */
function initSliders() {
  const sliders = document.querySelectorAll('.slider');
  sliders.forEach(slider => {
    const valueDisplay = document.getElementById(`${slider.id}_val`);
    if (valueDisplay) {
      slider.addEventListener('input', e => {
        const unit = slider.getAttribute('data-unit') || '';
        valueDisplay.textContent = `${e.target.value}${unit}`;
      });
    }
  });
}

/**
 * File Upload & Data Allowance Calculation
 */
function initFileUpload() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('custom_file_input');
  const fileInfo = document.getElementById('file_info_badge');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    const fileNameEl = document.getElementById('selected_file_name');
    const fileSizeEl = document.getElementById('selected_file_size');
    const allowanceEl = document.getElementById('allowance_usage_text');

    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileSizeEl) fileSizeEl.textContent = `${sizeMb} MB`;
    if (allowanceEl) allowanceEl.textContent = `${sizeMb} MB / 1,024 MB Allowance Used`;

    if (fileInfo) fileInfo.style.display = 'block';
  }
}


/**
 * Form Submission & Spinner Overlay
 */
function initFormSubmission() {
  const form = document.getElementById('analysis_form');
  const overlay = document.getElementById('loading_overlay');

  if (form && overlay) {
    form.addEventListener('submit', () => {
      overlay.classList.add('active');
    });
  }

  window.loadDemoPreset = function(mode) {
    const modeInput = document.getElementById('dataset_mode');
    if (modeInput) modeInput.value = mode;
    if (overlay) overlay.classList.add('active');
    if (form) form.submit();
  };
}

/**
 * Table Filtering (Normal / Anomaly)
 */
function initTableFilters() {
  const filterBtns = document.querySelectorAll('.table-filter-btn');
  const rows = document.querySelectorAll('.telemetry-table tbody tr');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.getAttribute('data-filter');

      rows.forEach(row => {
        const isAnomaly = row.getAttribute('data-anomaly') === 'true';
        if (filter === 'all') {
          row.style.display = '';
        } else if (filter === 'anomaly') {
          row.style.display = isAnomaly ? '' : 'none';
        } else if (filter === 'normal') {
          row.style.display = !isAnomaly ? '' : 'none';
        }
      });
    });
  });
}

/**
 * Render Interactive Plotly Telemetry Strip (Clean, Medical-Grade, De-cluttered)
 */
function renderECGPlot(containerId, telemetry) {
  const container = document.getElementById(containerId);
  if (!container || !window.Plotly) return;

  const { time, signal, r_peaks, predictions, anomaly_scores, lead_name } = telemetry;
  if (!time || time.length === 0) return;

  const maxTime = time[time.length - 1];
  const defaultWindow = Math.min(8.0, maxTime);

  const traces = [];

  // 1. Sleek Continuous ECG Waveform Line
  traces.push({
    x: time,
    y: signal,
    type: 'scatter',
    mode: 'lines',
    name: 'ECG Waveform',
    line: { color: '#0284c7', width: 1.8 },
    hoverinfo: 'x+y'
  });

  // Separate normal vs anomalous peaks
  const normX = [];
  const normY = [];
  const normText = [];

  const anoX = [];
  const anoY = [];
  const anoText = [];

  r_peaks.forEach((peakIdx, i) => {
    const t = time[peakIdx];
    const y = signal[peakIdx];
    const isAno = predictions[i] === 1;
    const score = anomaly_scores[i] !== undefined ? anomaly_scores[i].toFixed(3) : '0.000';

    if (isAno) {
      anoX.push(t);
      anoY.push(y);
      anoText.push(`<b>Flagged Arrhythmia Beat #${i + 1}</b><br>Time: ${t.toFixed(2)}s<br>Anomaly Score: ${score}`);
    } else {
      normX.push(t);
      normY.push(y);
      normText.push(`<b>Normal Sinus Beat #${i + 1}</b><br>Time: ${t.toFixed(2)}s<br>Anomaly Score: ${score}`);
    }
  });

  // 2. Normal Beat Markers (Subtle, clean emerald dots)
  if (normX.length > 0) {
    traces.push({
      x: normX,
      y: normY,
      type: 'scatter',
      mode: 'markers',
      name: 'Normal Beat',
      showlegend: false,
      marker: {
        symbol: 'circle',
        size: 6,
        color: '#10b981',
        line: { color: '#ffffff', width: 1 }
      },
      text: normText,
      hoverinfo: 'text'
    });
  }

  // 3. Anomalous Beat Markers (Crisp alert red diamonds - NO overlapping ⚠ text!)
  if (anoX.length > 0) {
    traces.push({
      x: anoX,
      y: anoY,
      type: 'scatter',
      mode: 'markers',
      name: 'Flagged Arrhythmia',
      showlegend: false,
      marker: {
        symbol: 'diamond',
        size: 8,
        color: '#ef4444',
        line: { color: '#ffffff', width: 1.5 }
      },
      text: anoText,
      hoverinfo: 'text'
    });
  }

  const layout = {
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#0f172a', family: 'Inter, -apple-system, sans-serif' },
    margin: { l: 50, r: 25, t: 15, b: 35 },
    height: 400,
    showlegend: false, // Prevents duplicate legend colliding with the card header
    xaxis: {
      title: { text: 'Time (seconds)', font: { size: 12, color: '#64748b' } },
      gridcolor: '#f8fafc',
      zerolinecolor: '#e2e8f0',
      range: [0, defaultWindow], // Crisp 8-second window by default instead of squished 60s
      rangeslider: {
        visible: true,
        thickness: 0.07,
        bgcolor: '#f8fafc',
        bordercolor: '#e2e8f0',
        borderwidth: 1
      }
    },
    yaxis: {
      title: { text: 'Amplitude (Normalized mV)', font: { size: 12, color: '#64748b' } },
      gridcolor: '#f8fafc',
      zerolinecolor: '#e2e8f0'
    },
    hovermode: 'closest'
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toggleSpikelines']
  };

  Plotly.newPlot(container, traces, layout, config).then(() => {
    updateRangeIndicator(container, maxTime);
  });

  // Listen to range changes from user dragging range slider or panning
  container.on('plotly_relayout', function() {
    updateRangeIndicator(container, maxTime);
  });

  // Global window adjustment handlers
  window.setECGWindow = function(windowSec, btnElement) {
    if (btnElement) {
      document.querySelectorAll('.telemetry-toolbar .btn-chip[data-window]').forEach(b => b.classList.remove('active'));
      btnElement.classList.add('active');
    }

    const currentRange = container.layout && container.layout.xaxis ? container.layout.xaxis.range : [0, defaultWindow];
    let start = currentRange ? currentRange[0] : 0;
    let end;

    if (windowSec === 'all') {
      start = 0;
      end = maxTime;
    } else {
      const sec = parseFloat(windowSec);
      end = Math.min(maxTime, start + sec);
      if (end - start < sec) {
        start = Math.max(0, end - sec);
      }
    }

    Plotly.relayout(container, { 'xaxis.range': [start, end] });
  };

  window.shiftECGWindow = function(direction) {
    const currentRange = container.layout && container.layout.xaxis ? container.layout.xaxis.range : [0, defaultWindow];
    if (!currentRange) return;

    const span = currentRange[1] - currentRange[0];
    const shift = span * 0.75 * direction;
    let newStart = Math.max(0, currentRange[0] + shift);
    let newEnd = newStart + span;

    if (newEnd > maxTime) {
      newEnd = maxTime;
      newStart = Math.max(0, newEnd - span);
    }

    Plotly.relayout(container, { 'xaxis.range': [newStart, newEnd] });
  };
}

/**
 * Updates the range badge indicator text
 */
function updateRangeIndicator(container, maxTime) {
  const badge = document.getElementById('window_range_badge');
  if (!badge || !container.layout || !container.layout.xaxis) return;

  const range = container.layout.xaxis.range;
  if (!range) return;

  const start = Math.max(0, range[0]).toFixed(1);
  const end = Math.min(maxTime, range[1]).toFixed(1);
  badge.textContent = `${start}s – ${end}s of ${maxTime.toFixed(1)}s`;
}

/**
 * Render RR Tachogram & Anomaly Score Subplots
 */
function renderTachogramPlot(containerId, telemetry) {
  const container = document.getElementById(containerId);
  if (!container || !window.Plotly) return;

  const { time, r_peaks, predictions, anomaly_scores, threshold } = telemetry;
  if (!r_peaks || r_peaks.length < 2) return;

  const beatTimes = r_peaks.map(idx => time[idx]);
  const rrIntervals = [];
  const rrTimes = [];

  for (let i = 1; i < r_peaks.length; i++) {
    const rrMs = (time[r_peaks[i]] - time[r_peaks[i - 1]]) * 1000;
    rrIntervals.push(rrMs);
    rrTimes.push(time[r_peaks[i]]);
  }

  const traceRR = {
    x: rrTimes,
    y: rrIntervals,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'RR Interval (ms)',
    line: { color: '#2563eb', width: 1.5 },
    marker: { size: 5, color: '#0284c7' },
    xaxis: 'x',
    yaxis: 'y'
  };

  const traceScores = {
    x: beatTimes,
    y: anomaly_scores,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Anomaly Score [0, 1]',
    line: { color: '#d97706', width: 1.5 },
    marker: {
      size: 5,
      color: predictions.map(p => (p === 1 ? '#dc2626' : '#059669'))
    },
    xaxis: 'x2',
    yaxis: 'y2'
  };

  const layout = {
    grid: { rows: 2, columns: 1, pattern: 'independent' },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#0f172a', family: 'Inter, sans-serif' },
    height: 380,
    margin: { l: 50, r: 25, t: 20, b: 40 },
    xaxis: { gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0' },
    yaxis: { title: 'RR (ms)', gridcolor: '#f1f5f9' },
    xaxis2: { title: 'Time (seconds)', gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0' },
    yaxis2: { title: 'Score', range: [-0.05, 1.05], gridcolor: '#f1f5f9' },
    showlegend: true,
    legend: { orientation: 'h', y: 1.08, x: 1, xanchor: 'right', bgcolor: 'rgba(255, 255, 255, 0.95)' },
    shapes: [
      {
        type: 'line',
        xref: 'x2',
        yref: 'y2',
        x0: beatTimes[0],
        x1: beatTimes[beatTimes.length - 1],
        y0: threshold || 0.5,
        y1: threshold || 0.5,
        line: { color: '#dc2626', dash: 'dash', width: 1.5 }
      }
    ]
  };

  Plotly.newPlot(container, [traceRR, traceScores], layout, { responsive: true, displaylogo: false });
}
