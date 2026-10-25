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

  // Allowance Slicing Mode Toggle
  const allowanceRadios = document.querySelectorAll('input[name="allowance_mode"]');
  const windowControls = document.getElementById('window_allowance_controls');

  allowanceRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      if (windowControls) {
        windowControls.style.display = radio.value === 'window' ? 'block' : 'none';
      }
    });
  });
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
 * Render Interactive Plotly Telemetry Strip
 */
function renderECGPlot(containerId, telemetry) {
  const container = document.getElementById(containerId);
  if (!container || !window.Plotly) return;

  const { time, signal, r_peaks, predictions, anomaly_scores, lead_name } = telemetry;

  const traces = [];

  // 1. Base ECG Trace
  traces.push({
    x: time,
    y: signal,
    type: 'scatter',
    mode: 'lines',
    name: `ECG (${lead_name || 'Lead'})`,
    line: { color: '#0284c7', width: 1.5 },
    hoverinfo: 'x+y'
  });

  // Separate normal vs anomalous peaks
  const normX = [];
  const normY = [];
  const normText = [];

  const anoX = [];
  const anoY = [];
  const anoText = [];

  const shadedRects = [];

  r_peaks.forEach((peakIdx, i) => {
    const t = time[peakIdx];
    const y = signal[peakIdx];
    const isAno = predictions[i] === 1;
    const score = anomaly_scores[i] ? anomaly_scores[i].toFixed(3) : '0.000';

    if (isAno) {
      anoX.push(t);
      anoY.push(y);
      anoText.push(`⚠ Flagged Anomaly<br>Beat #${i+1}<br>Score: ${score}`);

      // Add soft red highlight window
      shadedRects.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: Math.max(0, t - 0.20),
        x1: Math.min(time[time.length - 1], t + 0.20),
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(239, 68, 68, 0.12)',
        line: { width: 0 },
        layer: 'below'
      });
    } else {
      normX.push(t);
      normY.push(y);
      normText.push(`Normal Beat #${i+1}<br>Score: ${score}`);
    }
  });

  // 2. Normal Beat Markers
  if (normX.length > 0) {
    traces.push({
      x: normX,
      y: normY,
      type: 'scatter',
      mode: 'markers',
      name: 'Normal Beat',
      marker: {
        symbol: 'circle',
        size: 7,
        color: '#10b981',
        line: { color: '#ffffff', width: 1 }
      },
      text: normText,
      hoverinfo: 'text+x+y'
    });
  }

  // 3. Anomalous Beat Markers
  if (anoX.length > 0) {
    traces.push({
      x: anoX,
      y: anoY,
      type: 'scatter',
      mode: 'markers+text',
      name: 'Flagged Anomaly',
      marker: {
        symbol: 'diamond',
        size: 11,
        color: '#dc2626',
        line: { color: '#ffffff', width: 1.5 }
      },
      text: anoX.map(() => '⚠'),
      textposition: 'top center',
      textfont: { color: '#dc2626', size: 13 },
      hovertext: anoText,
      hoverinfo: 'text+x+y'
    });
  }

  const layout = {
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#0f172a', family: 'Inter, sans-serif' },
    margin: { l: 50, r: 25, t: 30, b: 40 },
    height: 420,
    xaxis: {
      title: 'Time (seconds)',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
      rangeslider: { visible: true, thickness: 0.08, bgcolor: '#f8fafc' }
    },
    yaxis: {
      title: 'Amplitude (Normalized mV)',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0'
    },
    legend: {
      orientation: 'h',
      y: 1.05,
      x: 1,
      xanchor: 'right',
      bgcolor: 'rgba(255, 255, 255, 0.95)',
      bordercolor: '#e2e8f0',
      borderwidth: 1
    },
    shapes: shadedRects
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  };

  Plotly.newPlot(container, traces, layout, config);
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
