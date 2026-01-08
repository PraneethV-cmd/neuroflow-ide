import React, { useMemo, useRef } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import './ModelVisualizerNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';

const width = 400;
const height = 300;
const padding = 40;

// --- Helper Math Functions ---

function calculateRSquared(actual, predicted) {
  if (actual.length === 0) return 0;
  const actualMean = actual.reduce((a, b) => a + b, 0) / actual.length;
  const ssRes = actual.reduce((sum, y, i) => sum + Math.pow(y - predicted[i], 2), 0);
  const ssTot = actual.reduce((sum, y) => sum + Math.pow(y - actualMean, 2), 0);
  if (ssTot === 0) return 1;
  return 1 - (ssRes / ssTot);
}

function calculateClassificationMetrics(actual, predicted, threshold = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  let correct = 0;

  actual.forEach((y, i) => {
    const p = predicted[i];
    if (y === p) correct++;
    if ((y === 0 || y === 1) && (p === 0 || p === 1)) {
      if (y === 1 && p === 1) tp++;
      else if (y === 0 && p === 0) tn++;
      else if (y === 0 && p === 1) fp++;
      else if (y === 1 && p === 0) fn++;
    }
  });

  const accuracy = correct / actual.length;
  return { accuracy, tp, tn, fp, fn };
}

function safeDomain(domain) {
  if (!domain || !Array.isArray(domain) || domain.length < 2) return [0, 1];
  const [min, max] = domain;
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  const range = max - min;
  return [min - range * 0.1, max + range * 0.1];
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Helper to generate polynomial features (matching Scikit-Learn order)
function generatePolynomialFeatures(X, degree, includeBias, interactionOnly) {
  if (!Array.isArray(X)) X = [X];
  const n_features = X.length;

  // 1. Generate combinations of indices for terms
  // Scikit-learn orders by degree, then lexicographically
  // e.g. [a, b] deg 2 => [1, a, b, a^2, ab, b^2]
  // BUT coefficients from sklearn linear model fit on poly features usually follow the output of PolynomialFeatures.fit_transform
  // PolynomialFeatures output order:
  // if bias: [1]
  // then degree 1 terms: [x0, x1, ...]
  // then degree 2 terms: [x0^2, x0x1, x0x2, ..., x1^2, x1x2, ..., xn^2]
  // Wait, sklearn default order is: combinations with replacement.
  // It iterates degrees 0 to degree.
  // For each degree d: generate combinations of length d.

  // Implementation of combinations with replacement
  function getCombinations(start, k, current) {
    if (current.length === k) {
      // Calculate term
      let val = 1;
      let isInteraction = false;
      const seen = new Set();
      for (const idx of current) {
        val *= X[idx];
        if (interactionOnly) {
          if (seen.has(idx)) {
            // duplicate index in term => x^2 => skip if interaction_only
            return null;
          }
          seen.add(idx);
        }
      }
      return val;
    }

    // Recursive
    // Scikit-learn uses "combinations_with_replacement" iterator order
    // Which produces indices sorted. e.g. (0,0), (0,1), (0,2), (1,1)...
    // We iterate i from start to n_features
    let results = [];
    for (let i = start; i < n_features; i++) {
      const res = getCombinations(i, k - 1, [...current, i]);
      if (res !== null) {
        if (Array.isArray(res)) results.push(...res);
        else results.push(res);
      }
    }
    return results;
  }

  const features = [];
  if (includeBias) features.push(1);

  for (let d = 1; d <= degree; d++) {
    // Generate terms for degree d
    // We need to match sklearn order
    // Sklearn (v1.0+): order='C' (default).
    // It generates powers:
    // iterates items, then combinations with replacement of items.
    // Actually, simple observation:
    // generatePolynomialFeatures([a,b], 2)
    // -> [1, a, b, a^2, ab, b^2]

    // My recursive strategy:
    // combinationsWithReplacement(indices, d)
    // indices = 0..n-1

    const combs = [];
    const f = (start, depth, prefix) => {
      if (depth === 0) {
        let val = 1;
        let isInter = false;
        const seen = new Set();
        let valid = true;
        for (const idx of prefix) {
          val *= X[idx];
          if (interactionOnly) {
            if (seen.has(idx)) valid = false;
            seen.add(idx);
          }
        }
        if (valid) features.push(val);
        return;
      }
      for (let i = start; i < n_features; i++) {
        f(i, depth - 1, [...prefix, i]);
      }
    };
    f(0, d, []);
  }

  return features;
}

function predictKNN(x, model, isClassification = false) {
  const { X_train, y_train, k, distance_metric, minkowski_p } = model;
  if (!X_train || !y_train) return 0;
  // Safety check just in case
  if (X_train.length === 0) return 0;

  const distances = X_train.map((x_train, i) => {
    let dist = 0;
    const x_vec = Array.isArray(x) ? x : [x];
    const xt_vec = Array.isArray(x_train) ? x_train : [x_train];

    // Ensure same length
    const dim = Math.min(x_vec.length, xt_vec.length);

    if (distance_metric === 'manhattan') {
      for (let j = 0; j < dim; j++) dist += Math.abs(x_vec[j] - xt_vec[j]);
    } else if (distance_metric === 'minkowski' && minkowski_p) {
      let sum = 0;
      for (let j = 0; j < dim; j++) sum += Math.pow(Math.abs(x_vec[j] - xt_vec[j]), minkowski_p);
      dist = Math.pow(sum, 1 / minkowski_p);
    } else if (distance_metric === 'chebyshev') {
      let maxD = 0;
      for (let j = 0; j < dim; j++) maxD = Math.max(maxD, Math.abs(x_vec[j] - xt_vec[j]));
      dist = maxD;
    } else {
      let sum = 0;
      for (let j = 0; j < dim; j++) sum += Math.pow(x_vec[j] - xt_vec[j], 2);
      dist = Math.sqrt(sum);
    }
    return { index: i, dist };
  });

  distances.sort((a, b) => a.dist - b.dist);
  const kIndices = distances.slice(0, k).map(d => d.index);
  const kLabels = kIndices.map(idx => y_train[idx]);

  if (isClassification) {
    const counts = {};
    kLabels.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
    let maxCount = 0;
    let prediction = kLabels[0];
    for (const l in counts) {
      if (counts[l] > maxCount) {
        maxCount = counts[l];
        prediction = parseFloat(l);
      }
    }
    return prediction;
  } else {
    const sum = kLabels.reduce((s, l) => s + l, 0);
    return sum / k;
  }
}

function predictNaiveBayes(x, model) {
  const { classes, class_means, class_vars, class_priors } = model;
  // classes: array, class_means: dict {cls: [means]}, ...

  let bestClass = null;
  let maxLogProb = -Infinity;

  const xVec = Array.isArray(x) ? x : [x];

  // Iterate over each class to find the one with max posterior probability
  for (const cls of classes) {
    const means = class_means[cls];
    const vars = class_vars[cls];
    const prior = class_priors[cls];

    // log(P(C|x)) ~ log(P(C)) + sum(log(P(xi|C)))
    let logProb = Math.log(prior + 1e-10);

    for (let i = 0; i < xVec.length; i++) {
      const mu = means[i];
      const sigma2 = vars[i]; // Variance
      const xi = xVec[i];

      // Gaussian Log PDF
      // -0.5 * log(2*pi*sigma2) - (x-mu)^2 / (2*sigma2)
      const safeSigma2 = Math.max(sigma2, 1e-9);
      logProb += -0.5 * Math.log(2 * Math.PI * safeSigma2);
      logProb += -0.5 * Math.pow(xi - mu, 2) / safeSigma2;
    }

    if (logProb > maxLogProb) {
      maxLogProb = logProb;
      bestClass = cls;
    }
  }

  return parseFloat(bestClass);
}

// --- Plot Components ---

function SeabornAxes({ xDomain, yDomain, xLabel, yLabel, isGrid = true }) {
  const safeX = safeDomain(xDomain);
  const safeY = safeDomain(yDomain);
  const scaleX = (x) => padding + ((x - safeX[0]) / (safeX[1] - safeX[0] || 1)) * (width - 2 * padding);
  const scaleY = (y) => height - padding - ((y - safeY[0]) / (safeY[1] - safeY[0] || 1)) * (height - 2 * padding);

  const xTicks = [];
  const yTicks = [];
  const gridLines = [];

  const xStep = (safeX[1] - safeX[0]) / 5;
  for (let i = 0; i <= 5; i++) {
    const val = safeX[0] + i * xStep;
    const pos = scaleX(val);
    xTicks.push(
      <g key={`x-${i}`}>
        <text x={pos} y={height - padding + 15} textAnchor="middle" fontSize="10" fill="#333">{val.toFixed(1)}</text>
      </g>
    );
    if (isGrid) {
      gridLines.push(<line key={`xg-${i}`} x1={pos} y1={padding} x2={pos} y2={height - padding} stroke="#fff" strokeWidth="1" />);
    }
  }

  const yStep = (safeY[1] - safeY[0]) / 5;
  for (let i = 0; i <= 5; i++) {
    const val = safeY[0] + i * yStep;
    const pos = scaleY(val);
    yTicks.push(
      <g key={`y-${i}`}>
        <text x={padding - 10} y={pos + 3} textAnchor="end" fontSize="10" fill="#333">{val.toFixed(1)}</text>
      </g>
    );
    if (isGrid) {
      gridLines.push(<line key={`yg-${i}`} x1={padding} y1={pos} x2={width - padding} y2={pos} stroke="#fff" strokeWidth="1" />);
    }
  }

  return (
    <>
      <rect x={padding} y={padding} width={width - 2 * padding} height={height - 2 * padding} fill="#EAEAF2" />
      {gridLines}
      {xTicks}
      {yTicks}
      <text x={width / 2} y={height - 5} textAnchor="middle" fontSize="12" fill="#333">{xLabel}</text>
      <text x={10} y={height / 2} textAnchor="middle" fontSize="12" fill="#333" transform={`rotate(-90, 10, ${height / 2})`}>{yLabel}</text>
    </>
  );
}

function ScatterPlot({ points, curvePoints, xDomain, yDomain, modelType, decisionGrid, xLabel, yLabel }) {
  const safeX = safeDomain(xDomain);
  const safeY = safeDomain(yDomain);
  const scaleX = (x) => padding + ((x - safeX[0]) / (safeX[1] - safeX[0] || 1)) * (width - 2 * padding);
  const scaleY = (y) => height - padding - ((y - safeY[0]) / (safeY[1] - safeY[0] || 1)) * (height - 2 * padding);

  let gridRects = null;
  const bgColors = ['#e3f2fd', '#ffebee', '#e8f5e9', '#fffde7', '#f3e5f5'];

  if (decisionGrid && decisionGrid.length > 0) {
    const cellW = (width - 2 * padding) / 40; // Approx matching grid size
    const cellH = (height - 2 * padding) / 40;
    gridRects = decisionGrid.map((p, i) => {
      const colorIdx = Math.abs(parseInt(p[2] || 0)) % bgColors.length;
      return (
        <rect key={`g-${i}`} x={scaleX(p[0]) - cellW / 2} y={scaleY(p[1]) - cellH / 2} width={cellW + 1} height={cellH + 1}
          fill={bgColors[colorIdx] || '#eee'} opacity={0.6} />
      );
    });
  }

  const colors = ['#1976d2', '#d32f2f', '#388e3c', '#fbc02d', '#8e24aa'];
  const dots = points.map((p, i) => {
    let color = '#4c72b0';
    if (modelType === 'knnClassification' && p.length > 2) {
      const cIdx = Math.abs(parseInt(p[2] || 0)) % colors.length;
      color = colors[cIdx];
    }
    return <circle key={i} cx={scaleX(p[0])} cy={scaleY(p[1])} r={3.5} fill={color} stroke="#fff" strokeWidth={0.5} />;
  });

  let curve = null;
  if (curvePoints && curvePoints.length > 1) {
    const d = curvePoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p[0])} ${scaleY(p[1])}`).join(' ');
    curve = <path d={d} fill="none" stroke="#c44e52" strokeWidth="2.5" />;
  }

  return (
    <svg width={width} height={height}>
      <SeabornAxes xDomain={xDomain} yDomain={yDomain} xLabel={xLabel} yLabel={yLabel} />
      {gridRects && <g>{gridRects}</g>}
      {dots}
      {curve}
    </svg>
  );
}

function ResidualHistogram({ residuals }) {
  if (!residuals || residuals.length === 0) return null;
  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  // If all same
  const same = min === max;
  const bins = 15;
  const step = same ? 1 : (max - min) / bins;

  // Create bins
  const histogram = new Array(bins).fill(0);
  residuals.forEach(r => {
    const binIdx = same ? 0 : Math.min(Math.floor((r - min) / step), bins - 1);
    histogram[binIdx]++;
  });
  const maxCount = Math.max(...histogram);

  const scaleX = (val) => padding + ((val - min) / ((same ? 1 : max - min) || 1)) * (width - 2 * padding);
  const scaleY = (count) => height - padding - (count / (maxCount || 1)) * (height - 2 * padding);

  const bars = histogram.map((count, i) => {
    const x = min + i * step;
    const xPos = scaleX(x);
    const yPos = scaleY(count);
    const barW = (width - 2 * padding) / bins - 1;
    const barH = height - padding - yPos;
    return (
      <rect key={i} x={xPos} y={yPos} width={Math.max(0, barW)} height={Math.max(0, barH)} fill="#4c72b0" opacity={0.8} stroke="#fff" />
    );
  });

  return (
    <svg width={width} height={height}>
      <SeabornAxes xDomain={[min, same ? min + 1 : max]} yDomain={[0, maxCount]} xLabel="Residuals" yLabel="Frequency" />
      {bars}
    </svg>
  );
}

function ConfusionMatrixPlot({ actual, predicted }) {
  if (!actual || !predicted || actual.length === 0) return null;
  const classes = [...new Set([...actual, ...predicted])].sort((a, b) => a - b);
  const map = {};
  classes.forEach((c, i) => map[c] = i);
  const n = classes.length;

  const matrix = Array(n).fill(0).map(() => Array(n).fill(0));
  actual.forEach((a, i) => {
    const p = predicted[i];
    if (map[a] !== undefined && map[p] !== undefined) {
      matrix[map[a]][map[p]]++;
    }
  });

  const maxVal = Math.max(...matrix.flat()) || 1;
  const cellSize = Math.min((width - 2 * padding) / n, (height - 2 * padding) / n);
  const offsetX = padding + ((width - 2 * padding) - n * cellSize) / 2;
  const offsetY = padding + ((height - 2 * padding) - n * cellSize) / 2;

  const cells = [];
  const getColor = (val) => {
    const t = val / maxVal;
    const r = Math.round(255 + (8 - 255) * t);
    const g = Math.round(255 + (48 - 255) * t);
    const b = Math.round(255 + (107 - 255) * t);
    return `rgb(${r},${g},${b})`;
  }

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val = matrix[r][c];
      const x = offsetX + c * cellSize;
      const y = offsetY + r * cellSize;
      cells.push(
        <g key={`${r}-${c}`}>
          <rect x={x} y={y} width={cellSize} height={cellSize} fill={getColor(val)} stroke="#fff" />
          <text x={x + cellSize / 2} y={y + cellSize / 2 + 5} textAnchor="middle" fill={val > maxVal / 2 ? '#fff' : '#000'} fontSize="14" fontWeight="bold">
            {val}
          </text>
        </g>
      );
    }
  }

  const xLabels = classes.map((cls, i) => (
    <text key={`xl-${i}`} x={offsetX + i * cellSize + cellSize / 2} y={offsetY - 10} textAnchor="middle" fontSize="12">{cls}</text>
  ));
  const yLabels = classes.map((cls, i) => (
    <text key={`yl-${i}`} x={offsetX - 10} y={offsetY + i * cellSize + cellSize / 2 + 4} textAnchor="end" fontSize="12">{cls}</text>
  ));

  return (
    <svg width={width} height={height}>
      <rect x="0" y="0" width={width} height={height} fill="#fff" />
      {cells}
      {xLabels}
      {yLabels}
      <text x={width / 2} y={offsetY - 30} textAnchor="middle" fontWeight="bold" fontSize="14">Predicted</text>
      <text x={offsetX - 35} y={height / 2} textAnchor="middle" transform={`rotate(-90, ${offsetX - 35}, ${height / 2})`} fontWeight="bold" fontSize="14">Actual</text>
    </svg>
  );
}

// --- Main Node Component ---

function findCsvNodeUpstream(nodeId, edges, nodes, visited = new Set()) {
  if (visited.has(nodeId)) return null;
  visited.add(nodeId);
  const incoming = edges.filter((e) => e.target === nodeId);
  for (const e of incoming) {
    const src = nodes.get(e.source);
    if (!src) continue;
    if (src.type === 'csvReader' && src.data?.file && src.data?.headers) return { file: src.data.file, headers: src.data.headers };
    if (['encoder', 'normalizer', 'dataCleaner', 'featureSelector', 'pca'].includes(src.type)) {
      const result = findCsvNodeUpstream(src.id, edges, nodes, visited);
      if (result) return result;
    }
  }
  return null;
}

function ModelVisualizerNode({ id, data, isConnectable }) {
  const [activeTab, setActiveTab] = React.useState('scatter');
  const prevUpstreamRef = useRef(null);

  const upstream = useStore((store) => {
    const edges = Array.from(store.edges.values());
    const nodes = store.nodeInternals;
    const visited = new Set();
    const stack = [id];
    const result = { model: null, csv: null, modelNodeId: null };

    // Traversal logic
    while (stack.length) {
      const targetId = stack.pop();
      if (visited.has(targetId)) continue;
      visited.add(targetId);
      const incoming = edges.filter((e) => e.target === targetId);
      for (const e of incoming) {
        const src = nodes.get(e.source);
        if (!src) continue;
        if (!result.model && ['linearRegression', 'multiLinearRegression', 'polynomialRegression', 'knnRegression', 'logisticRegression', 'knnClassification', 'naiveBayes'].includes(src.type) && src.data?.model) {
          result.model = { ...src.data.model, type: src.type };
          result.modelNodeId = src.id;
          if (!result.csv) {
            const c = findCsvNodeUpstream(src.id, edges, nodes, new Set());
            if (c) result.csv = c;
          }
        }
        if (!result.csv && src.type === 'csvReader' && src.data?.file && src.data?.headers) result.csv = { file: src.data.file, headers: src.data.headers };
        if (result.modelNodeId && !result.csv) {
          const c = findCsvNodeUpstream(src.id, edges, nodes, new Set());
          if (c) result.csv = c;
        }
        if (!(result.model && result.csv)) stack.push(src.id);
      }
      if (result.model && result.csv) break;
    }
    return result;
  });

  const [viz, setViz] = React.useState({
    points: [], curvePoints: [], xDomain: [0, 1], yDomain: [0, 1], metrics: {}, residuals: [], actualVsPredicted: [], decisionGrid: null, actualClasses: [], predictedClasses: []
  });

  React.useEffect(() => {
    const prev = prevUpstreamRef.current;

    // Deepish Check to avoid re-running on simple reference changes
    const sameModel = prev?.model?.k === upstream.model?.k &&
      prev?.model?.coefficients === upstream.model?.coefficients &&
      prev?.model?.intercept === upstream.model?.intercept &&
      prev?.model?.type === upstream.model?.type;
    const sameCsv = prev?.csv?.file === upstream.csv?.file;

    if (sameModel && sameCsv && prev) return;

    prevUpstreamRef.current = upstream;

    let cancelled = false;
    async function load() {
      if (!upstream.model || !upstream.csv) { if (!cancelled) setViz(prev => ({ ...prev, points: [] })); return; }
      const { file } = upstream.csv;
      const { xCol, yCol, type } = upstream.model;
      const xColNames = upstream.model.xCols || (xCol ? [xCol] : []);
      if (xColNames.length === 0 || !yCol) return;

      try {
        const parsed = await parseFullTabularFile(file);
        if (!parsed) return;
        const { headers: hs, rows } = parsed;

        const xIndices = xColNames.map(n => hs.indexOf(n));
        const yIndex = hs.indexOf(yCol);
        if (xIndices.includes(-1) || yIndex === -1) return;

        const pts = [], actual = [], predicted = [], avp = [];
        const isClass = type === 'knnClassification' || type === 'logisticRegression' || type === 'naiveBayes';

        // Limit rows for performance if needed, but usually 1000 is fine
        const processRows = rows.slice(0, 1000);

        for (const r of processRows) {
          const yVal = parseFloat(r[yIndex]);
          if (!Number.isFinite(yVal)) continue;
          const xVals = xIndices.map(i => parseFloat(r[i]));
          if (xVals.some(v => !Number.isFinite(v))) continue;

          let predVal = 0;
          if (type === 'linearRegression') predVal = upstream.model.slope * xVals[0] + upstream.model.intercept;
          else if (type === 'multiLinearRegression') {
            predVal = upstream.model.intercept;
            xVals.forEach((v, i) => { predVal += upstream.model.coefficients[i] * v; });
          } else if (type === 'polynomialRegression') {
            if (xVals.length === 1) {
              const features = generatePolynomialFeatures(xVals, upstream.model.degree, upstream.model.include_bias, upstream.model.interaction_only);
              predVal = features.reduce((sum, feat, i) => sum + (upstream.model.coefficients[i] || 0) * feat, 0) + (upstream.model.intercept || 0);
            }
          } else if (type === 'knnRegression') predVal = predictKNN(xVals, upstream.model, false);
          else if (type === 'logisticRegression') {
            let z = upstream.model.intercept;
            xVals.forEach((v, i) => { z += upstream.model.coefficients[i] * v; });
            predVal = sigmoid(z);
          } else if (type === 'knnClassification') {
            predVal = predictKNN(xVals, upstream.model, true);
          } else if (type === 'naiveBayes') {
            predVal = predictNaiveBayes(xVals, upstream.model);
          }

          actual.push(yVal);
          let predLabel = predVal;
          if (type === 'logisticRegression') predLabel = predVal >= 0.5 ? 1 : 0;
          predicted.push(isClass ? predLabel : predVal);

          avp.push({ actual: yVal, predicted: isClass ? predLabel : predVal });

          if (type === 'knnClassification' && xVals.length === 2) pts.push([xVals[0], xVals[1], yVal]);
          else if (type === 'naiveBayes' && xVals.length === 2) pts.push([xVals[0], xVals[1], yVal]);
          else if (xVals.length === 1) pts.push([xVals[0], yVal]);
        }

        let metrics = {};
        if (isClass) metrics = calculateClassificationMetrics(actual, predicted);
        else {
          const rSquared = calculateRSquared(actual, predicted);
          const mse = actual.reduce((s, a, i) => s + Math.pow(a - predicted[i], 2), 0) / actual.length;
          metrics = { rSquared, mse, mae: 0 };
        }

        const residuals = isClass ? [] : actual.map((a, i) => a - predicted[i]);

        let curvePoints = [], decisionGrid = null;
        let xDom = [0, 1], yDom = [0, 1];
        if (pts.length > 0) {
          const xs = pts.map(p => p[0]);
          xDom = [Math.min(...xs), Math.max(...xs)];
          if (pts[0].length === 3) {
            const ys = pts.map(p => p[1]);
            yDom = [Math.min(...ys), Math.max(...ys)];
          } else yDom = [Math.min(...actual), Math.max(...actual)];

          if (isClass && type === 'logisticRegression') yDom = [-0.1, 1.1];
        }

        if (xColNames.length === 1 && pts.length > 0) {
          const step = (xDom[1] - xDom[0]) / 100;
          for (let i = 0; i <= 100; i++) {
            const x = xDom[0] + i * step;
            let y = 0;
            if (type === 'linearRegression') y = upstream.model.slope * x + upstream.model.intercept;
            else if (type === 'polynomialRegression') {
              const f = generatePolynomialFeatures([x], upstream.model.degree, upstream.model.include_bias, upstream.model.interaction_only);
              y = f.reduce((s, ft, idx) => s + (upstream.model.coefficients[idx] || 0) * ft, 0) + (upstream.model.intercept || 0);
            }
            else if (type === 'logisticRegression') y = sigmoid(upstream.model.intercept + upstream.model.coefficients[0] * x);
            else if (type === 'knnRegression') y = predictKNN([x], upstream.model, false);
            else if (type === 'naiveBayes') y = predictNaiveBayes([x], upstream.model);

            curvePoints.push([x, y]);
          }
        }

        // Optimize Grid: 40x40
        if ((type === 'knnClassification' || type === 'naiveBayes') && xColNames.length === 2) {
          decisionGrid = [];
          const safeX = safeDomain(xDom), safeY = safeDomain(yDom);
          const xStep = (safeX[1] - safeX[0]) / 40, yStep = (safeY[1] - safeY[0]) / 40;
          // Async break? No, just sync. 1600 iterations.
          for (let i = 0; i < 40; i++) for (let j = 0; j < 40; j++) {
            const gx = safeX[0] + i * xStep + xStep / 2, gy = safeY[0] + j * yStep + yStep / 2;
            let pVal = 0;
            if (type === 'knnClassification') pVal = predictKNN([gx, gy], upstream.model, true);
            else if (type === 'naiveBayes') pVal = predictNaiveBayes([gx, gy], upstream.model);
            decisionGrid.push([gx, gy, pVal]);
          }
        }

        if (!cancelled) setViz({ points: pts, curvePoints, xDomain: xDom, yDomain: yDom, metrics, residuals, actualVsPredicted: avp, decisionGrid, actualClasses: actual, predictedClasses: predicted });
      } catch (err) { console.error("Viz Error", err); }
    }
    load();
    return () => { cancelled = true; };
  }, [upstream]);

  const hasEverything = upstream.model && upstream.csv && (viz.points.length > 0 || viz.actualVsPredicted.length > 0);
  const isClass = upstream.model?.type === 'logisticRegression' || upstream.model?.type === 'knnClassification' || upstream.model?.type === 'naiveBayes';
  const showScatter = !((upstream.model?.type === 'knnClassification' || upstream.model?.type === 'naiveBayes') && upstream.model.xCols?.length > 2) && !(upstream.model?.type === 'multiLinearRegression');

  return (
    <div className="model-visualizer-node">
      <div className="mv-title">Model Visualizer</div>
      {hasEverything && (
        <div className="mv-controls">
          <div className="mv-tabs">
            <button className={`mv-tab ${activeTab === 'scatter' ? 'active' : ''}`} onClick={() => setActiveTab('scatter')} disabled={!showScatter}>
              {(upstream.model.type === 'knnClassification' || upstream.model.type === 'naiveBayes') && upstream.model.xCols?.length === 2 ? 'Boundary' : 'Scatter'}
            </button>
            <button className={`mv-tab ${activeTab === 'actualVsPred' ? 'active' : ''}`} onClick={() => setActiveTab('actualVsPred')}>
              {isClass ? 'Class Plot' : 'Act vs Pred'}
            </button>
            {isClass ? (
              <button className={`mv-tab ${activeTab === 'confusion' ? 'active' : ''}`} onClick={() => setActiveTab('confusion')}>Confusion</button>
            ) : (
              <>
                <button className={`mv-tab ${activeTab === 'residuals' ? 'active' : ''}`} onClick={() => setActiveTab('residuals')}>Resid.</button>
                <button className={`mv-tab ${activeTab === 'residDist' ? 'active' : ''}`} onClick={() => setActiveTab('residDist')}>Dist.</button>
              </>
            )}
          </div>
        </div>
      )}
      <div className="mv-content">
        {hasEverything ? (
          <>
            {activeTab === 'scatter' && (
              <ScatterPlot
                points={viz.points} curvePoints={viz.curvePoints}
                xDomain={viz.xDomain} yDomain={viz.yDomain}
                modelType={upstream.model.type} decisionGrid={viz.decisionGrid}
                xLabel={upstream.model.xCols?.[0] || 'X'} yLabel={upstream.model.yCol || 'Y'}
              />
            )}
            {activeTab === 'actualVsPred' && (
              <ScatterPlot
                points={viz.actualVsPredicted.map(d => [d.actual, d.predicted])}
                xDomain={isClass ? (upstream.model.type === 'logisticRegression' ? [-0.1, 1.1] : viz.yDomain) : viz.yDomain}
                yDomain={isClass ? [-0.1, 1.1] : viz.yDomain}
                modelType={upstream.model.type}
                xLabel="Actual" yLabel="Predicted"
              />
            )}
            {activeTab === 'residuals' && (
              <ScatterPlot
                points={viz.residuals.map((r, i) => [i, r])}
                xDomain={[0, viz.residuals.length]}
                yDomain={[Math.min(...viz.residuals), Math.max(...viz.residuals)]}
                modelType="none" xLabel="Index" yLabel="Residual"
              />
            )}
            {activeTab === 'residDist' && <ResidualHistogram residuals={viz.residuals} />}
            {activeTab === 'confusion' && <ConfusionMatrixPlot actual={viz.actualClasses} predicted={viz.predictedClasses} />}
          </>
        ) : <div className="mv-placeholder">Waiting for data...</div>}
      </div>
      {hasEverything && (
        <div className="mv-metrics">
          {isClass ? (
            <div className="metric"><span className="metric-label">Acc:</span><span className="metric-value">{(viz.metrics.accuracy * 100).toFixed(1)}%</span></div>
          ) : (
            <div className="metric"><span className="metric-label">R²:</span><span className="metric-value">{viz.metrics.rSquared?.toFixed(3)}</span></div>
          )}
        </div>
      )}
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} className="custom-handle" />
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} className="custom-handle" />
    </div>
  );
}

export default ModelVisualizerNode;
