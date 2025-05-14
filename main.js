import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ShapeUtils } from 'three';
// Use a CDN for BufferGeometryUtils if direct import fails
import * as BufferGeometryUtils from 'https://cdn.jsdelivr.net/npm/three/examples/jsm/utils/BufferGeometryUtils.js';
import GUI from 'lil-gui';

// =============================================================================
// Helper Functions
// =============================================================================

function roundN(value, decimals) {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
}

function round6(value) {
    return roundN(value, 6);
}

function roundForVertexMerge(value) { // New function for vertex merging
    return roundN(value, config.vertexMergeDecimals); // Round to 5 decimal places for looser vertex merging
}

// New helper function to calculate the centroid of a polygon
function calculatePolygonCentroid(polygonVertices) {
    let centroidX = 0;
    let centroidY = 0;
    let signedArea = 0;
    const numVertices = polygonVertices.length;

    for (let i = 0; i < numVertices; i++) {
        const v0 = polygonVertices[i];
        const v1 = polygonVertices[(i + 1) % numVertices];

        const a = v0.x * v1.y - v1.x * v0.y;
        signedArea += a;
        centroidX += (v0.x + v1.x) * a;
        centroidY += (v0.y + v1.y) * a;
    }

    if (Math.abs(signedArea) < 1e-9) { // Avoid division by zero for degenerate polygons
        // Return the average of vertices if area is zero
        let avgX = 0;
        let avgY = 0;
        for (let i = 0; i < numVertices; i++) {
            avgX += polygonVertices[i].x;
            avgY += polygonVertices[i].y;
        }
        return { x: avgX / numVertices, y: avgY / numVertices };
    }

    signedArea *= 0.5;
    centroidX /= (6 * signedArea);
    centroidY /= (6 * signedArea);

    return { x: centroidX, y: centroidY };
}

// New helper function to calculate the internal angles of a polygon
function calculatePolygonAngles(polygonVertices) {
    const angles = [];
    const numVertices = polygonVertices.length;
    if (numVertices < 3) return []; // Not a polygon

    for (let i = 0; i < numVertices; i++) {
        const p1 = polygonVertices[i];
        const p2 = polygonVertices[(i + 1) % numVertices];
        const p3 = polygonVertices[(i + 2) % numVertices];

        const v1 = new THREE.Vector2(p1.x - p2.x, p1.y - p2.y);
        const v2 = new THREE.Vector2(p3.x - p2.x, p3.y - p2.y);

        const angle = Math.atan2(v2.y, v2.x) - Math.atan2(v1.y, v1.x);
        let degrees = THREE.MathUtils.radToDeg(angle);

        // Ensure positive angle
        degrees = (degrees <= 0) ? degrees + 360 : degrees;
        // Handle cases where angle might be > 180 for convex polygon vertex (due to winding)
        // For a simple convex polygon, internal angles should be < 180.
        // However, the order of vertices from the current algorithm might not guarantee consistent winding
        // for this specific angle calculation method.
        // For now, we take the direct angle. If issues arise, this might need refinement
        // based on guaranteed winding order or by checking convexity.
        // A common approach is to ensure angles are within (0, 360) and then, if needed,
        // subtract from 360 if it's a reflex angle in a concave polygon, or ensure it's the smaller angle.
        // Given the current context of forming mostly convex rhombs, we'll assume the direct angle is mostly okay.
        // Let's ensure it's the smaller of the two possible angles (e.g. 270 vs 90) by taking modulo 360 and potentially 360 - angle.
        // However, the standard formula for internal angle uses dot product and magnitudes for unsigned angle.
        // The current atan2 method gives signed angle, which is good.

        // Let's normalize to [0, 360)
        degrees = (degrees % 360 + 360) % 360;

        // For internal angles of a simple polygon, they should be < 360.
        // If we expect convex polygons, they should be < 180.
        // The current `performMultigridGeneration` filters for 4-sided polygons (quadrilaterals).
        // These are expected to be rhombs or similar, which are convex.
        // Thus, we can ensure the angle is < 180 or 360 - angle if it's > 180.
        // This step is actually tricky without knowing winding order.
        // Let's use a simpler approach based on dot product for unsigned angle.

        v1.normalize();
        v2.normalize();
        let dot = v1.dot(v2);
        dot = Math.max(-1, Math.min(1, dot)); // Clamp dot product to avoid acos errors
        let internalAngleRad = Math.acos(dot);

        // The cross product z-component can determine if it's a left or right turn.
        // (v1.x * v2.y - v1.y * v2.x) tells about the orientation.
        // For a counter-clockwise wound polygon, internal angle is (PI - acos(dot)).
        // For clockwise, it's (PI + acos(dot)) or (acos(dot) - PI adjusted).
        // Or, more simply, the angle is acos(dot). The sum of interior angles is (n-2)*180.

        // Let's use the property that for a CCW polygon, the sum of exterior angles is 360.
        // The angle from atan2 is correct if vectors are defined from the vertex outwards.
        // Let p_prev = polygonVertices[(i + numVertices - 1) % numVertices];
        // Let p_curr = polygonVertices[i];
        // Let p_next = polygonVertices[(i + 1) % numVertices];
        // vecA = p_prev - p_curr
        // vecB = p_next - p_curr
        // Angle between vecA and vecB is the internal angle.

        const p_curr = polygonVertices[i];
        const p_prev = polygonVertices[(i + numVertices - 1) % numVertices];
        const p_next = polygonVertices[(i + 1) % numVertices];

        const vec_curr_prev = new THREE.Vector2(p_prev.x - p_curr.x, p_prev.y - p_curr.y);
        const vec_curr_next = new THREE.Vector2(p_next.x - p_curr.x, p_next.y - p_curr.y);

        // Angle from vec_curr_prev to vec_curr_next
        let angleRad = Math.atan2(vec_curr_next.y, vec_curr_next.x) - Math.atan2(vec_curr_prev.y, vec_curr_prev.x);

        // Normalize to [0, 2*PI)
        if (angleRad < 0) {
            angleRad += 2 * Math.PI;
        }
        // For simple polygons, the internal angle is usually what's desired.
        // If the polygon winding is known (e.g., CCW), then the angle is correct.
        // If winding is CW, angle would be 2*PI - angleRad.
        // Assuming vertices are ordered (e.g. CCW from the algorithm)

        angles.push(THREE.MathUtils.radToDeg(angleRad));
    }
    return angles;
}

// --- Dome Extrusion Utility Functions ---
function heightProfile(u) {
  const R = config.domeRadius;
  u = Math.max(0, Math.min(1, u));
  switch (config.profileType) {
    case 'spherical':
      return R - Math.sqrt(Math.max(0, R*R - (u*r_max)**2));
    case 'eased':
      return R * u * u;
    case 'stepped':
      const tier = Math.floor(config.tierCount * u);
      return Math.min(R, config.stepHeight * tier);
    case 'cascading':
      const k = config.cascadeSteps;
      const stepId = Math.min(k-1, Math.floor(u * k));
      const dz = config.cascadeDrop * r_max / k;
      return dz * stepId;
    default:
      return 0;
  }
}

function faceTiltDeg(u) {
  if (config.profileType !== 'cascading') return config.tiltDeg;
  // interpolate between inner and outer tilts
  const t0 = config.tiltInnerDeg, t1 = config.tiltOuterDeg;
  u = Math.max(0, Math.min(1, u));
  return t1 + (t0 - t1)*(1 - u);
}

// quaternion to tilt a face toward the origin by angleRad
function getTiltQuaternion(centroid, angleRad) {
  const q = new THREE.Quaternion();
  if (Math.abs(angleRad) < 1e-6) return q;
  // direction from centroid to origin
  const d = new THREE.Vector3(-centroid.x, -centroid.y, 0).normalize();
  if (d.lengthSq() < 1e-12) return q;
  // rotation axis = d × z-axis
  const axis = new THREE.Vector3(d.y, -d.x, 0).normalize();
  return q.setFromAxisAngle(axis, angleRad);
}
// --- End Dome Extrusion Utility Functions ---

// =============================================================================
// Configuration & Global State (Revision 2)
// =============================================================================

const config = {
    // --- Multigrid Algorithm Parameters (from Rev 2 Spec Table 1) ---
    N: 5,                   // integer, 3 ... 33, "Symmetry"
    phi: 0.2,               // float, 0 <= phi < 1, "Phase (pattern)"
    Delta: 0.0,             // float, 0 <= Delta <= 1, "Disorder"
    seed: "0",              // string/int, PRNG seed, "Random-seed"
    R_param: 6,            // integer, 1 ... 200, "Radius (# lines)" (used as R in spec sections)
    alpha_rot: 0.0,         // float (deg), global rotation, "Global rotation"
    zeta: 1.0,              // float, > 0, global scale (zoom), "Zoom"
    panX: 0.0,              // float, global translation pX (part of p)
    panY: 0.0,              // float, global translation pY (part of p)
    // epsilon: 1e-6,          // float, numerical tolerance (fixed, not UI) // Will be replaced by UI version
    // q_offset_epsilon: 1e-4, // float, for Q_point offset. REVERTING to 1e-4. // Will be replaced by UI version

    // --- Drawing Parameters ---
    maxDrawRadius: 24,       // float, max distance of vertices from origin to be drawn

    // --- UI-Exposed Tuning Parameters ---
    vertexMergeDecimals: 9,   // was 7
    epsilon_ui: 1e-9,         // was 1e-6
    q_offset_epsilon_ui: 1e-6, // was 1e-4

    // --- Visualization Parameters ---
    showVertices: true,
    showEdges: true,
    showFaces:    true,
    faceOpacity:  0.9,
    faceColors: [
        '#007f7f', // Darkest Cyan
        '#009999',
        '#00b2b2',
        '#00cccc',
        '#00e5e5',
        '#00ffff', // Pure Cyan
        '#19ffff',
        '#33ffff',
        '#4cffff',
        '#66ffff',
        '#7fffff',
        '#99ffff'  // Lightest Cyan
    ], // Default 12 shades of cyan for face types

    // --- Dome Extrusion Parameters ---
    extrudeDome: false,
    domeRadius: 6.0,
    profileType: 'spherical',    // ['spherical','eased','stepped','cascading']
    tierCount: 5,
    stepHeight: 1.0,
    tiltDeg: 0.0,
    cascadeSteps: 12,
    cascadeDrop: 0.6,
    tiltInnerDeg: 55,
    tiltOuterDeg: 10,

    // --- Color Scheme Parameters ---
    globalHueShift: 0.0, // 0.0 to 1.0, maps to 0-360 degrees
};

// --- Global Three.js Variables ---
let scene, camera, renderer, controls;
let mainVertexPointsObject = null;
let mainEdgesObject        = null;
let mainFacesObject        = null;   // NEW
let gui;

// --- Global Multigrid Data ---
let tilesData = {};
let uniqueVertexMap = new Map(); // Used for unique dual vertices for Points object
let indexedPositions = [];   // Will store unique dual vertex positions

// --- Base HSL colors for hue shifting ---
let baseFaceColorsHSL = [];

// --- PRNG instance ---
let prng;

// --- Max radius of dual vertices ---
let r_max = 0;

// --- Dome geometry objects ---
let domeMeshObject = null;
let domeEdgesObject = null;

// --- GUI Controllers for dynamic updates ---
let domeRadiusController = null;
let stepHeightController = null;

// Scratch objects
const _vec2_1 = new THREE.Vector2();
const _vec2_2 = new THREE.Vector2();

// -----------------------------------------------------------------------------
// Camera-based radius helper
// -----------------------------------------------------------------------------
function getViewportRadius() {
    // Calculate the visible height/width at the z=0 plane where the grid lies
    const halfHeightWorld = camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const halfWidthWorld = halfHeightWorld * camera.aspect;
    // Convert to grid-space units (which are pre-zoom) by dividing by zeta, then add a safety margin
    return Math.ceil(Math.max(halfWidthWorld, halfHeightWorld) / config.zeta) + 2;
}

function requiredGridRadius() {
    // how many grid units from (0,0) to the farthest screen corner *after* zoom
    const halfH = camera.position.z *
        Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const halfW = halfH * camera.aspect;
    return Math.ceil(Math.hypot(halfW, halfH) / config.zeta) + 2;
}

function viewportCornersInGridSpace() {
    // world-space half-sizes of the visible rectangle at z = 0
    const halfH = camera.position.z *
        Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const halfW = halfH * camera.aspect;

    // four corners in world space
    const corners = [
        new THREE.Vector2(-halfW, -halfH),
        new THREE.Vector2(halfW, -halfH),
        new THREE.Vector2(halfW, halfH),
        new THREE.Vector2(-halfW, halfH)
    ];

    // inverse global transform: 1) un-pan, 2) un-rotate, 3) un-scale
    const invScale = 1 / config.zeta;
    const sinA = Math.sin(-THREE.MathUtils.degToRad(config.alpha_rot));
    const cosA = Math.cos(-THREE.MathUtils.degToRad(config.alpha_rot));
    const invPan = new THREE.Vector2(-config.panX, -config.panY);

    return corners.map(c => {
        // undo pan
        c.add(invPan);
        // undo rotation
        const x = c.x * cosA - c.y * sinA;
        const y = c.x * sinA + c.y * cosA;
        // undo scale
        return new THREE.Vector2(x * invScale, y * invScale);
    });
}

// -----------------------------------------------------------------------------
// For a direction  n = (cosθ, sinθ)  return the closed interval of integer
// indices  k  with a line  ⟨n , (x,y)⟩ = k + o  that crosses the current
// viewport (safety-belt = 2 grid–units all round)
// -----------------------------------------------------------------------------
function kRangeForFamily(n_vec, o_i) { // n_vec is a THREE.Vector2
    // world-space half-sizes of the visible rectangle at z = 0
    const halfH = camera.position.z *
        Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const halfW = halfH * camera.aspect;

    // the four corners in world space
    const corners = [
        new THREE.Vector2(-halfW, -halfH),
        new THREE.Vector2(halfW, -halfH),
        new THREE.Vector2(halfW, halfH),
        new THREE.Vector2(-halfW, halfH)
    ];

    // project the corners on n_vec to obtain the min / max scalar values
    let min_proj = Infinity;
    let max_proj = -Infinity;
    for (const c of corners) {
        const s = n_vec.dot(c);
        if (s < min_proj) min_proj = s;
        if (s > max_proj) max_proj = s;
    }

    // convert world-space scalars to *grid* indices, add safety belt
    return {
        kMin: Math.floor(min_proj - o_i) - 2,
        kMax: Math.ceil(max_proj - o_i) + 2
    };
}

function clampKR(range, R) {
    const a = Math.max(range.kMin, -R);
    const b = Math.min(range.kMax, R);
    return (a > b)                    // nothing of this family is visible
        ? null
        : { kMin: a, kMax: b };
}

// =============================================================================
// PRNG (Seedable LCG - PCG32 can be subbed in if available/required)
// =============================================================================
function stringToSeed(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash |= 0;
    }
    return hash;
}

function LCG(seedValue) { // Simulates PRNG with .uniform()
    let state = typeof seedValue === 'string' ? stringToSeed(seedValue) : parseInt(seedValue, 10) || 0;
    if (state === 0) state = 1;
    return {
        uniform: function () {
            state = (1664525 * state + 1013904223) % 4294967296;
            return state / 4294967296;
        }
    };
}

function initializePRNG() {
    prng = LCG(config.seed);
}

// =============================================================================
// Multigrid Tiling Algorithm (Implementing Revision 2 Spec)
// =============================================================================

// --- Variables for shape type classification ---
let shapeToTypeIdMap = new Map();
let nextTypeId = 0;

function performMultigridGeneration() {
    console.log("Starting Multigrid generation (Rev 2 Spec)...");
    const startTime = performance.now();

    // Reset shape type classification for this generation run
    shapeToTypeIdMap.clear();
    nextTypeId = 0;

    // --- Add counters for skipped tiles ---
    let countSkippedNotQuad = 0;
    // let countSkippedSmallArea = 0; // REMOVED
    // --- End of counter addition ---

    initializePRNG();

    // --- 1. User-controlled parameters ---
    const N_param = Math.floor(config.N);
    if (N_param < 3 || N_param > 33) {
        console.error("N must be between 3 and 33.");
        clearGeometry(); return;
    }
    const phi_param = config.phi;
    const Delta_param = config.Delta;

    // R_user_setting is still validated as it's a UI parameter, though not directly used for k/l loops anymore.
    const R_user_setting = Math.floor(config.R_param);
    if (R_user_setting < 1 || R_user_setting > 200) {
        console.warn(`R_param from UI (${R_user_setting}) is out of typical 1-200 range, but k_ranges will govern line generation.`);
        // clearGeometry(); return; // Decided not to hard error, k_ranges will take precedence.
    }

    const epsilon_val = config.epsilon_ui; // ε from spec §0

    // --- 2. Pre-compute family directions {n_i} ---
    const theta_arr = []; // θ_i
    const n_arr = [];     // n_i (array of THREE.Vector2)
    for (let i = 0; i < N_param; i++) {
        const theta_i = (2 * Math.PI * i) / N_param;
        theta_arr.push(theta_i);
        n_arr.push(new THREE.Vector2(Math.cos(theta_i), Math.sin(theta_i)));
    }

    // --- 3. One scalar offset per family {o_i} ---
    const o_arr = [];
    for (let i = 0; i < N_param; i++) {
        let o_i = phi_param + Delta_param * (prng.uniform() - 0.5);
        o_i = ((o_i % 1.0) + 1.0) % 1.0;
        o_arr.push(o_i);
    }

    // --- 3½. k-ranges: only the lines that can hit the screen ---------------
    const k_ranges = Array.from({ length: N_param }, () => ({
        kMin: -R_user_setting,
        kMax:  R_user_setting
    }));

    console.log("Effective k_ranges (fixed by R_user_setting):");
    k_ranges.forEach((r, idx) => console.log(`  Fam ${idx}: [${r.kMin}, ${r.kMax}], Lines: ${Math.max(0, r.kMax - r.kMin + 1)}`));

    // --- 4. Raw multigrid L (conceptual) ---

    // --- 5. Enumerate intersection vertices V ---
    const V_map = new Map();

    // Re-introduce world-space half-sizes for point-clipping
    const halfHeightWorld_clip = camera.position.z * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const halfWidthWorld_clip = halfHeightWorld_clip * camera.aspect;

    // 5.1 Pairwise intersections
    for (let i = 0; i < N_param; i++) {
        const n_i = n_arr[i];
        const kr_i = k_ranges[i];
        if (kr_i.kMax < kr_i.kMin) continue; // Skip if family i has no lines in the effective range

        for (let j = i + 1; j < N_param; j++) {
            const n_j = n_arr[j];
            const kr_j = k_ranges[j];
            if (kr_j.kMax < kr_j.kMin) continue; // Skip if family j has no lines in the effective range

            const den = n_i.x * n_j.y - n_i.y * n_j.x;
            if (Math.abs(den) <= epsilon_val) continue;

            for (let k = kr_i.kMin; k <= kr_i.kMax; k++) {
                for (let l = kr_j.kMin; l <= kr_j.kMax; l++) {
                    const val_i = k + o_arr[i];
                    const val_j = l + o_arr[j];

                    const x = (val_i * n_j.y - val_j * n_i.y) / den;
                    const y = (val_j * n_i.x - val_i * n_j.x) / den;

                    // Filter vertices beyond maxDrawRadius
                    if (Math.hypot(x, y) > config.maxDrawRadius) continue;

                    // Point-clip: Compare world-scaled grid coordinates to world limits
                    const clipMargin = 4; // Widened margin
                    if (Math.abs(x * config.zeta) > halfWidthWorld_clip + clipMargin ||
                        Math.abs(y * config.zeta) > halfHeightWorld_clip + clipMargin) continue;

                    // 5.2 Vertex map
                    const xr = roundForVertexMerge(x);
                    const yr = roundForVertexMerge(y);
                    const v_key = xr + "," + yr; // Simple string key is fine

                    let v_record = V_map.get(v_key);
                    if (!v_record) {
                        v_record = {
                            point: new THREE.Vector2(x, y), // Store original precise point for calcs
                            lines: new Set()
                        };
                        V_map.set(v_key, v_record);
                    }
                    v_record.lines.add(JSON.stringify([i, k]));
                    v_record.lines.add(JSON.stringify([j, l]));
                }
            }
        }
    }
    console.log('Step 5: Found ' + V_map.size + ' unique intersection pre-vertices.');

    // --- 6. Complete line membership for each vertex ---
    V_map.forEach(v_record => {
        const v_point = v_record.point; // This is THREE.Vector2
        // The following block is removed as per user request:
        // for (let i_fam = 0; i_fam < N_param; i_fam++) {
        //     const t_val = n_arr[i_fam].dot(v_point) - o_arr[i_fam];
        //     if (Math.abs(t_val - Math.round(t_val)) < epsilon_val) {
        //         v_record.lines.add(JSON.stringify([i_fam, Math.round(t_val)]));
        //     }
        // }
        // v_record.lines is now solely populated by Step 5.
    });

    // --- 7. Build the dual rhomb T(v) ---
    tilesData = {};
    const tempTiles = {};
    let temp_debug_counter_step7 = 0; // DEBUG

    V_map.forEach((v_record, v_key) => {
        const v_point_vec2 = v_record.point; // THREE.Vector2
        const v_lines_set = v_record.lines; // Set of JSON.stringified [i,k] tuples
        temp_debug_counter_step7++; // DEBUG

        // 7.1 Angular list A
        const family_indices_for_A = [...v_lines_set].map(s => JSON.parse(s)[0]);
        let A_angles = family_indices_for_A.flatMap(fam_idx => [theta_arr[fam_idx], theta_arr[fam_idx] + Math.PI]);
        A_angles = [...new Set(A_angles.map(a => (a % (2*Math.PI) + 2*Math.PI) % (2*Math.PI)))].sort((a,b) => a - b);

        if (temp_debug_counter_step7 <= 5) { // DEBUG log for first 5 vertices
            console.log(`DEBUG Step 7 (${temp_debug_counter_step7}): Vertex (${v_point_vec2.x.toFixed(3)}, ${v_point_vec2.y.toFixed(3)})`);
            console.log(`  v_lines_set size: ${v_lines_set.size}, Sample: ${[...v_lines_set].slice(0, 3).join('; ')}`);
            console.log(`  A_angles.length (after unique/sort): ${A_angles.length}`);
        }

        if (A_angles.length === 0) {
            if (temp_debug_counter_step7 <= 5) console.log(`  Skipping: A_angles.length is 0`); // DEBUG
            return;
        }

        // 7.2 Offset points Q (ε-inset)
        const Q_points = []; // Array of THREE.Vector2
        for (let k_idx = 0; k_idx < A_angles.length; k_idx++) {
            const a_k = A_angles[k_idx];
            _vec2_1.set(-Math.sin(a_k), Math.cos(a_k)).multiplyScalar(config.q_offset_epsilon_ui);
            Q_points.push(v_point_vec2.clone().add(_vec2_1));
        }

        // 7.3 Midpoints M of consecutive Q
        const M_pts = []; // Array of THREE.Vector2
        const num_A = A_angles.length;
        for (let k_idx = 0; k_idx < num_A; k_idx++) {
            _vec2_1.copy(Q_points[k_idx]);
            _vec2_2.copy(Q_points[(k_idx + 1) % num_A]);
            M_pts.push(_vec2_1.add(_vec2_2).multiplyScalar(0.5).clone());
        }

        // --- Log M_pts for debugging ---
        if (temp_debug_counter_step7 <= 1) { // DEBUG for first vertex only
            console.log(`  M_pts for vertex ${temp_debug_counter_step7}:`);
            M_pts.forEach((m, idx) => console.log(`    M_pt[${idx}]: (${m.x.toFixed(9)}, ${m.y.toFixed(9)})`));
        }
        // --- End log M_pts ---

        // 7.4 Dual vertices D (integer grid projection)
        const D_vertices_vec2 = []; // Array of THREE.Vector2
        if (temp_debug_counter_step7 <= 1) { // DEBUG for first vertex only
            console.log(`  Detail for D_vertices calc (vertex ${temp_debug_counter_step7}):`);
            console.log(`    v_point: (${v_point_vec2.x.toFixed(6)}, ${v_point_vec2.y.toFixed(6)})`);
            console.log(`    Offsets o_arr: [${o_arr.map(val => val.toFixed(6)).join(', ')}]`);
        }

        for (let k_idx = 0; k_idx < M_pts.length; k_idx++) {
            const m_k_pt_vec2 = M_pts[k_idx];
            _vec2_1.set(0, 0);
            if (temp_debug_counter_step7 <= 1) { // DEBUG
                console.log(`    M_pt[${k_idx}]: (${m_k_pt_vec2.x.toFixed(6)}, ${m_k_pt_vec2.y.toFixed(6)})`);
            }
            for (let i_proj = 0; i_proj < N_param; i_proj++) {
                const val_before_floor = n_arr[i_proj].dot(m_k_pt_vec2) - o_arr[i_proj];
                const floor_val = Math.floor(val_before_floor);
                // const floor_val = Math.floor(val_before_floor + config.epsilon * 0.5);
                if (temp_debug_counter_step7 <= 1) { // DEBUG
                    console.log(`      i_proj=${i_proj}: n_i=(${n_arr[i_proj].x.toFixed(3)},${n_arr[i_proj].y.toFixed(3)}), <n_i,M_k>-o_i = ${val_before_floor.toFixed(6)}, floor_val_nudged = ${floor_val}`);
                }
                _vec2_2.copy(n_arr[i_proj]).multiplyScalar(floor_val);
                _vec2_1.add(_vec2_2);
            }
            D_vertices_vec2.push(_vec2_1.clone());
            if (temp_debug_counter_step7 <= 1) { // DEBUG
                console.log(`    Calculated D_vertex[${k_idx}]: (${_vec2_1.x.toFixed(6)}, ${_vec2_1.y.toFixed(6)})`);
            }
        }

        if (temp_debug_counter_step7 <= 5) {
            console.log(`  D_vertices_vec2.length: ${D_vertices_vec2.length}`);
        }

        // Filter dual vertices by config.maxDrawRadius before deduplication
        const D_vertices_vec2_filtered = D_vertices_vec2.filter(
            dv => Math.hypot(dv.x, dv.y) <= config.maxDrawRadius
        );

        if (temp_debug_counter_step7 <= 5) {
            console.log(`  D_vertices_vec2.length (after radius filter): ${D_vertices_vec2_filtered.length}`);
        }

        // ↓ NEW CODE:   accept any ≥3-gon, but first de-duplicate vertices
        const uniq = [];
        const seen = new Set();
        for (const v of D_vertices_vec2_filtered) { // Use the filtered list
            const key = roundForVertexMerge(v.x) + ',' + roundForVertexMerge(v.y);
            if (!seen.has(key)) {
                uniq.push(v);
                seen.add(key);
            }
        }
        const vCount = uniq.length; // This is the vCount that will be used henceforth
        if (vCount < 3) {             // truly degenerate → ignore
            if (temp_debug_counter_step7 <= 5) { // Optional: log skipped degenerate polygons
                console.log(`  Skipping tile construction for unique vCount = ${vCount} (degenerate).`);
            }
            // Consider adding to a counter for degenerate tiles if needed
            // countSkippedDegenerate++; 
            return;
        }
        // continue with 'uniq' instead of D_vertices_vec2

        // Now, we handle any polygon with vCount >= 3.
        // The area calculation using Shoelace formula is general.
        let area = 0;
        for (let k = 0; k < vCount; k++) {
            const p = uniq[k];
            const q = uniq[(k + 1) % vCount]; // Loop correctly for any vCount
            area += p.x * q.y - p.y * q.x;
        }
        area = Math.abs(area) * 0.5;

        // MODIFIED: Construct detailed tile object
        const plainVertices = uniq.map(v => ({ x: v.x, y: v.y }));
        const centroid = calculatePolygonCentroid(uniq); // Use uniq (array of THREE.Vector2)
        const angles = calculatePolygonAngles(uniq);   // Use uniq (array of THREE.Vector2)

        // --- Assign a typeId based on the shape (angles) ---
        let typeId;
        const shapeKey = angles.map(a => roundN(a, 2)).sort().join('_');

        if (shapeToTypeIdMap.has(shapeKey)) {
            typeId = shapeToTypeIdMap.get(shapeKey);
        } else {
            if (nextTypeId < config.faceColors.length) {
                typeId = nextTypeId;
                shapeToTypeIdMap.set(shapeKey, typeId);
                nextTypeId++;
            } else {
                typeId = config.faceColors.length - 1; // Fallback to the last color
            }
        }
        // --- End of typeId assignment ---

        tempTiles[v_key] = { // Use v_key as the key for the tile
            center: { x: v_point_vec2.x, y: v_point_vec2.y }, // Primal vertex center
            vertices: plainVertices, // Dual polygon vertices {x,y} from 'uniq'
            gridLines: [...v_lines_set].map(s => JSON.parse(s)), // Array of [fam_idx, k_val]
            properties: {
                area: area, // Use generalized area calculation
                angles: angles, // Array of internal angles in degrees
                numVertices: vCount, // Actual number of unique vertices
                centroid: centroid, // {x,y} of the dual polygon's centroid
                typeId: typeId // Store the assigned typeId
            }
        };
        // The generic path for 3-, 5-, 6-, 8-, ...-gons is now omitted / unreachable.

    });
    tilesData = tempTiles;
    console.log('Step 7: Constructed ' + Object.keys(tilesData).length + ' raw dual polygons (quadrilaterals only).');
    console.log(`  Skipped (non-quadrilateral): ${countSkippedNotQuad}`);

    // --- 8. Global post-transforms (Handled by applyGlobalTransforms for vertex points) ---

    // Populate indexedPositions with all unique dual vertices for visualization
    uniqueVertexMap.clear(); // To keep track of unique vertices for visualization
    indexedPositions = [];   // Reset for new points

    Object.values(tilesData).forEach(tile => {
        tile.vertices.forEach(vert => { // tile.vertices are {x, y} objects
            const key = roundForVertexMerge(vert.x) + "," + roundForVertexMerge(vert.y);
            if (!uniqueVertexMap.has(key)) {
                uniqueVertexMap.set(key, indexedPositions.length / 3); // Store index of the point
                indexedPositions.push(vert.x, vert.y, 0); // Add to positions for Points object
            }
        });
    });
    console.log('Populated dual vertices for visualization: ' + (indexedPositions.length / 3) + ' unique points.');

    // 1. For every tile, record its base‐vertex indices (vertexOrder)
    Object.values(tilesData).forEach(tile => {
      tile.vertexOrder = tile.vertices.map(v => { // tile.vertices are {x,y} objects
        const key = roundForVertexMerge(v.x) + "," + roundForVertexMerge(v.y);
        return uniqueVertexMap.get(key);  // global index into indexedPositions
      });
    });

    // Mesh and face generation (formerly Step 9 and updateMainMeshObject call) removed.

    const totalEndTime = performance.now();
    console.log('Full multigrid generation finished in ' + (totalEndTime - startTime).toFixed(2) + ' ms.');
    console.info('Multigrid summary: Dual Polygons (quads): ' + Object.keys(tilesData).length + ', Unique Dual Vertices: ' + uniqueVertexMap.size);

    // --- Compute r_max from dual vertices ---
    r_max = 0; // Reset r_max for the current generation
    uniqueVertexMap.forEach((idx, key) => {
        const [x,y] = key.split(',').map(Number);
        r_max = Math.max(r_max, Math.hypot(x, y));
    });
    console.log("r_max =", r_max);
    // --- End r_max computation ---

    // --- Update GUI slider ranges that depend on r_max ---
    if (domeRadiusController) {
        domeRadiusController.max(Math.max(0.2, 3 * r_max)); // Ensure max is not less than min (0.1)
        domeRadiusController.updateDisplay();
    }
    if (stepHeightController) {
        stepHeightController.max(Math.max(0.02, r_max)); // Ensure max is not less than min (0.01)
        stepHeightController.updateDisplay();
    }
    // --- End GUI slider update ---

    updateVertexPointsObject(); // Create/update vertex points object
    updateMainEdgesObject(); // Create/update main edges object
    updateMainFacesObject();   // NEW – build / refresh face mesh
    updateDomeGeometry();      // NEW - build / refresh dome if enabled
}

function clearGeometry() {
    tilesData = {};
    uniqueVertexMap.clear();
    indexedPositions = [];

    // mainMeshObject related clearing removed
    // mainEdgesObject related clearing removed

    if (mainVertexPointsObject) {
        scene.remove(mainVertexPointsObject);
        if (mainVertexPointsObject.geometry) mainVertexPointsObject.geometry.dispose();
        if (mainVertexPointsObject.material) mainVertexPointsObject.material.dispose();
        mainVertexPointsObject = null;
    }
    if (mainEdgesObject) {
        scene.remove(mainEdgesObject);
        if (mainEdgesObject.geometry) mainEdgesObject.geometry.dispose();
        if (mainEdgesObject.material) mainEdgesObject.material.dispose();
        mainEdgesObject = null;
    }
    if (mainFacesObject) {
        scene.remove(mainFacesObject);
        if (mainFacesObject.geometry) mainFacesObject.geometry.dispose();
        if (mainFacesObject.material) mainFacesObject.material.dispose();
        mainFacesObject = null;
    }
    // faceMaterialsCache clearing removed as cache is removed
}


// =============================================================================
// Visualization Update Functions
// =============================================================================
// const faceMaterialsCache = {}; // REMOVED

// function getMaterialForType(type_id) { ... } // REMOVED
// function resetMaterialCache() { ... } // REMOVED
// function updateMainMeshObject(facesWithTypes) { ... } // REMOVED

function updateVertexPointsObject() {
    if (mainVertexPointsObject) {
        scene.remove(mainVertexPointsObject);
        if (mainVertexPointsObject.geometry) mainVertexPointsObject.geometry.dispose();
        if (mainVertexPointsObject.material) mainVertexPointsObject.material.dispose();
        mainVertexPointsObject = null;
    }

    if (!config.showVertices || indexedPositions.length === 0) {
        return; // Don't create if not showing or no data
    }

    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(indexedPositions, 3));

    const pointsMaterial = new THREE.PointsMaterial({
        color: 0x00ff00, // Bright green
        size: 2,
        sizeAttenuation: false // Points will have a fixed size in pixels
    });

    mainVertexPointsObject = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(mainVertexPointsObject);

    // Ensure transforms are applied if created after initial transform pass
    mainVertexPointsObject.rotation.set(0, 0, 0);
    mainVertexPointsObject.scale.set(1, 1, 1);
    mainVertexPointsObject.position.set(0, 0, 0);
    mainVertexPointsObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
    mainVertexPointsObject.scale.set(config.zeta, config.zeta, config.zeta);
    mainVertexPointsObject.position.set(config.panX, config.panY, 0);
}

function updateMainEdgesObject() {
    if (mainEdgesObject) {
        scene.remove(mainEdgesObject);
        if (mainEdgesObject.geometry) mainEdgesObject.geometry.dispose();
        if (mainEdgesObject.material) mainEdgesObject.material.dispose();
        mainEdgesObject = null;
    }

    if (!config.showEdges || Object.keys(tilesData).length === 0) {
        return; // Don't create if not showing or no data
    }

    const edgeSet = new Set();
    for (const tileKey in tilesData) {
        const tile = tilesData[tileKey];
        const verts = tile.vertices; // [{x,y}, ...] length 4
        for (let i = 0; i < verts.length; i++) {
            const a = verts[i];
            const b = verts[(i + 1) % verts.length];
            
            // Create a key that is the same regardless of winding order
            // by sorting the coordinates of the two points.
            // The key combines x and y of both points, after rounding.
            const p1x = roundForVertexMerge(a.x);
            const p1y = roundForVertexMerge(a.y);
            const p2x = roundForVertexMerge(b.x);
            const p2y = roundForVertexMerge(b.y);

            let key;
            // Consistent key: sort by x, then y if x is equal.
            if (p1x < p2x || (p1x === p2x && p1y < p2y)) {
                key = `${p1x},${p1y},${p2x},${p2y}`;
            } else {
                key = `${p2x},${p2y},${p1x},${p1y}`;
            }
            edgeSet.add(key);
        }
    }

    const edgePositions = [];
    edgeSet.forEach(key => {
        const coords = key.split(',').map(Number);
        edgePositions.push(coords[0], coords[1], 0, coords[2], coords[3], 0);
    });

    if (edgePositions.length === 0) return;

    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(edgePositions, 3)
    );

    const edgeMat = new THREE.LineBasicMaterial({
        color: 0xffffff, // White edges
        linewidth: 1     // Note: linewidth > 1 might not work on all platforms/drivers with WebGL1
    });

    mainEdgesObject = new THREE.LineSegments(edgeGeo, edgeMat);
    scene.add(mainEdgesObject);

    // Apply current global transforms immediately upon creation
    mainEdgesObject.rotation.set(0, 0, 0);
    mainEdgesObject.scale.set(1, 1, 1);
    mainEdgesObject.position.set(0, 0, 0);
    mainEdgesObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
    mainEdgesObject.scale.set(config.zeta, config.zeta, config.zeta);
    mainEdgesObject.position.set(config.panX, config.panY, 0);
}

function updateMainFacesObject() {
    // ----- destroy old mesh -----
    if (mainFacesObject) {
        scene.remove(mainFacesObject);
        if (mainFacesObject.geometry) mainFacesObject.geometry.dispose();
        if (mainFacesObject.material) mainFacesObject.material.dispose();
        mainFacesObject = null;
    }

    if (!config.showFaces || Object.keys(tilesData).length === 0) return;

    // 1) build a global vertex-to-index map
    // The key now includes typeId to differentiate vertices for coloring
    const globalVertexMap = new Map();
    const uniquePositions = [];
    const uniqueColors    = [];

    let nextIndex = 0;
    for (const tile of Object.values(tilesData)) {
        const typeId = tile.properties.typeId !== undefined ? tile.properties.typeId : 0;
        const colorValue = config.faceColors[typeId % config.faceColors.length];
        const color = new THREE.Color(colorValue);

        for (const v_geom of tile.vertices) { // v_geom is {x, y} from tile.vertices
            const posKeyPart = roundForVertexMerge(v_geom.x) + ',' + roundForVertexMerge(v_geom.y);
            const fullKey = posKeyPart + '_type_' + typeId;

            if (!globalVertexMap.has(fullKey)) {
                globalVertexMap.set(fullKey, nextIndex++);
                uniquePositions.push(v_geom.x, v_geom.y, 0);
                uniqueColors.push(color.r, color.g, color.b);
            }
        }
    }

    // 2) build the big index list
    const indices = [];
    for (const tile of Object.values(tilesData)) {
        const typeId = tile.properties.typeId !== undefined ? tile.properties.typeId : 0;
        
        // Vertices of the current tile, used for triangulation
        let verts_for_triangulation = tile.vertices.map(v => new THREE.Vector2(v.x, v.y));
        
        // Ensure CCW for ShapeUtils.triangulateShape
        // Note: if tile.vertices could form a self-intersecting polygon, triangulation might be unpredictable.
        // Assuming simple polygons from the upstream algorithm.
        if (THREE.ShapeUtils.area(verts_for_triangulation) < 0) {
            verts_for_triangulation.reverse();
        }
        
        const tris = THREE.ShapeUtils.triangulateShape(verts_for_triangulation, []);

        for (const tri of tris) { // tri is [local_idx0, local_idx1, local_idx2] referring to verts_for_triangulation
            const tri_global_indices = tri.map(local_idx_in_tile => {
                const v_from_tile_for_key = verts_for_triangulation[local_idx_in_tile]; // This is a THREE.Vector2
                const posKeyPart = roundForVertexMerge(v_from_tile_for_key.x) + ',' + roundForVertexMerge(v_from_tile_for_key.y);
                const fullKey = posKeyPart + '_type_' + typeId; // Use current tile's typeId for the key
                return globalVertexMap.get(fullKey);
            });
            indices.push(tri_global_indices[0], tri_global_indices[1], tri_global_indices[2]);
        }
    }

    // 3) build one indexed geometry
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(uniquePositions, 3));
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(uniqueColors, 3));
    geo.setIndex(indices);

    console.log(`Vertex count before mergeVertices: ${geo.attributes.position.count}`);

    // optional: remove any near-duplicates one last time
    // This will merge vertices if position AND color (and other attributes) are identical.
    const merged = BufferGeometryUtils.mergeVertices(geo, config.epsilon_ui); // Use UI epsilon for merging
    
    console.log(`Vertex count after mergeVertices: ${merged.attributes.position.count}`);
    
    merged.computeVertexNormals();

    // 4) single mesh
    const mat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent:  config.faceOpacity < 1,
        opacity:      config.faceOpacity,
        side:         THREE.DoubleSide
    });
    mainFacesObject = new THREE.Mesh(merged, mat);
    scene.add(mainFacesObject);

    // ----- apply global transforms -----
    mainFacesObject.rotation.set(0, 0, 0);
    mainFacesObject.scale.set(1, 1, 1);
    mainFacesObject.position.set(0, 0, 0);
    mainFacesObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
    mainFacesObject.scale.set(config.zeta, config.zeta, config.zeta);
    mainFacesObject.position.set(config.panX, config.panY, 0);
}

function applyGlobalTransforms() {
    if (mainVertexPointsObject) {
        mainVertexPointsObject.rotation.set(0, 0, 0);
        mainVertexPointsObject.scale.set(1, 1, 1);
        mainVertexPointsObject.position.set(0, 0, 0);

        mainVertexPointsObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
        mainVertexPointsObject.scale.set(config.zeta, config.zeta, config.zeta);
        mainVertexPointsObject.position.set(config.panX, config.panY, 0);
    }

    if (mainEdgesObject) {
        mainEdgesObject.rotation.set(0, 0, 0);
        mainEdgesObject.scale.set(1, 1, 1);
        mainEdgesObject.position.set(0, 0, 0);

        mainEdgesObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
        mainEdgesObject.scale.set(config.zeta, config.zeta, config.zeta);
        mainEdgesObject.position.set(config.panX, config.panY, 0);
    }

    if (mainFacesObject) {
        mainFacesObject.rotation.set(0, 0, 0);
        mainFacesObject.scale.set(1, 1, 1);
        mainFacesObject.position.set(0, 0, 0);

        mainFacesObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
        mainFacesObject.scale.set(config.zeta, config.zeta, config.zeta);
        mainFacesObject.position.set(config.panX, config.panY, 0);
    }

    if (domeMeshObject) {
        domeMeshObject.rotation.set(0,0,0);
        domeMeshObject.scale.set(1,1,1);
        domeMeshObject.position.set(0,0,0);
        domeMeshObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
        domeMeshObject.scale.set(config.zeta, config.zeta, config.zeta);
        domeMeshObject.position.set(config.panX, config.panY, 0);
    }

    if (domeEdgesObject) {
        domeEdgesObject.rotation.set(0,0,0);
        domeEdgesObject.scale.set(1,1,1);
        domeEdgesObject.position.set(0,0,0);
        domeEdgesObject.rotation.z = THREE.MathUtils.degToRad(config.alpha_rot);
        domeEdgesObject.scale.set(config.zeta, config.zeta, config.zeta);
        domeEdgesObject.position.set(config.panX, config.panY, 0);
    }
}

function updateVisibility() {
    const showD = config.extrudeDome;

    if (mainVertexPointsObject) {
        mainVertexPointsObject.visible = config.showVertices && !showD;
    }
    if (mainEdgesObject) {
        mainEdgesObject.visible = config.showEdges && !showD;
    }
    if (mainFacesObject) {
        mainFacesObject.visible = config.showFaces && !showD;
    }

    if (domeMeshObject) {
        domeMeshObject.visible = config.showFaces && showD;
    }
    if (domeEdgesObject) {
        domeEdgesObject.visible = config.showEdges && showD;
    }
}

function updateDomeGeometry() {
  // 4.1  Dispose previous dome
  [domeMeshObject, domeEdgesObject].forEach(obj => {
    if (obj) {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(mat => mat.dispose());
        } else {
          obj.material.dispose();
        }
      }
      scene.remove(obj);
    }
  });
  domeMeshObject = domeEdgesObject = null;
  
  if (!config.extrudeDome || !tilesData || Object.keys(tilesData).length === 0 || r_max === 0) {
    updateVisibility(); // Ensure dome objects are hidden if they were visible
    return;
  }

  // 4.2  Build combined position buffer
  const baseCount = indexedPositions.length / 3; // Number of unique 2D vertices
  const positions = [...indexedPositions]; // flat [x,y,0,…] from original 2D tiling

  const faceTopIndexMap = new Map(); // Maps tile object to array of its top vertex global indices
  let topIndex = baseCount; // Start indexing for new top vertices after base vertices
  
  Object.values(tilesData).forEach(tile => {
    const C = tile.properties.centroid; // {x, y}
    const u = Math.hypot(C.x, C.y) / r_max;
    const z = heightProfile(u);
    const tilt = THREE.MathUtils.degToRad(faceTiltDeg(u));
    const centroidV3 = new THREE.Vector3(C.x, C.y, 0); // Convert to Vector3
    const q = getTiltQuaternion(centroidV3, tilt);

    const localTopIndices = [];
    tile.vertices.forEach(v_obj => { // v_obj is {x,y}
      const p = new THREE.Vector3(v_obj.x, v_obj.y, 0)
        .sub(new THREE.Vector3(C.x, C.y, 0)) // Translate to origin for rotation
        .applyQuaternion(q)
        .add(new THREE.Vector3(C.x, C.y, z)); // Translate back and lift
      positions.push(p.x, p.y, p.z);
      localTopIndices.push(topIndex++);
    });
    faceTopIndexMap.set(tile, localTopIndices);
  });

  if (Object.keys(tilesData).length > 0) {
      const sampleTileKey = Object.keys(tilesData)[0];
      const sampleTile = tilesData[sampleTileKey];
      if (sampleTile && sampleTile.properties) { // Ensure sampleTile and its properties exist
        console.log(
          'Sample tile centroid:', sampleTile.properties.centroid,
          '→ top indices:', faceTopIndexMap.get(sampleTile)
        );
      }
  }

  // 4.3  Build index arrays
  const wallIndices = [];
  const roofIndices = [];

  // --- Walls: for each tile, connect base edge → top edge ---
  Object.values(tilesData).forEach(tile => {
    if (!tile.vertexOrder || !faceTopIndexMap.has(tile)) {
      console.warn("Skipping wall generation for a tile due to missing vertexOrder or top indices", tile);
      return;
    }
    const baseIdx = tile.vertexOrder;               // e.g. [i0, i1, i2, i3]
    const topIdx  = faceTopIndexMap.get(tile);      // e.g. [i4, i5, i6, i7]

    if (baseIdx.length !== topIdx.length) {
        console.warn("Skipping wall for tile: baseIdx and topIdx length mismatch.", tile, baseIdx, topIdx);
        return;
    }

    for (let e = 0; e < baseIdx.length; e++) {
      const b0 = baseIdx[e];
      const b1 = baseIdx[(e + 1) % baseIdx.length];
      const t0 = topIdx[e];
      const t1 = topIdx[(e + 1) % topIdx.length];

      // two triangles per quad‐face edge
      wallIndices.push(b0, b1, t1);
      wallIndices.push(b0, t1, t0);
    }
  });

  // --- Roofs --- 
  Object.values(tilesData).forEach(tile => {
    if (!faceTopIndexMap.has(tile)) {
        console.warn("Skipping roof generation for a tile due to missing top indices", tile);
        return;
    }
    const top_global_indices = faceTopIndexMap.get(tile); // These are already global indices
    // Assuming tiles are quads, as current code filters for them.
    // Triangulate quad [0,1,2,3] into (0,1,2) and (0,2,3) using local order of top_global_indices
    if (top_global_indices.length === 4) { // Explicitly check for quads for safety
        roofIndices.push(top_global_indices[0], top_global_indices[1], top_global_indices[2]);
        roofIndices.push(top_global_indices[0], top_global_indices[2], top_global_indices[3]);
    } else {
        console.warn("Skipping roof for non-quad tile (top_global_indices.length !== 4):", top_global_indices.length);
        // For robust general polygon triangulation, use THREE.ShapeUtils.triangulateShape
        // on the top_global_indices after converting them to Vector2 points if needed,
        // then remap local triangulated indices back to global top_global_indices.
    }
  });

  const combinedIndices = [...wallIndices, ...roofIndices];
  if (combinedIndices.length === 0) {
      console.warn("No indices generated for dome geometry.");
      return;
  }

  console.log(
    'Dome build:',
    'baseVerts=', baseCount,
    'topVerts=', (positions.length/3 - baseCount),
    'wallTris=', wallIndices.length/3,
    'roofTris=', roofIndices.length/3
  );

  // 4.4  Create geometry & materials
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(positions), 3));
  geom.setIndex(combinedIndices);

  const wallMat = new THREE.MeshBasicMaterial({ color:0x404040, side:THREE.DoubleSide, name: 'DomeWall' });
  
  const distinctRoofTypeIds = [...new Set(Object.values(tilesData).map(t => t.properties.typeId))].sort((a,b)=>a-b);
  const preparedRoofMats = [];
  const typeIdToMaterialArrayIndex = {}; // Maps original tile.properties.typeId to index in preparedRoofMats array
  
  distinctRoofTypeIds.forEach((id, arrayIndex) => {
      const colorValue = config.faceColors[id % config.faceColors.length];
      const roofMaterial = new THREE.MeshBasicMaterial({
          color: new THREE.Color(colorValue),
          side: THREE.DoubleSide,
          name: `DomeRoof_Type${id}`
      });
      preparedRoofMats.push(roofMaterial);
      typeIdToMaterialArrayIndex[id] = arrayIndex; // Store index in the specific preparedRoofMats array
  });

  const allMeshMaterials = [wallMat, ...preparedRoofMats];
  domeMeshObject = new THREE.Mesh(geom, allMeshMaterials);
  scene.add(domeMeshObject);

  // Add geometry groups for multi-material mesh
  geom.clearGroups();
  if (wallIndices.length > 0) {
    geom.addGroup(0, wallIndices.length, 0); // Walls use material at index 0 (wallMat)
  }
  
  let roofStartIndex = wallIndices.length;
  Object.values(tilesData).forEach(tile => {
    if (faceTopIndexMap.has(tile) && faceTopIndexMap.get(tile).length === 4) { // Only add group if roof was generated
        const typeId = tile.properties.typeId;
        const materialArrayIndex = typeIdToMaterialArrayIndex[typeId];
        if (materialArrayIndex !== undefined) {
            const roofMaterialOverallIndex = materialArrayIndex + 1; // +1 because wallMat is the first material
            geom.addGroup(roofStartIndex, 6, roofMaterialOverallIndex); // 6 indices per quad roof
            roofStartIndex += 6;
        }
    }
  });
  
  geom.computeVertexNormals(); // Compute normals after indices and groups are set

  // 4.5  Add edges
  // Create domeEdgesObject if domeMeshObject and its geometry exist.
  // Visibility is handled by updateVisibility().
  if (domeMeshObject && domeMeshObject.geometry) { 
    const edgesGeom = new THREE.EdgesGeometry(domeMeshObject.geometry, 30); // Angle threshold for EdgesGeometry
    domeEdgesObject = new THREE.LineSegments(
      edgesGeom,
      new THREE.LineBasicMaterial({ color: 0xffffff, name: 'DomeEdges' })
    );
    scene.add(domeEdgesObject);
  }

  applyGlobalTransforms(); // Apply transforms to the new dome objects
  updateVisibility(); // Update visibility status
}

// =============================================================================
// User Interface Setup (lil-gui)
// =============================================================================

function setupGUI() {
    if (gui) gui.destroy();
    gui = new GUI();
    gui.title("Multigrid Tiling (Rev 2)");

    const regen = () => {
        performMultigridGeneration();
    };

    const updateTransforms = () => { // Renamed from updateTransformsAndEdges
        applyGlobalTransforms();
        // updateMainEdgesObject(); // REMOVED Call
    };

    const updateBasicVis = () => { // Renamed from updateVisAndEdges
        // Face material opacity updates removed
        // Edge color updates removed
        updateVisibility(); // This now only handles vertex points
        // updateMainEdgesObject(); // REMOVED Call
    };

    const genFolder = gui.addFolder('Generation Parameters (§1)');
    genFolder.add(config, 'N', 3, 33, 1).name('Symmetry (N)').onFinishChange(regen);
    genFolder.add(config, 'phi', 0, 0.9999, 0.001).name('Phase (φ)').onFinishChange(regen); // Max < 1
    genFolder.add(config, 'Delta', 0, 1, 0.01).name('Disorder (Δ)').onFinishChange(regen);
    genFolder.add(config, 'seed').name('Random Seed').onFinishChange(regen);
    genFolder.add(config, 'R_param', 1, 200, 1).name('Radius (R lines)').onFinishChange(regen);
    genFolder.add(config, 'maxDrawRadius', 1, 500, 1).name('Max Draw Radius').onFinishChange(regen);
    // Epsilon is not a UI parameter per spec table 1

    const transformFolder = gui.addFolder('Global Transforms (§8)');
    transformFolder.add(config, 'alpha_rot', -360, 360, 1).name('Rotation (α deg)').onFinishChange(updateTransforms);
    transformFolder.add(config, 'zeta', 0.01, 10, 0.01).name('Zoom (ζ)').onFinishChange(updateTransforms);
    transformFolder.add(config, 'panX', -100, 100, 0.1).name('Pan X (p.x)').onFinishChange(updateTransforms);
    transformFolder.add(config, 'panY', -100, 100, 0.1).name('Pan Y (p.y)').onFinishChange(updateTransforms);

    const vizFolder = gui.addFolder('Visualization (§9)');
    vizFolder.add(config, 'showVertices').name('Show Vertices').onChange(() => {
        updateVertexPointsObject(); // Create/destroy points object
        updateVisibility();       // Then ensure its visibility is set
    });
    vizFolder.add(config, 'showEdges').name('Show Edges').onChange(() => {
        updateMainEdgesObject(); // Create/destroy edges object
        updateVisibility();    // Then ensure its visibility is set
    });
    vizFolder.add(config, 'showFaces').name('Show Faces').onChange(() => {
        updateMainFacesObject(); // create/destroy
        updateVisibility();      // then apply flag
    });
    vizFolder.add(config, 'faceOpacity', 0.05, 1, 0.05).name('Face Opacity').onFinishChange(() => {
        if (mainFacesObject && mainFacesObject.material) mainFacesObject.material.opacity = config.faceOpacity;
        // Also need to update transparency flag if opacity hits 1 or drops below 1
        if (mainFacesObject && mainFacesObject.material) mainFacesObject.material.transparent = config.faceOpacity < 1;
    });

    vizFolder.add(config, 'globalHueShift', 0, 1, 0.01).name('Global Hue Shift').onFinishChange(applyHueShiftToFaceColors);

    const faceColorsFolder = vizFolder.addFolder('Face Type Colors (Base)'); // Renamed for clarity
    config.faceColors.forEach((color, index) => {
        const colorObject = { color: color }; // lil-gui operates on object properties
        faceColorsFolder.addColor(colorObject, 'color')
            .name(`Type ${index + 1} Color`)
            .onChange(newColor => {
                config.faceColors[index] = newColor;
                updateMainFacesObject(); // Re-render faces with the new color
            });
    });

    const tuningFolder = gui.addFolder('Fine Tuning');
    tuningFolder.add(config, 'vertexMergeDecimals', 3, 10, 1).name('Vertex Merge Precision').onFinishChange(regen);
    tuningFolder.add(config, 'epsilon_ui', 1e-8, 1e-3, 1e-7).name('Numeric Tolerance (ε)').onFinishChange(regen);
    tuningFolder.add(config, 'q_offset_epsilon_ui', 1e-6, 1e-1, 1e-5).name('Q Offset (εQ)').onFinishChange(regen);

    // --- Dome Extrusion GUI --- 
    const initial_r_max_for_gui = config.R_param > 0 ? config.R_param : 20; // Default if R_param is 0

    const domeF = gui.addFolder('Dome Extrusion');
    domeF.add(config, 'extrudeDome').name('Enable Dome').onChange(updateDomeGeometry);
    
    domeRadiusController = domeF.add(config, 'domeRadius', 0.1, Math.max(0.2, 3 * initial_r_max_for_gui), 0.1).name('Radius').onFinishChange(updateDomeGeometry);
    domeF.add(config, 'profileType', ['spherical','eased','stepped','cascading']).name('Profile').onFinishChange(updateDomeGeometry);
    stepHeightController = domeF.add(config, 'stepHeight', 0.01, Math.max(0.02, initial_r_max_for_gui), 0.01).name('Step Height').onFinishChange(updateDomeGeometry);
    domeF.add(config, 'tierCount', 1, 20, 1).name('Tiers (stepped)').onFinishChange(updateDomeGeometry); // Updated name for clarity
    domeF.add(config, 'tiltDeg', -80, 80, 0.5).name('Tilt ° (not cascading)').onFinishChange(updateDomeGeometry); // Updated name

    const casF = domeF.addFolder('Cascading Profile Settings');
    casF.add(config, 'cascadeSteps', 1, 50, 1).name('Steps').onFinishChange(updateDomeGeometry);
    casF.add(config, 'cascadeDrop', 0.01, 2, 0.01).name('Drop Factor (× r_max)').onFinishChange(updateDomeGeometry); // Clarified name
    casF.add(config, 'tiltInnerDeg', -90, 90, 1).name('Inner Tilt °').onFinishChange(updateDomeGeometry); // Clarified name
    casF.add(config, 'tiltOuterDeg', -90, 90, 1).name('Outer Tilt °').onFinishChange(updateDomeGeometry); // Clarified name

    genFolder.open();
    transformFolder.open();
    vizFolder.open(); // Let's open this too by default
    tuningFolder.open();
    domeF.open(); // Open dome folder by default
}

function applyHueShiftToFaceColors() {
    if (!baseFaceColorsHSL || baseFaceColorsHSL.length === 0) {
        console.warn("Base HSL colors not initialized for hue shift.");
        return;
    }

    const tempColor = new THREE.Color();
    for (let i = 0; i < baseFaceColorsHSL.length; i++) {
        const baseHSL = baseFaceColorsHSL[i];
        tempColor.setHSL(baseHSL.h, baseHSL.s, baseHSL.l);
        
        // Get current HSL of the tempColor (which is based on baseHSL)
        let currentHSL = { h: 0, s: 0, l: 0 };
        tempColor.getHSL(currentHSL);

        // Apply shift
        let newHue = (currentHSL.h + config.globalHueShift) % 1.0;
        if (newHue < 0) newHue += 1.0; // Ensure hue is in [0, 1)

        tempColor.setHSL(newHue, currentHSL.s, currentHSL.l);
        config.faceColors[i] = '#' + tempColor.getHexString();
    }

    // Refresh objects that use these colors
    updateMainFacesObject();
    updateDomeGeometry(); 
}

// =============================================================================
// Three.js Scene Initialization & Rendering Loop
// =============================================================================

function init() {
    scene = new THREE.Scene();
    // scene.background set by HTML/CSS body style

    // --- Initialize base HSL colors for hue shifting ---
    const tempColor = new THREE.Color();
    baseFaceColorsHSL = config.faceColors.map(hexColor => {
        tempColor.set(hexColor);
        let hsl = { h: 0, s: 0, l: 0 };
        tempColor.getHSL(hsl);
        return hsl;
    });
    // --- End HSL initialization ---

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 5000);
    camera.position.set(0, 0, 150); // Initial distance suitable for R_param around 75

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // alpha true for CSS background
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true;
    controls.target.set(0, 0, 0);
    controls.autoRotate = false; // controllable by UI later if needed
    controls.autoRotateSpeed = 0.5;
    controls.update();

    const ambientLight = new THREE.AmbientLight(0x707070);
    scene.add(ambientLight);
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight1.position.set(1, 1.5, 1.2).normalize();
    scene.add(directionalLight1);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight2.position.set(-1, -0.8, 0.5).normalize();
    scene.add(directionalLight2);

    const axesHelper = new THREE.AxesHelper(config.R_param / 5); // Scale helper to R_param
    scene.add(axesHelper);

    setupGUI();
    performMultigridGeneration();

    window.addEventListener('resize', onWindowResize, false);
    // Start animation loop
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    render();
}

function render() {
    renderer.render(scene, camera);
}

// =============================================================================
// Main Execution
// =============================================================================

init(); 

/*
TODO: Entrypoint
- Step increase needs to work
- Bevel
- Half sphere, quarter sphere.
*/