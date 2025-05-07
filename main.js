/**
 * Interactive Penrose Tiling and Dome Generator
 *
 * Overview:
 * This script generates an interactive 2D Penrose tiling with 5-fold symmetry
 * using the "cut-and-project" method from a 5-dimensional (Z^5) integer lattice.
 * It also provides an option to extrude this 2D tiling into a 3D dome structure
 * with configurable height profiles.
 *
 * Core 2D Tiling Generation (Cut-and-Project):
 * 1. Basis Definition (calculateProjectionMatrices):
 *    - Orthonormal basis vectors for the 2D physical subspace (E_phys) and
 *      the 3D internal subspace (E_int) are defined within R^5.
 *    - The acceptance window (W) in E_int is defined as the projection
 *      of the 5D unit hypercube (a rhombic triacontahedron for Penrose).
 *    - The window center is slightly perturbed to ensure generic projections.
 *
 * 2. Point Acceptance (performGeneration):
 *    - Integer points p ∈ Z⁵ are scanned within a configurable range.
 *    - Each point is projected into E_int (p_int). If p_int falls within W
 *      (and its UI-controlled shifts/scale), its projection into E_phys (p_phys)
 *      is accepted.
 *    - Accepted points (vertices of the tiling) are stored with their unique
 *      IDs and 2D physical coordinates.
 *
 * 3. Tiling Connectivity (generateConnectivity):
 *    - Edges: Connect pairs of accepted points whose 5D preimages differ by ±e_k
 *      (a 5D standard basis vector).
 *    - Faces (Rhombi): Formed from parallelograms in Z^5 (defined by an accepted
 *      point and two basis vectors e_i, e_j) where all four 5D vertices project
 *      into the acceptance window. Two types of Penrose rhombi (thick and thin)
 *      are identified.
 *
 * 3D Dome Extrusion (Optional, updateDomeGeometry):
 * 1. Height Profile:
 *    - A Z-height is calculated for each 2D rhombus based on the radial distance
 *      of its centroid and a selected height profile (e.g., spherical, eased, stepped).
 * 2. Roof Construction:
 *    - Each 2D rhombus is lifted to create a flat, planar roof tile. Top vertices
 *      for each rhombus are generated independently (not shared between tiles)
 *      to ensure planarity, preventing warping at shared edges.
 * 3. Wall Construction:
 *    - Vertical walls connect the base vertices of the 2D tiling to the
 *      corresponding vertices of their respective lifted roof tiles.
 *
 * Rendering and Interaction (Three.js & lil-gui):
 * - 2D Tiling: Vertices (THREE.Points), edges (THREE.LineSegments), and faces
 *   (THREE.Mesh with multiple materials for rhombi types) are rendered in the XY plane.
 * - 3D Dome: Extruded rhombi (roofs and walls) are rendered as a single THREE.Mesh
 *   with multiple materials. Dome edges can also be shown (THREE.LineSegments).
 * - UI Controls: Allow modification of generation parameters (e.g., lattice extent,
 *   window properties, dome profile) and visualization settings (e.g., visibility
 *   of points/edges/faces, colors, sizes, opacities).
 * - All changes dynamically trigger regeneration or visual updates of the relevant
 *   Three.js objects.
 *
 * Goals:
 *  - Clearly demonstrate the cut-and-project algorithm for generating Penrose tilings.
 *  - Provide a visually engaging and interactive 2D tiling.
 *  - Offer an optional 3D dome extrusion feature, showcasing how the 2D tiling
 *    can be extended into a three-dimensional structure with flat, well-defined facets.
 *  - Maintain a readable codebase that illustrates these geometric concepts.
 *
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js'; // No longer needed?
import GUI from 'lil-gui';

// =============================================================================
// Configuration & Global State
// =============================================================================

// const MAX_ACCEPTED_VERTS = 50000; // REMOVED: Performance guard, less needed for multigrid?

const config = {
    // --- Mathematical Constants ---
    goldenRatio: (1 + Math.sqrt(5)) / 2,
    tau: (1 + Math.sqrt(5)) / 2, // Explicit tau for clarity

    // --- NEW: Symmetry Parameter ---
    nSym: 5, // Desired n-fold symmetry (≥3)

    // --- Basis Vectors (calculated in calculateProjectionMatrices) ---
    parVecs: [], // Physical space basis vectors (2D) - ROWS of P_PHYS
    ortVecs: [], // Internal space basis vectors (3D) - ROWS of P_INT
    P_PHYS: null, // 2x5 Physical projection matrix
    P_INT: null,  // 3x5 Internal projection matrix
    windowCenterInternalPerturbed: null,        // For genericity
    internalProjectionNormals: [], // Normals for window planes (projections of 5D basis)

    // --- Generation Parameters (UI controllable) ---
    extent: 4,                  // Range [-extent, extent] for 5D lattice search
    windowShiftInternal: [0, 0, 0], // NEW: Array for N-2 dimensions. Size will be adapted if needed.
    windowScale: 1.0,                  // NEW: Scale factor for the acceptance window size

    // --- Extrusion Parameters (UI controllable) ---
    extrudeDome: false,         // Toggle extrusion view
    generatorMode : 'multigrid',   // 'candp' | 'multigrid'
    domeRadius: 6.0,           // Target dome radius (TEMPORARY default, calculated later)
    profileType: 'cascading',   // 'spherical', 'eased', 'stepped', 'cascading'
    tierCount: 5,               // For 'stepped' profile
    stepHeight: 1.0,           // For 'stepped' profile (TEMPORARY default, calculated later)
    _stepHeightUserSet: false, // Internal flag to track if user changed stepHeight
    tiltDeg: 0.0, // NEW: For cone-tilt of rhombi roofs
    // wallThickness: 0,        // Future: for solid cells
    cascadeSteps: 12,          // NEW: number of tiers for cascading
    cascadeDrop: 0.6,          // NEW: total height (× r_max) to distribute for cascading
    tiltInnerDeg: 55,          // NEW: tilt at the very centre for cascading
    tiltOuterDeg: 10,          // NEW: tilt at the rim for cascading

    // --- Visualization Parameters (UI controllable) ---
    vertexColor: '#ffffff',     // Color of the generated points
    vertexSize: 0.05,           // Size of the generated points
    edgeColor: '#ff00ff',       // Color for edges
    faceColor1: '#00eeee',      // Color for thin rhombi (Cyan)
    faceColor2: '#ee00ee',      // Color for thick rhombi (Magenta)
    faceOpacity: 0.6,           // Opacity for faces
    showPoints: true,           // Toggle visibility for points
    showEdges: true,            // Toggle visibility for edges
    showFaces: true,            // Toggle visibility for faces

    // --- Fixed Parameters ---\
    physDimension: 2, // Dimension of the physical space R^d_phys
    windowPerturbationMagnitude: 1e-6, // Small random offset for window center
    epsilonComparison: 1e-12,          // Small value for floating point comparisons (INCREASED PRECISION)

    // --- Tolerances for Face Type Identification (Optional: could be used for coloring) ---\
    rhombAngleTol: 0.01, // Tolerance for distinguishing thick/thin rhombi based on angle
};

// --- Global Three.js Variables ---
let scene, camera, renderer, controls;
let pointsObject = null;      // Holds the THREE.Points object for the quasicrystal vertices
let edgesObject = null;       // Holds the THREE.LineSegments object for edges
let facesObject = null;       // Holds the THREE.Mesh object for faces (potentially split by type)
let domeMeshObject = null;    // NEW: Holds the extruded dome mesh
let domeEdgesObject = null;   // NEW: Holds the edges for the dome mesh
let domeMaterials = {};       // NEW: References to dome materials
let r_max = 1.0;              // NEW: Max radial distance of flat vertices
let gui;
let guiControllers = {};      // NEW: Object to hold references to specific controllers

// --- Global Generation Data ---
let acceptedPointsData = []; // Stores { id, lattice(N-D), phys(2D), internal((N-2)-D) } records
let generatedEdges = [];    // Stores { v1: id1, v2: id2 }
let generatedFaces = [];    // Stores { vertices: [id0, id1, id2, id3], type: string }
let windowPlanes = [];      // Stores { normal: Float64Array (in E_int), offset: number } for the window

// --- NEW: Global Material Caches ---
let faceMaterialsCache = {};    // For 2D flat faces
let domeRoofMaterialsCache = {}; // For 3D dome roofs
const degenerateMaterialTypeString = 'degenerate'; // Consistent key for degenerate material

// --- Scratch objects for math to reduce GC ---
const _tmpVec3 = new THREE.Vector3();
const _tmpQuat = new THREE.Quaternion();
const _centroidHelper = new THREE.Vector3(); // Helper for centroid calculation
const _vertexPosHelper = new THREE.Vector3(); // Helper for vertex transformations

// =============================================================================
// Mathematical Utilities & Projection Logic (Adapted for 5D -> 2D/3D)
// =============================================================================

// --- Vector Math Helpers (Assume vectors are arrays of numbers) ---

function dot(v1, v2) {
    let sum = 0;
    const len = Math.min(v1.length, v2.length); // Handle different lengths safely
    for (let i = 0; i < len; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

function vecScale(v, scalar) { // Renamed from scale
    return v.map(x => x * scalar);
}

function vecSubtract(v1, v2) { // Renamed from subtract
    const len = Math.min(v1.length, v2.length);
    const result = new Array(len);
    for (let i = 0; i < len; i++) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}


function vecAdd(v1, v2) { // Renamed from add
    const len = Math.min(v1.length, v2.length);
    const result = new Array(len);
    for (let i = 0; i < len; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}


function normalize(v) {
    const magnitude = Math.sqrt(dot(v, v));
    if (magnitude < config.epsilonComparison) { // Avoid division by zero
        return v.map(() => 0);
    }
    return vecScale(v, 1 / magnitude); // Use renamed vecScale
}

// --- NEW: Internal Space Vector Math Helpers (for Float64Array) ---
function internalDot(v1, v2) {
    let sum = 0;
    const len = Math.min(v1.length, v2.length);
    for (let i = 0; i < len; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

function internalLengthSq(v) {
    return internalDot(v, v);
}

function internalSubtract(v1, v2, out) {
    const len = Math.min(v1.length, v2.length, out.length);
    for (let i = 0; i < len; i++) {
        out[i] = v1[i] - v2[i];
    }
    return out;
}

/**
 * Calculates orthonormal 2D physical (E_phys) and (N-2)-D internal (E_int) space basis vectors
 * for projecting from Z^N lattice for generalized n-fold symmetric tiling.
 */
function calculateProjectionMatrices() {
    const N = config.nSym;
    if (N < 3) {
        console.error(`nSym (N) must be >= 3. Received ${N}. Defaulting to N=5.`);
        config.nSym = 5; // Fallback to a valid N
        // Optionally, update GUI display here if possible
        // guiControllers.nSym?.setValue(5); // Example, if nSym controller is stored
        // return calculateProjectionMatrices(); // Recurse with corrected N
    }
    const internalDim = N - 2;

    console.log(`Setting up generic Fourier basis for N=${N} (${internalDim}D internal space)...`);
    const norm = 1; // we keep columns orthonormal later – do not pre-shrink

    // --- Physical plane (m = 1 Fourier mode) ---
    const P_PHYS_matrix = [[], []]; // 2xN
    for (let k = 0; k < N; k++) {
        const theta = 2 * Math.PI * k / N;
        P_PHYS_matrix[0][k] = Math.cos(theta) * norm;
        P_PHYS_matrix[1][k] = Math.sin(theta) * norm;
    }

    // --- Internal space (all other Fourier modes) ---
    const P_INT_matrix = []; // (N-2)xN

    // --- ADD m=0 row (constant) required for odd N to ensure correct dimension ---
    if (N % 2 !== 0 && N >= 3) { // Add for odd N >= 3
        const row0 = new Array(N).fill(norm); // m=0 cos row
        P_INT_matrix.push(row0);
        console.log(" -> Added m=0 row for odd N.");
    }

    // --- Add m=2 up to floor(N/2) rows ---
    for (let m = 2; m <= Math.floor(N / 2); m++) {
        const rowC = new Array(N);
        const rowS = new Array(N);
        for (let k = 0; k < N; k++) {
            const theta_m = 2 * Math.PI * m * k / N;
            rowC[k] = Math.cos(theta_m) * norm;
            rowS[k] = Math.sin(theta_m) * norm;
        }
        P_INT_matrix.push(rowC);
        if (2 * m !== N) { // Skip sine row exactly at Nyquist for even N
            P_INT_matrix.push(rowS);
        }
    }

    if (P_INT_matrix.length !== internalDim && N >=3) { // N < 3 handled above
        console.error(`Internal space dimension mismatch! Expected ${internalDim}, got ${P_INT_matrix.length} for N=${N}.`);
        // This case should ideally not be reached if logic is correct.
        // Fallback or error handling might be needed here.
        // For now, log and continue, but projections might be incorrect.
    }
    
    config.P_PHYS = P_PHYS_matrix;
    config.P_INT = P_INT_matrix;

    config.parVecs = P_PHYS_matrix.map(row => Object.freeze([...row]));
    config.ortVecs = P_INT_matrix.map(row => Object.freeze([...row]));

    console.log(`Phys Basis Vectors (Rows of P_PHYS, ${config.parVecs.length}x${N}):`, config.parVecs);
    console.log(`Int Basis Vectors (Rows of P_INT, ${config.ortVecs.length}x${N}):`, config.ortVecs);

    // --- Calculate perturbed window center for (N-2)D internal space ---
    if (internalDim > 0) {
        config.windowCenterInternalPerturbed = new Array(internalDim);
        for (let i = 0; i < internalDim; i++) {
            config.windowCenterInternalPerturbed[i] = (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude;
        }
    } else { // For N=2, internalDim is 0. Handle this edge case.
        config.windowCenterInternalPerturbed = []; // No internal space, no perturbation.
    }
    // Ensure windowShiftInternal also matches internalDim
    // For now, we assume it's a THREE.Vector3 and only use x,y,z if internalDim >=3
    // This will be properly handled when isInWindow is updated.
    // If internalDim < 3, some components of windowShiftInternal might be unused or cause errors if not handled.
    // For simplicity now, we'll let projectToInternal and isInWindow manage this.

    // --- Window planes will be computed dynamically in Step 5 ---
    // Clear any old hardcoded planes. The global `windowPlanes` will be repopulated later.
    windowPlanes = []; 
    console.log("Projection matrices and basis vectors updated for N=" + N + ".");
    // console.log("Dynamic window planes will be computed in the next steps.");
}

/**
 * Projects an N-D vector onto the 2D physical subspace (E_phys).
 * @param {number[]} vecND - The input N-D vector.
 * @returns {THREE.Vector2} The resulting 2D vector in physical space.
 */
function projectToPhysical(vecND) {
    // Dot product of vecND with each ROW of P_PHYS
    const x = dot(vecND, config.parVecs[0]);
    const y = dot(vecND, config.parVecs[1]);
    return new THREE.Vector2(x, y);
}

/**
 * Projects an N-D vector onto the (N-2)-D internal subspace (E_int).
 * @param {number[]} vecND - The input N-D vector.
 * @returns {Float64Array} The resulting (N-2)-D vector in internal space.
 */
function projectToInternal(vecND) {
    const rows = config.P_INT;
    const internalDim = rows.length;
    const arr = new Float64Array(internalDim);
    for (let r = 0; r < internalDim; r++) {
        arr[r] = dot(vecND, rows[r]); // dot is the existing 5D helper, which is fine here.
    }
    return arr;
}

/**
 * Checks if the projection of an N-D point into internal space (vecInternal)
 * falls within the acceptance window (projected N-D hypercube).
 * Condition: |N_k ⋅ (p_int - p_shift)| <= d_k * scale for k=1..N
 * where N_k is the UNNORMALIZED projection Π_int(e_k) and d_k is its calculated offset.
 * @param {Float64Array} vecInternal - The (N-2)-D point in internal space.
 * @returns {boolean} True if the point is within the window, false otherwise.
 */
function isInWindow(vecInternal) {
    if (windowPlanes.length === 0) {
        // This might happen if N<3, where internalDim is 0 or 1, and windowPlanes might not be well-defined yet.
        // Or if called before windowPlanes are computed in step 5.
        // For N=2 (internalDim=0), all points could be considered "in window" or an error.
        // For N=3 (internalDim=1), windowPlanes should have 3 planes.
        if (config.nSym < 3 && vecInternal.length === 0) return true; // Tentative: if no internal space, all points in.
        console.warn("isInWindow called but windowPlanes is empty. Defaulting to false.");
        return false;
    }

    const internalDim = vecInternal.length;
    let effectiveVec = new Float64Array(internalDim); // To store intermediate results

    // Adjust point by window center perturbation
    internalSubtract(vecInternal, config.windowCenterInternalPerturbed, effectiveVec);

    // Adjust by UI shift (config.windowShiftInternal is now an array)
    // Ensure shift array is compatible with internalDim
    let currentShift = config.windowShiftInternal;
    let tempShift = new Float64Array(internalDim);
    for (let i = 0; i < internalDim; i++) {
        tempShift[i] = (i < currentShift.length) ? currentShift[i] : 0; // Use value or 0 if array too short
    }
    
    // Subtract the adapted shift from the current effectiveVec
    let P_minus_S = new Float64Array(internalDim);
    internalSubtract(effectiveVec, tempShift, P_minus_S);
    effectiveVec = P_minus_S; // effectiveVec now holds (p_int - p_perturb - p_shift_ui)

    // Check against the N plane conditions |N_k . x_effective| <= d_k * scale
    // windowPlanes will contain {normal: Float64Array, offset: number}
    for (const plane of windowPlanes) {
        const scaledOffset = plane.offset * config.windowScale;
        // plane.normal is an array, effectiveVec is an array
        if (Math.abs(internalDot(plane.normal, effectiveVec)) > scaledOffset + config.epsilonComparison) {
            return false; // Outside this pair of planes defined by N_k
        }
    }

    return true; // Inside all planes
}

/**
 * Calculates a quaternion for tilting a rhombus roof towards the dome center.
 * @param {THREE.Vector3} centroidXY - The XY centroid of the rhombus (z component will be ignored, assumed 0).
 * @param {number} tiltAngleRad - The desired tilt angle in radians.
 * @param {THREE.Quaternion} outQuaternion - The quaternion to store the result.
 * @returns {THREE.Quaternion} The resulting rotation quaternion (outQuaternion).
 */
function getTiltQuaternion(centroidXY, tiltAngleRad, outQuaternion) {
    outQuaternion.identity(); // Reset before use

    if (Math.abs(tiltAngleRad) < config.epsilonComparison) {
        return outQuaternion; // No tilt, return identity
    }

    // Direction toward centre in XY plane (d = -normalize(c_xy))
    // Use a temporary vector for this calculation to avoid modifying centroidXY if it's reused
    _tmpVec3.set(centroidXY.x, centroidXY.y, 0).multiplyScalar(-1);

    if (_tmpVec3.lengthSq() < config.epsilonComparison * config.epsilonComparison) {
        // Centroid is at origin (or very close), no defined direction to tilt towards.
        return outQuaternion; // Return identity, no tilt
    }
    _tmpVec3.normalize(); // This is vector d (direction from centroid to origin in XY plane)

    // Rotation axis a = d x k, where k = (0,0,1) (world Z axis)
    // This means the rotation axis is in the XY plane, perpendicular to d.
    // If d = (dx, dy, 0), then a = (dy, -dx, 0)
    const rotationAxis = _vertexPosHelper; // Reuse another helper vector
    rotationAxis.set(_tmpVec3.y, -_tmpVec3.x, 0);
    // No need to normalize axis for setFromAxisAngle if it's derived from normalized d like this,
    // as its length will be 1 if d was normalized and in XY plane.
    // However, to be safe, especially if d might not be perfectly XY or normalized due to float issues:
    if (rotationAxis.lengthSq() < config.epsilonComparison * config.epsilonComparison) {
         // This implies d was (0,0,0) or parallel to Z, which shouldn't happen for XY d.
        return outQuaternion; // Cannot determine axis, return identity
    }
    rotationAxis.normalize();

    outQuaternion.setFromAxisAngle(rotationAxis, tiltAngleRad);
    return outQuaternion;
}

/**
 * Calculates the tilt angle in degrees for a face based on its normalized radial distance (u).
 * If the profile type is not 'cascading', it returns the general tiltDeg.
 * @param {number} u - Normalized radial distance (r / r_max), range [0, 1].
 * @returns {number} The calculated tilt angle in degrees.
 */
function faceTiltDeg(u) {
    if (config.profileType !== 'cascading') return config.tiltDeg;
    const t0 = config.tiltInnerDeg;
    const t1 = config.tiltOuterDeg;
    // Ensure u is clamped between 0 and 1 for the interpolation
    const clampedU = Math.max(0, Math.min(1, u));
    return t1 + (t0 - t1) * (1 - clampedU);   // larger tilt near centre (u=0)
}


// =============================================================================
// Quasicrystal Generation Logic (Adapted for 5D -> 2D)
// =============================================================================

/**
 * Performs the main generation process:
 * 1. Iterates through points in an N-D integer lattice (Z^N).
 * 2. Projects each N-D point into (N-2)-D internal and 2D physical spaces.
 * 3. Accepts points if their internal projection is within the acceptance window.
 * 4. Stores accepted points with their N-D lattice coordinates and 2D physical coords.
 * 5. Generates edges and faces based on N-D connectivity.
 * 6. Creates/updates the Three.js objects for points, edges, and faces in the XY plane.
 */
function performGeneration() {
    console.log(`Starting new generation cycle (${config.nSym}D -> 2D)...`);
    const startTime = performance.now();

    // --- Clear previous generated data ---
    acceptedPointsData = [];
    generatedEdges = [];
    generatedFaces = [];
    // windowPlanes = []; // Clearing is already done in calculateProjectionMatrices, but good to be explicit if called independently.
                       // However, per instructions, it's populated here now.

    const N = config.nSym;
    const internalDim = N - 2;

    // --- Dynamically compute acceptance window planes ---
    windowPlanes = []; // Ensure it's clear before populating
    if (N >= 3 && config.P_INT && config.P_INT.length === internalDim) {
        console.log(`Computing ${N} window planes for ${internalDim}D internal space...`);
        for (let k = 0; k < N; k++) {
            // N_k is the kth column of P_INT, seen as a vector in internal space
            const N_k_array = new Float64Array(internalDim);
            for (let i = 0; i < internalDim; i++) {
                if (config.P_INT[i] && k < config.P_INT[i].length) {
                    N_k_array[i] = config.P_INT[i][k];
                } else {
                    // This case should ideally not happen if P_INT is correctly formed
                    console.error(`Error accessing P_INT[${i}][${k}] for N_k construction. N=${N}, internalDim=${internalDim}`)
                    N_k_array[i] = 0; // Default to 0 to avoid crashing, though results might be wrong
                }
            }
            // const offset = 0.5 * internalDot(N_k_array, N_k_array); // OLD: ||N_k||² / 2
            const offset = 0.5;      // half-width of each acceptance strip in raw coordinates
            windowPlanes.push({ normal: N_k_array, offset: offset });
        }
        console.log(` -> ${windowPlanes.length} window planes computed.`);
    } else {
        console.warn(`Skipping window plane computation. N=${N}, P_INT:`, config.P_INT);
        // If N < 3 or P_INT is not set up, windowPlanes will remain empty.
        // isInWindow has a check for empty windowPlanes.
    }

    const maxCoord = Math.max(1, Math.round(config.extent));
    const minCoord = -maxCoord;
    let acceptedCount = 0;
    let processedCount = 0;
    let nextPointId = 0;

    console.log(`Scanning ${config.nSym}D integer lattice Z^${config.nSym} within extent: [${minCoord}, ${maxCoord}]`);

    // Performance Tweak: Precompute max internal radius squared for sphere pre-check
    // Use a slightly generous bound based on the window planes offset
    const maxInternalNormBound = windowPlanes.length > 0 && internalDim > 0
        ? windowPlanes.reduce((max, p) => {
            const normalLengthSq = internalLengthSq(p.normal);
            if (normalLengthSq === 0) return max; // Avoid division by zero for zero-length normals
            return Math.max(max, p.offset / Math.sqrt(normalLengthSq));
          }, 0) * config.windowScale * 1.5 // 1.5 safety factor
        : (config.extent + 0.5) * Math.sqrt(internalDim > 0 ? internalDim : 1); // Fallback if planes not ready or internalDim is 0
    const maxInternalRadiusSq = maxInternalNormBound * maxInternalNormBound;
    let preCheckSkipped = 0;
    const HARD_VERTEX_LIMIT = 75000;

    // --- Iterate through the N-D integer lattice ---\
    // Recursive generator for iterating through N-dimensional hypercube
    function* latticePoints(currentDim, currentPoint) {
        if (currentDim > N) { // Base case: yield the complete N-D point
            processedCount++;
            yield currentPoint;
            return;
        }

        for (let coord = minCoord; coord <= maxCoord; coord++) {
            currentPoint[currentDim - 1] = coord; // Set coordinate for current dimension (0-indexed)
            yield* latticePoints(currentDim + 1, currentPoint);
        }
    }

    // Start the iteration
    for (const pND of latticePoints(1, new Array(N))) {
         // Optional: Add progress logging for very large extents
        // if (processedCount % 100000 === 0) {
        //    console.log(`... scanned ${processedCount} points`);
        // }

        const pInternal = projectToInternal(pND);

        // Performance Tweak: Sphere pre-check
        // Skip if point's internal projection is definitely outside a bounding sphere around the window
        if (internalLengthSq(pInternal) > maxInternalRadiusSq) {
            preCheckSkipped++;
            continue;
        }

        if (isInWindow(pInternal)) {
            const pPhysical = projectToPhysical(pND); // Project to 2D

            acceptedPointsData.push({
                id: nextPointId++,
                lattice: [...pND], // Store a copy of the N-D lattice coordinates
                phys: pPhysical, // Store the 2D physical coordinates
                internal: pInternal // Store (N-2)D internal coordinates (optional)
            });
            acceptedCount++;
            if (acceptedCount >= HARD_VERTEX_LIMIT) { console.warn(`C&P vertex cap (${HARD_VERTEX_LIMIT}) hit – stopping scan`); break; }
        }
        // --- Check if the C&P HARD_VERTEX_LIMIT break needs to propagate upwards ---
        if (acceptedCount >= HARD_VERTEX_LIMIT) {
             break; // Exit the generator loop immediately
        }
    }

    const scanEndTime = performance.now();
    console.log(`Lattice scan complete in ${(scanEndTime - startTime).toFixed(2)} ms.`);
    console.log(` -> Processed ${processedCount} total ${config.nSym}D lattice points (skipped ${preCheckSkipped} by pre-check).`);
    console.log(` -> Accepted ${acceptedCount} points.`);

    if (acceptedPointsData.length > 0) {
        // --- Calculate r_max for extrusion ---
        r_max = 0;
        acceptedPointsData.forEach(pt => {
            r_max = Math.max(r_max, pt.phys.length());
        });
        console.log(` -> Max radial extent (r_max): ${r_max.toFixed(4)}`);

        // --- Set Default Dome Radius if needed ---
        if (config.domeRadius === null) {
            // Default R puts rim roughly 36 deg down hemisphere (sin(pi/5) ~ 0.5878)
            config.domeRadius = r_max / Math.sin(Math.PI / 5); // approx r_max * 1.701
            console.log(` -> Default Dome Radius (R) set to: ${config.domeRadius.toFixed(4)}`);
            // Update GUI listener if available
            const controllerR = gui?.controllers.find(c => c.property === 'domeRadius');
            if(controllerR) controllerR.updateDisplay();
        }
        if (config.stepHeight === null || config.stepHeight === 1.0) { // Check for initial null or temp default
            config.stepHeight = 0.1 * config.domeRadius; // Default step height 10% of R
             const controllerSH = gui?.controllers.find(c => c.property === 'stepHeight');
             if(controllerSH) controllerSH.updateDisplay();
        } else {
            // User likely set a value before r_max was known, mark it as set
            config._stepHeightUserSet = true;
        }

        // --- C-1: Update GUI slider ranges based on r_max ---
        if (gui && guiControllers.domeRadius && guiControllers.stepHeight && guiControllers.tierCount) {
             const Rmin = 0.5 * r_max; const Rmax = 3 * r_max;
             // Clamp current R to new range *before* updating GUI controller range
             config.domeRadius = THREE.MathUtils.clamp(config.domeRadius, Rmin, Rmax);

             // Update Dome Radius controller
             guiControllers.domeRadius.min(Rmin).max(Rmax).updateDisplay();

             // Update Step Height controller
             guiControllers.stepHeight.min(0.01).max(r_max).updateDisplay();

             // Update Tier Count controller
             if (config.stepHeight > 0) {
                 const maxTiers = Math.max(1, Math.floor(config.domeRadius / config.stepHeight));
                 // config.tierCount = THREE.MathUtils.clamp(config.tierCount, 1, maxTiers);
                 guiControllers.tierCount.max(maxTiers).updateDisplay();
             }
        }

        // --- A-1: Pre-compute top layer vertex positions ---
        const topPositions = []; // Will store Vec3 for top vertices
        for (const pt of acceptedPointsData) {
            const r = pt.phys.length();
            const u = r_max > 0 ? r / r_max : 0;
            const z = heightProfile(u);
            topPositions.push(pt.phys.x, pt.phys.y, z); // Corrected: push individual coords, not new THREE.Vector3
        }

        generateConnectivity(); // Generate edges and faces based on 5D rules

        // Update Three.js objects (rendered in XY plane)
        updatePointsObject();
        updateEdgesObject();
        updateFacesObject();

        // --- Initial Dome Generation/View Update ---
        updateDomeGeometry(); // Create dome geometry
        updateVisibility();   // Set initial visibility correctly

    } else {
        console.log("No vertices accepted. Clearing visual objects.");
        updatePointsObject();
        updateEdgesObject();
        updateFacesObject();
    }

    const totalEndTime = performance.now();
    console.log(`Full generation cycle finished in ${(totalEndTime - startTime).toFixed(2)} ms.`);
    console.info(`Generation summary: verts=${acceptedPointsData.length}, edges=${generatedEdges.length}, faces=${generatedFaces.length}`);
}

// --- NEW: Generator Wrapper ---
function generateGeometry() {
    console.log(`Generating geometry using mode: ${config.generatorMode}`);
    if (config.generatorMode === 'candp') {
        performGeneration();        // old cut-and-project
    } else {
        performMultigridGeneration();  // new function, see step 5
    }
}

// =============================================================================
// Connectivity Generation (Adapted for Penrose from Z^N)
// =============================================================================

/**
 * Generates edges and faces (generalized rhombi) based on the N-D lattice
 * connectivity of the accepted points.
 */
function generateConnectivity() {
    if (acceptedPointsData.length === 0) {
        console.log("No points accepted, skipping connectivity generation.");
        generatedEdges = [];
        generatedFaces = [];
        return;
    }

    const startTime = performance.now();
    console.log("Generating connectivity (edges and faces) for Z^N lattice...");

    // --- 1. Build LookupMap from N-D lattice coords to point data ---\
    const lookupMap = new Map();
    acceptedPointsData.forEach(pt => {
        lookupMap.set(pt.lattice.join(','), pt);
    });
    console.log(` -> Built LookupMap with ${lookupMap.size} entries.`);

    // --- 2. Generate Edges (Z^N rule: neighbors differ by +/- e_i) ---\
    generatedEdges = [];
    let edgeCount = 0;
    const checkedEdges = new Set(); // Avoid duplicate checks (e.g., 1-2 vs 2-1)
    const N = config.nSym; // 5
    const standardBasisND = [];
     for(let i=0; i<N; ++i) {
         standardBasisND.push(Object.freeze(Array(N).fill(0).map((_, idx) => idx === i ? 1 : 0)));
     }


    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;

        // Iterate through each dimension i
        for (let i = 0; i < N; i++) {
            // Check neighbor v0 + e_i
            const neighborLatticePos = vecAdd(v0_lattice, standardBasisND[i]); // Use renamed vecAdd
            const neighborKeyPos = neighborLatticePos.join(',');
            if (lookupMap.has(neighborKeyPos)) {
                const neighborPt = lookupMap.get(neighborKeyPos);
                const edgeKey = [pt.id, neighborPt.id].sort().join('-');
                if (!checkedEdges.has(edgeKey)) {
                    checkedEdges.add(edgeKey);
                    generatedEdges.push({ v1: pt.id, v2: neighborPt.id });
                    edgeCount++;
                }
            }
             // Check neighbor v0 - e_i (implicitly covered when iterating through neighborPt)
        }
    }
    console.log(` -> Generated ${edgeCount} edges (Z^N rule).`);

    // --- 3. Generate Faces (Generalized Rhombi from Z^N parallelograms) ---
    // A rhombus is formed by v0, v0+e_i, v0+e_j, v0+e_i+e_j if all are accepted points.
    generatedFaces = [];
    let faceCount = 0;
    const checkedFaces = new Set(); // Avoid duplicates

    // Pre-calculate 2D projections of N-D basis vectors for angle check
    const physProjections = standardBasisND.map(e_i => projectToPhysical(e_i));

    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;
        const p0 = pt; // Point data for v0

        // Iterate through distinct pairs of dimensions (i, j)
        for (let i = 0; i < N; i++) {
            const step_i = standardBasisND[i];
            const v1_lattice = vecAdd(v0_lattice, step_i); // Use renamed vecAdd
            const key1 = v1_lattice.join(',');

            // Check if v0 + e_i is accepted
            if (!lookupMap.has(key1)) continue;
            const p1 = lookupMap.get(key1); // Point data for v1

            for (let j = i + 1; j < N; j++) {
                const step_j = standardBasisND[j];
                const v2_lattice = vecAdd(v0_lattice, step_j); // Use renamed vecAdd
                const key2 = v2_lattice.join(',');

                const v3_lattice = vecAdd(v1_lattice, step_j); // Use renamed vecAdd
                const key3 = v3_lattice.join(',');

                // Check if v0+e_j and v0+e_i+e_j are also accepted points
                if (lookupMap.has(key2) && lookupMap.has(key3)) {
                    const p2 = lookupMap.get(key2); // Point data for v2 = v0+ej
                    const p3 = lookupMap.get(key3); // Point data for v3 = v0+ei+ej

                    // Found a valid parallelogram in 5D projecting to a rhombus in 2D.
                    // Vertices in order: p0, p1, p3, p2 (corresponds to v0, v0+ei, v0+ei+ej, v0+ej)
                    const ids = [p0.id, p1.id, p3.id, p2.id];

                    // Deduplicate using a canonical key (sorted IDs)
                    const faceKey = ids.slice().sort((a, b) => a - b).join('-');
                    if (!checkedFaces.has(faceKey)) {
                        checkedFaces.add(faceKey);

                        // --- Determine Rhombus Type (Thick/Thin) ---\
                        // Based on the angle between the projected edge vectors originating from p0.
                        const edgeVec1 = new THREE.Vector2().subVectors(p1.phys, p0.phys); // Vector p0 -> p1 (proj of e_i)
                        const edgeVec2 = new THREE.Vector2().subVectors(p2.phys, p0.phys); // Vector p0 -> p2 (proj of e_j)

                        const len1Sq = edgeVec1.lengthSq();
                        const len2Sq = edgeVec2.lengthSq();
                        let type = 'unknown';

                        if (len1Sq > config.epsilonComparison && len2Sq > config.epsilonComparison) {
                            // Use integer index difference for classification
                            const diff = (j - i + N) % N; // N=config.nSym. Ensures diff is 0 to N-1
                                                          // For parallelograms from e_i, e_j with i < j, diff will be j-i or N-(j-i)
                                                          // We are interested in the smaller angle, so usually j-i if j-i <= N/2
                                                          // The problem defines d = (j - i + N) % N. Let's stick to this for now.
                                                          // However, typically d=0 is not a valid parallelogram for distinct e_i, e_j.
                                                          // And for tile types, we'd usually consider d and N-d as equivalent (same shape, diff orientation)
                                                          // Let's use a canonical diff: min(j-i, N-(j-i)) for N distinct types for N-gons (up to N/2 types of rhombs)
                           
                            let canonicalDiff = (j - i); // since j > i, this is j-i
                            // Smallest angle corresponds to smallest index difference, or N minus that difference.
                            // Example N=5: j-i can be 1,2,3,4.
                            // (0,1) -> diff=1. (0,2)->diff=2. (0,3)->diff=3 (same as N-2). (0,4)->diff=4 (same as N-1).
                            // Types are usually based on min(diff, N-diff). For N=5, d=1 (thick), d=2 (thin).
                            // For N=8: d=1, d=2, d=3. (d=4 is a square, distinct)
                            // The instructions simply say: const type = `rhomb_d${diff}`; where diff = (j - i + N) % N
                            // This will create types like rhomb_d1, rhomb_d2, ... up to rhomb_d(N-1) if i=0, j=N-1.
                            // This might create more types than geometrically distinct rhombi for some N.
                            // Let's follow the instruction first: d = (j - i + N) % N
                            // Since j > i, j-i is always positive. So (j-i+N)%N is just j-i.
                            // This means diff will range from 1 (e.g. i=0,j=1) to N-1 (e.g. i=0,j=N-1).
                            // This will give N-1 types. This is likely too many.
                            // The table given: N=5 -> 2 types (d=1, d=2). N=8 -> 3 types (d=1,2,3). N=12 -> 5 types (d=1..5)
                            // This implies diff should be min( (j-i), N-(j-i) ), and should go up to floor((N-1)/2) if we exclude N/2 for even N (squares)
                            // Or more simply, diff goes from 1 up to floor(N/2). Let's use this simpler interpretation: d = j-i. Max j-i is N-1. We need to map this. 

                            // Let's use the definition d = (j-i). This will give values 1, 2, ..., N-1.
                            // The prompt's table suggests fewer distinct types, up to floor(N/2).
                            // Type 1: (j-i) = 1 or N-1
                            // Type 2: (j-i) = 2 or N-2
                            // ... up to Type floor(N/2)
                            let actualDiff = j - i; //  1 <= actualDiff <= N-1 because i < j
                            let typeIndex = Math.min(actualDiff, N - actualDiff); // This gives 1, 2, ..., floor(N/2)

                            if (typeIndex > 0 && typeIndex <= Math.floor(N/2)) {
                                type = `rhomb_d${typeIndex}`;
                            } else {
                                // This case should not be hit if N >= 3 and i != j
                                console.warn(`Unexpected typeIndex in face gen: i=${i}, j=${j}, N=${N}, actualDiff=${actualDiff}, typeIndex=${typeIndex}`);
                                type = 'degenerate';
                            }

                            // Original instruction from prompt: const type = `rhomb_d${diff}`; with diff = (j - i + N) % N
                            // Let's re-evaluate. If N=5, i=0:
                            // j=1: diff=(1-0+5)%5 = 1. type=rhomb_d1
                            // j=2: diff=(2-0+5)%5 = 2. type=rhomb_d2
                            // j=3: diff=(3-0+5)%5 = 3. type=rhomb_d3
                            // j=4: diff=(4-0+5)%5 = 4. type=rhomb_d4
                            // This generates N-1 types. The table shows for N=5, d=1 (thick), d=2 (thin). These correspond to typeIndex=1 and typeIndex=2.

                            // So, the rule for type should be `rhomb_d${typeIndex}`
                            // where typeIndex = min(j-i, N-(j-i))

                        } else {
                            // console.warn(`Zero length edge vector for face ${ids.join(',')}`);
                            type = 'degenerate';
                        }

                        // Only add non-degenerate faces
                        if (type !== 'degenerate' && type !== 'unknown') {
                            generatedFaces.push({ vertices: ids, type: type });
                            faceCount++;
                        }
                    }
                }
            }
        }
    }
    console.log(` -> Generated ${faceCount} faces (Penrose rhombi).`);

    const endTime = performance.now();
    console.log(`Connectivity generation finished in ${(endTime - startTime).toFixed(2)} ms.`);
}


// =============================================================================
// Visualization Update Functions (Rendering in XY Plane)
// =============================================================================

/** Cleans up a Three.js object */
function disposeObject(object) {
    if (!object) return;
    if (object.geometry) object.geometry.dispose();
    if (object.material) {
        if (Array.isArray(object.material)) {
            object.material.forEach(m => { if(m && m.dispose) m.dispose(); }); // Check m exists
        } else if (object.material.dispose) {
            object.material.dispose();
        }
    }
     // More robust removal
     if (object.parent) {
        object.parent.remove(object);
     } else if (scene.children.includes(object)) {
        scene.remove(object);
     }
}


/** Creates/updates the THREE.Points object for vertices */
function updatePointsObject() {
    disposeObject(pointsObject);
    pointsObject = null;

    if (!config.showPoints || acceptedPointsData.length === 0) {
        // console.log("Points hidden or no data.");
        return;
    }

    // console.log("Updating points object...");
    const positions = [];
    // Project 2D points into XY plane (z=0)
    acceptedPointsData.forEach(pt => positions.push(pt.phys.x, pt.phys.y, 0)); // RESET to z=0

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: config.vertexColor,
        size: config.vertexSize,
        sizeAttenuation: true // Points scale with distance
    });

    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);
    // console.log(" -> Points object updated.");
}

/** Creates/updates the THREE.LineSegments object for edges */
function updateEdgesObject() {
    disposeObject(edgesObject);
    edgesObject = null;

    if (!config.showEdges || generatedEdges.length === 0 || acceptedPointsData.length === 0) {
         // console.log("Edges hidden or no data.");
         return;
    }

    // console.log("Updating edges object...");
    const idToPointMap = new Map();
    acceptedPointsData.forEach(pt => idToPointMap.set(pt.id, pt));

    const linePositions = [];
    generatedEdges.forEach(edge => {
        const p1Data = idToPointMap.get(edge.v1);
        const p2Data = idToPointMap.get(edge.v2);
        if (p1Data && p2Data) {
             // Use 2D coords, set z=0
             linePositions.push(p1Data.phys.x, p1Data.phys.y, 0); // RESET to z=0
             linePositions.push(p2Data.phys.x, p2Data.phys.y, 0); // RESET to z=0
        } else {
            console.warn(`Edge references missing point ID: ${edge.v1} or ${edge.v2}`);
        }
    });

    if (linePositions.length === 0) {
         // console.log(" -> No valid edge positions found.");
         return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

    const material = new THREE.LineBasicMaterial({ color: config.edgeColor });

    edgesObject = new THREE.LineSegments(geometry, material);
    scene.add(edgesObject);
    // console.log(" -> Edges object updated.");
}

// --- NEW: Color Generation for Rhomb Types ---
function getRhombColor(typeIndex, maxTypeIndex) {
    // typeIndex is 1, 2, ..., maxTypeIndex
    // maxTypeIndex is typically Math.floor(config.nSym / 2)
    const hue = (typeIndex -1) / Math.max(1, maxTypeIndex); // Normalized hue 0 to <1
    return new THREE.Color().setHSL(hue, 0.7, 0.6); // Saturation 0.7, Lightness 0.6
}

/** Creates/updates the THREE.Mesh object for faces (rhombi) using geometry groups */
function updateFacesObject() {
    disposeObject(facesObject);
    facesObject = null;

    if (!config.showFaces || generatedFaces.length === 0 || acceptedPointsData.length === 0) {
        return;
    }

    const idToPointMap = new Map();
    acceptedPointsData.forEach(pt => idToPointMap.set(pt.id, pt));

    const fullPositions = [];
    const allIndices = [];
    const vertexMapFull = new Map();

    function addVertex(ptData) {
        if (!vertexMapFull.has(ptData.id)) {
            const index = vertexMapFull.size;
            vertexMapFull.set(ptData.id, index);
            fullPositions.push(ptData.phys.x, ptData.phys.y, 0); // z=0 for flat faces
            return index;
        }
        return vertexMapFull.get(ptData.id);
    }

    generatedFaces.forEach(face => {
        const p0Data = idToPointMap.get(face.vertices[0]);
        const p1Data = idToPointMap.get(face.vertices[1]);
        const p2Data = idToPointMap.get(face.vertices[2]);
        const p3Data = idToPointMap.get(face.vertices[3]);

        if (p0Data && p1Data && p2Data && p3Data) {
            const i0 = addVertex(p0Data);
            const i1 = addVertex(p1Data);
            const i2 = addVertex(p2Data);
            const i3 = addVertex(p3Data);
            allIndices.push(i0, i1, i2); // Triangle 1
            allIndices.push(i0, i2, i3); // Triangle 2
        } else {
            // console.warn("Face references missing point ID for flat faces:", face.vertices);
        }
    });

    if (vertexMapFull.size === 0 || allIndices.length === 0) {
        return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(fullPositions, 3));
    geometry.setIndex(allIndices);
    geometry.clearGroups();

    const activeMaterialsList = [];
    const materialTypeToIndexMap = new Map(); // Map type string to index in activeMaterialsList
    let nextMaterialIndex = 0;
    const maxRhombTypeIndex = Math.floor(config.nSym / 2);

    let currentFaceTriangleStart = 0;
    generatedFaces.forEach(face => {
        // Ensure face was valid (all vertices existed) before adding group
        const p0Data = idToPointMap.get(face.vertices[0]);
        if (!p0Data || !idToPointMap.get(face.vertices[1]) || !idToPointMap.get(face.vertices[2]) || !idToPointMap.get(face.vertices[3])) {
            return; // Skip if any vertex was missing during index creation
        }

        let materialType = face.type; // e.g., "rhomb_d1", "degenerate"

        if (!faceMaterialsCache[materialType]) {
            let color;
            let standardProps = {
                opacity: config.faceOpacity,
                transparent: config.faceOpacity < 1.0,
                side: THREE.DoubleSide,
                metalness: 0.1,
                roughness: 0.7,
                polygonOffset: true,
                polygonOffsetFactor: 1,
                polygonOffsetUnits: 1
            };

            if (materialType.startsWith('rhomb_d')) {
                const typeIndex = parseInt(materialType.substring('rhomb_d'.length));
                color = getRhombColor(typeIndex, maxRhombTypeIndex);
                faceMaterialsCache[materialType] = new THREE.MeshStandardMaterial({ ...standardProps, color: color });
            } else { // Default to degenerate for any other type
                materialType = degenerateMaterialTypeString; // Ensure consistent key
                if (!faceMaterialsCache[materialType]) { // Check again with consistent key
                    faceMaterialsCache[materialType] = new THREE.MeshStandardMaterial({
                        ...standardProps,
                        color: 0x808080, // Grey
                        opacity: config.faceOpacity * 0.5, // More transparent
                        transparent: (config.faceOpacity * 0.5) < 1.0,
                    });
                }
            }
        }
        
        const material = faceMaterialsCache[materialType];
        // Update opacity from GUI (already done if newly created, but good for existing)
        material.opacity = (materialType === degenerateMaterialTypeString) ? config.faceOpacity * 0.5 : config.faceOpacity;
        material.transparent = material.opacity < 1.0;

        let matIndexInMesh = materialTypeToIndexMap.get(materialType);
        if (matIndexInMesh === undefined) {
            activeMaterialsList.push(material);
            matIndexInMesh = nextMaterialIndex++;
            materialTypeToIndexMap.set(materialType, matIndexInMesh);
        }
        
        geometry.addGroup(currentFaceTriangleStart, 6, matIndexInMesh); // Each face is 2 triangles = 6 indices
        currentFaceTriangleStart += 6;
    });

    if (activeMaterialsList.length > 0) {
        geometry.computeVertexNormals();
        facesObject = new THREE.Mesh(geometry, activeMaterialsList);
        scene.add(facesObject);
    }
}


// =============================================================================
// Dome Extrusion Logic
// =============================================================================

/**
 * Calculates the Z-height based on normalized radial distance (u) and profile type.
 * @param {number} u - Normalized radial distance (r / r_max), range [0, 1].
 * @returns {number} The calculated Z-coordinate.
 */
function heightProfile(u) {
    const R = config.domeRadius;
    if (u * r_max > R && config.profileType !== 'cascading') return 0; // Clamp height if point is beyond dome radius footprint, except for cascading
    // Ensure u is within [0,1] for profile functions to avoid issues like sqrt(negative)
    u = Math.max(0, Math.min(1, u));

    switch (config.profileType) {
        case 'spherical':
            // Ensure argument to sqrt is non-negative due to float precision
            const R2 = R * R;
            const r2 = (u * r_max) * (u * r_max);
            return R - Math.sqrt(Math.max(0, R2 - r2));
        case 'eased':
            // Parabolic profile: h(u) = R * u^2
            return R * u * u;
        case 'stepped':
            // Muqarnas-style steps
            const effectiveTierCount = Math.max(1, config.tierCount);
            const effectiveStepHeight = config.stepHeight > 0 ? config.stepHeight : (0.1 * R);
            const tier = Math.floor(effectiveTierCount * u);
            // Ensure z doesn't exceed R for the highest tier
            return Math.min(R, effectiveStepHeight * tier);
        case 'cascading': {
            const k = Math.max(1, config.cascadeSteps); // Ensure k is at least 1
            const stepId = Math.floor(u * k);           // 0 … k-1
            // If u is 1.0, stepId can be k. Clamp it to k-1 for the highest step.
            const clampedStepId = Math.min(stepId, k - 1);
            const dz = (config.cascadeDrop * r_max) / k; // total height is cascadeDrop * r_max
            return dz * clampedStepId;
        }
        default:
            console.warn("Unknown profile type:", config.profileType);
            return 0; // Default to flat
    }
}

/**
 * Creates or updates the extruded dome mesh.
 * REVISED VERSION with dynamic roof materials.
 */
function updateDomeGeometry() {
    // Ensure r_max and base geometry is ready
    if (acceptedPointsData.length === 0 || generatedFaces.length === 0) {
        disposeObject(domeMeshObject);
        disposeObject(domeEdgesObject);
        domeMeshObject = null;
        domeEdgesObject = null;
        return;
    }

    disposeObject(domeMeshObject);
    disposeObject(domeEdgesObject); // Also dispose edges
    domeMeshObject = null;
    domeEdgesObject = null;
    // Keep domeMaterials = {} for now, will repopulate at the end
    // But we use domeRoofMaterialsCache for creation/lookup

    if (!config.extrudeDome || acceptedPointsData.length === 0 || generatedFaces.length === 0) {
        console.log("Dome view disabled or no data.");
        return;
    }

    console.log("Updating dome geometry (dynamic roofs)...");
    const startTime = performance.now();

    // --- 1. Prepare Vertex Data ---
    const idToPointData = new Map();
    acceptedPointsData.forEach(pt => idToPointData.set(pt.id, pt));

    const idToBaseIndex = new Map();
    const baseVertexCount = acceptedPointsData.length;
    acceptedPointsData.forEach((pt, index) => { idToBaseIndex.set(pt.id, index); });

    const topOffsetStart = baseVertexCount;
    let runningTopIndex = 0;
    const faceTopIndexMap = new Map();
    const topPositions = []; // flat array of XYZ for top vertices

    // Calculate top vertex positions for each face
    generatedFaces.forEach((face, fIdx) => {
        const ptIds = face.vertices;
        _centroidHelper.set(0, 0, 0);
        let validPointsForCentroid = 0;
        ptIds.forEach(id => {
            const pData = idToPointData.get(id);
            if (pData) {
                _centroidHelper.x += pData.phys.x;
                _centroidHelper.y += pData.phys.y;
                validPointsForCentroid++;
            }
        });

        if (validPointsForCentroid < 4) {
            console.warn(`updateDomeGeometry: Face ${fIdx} (${face.type}) has only ${validPointsForCentroid} valid points. Skipping tilt/face.`);
            const localTopIndicesOnError = [];
            for (let i = 0; i < (ptIds.length || 4); i++) { // Assume 4 if ptIds missing?
                 topPositions.push(0,0,0); // Push degenerate flat vertices
                 localTopIndicesOnError.push(topOffsetStart + runningTopIndex++);
            }
            faceTopIndexMap.set(fIdx, localTopIndicesOnError);
            return; // Skip this face
        }
        _centroidHelper.divideScalar(validPointsForCentroid);

        const u = r_max > 0 ? _centroidHelper.length() / r_max : 0;
        const zProfile = heightProfile(u);
        const tiltAngleRad = THREE.MathUtils.degToRad(faceTiltDeg(u));
        const tiltQuat = getTiltQuaternion(_centroidHelper, tiltAngleRad, _tmpQuat);

        const localTopIndices = [];
        ptIds.forEach(id => {
            const pData = idToPointData.get(id);
            _vertexPosHelper.set(pData ? pData.phys.x : 0, pData ? pData.phys.y : 0, 0);
            _vertexPosHelper.sub(_centroidHelper).applyQuaternion(tiltQuat).add(_centroidHelper);
            _vertexPosHelper.z += zProfile;
            if (config.profileType === 'cascading') {
                const k = Math.max(1, config.cascadeSteps);
                const dz = (config.cascadeDrop * r_max) / k;
                _vertexPosHelper.z -= dz * 0.5;
            }
            topPositions.push(_vertexPosHelper.x, _vertexPosHelper.y, _vertexPosHelper.z);
            localTopIndices.push(topOffsetStart + runningTopIndex++);
        });
        faceTopIndexMap.set(fIdx, localTopIndices);
    });

    const totalVerts = baseVertexCount + runningTopIndex;
    const positions = new Float32Array(totalVerts * 3);
    // Copy base layer
    acceptedPointsData.forEach((pt, i) => {
        positions[3 * i] = pt.phys.x;
        positions[3 * i + 1] = pt.phys.y;
        positions[3 * i + 2] = 0;
    });
    // Copy top layer
    positions.set(topPositions, baseVertexCount * 3);

    // --- 2. Prepare Geometry & Materials ---
    const indices = [];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.clearGroups();

    const activeDomeMaterialsList = []; // Holds Wall, then Roof materials
    const domeMaterialTypeToIndexMap = new Map(); // Maps type string to index in activeDomeMaterialsList
    let nextDomeMaterialIndex = 0;

    // --- Wall Material (Index 0) ---
    const wallMaterialType = 'wall';
    if (!domeRoofMaterialsCache[wallMaterialType]) { // Use roof cache for wall too, or a separate one?
         domeRoofMaterialsCache[wallMaterialType] = new THREE.MeshStandardMaterial({
             color: 0xFF8C00, // Orange
             side: THREE.DoubleSide,
             metalness: 0.1, roughness: 0.8,
             opacity: config.faceOpacity, transparent: config.faceOpacity < 1.0,
             polygonOffset: true, polygonOffsetFactor: 0.5, polygonOffsetUnits: 1
         });
    }
    const wallMaterial = domeRoofMaterialsCache[wallMaterialType];
    wallMaterial.opacity = config.faceOpacity;
    wallMaterial.transparent = config.faceOpacity < 1.0;
    
    activeDomeMaterialsList.push(wallMaterial);
    domeMaterialTypeToIndexMap.set(wallMaterialType, nextDomeMaterialIndex++);
    const wallMaterialIndex = domeMaterialTypeToIndexMap.get(wallMaterialType);

    // --- 3. Generate Wall Faces ---
    const wallEdgeSeen = new Set();
    generatedEdges.forEach(e => {
        const key = e.v1 < e.v2 ? `${e.v1}_${e.v2}` : `${e.v2}_${e.v1}`;
        if (wallEdgeSeen.has(key)) return;
        wallEdgeSeen.add(key);
        const b0 = idToBaseIndex.get(e.v1);
        const b1 = idToBaseIndex.get(e.v2);
        if (b0 === undefined || b1 === undefined) return;

        const facesSharing = generatedFaces
            .map((f, idx) => ({ f, idx }))
            .filter(obj => obj.f.vertices.includes(e.v1) && obj.f.vertices.includes(e.v2));

        facesSharing.forEach(obj => {
            const idx = obj.idx;
            const vArr = obj.f.vertices;
            const localTop = faceTopIndexMap.get(idx);
            if (!localTop) return;
            const lf0 = vArr.indexOf(e.v1);
            const lf1 = vArr.indexOf(e.v2);
            if (lf0 === -1 || lf1 === -1 || lf0 >= localTop.length || lf1 >= localTop.length) return;
            const t0 = localTop[lf0];
            const t1 = localTop[lf1];
            indices.push(b0, b1, t1, b0, t1, t0); // Two triangles for wall quad
        });
    });
    const wallIndexCount = indices.length;
    if (wallIndexCount > 0) {
        geometry.addGroup(0, wallIndexCount, wallMaterialIndex); // Group 0 for walls
    }

    // --- 4. Generate Roof Faces & Materials ---
    const maxRhombTypeIndex = Math.floor(config.nSym / 2);
    let roofStartIndex = wallIndexCount; // Start index for roof groups

    generatedFaces.forEach((face, fIdx) => {
        const t = faceTopIndexMap.get(fIdx);
        if (!t || t.length !== 4) {
            return; // Skip if face vertices are not valid (e.g., from centroid warning)
        }

        let materialType = face.type;
        let isDegenerate = false;

        // Ensure type is valid, default to degenerate if not
        if (!materialType || !materialType.startsWith('rhomb_d')) {
            materialType = degenerateMaterialTypeString;
            isDegenerate = true;
        }

        // Create material if not cached
        if (!domeRoofMaterialsCache[materialType]) {
            let color;
            let standardProps = {
                opacity: isDegenerate ? config.faceOpacity * 0.5 : config.faceOpacity,
                transparent: (isDegenerate ? config.faceOpacity * 0.5 : config.faceOpacity) < 1.0,
                side: THREE.DoubleSide,
                metalness: 0.1,
                roughness: 0.7
            };
            if (isDegenerate) {
                domeRoofMaterialsCache[materialType] = new THREE.MeshStandardMaterial({ ...standardProps, color: 0x808080 });
            } else {
                const typeIndex = parseInt(materialType.substring('rhomb_d'.length));
                color = getRhombColor(typeIndex, maxRhombTypeIndex);
                domeRoofMaterialsCache[materialType] = new THREE.MeshStandardMaterial({ ...standardProps, color: color });
            }
        }

        // Update opacity from GUI
        const material = domeRoofMaterialsCache[materialType];
        material.opacity = isDegenerate ? config.faceOpacity * 0.5 : config.faceOpacity;
        material.transparent = material.opacity < 1.0;

        // Get index in the final material list for the mesh
        let matIndexInMesh = domeMaterialTypeToIndexMap.get(materialType);
        if (matIndexInMesh === undefined) {
            activeDomeMaterialsList.push(material); // Add to the list [wall, roof1, roof2, ...]
            matIndexInMesh = nextDomeMaterialIndex++; // Assign the next available index
            domeMaterialTypeToIndexMap.set(materialType, matIndexInMesh); // Store mapping type -> index
        }

        indices.push(t[0], t[1], t[2], t[0], t[2], t[3]); // Add roof quad indices
        geometry.addGroup(roofStartIndex, 6, matIndexInMesh); // Use the mapped index for the group
        roofStartIndex += 6;
    });

    // --- 5. Finalize Geometry and Create Mesh ---
    geometry.setIndex(indices);
    if (indices.length > 0) {
        geometry.computeVertexNormals(); // Compute normals if there's geometry
    }

    if (activeDomeMaterialsList.length > 0) {
        domeMeshObject = new THREE.Mesh(geometry, activeDomeMaterialsList);
        scene.add(domeMeshObject);
    } else {
        // console.log("No geometry or materials generated for dome mesh.");
    }

    // --- Update domeMaterials global object for GUI access ---
    // Store references to the active materials by type name for GUI updates (opacity/visibility)
    domeMaterials = {}; // Clear old references
    domeMaterialTypeToIndexMap.forEach((index, type) => {
        domeMaterials[type] = activeDomeMaterialsList[index];
    });

    // --- 6. Create Dome Edges ---
    if (domeMeshObject && domeMeshObject.geometry.attributes.position && domeMeshObject.geometry.attributes.position.count > 0) {
        disposeObject(domeEdgesObject); // Dispose previous before creating new
        const edgesGeom = new THREE.EdgesGeometry(domeMeshObject.geometry, 30); 
        const edgeMaterial = new THREE.LineBasicMaterial({ color: config.edgeColor });
        domeEdgesObject = new THREE.LineSegments(edgesGeom, edgeMaterial);
        scene.add(domeEdgesObject);
    } else {
        disposeObject(domeEdgesObject);
        domeEdgesObject = null;
    }

    const endTime = performance.now();
    console.log(`Dome geometry updated in ${(endTime - startTime).toFixed(2)} ms.`);
}

/**
 * Toggles visibility between flat Penrose objects and the extruded dome.
 */
function toggleDomeView() {
    // This function is now redundant, as updateVisibility handles everything.
    // Kept for potential future logic, but currently just calls updateVisibility.
    // console.log("toggleDomeView called (now likely redundant)");
    updateVisibility();
}

// NEW: Central function to control visibility of all objects
function updateVisibility() {
    const showDome = config.extrudeDome;

    // Lazy build/update dome if enabling and it doesn't exist
    if (showDome && !domeMeshObject) {
        console.log("Dome geometry missing, building...");
        updateDomeGeometry(); // Build it now
        // If building failed (e.g., no base data), domeMeshObject might still be null
        if (!domeMeshObject) {
            console.error("Failed to build dome geometry.");
            // Optionally force showDome to false?
            // config.extrudeDome = false; // Revert the toggle state
            // guiControllers.domeToggle?.updateDisplay();
            // showDome = false; // Use reverted state for visibility below
            return; // Exit early
        }
    }

    // Flat geometry visibility
    if (pointsObject) pointsObject.visible = !showDome && config.showPoints;
    if (edgesObject) edgesObject.visible = !showDome && config.showEdges;
    if (facesObject) facesObject.visible = !showDome && config.showFaces;

    // Dome geometry visibility
    if (domeMeshObject) {
        // Overall mesh container is visible if dome is on
        domeMeshObject.visible = showDome;
        // Control roof visibility via materials
        // if (domeMaterials.thinRoof) domeMaterials.thinRoof.visible = config.showFaces; // OLD
        // if (domeMaterials.thickRoof) domeMaterials.thickRoof.visible = config.showFaces; // OLD
        // if (domeMaterials.unknownRoof) domeMaterials.unknownRoof.visible = config.showFaces; // OLD
        Object.entries(domeMaterials).forEach(([key, mat])=>{ // NEW dynamic iteration
            if (!mat) return; // Skip if material somehow null/undefined
            if (key === 'wall') return;          // wall visibility is handled separately (always on)
            mat.visible = config.showFaces;
        });
        // Keep walls always visible when dome is visible (or add separate toggle)
        if (domeMaterials.wall) domeMaterials.wall.visible = true;
    }
    if (domeEdgesObject) {
        domeEdgesObject.visible = showDome && config.showEdges;
    }
}

// =============================================================================
// User Interface Setup (lil-gui)
// =============================================================================

/** Sets up the lil-gui panel */
function setupGUI() {
    if (gui) gui.destroy(); // Destroy previous GUI if exists
    gui = new GUI();
    guiControllers = {}; // Reset references when GUI is rebuilt
    gui.title("Penrose Tiling Controls");

    // Function to dispose existing material caches
    const disposeMaterialCaches = () => {
        const disposeMaterial = (mat) => { if (mat && mat.dispose) mat.dispose(); };
        Object.values(faceMaterialsCache).forEach(disposeMaterial);
        Object.values(domeRoofMaterialsCache).forEach(disposeMaterial);
        faceMaterialsCache = {};
        domeRoofMaterialsCache = {};
        // domeMaterials global reference is rebuilt in updateDomeGeometry, just clear caches
        console.log("Cleared and disposed material caches.");
    };

    // --- Generation Parameters Folder ---
    const genFolder = gui.addFolder('Generation Parameters');
    guiControllers.nSym = genFolder.add(config, 'nSym', 3, 12, 1).name('n-fold Symmetry (N)').onChange(() => {
        if (config.nSym > 7) { 
            const estimatedComplexity = Math.pow(config.extent, config.nSym);
            console.warn(`High nSym (${config.nSym}) with extent ${config.extent}. Est. complexity factor ~${estimatedComplexity.toExponential(2)}. May be slow!`);
        }
        // Clear material caches when N changes
        disposeMaterialCaches(); 

        // Recalculate basis and update window shift array size
        calculateProjectionMatrices(); 
        const internalDim = config.nSym - 2;
        // Ensure windowShiftInternal array has enough elements, padding with 0 if needed
        while (config.windowShiftInternal.length < internalDim) {
             config.windowShiftInternal.push(0);
        }
        // Note: We don't truncate the array if internalDim decreases, previous values remain.
        
        // Rebuild the shift GUI dynamically
        rebuildShiftGUI(genFolder);

        generateGeometry(); // NEW CALL
    });
    guiControllers.extent = genFolder.add(config, 'extent', 1, 7, 1).name('Lattice Extent (per dim)').onChange(generateGeometry); // NEW CALL
    guiControllers.windowScale = genFolder.add(config, 'windowScale', 0.1, 3.0, 0.05).name('Window Scale').onChange(generateGeometry); // NEW CALL

    // --- NEW: Generator Mode Selector ---
    guiControllers.generatorMode = genFolder.add(config, 'generatorMode', ['multigrid', 'candp'])
       .name('Generator')
       .onChange(value => {
            // Enable/disable C&P specific controls
            const isCandP = (value === 'candp');
            if (guiControllers.windowScale) guiControllers.windowScale.enable(isCandP);
            // Enable/disable shift controls (need access to the dynamically created ones)
            if (shiftFolder && shiftFolder.controllers) { // NEW Check
                 shiftFolder.controllers.forEach(c => c.enable(isCandP));
            }
            // Regenerate
            generateGeometry();
            // --- NEW: Disable pointer events on C&P controls when in multigrid mode ---
            if (guiControllers.windowScale && guiControllers.windowScale.domElement && guiControllers.windowScale.domElement.parentElement) {
                guiControllers.windowScale.domElement.parentElement.style.pointerEvents = isCandP ? 'auto' : 'none';
            }
            // Apply to shiftFolder controllers as well
            if (shiftFolder && shiftFolder.controllers) {
                shiftFolder.controllers.forEach(c => {
                    if (c.domElement && c.domElement.parentElement) {
                        c.domElement.parentElement.style.pointerEvents = isCandP ? 'auto' : 'none';
                    }
                });
            }
       });

    // --- Internal Window Shift Sub-Folder (Dynamically Built) ---
    let shiftFolder = null; // Keep reference to folder
    const shiftRange = 1.0; 

    function rebuildShiftGUI(parentFolder) {
        if (shiftFolder) {
            shiftFolder.destroy(); // Remove old folder and its controllers
        }
        shiftFolder = parentFolder.addFolder(`Window Shift (${config.nSym - 2}D Internal)`);
        const internalDim = config.nSym - 2;
        // Adapt shift array size if needed (redundant if done in nSym onChange, but safe)
         while (config.windowShiftInternal.length < internalDim) {
             config.windowShiftInternal.push(0);
         }

        for (let i = 0; i < internalDim; i++) {
            // Need to add controller for config.windowShiftInternal[i]
            // lil-gui doesn't directly support adding controllers for array elements easily.
            // Workaround: Add to a temporary proxy object, or rebuild GUI differently.
            // Simpler for now: Just show the first 3 if dim >= 3, like before, until better GUI strategy.
            // This avoids complex proxy objects or full GUI rebuild on N change.
            if (i < 3) { // Limit to 3 sliders for simplicity for now
                 const axis = ['X', 'Y', 'Z'][i];
                 shiftFolder.add(config.windowShiftInternal, i, -shiftRange, shiftRange, 0.01)
                     .name(`Shift ${axis}`)
                     .onChange(generateGeometry) // NEW CALL
                     .listen();
            } else {
                // Optionally add disabled display or hide for dims > 3?
                // shiftFolder.add({value: config.windowShiftInternal[i]}, 'value').name(`Shift Axis ${i+1}`).disable();
                 break; // Stop adding sliders after Z for now
            }
        }
         if (internalDim > 0) shiftFolder.open();
    }

    rebuildShiftGUI(genFolder); // Initial build
    // genFolder.open(); // Keep top-level folder open

    // --- Initial enable/disable based on default mode ---
    const initialIsCandP = (config.generatorMode === 'candp');
    if (guiControllers.windowScale) guiControllers.windowScale.enable(initialIsCandP);
    if (shiftFolder && shiftFolder.controllers) { // NEW Check
        shiftFolder.controllers.forEach(c => c.enable(initialIsCandP));
    }

    // --- Visualization Parameters Folder ---
    const vizFolder = gui.addFolder('Visualization');

    // --- Points Controls ---\
    const pointsFolder = vizFolder.addFolder('Points');
    pointsFolder.add(config, 'showPoints').name('Show Points').onChange(updateVisibility);
    pointsFolder.addColor(config, 'vertexColor').name('Color').onChange(() => {
        if (pointsObject && pointsObject.material) pointsObject.material.color.set(config.vertexColor);
    });
    pointsFolder.add(config, 'vertexSize', 0.001, 0.2, 0.001).name('Size').onChange(() => {
        if (pointsObject && pointsObject.material) pointsObject.material.size = config.vertexSize;
    });
    // pointsFolder.open(); // Default closed

    // --- Edges Controls ---\
    const edgesFolder = vizFolder.addFolder('Edges');
    edgesFolder.add(config, 'showEdges').name('Show Edges').onChange(updateVisibility);
    edgesFolder.addColor(config, 'edgeColor').name('Color').onChange(() => {
         if (edgesObject && edgesObject.material) edgesObject.material.color.set(config.edgeColor);
         // Also update dome edge color
         if (domeEdgesObject && domeEdgesObject.material) domeEdgesObject.material.color.set(config.edgeColor);
    });
    // edgesFolder.open(); // Default closed

    // --- Faces Controls ---\
    const facesFolder = vizFolder.addFolder('Faces (Rhombi)');
    facesFolder.add(config, 'showFaces').name('Show Faces').onChange(updateVisibility);
    facesFolder.add(config, 'faceOpacity', 0, 1, 0.01).name('Opacity').onChange(() => {
         const isTransparent = config.faceOpacity < 1.0;
         const degenerateOpacity = config.faceOpacity * 0.5;
         const isDegenerateTransparent = degenerateOpacity < 1.0;

         // Update flat face materials
         for (const type in faceMaterialsCache) {
             const material = faceMaterialsCache[type];
             if (type === degenerateMaterialTypeString) {
                 material.opacity = degenerateOpacity;
                 material.transparent = isDegenerateTransparent;
             } else {
                 material.opacity = config.faceOpacity;
                 material.transparent = isTransparent;
             }
         }

         // Update dome roof materials
         for (const type in domeRoofMaterialsCache) {
             const material = domeRoofMaterialsCache[type];
             if (type === degenerateMaterialTypeString) {
                 material.opacity = degenerateOpacity;
                 material.transparent = isDegenerateTransparent;
             } else {
                 material.opacity = config.faceOpacity;
                 material.transparent = isTransparent;
             }
         }

         // Update dome wall material opacity
         if (domeMaterials.wall) {
              domeMaterials.wall.opacity = config.faceOpacity; 
              domeMaterials.wall.transparent = isTransparent;
         }
     });
    facesFolder.open(); // Default open

    // --- Extrusion Controls ---
    const extrudeFolder = gui.addFolder('Dome Extrusion');
    guiControllers.domeToggle = extrudeFolder.add(config, 'extrudeDome').name('Enable Dome').onChange(updateVisibility);
    guiControllers.domeRadius = extrudeFolder.add(config, 'domeRadius', 0.1, 30, 0.1).name('Dome Radius (R)').onChange(updateDomeGeometry).listen(); // Use temporary wide range initially
    guiControllers.profileType = extrudeFolder.add(config, 'profileType', ['spherical', 'eased', 'stepped', 'cascading']).name('Profile Type').onChange(updateDomeGeometry);
    guiControllers.tierCount = extrudeFolder.add(config, 'tierCount', 1, 20, 1).name('Tier Count (Stepped)').onChange(updateDomeGeometry).listen(); // For stepped
    guiControllers.stepHeight = extrudeFolder.add(config, 'stepHeight', 0.01, 10, 0.01).name('Step Height (Stepped)') // Use temporary wide range initially
                 .onChange(v => { config._stepHeightUserSet = true; updateDomeGeometry(); })
                 .listen(); // For stepped

    // --- NEW: Cascading Profile Controls ---
    const cascadeFolder = extrudeFolder.addFolder('Cascading Profile');
    guiControllers.cascadeSteps = cascadeFolder.add(config, 'cascadeSteps', 1, 50, 1).name('Cascade Steps').onChange(updateDomeGeometry).listen();
    guiControllers.cascadeDrop = cascadeFolder.add(config, 'cascadeDrop', 0.01, 2.0, 0.01).name('Cascade Drop (× r_max)').onChange(updateDomeGeometry).listen();
    guiControllers.tiltInnerDeg = cascadeFolder.add(config, 'tiltInnerDeg', -90, 90, 0.5).name('Tilt Inner (°)')
        .onChange(updateDomeGeometry).listen();
    guiControllers.tiltOuterDeg = cascadeFolder.add(config, 'tiltOuterDeg', -90, 90, 0.5).name('Tilt Outer (°)')
        .onChange(updateDomeGeometry).listen();
    // cascadeFolder.open(); // Optionally open by default

    const tiltFolder = extrudeFolder.addFolder('Tilt (General)');
    guiControllers.tiltDeg = tiltFolder
       .add(config, 'tiltDeg', -80, 80, 0.5)
       .name('Tilt Toward Centre (°)')
       .onChange(updateDomeGeometry)
       .listen();
}


// =============================================================================
// Three.js Scene Initialization & Rendering Loop
// =============================================================================

/** Initializes the Three.js scene, camera, controls, etc. */
function init() {
    console.log("Initializing scene...");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // --- Camera Setup (Perspective, looking at XY plane) ---\
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    const initialExtent = config.extent || 4;
    const viewDistance = initialExtent * 3; // Adjust initial view distance
    camera.position.set(0, 0, Math.max(15, viewDistance)); // Look from Z-axis

    // --- Renderer Setup ---\
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    // --- Orbit Controls (Adjusted for 2D viewing) ---\
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.screenSpacePanning = true; // Allows easier panning for 2D
    controls.enableRotate = true; // Keep rotation enabled
    controls.minDistance = 1;
    controls.maxDistance = 500; // Increase max distance
    controls.target.set(0, 0, 0); // Target the origin in the XY plane
    controls.update();

    // --- Lighting (Simple setup for 2D) ---
    /* OLD LIGHTING - replaced
    const ambientLight = new THREE.AmbientLight(0x707070); // Slightly brighter ambient
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight.position.set(1, 1, 2).normalize(); // Light from above/side
    scene.add(directionalLight);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
     directionalLight2.position.set(-1, -1, 1).normalize(); // Fill light
     scene.add(directionalLight2);
    */

    // New lighting setup
    const newAmbientLight = new THREE.AmbientLight(0x606060); // Soft ambient
    scene.add(newAmbientLight);

    const keyLight = new THREE.DirectionalLight(0xffccaa, 1.0); // Warm, from above
    keyLight.position.set(0.5, 0.5, 2); // More directly from above
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xaaccff, 0.4); // Cool fill, from side
    fillLight.position.set(-1, 0.5, 0.5);
    scene.add(fillLight);


    // --- Axes Helper (Optional) ---
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // --- Initial Calculations & Setup ---\
    calculateProjectionMatrices(); // Must be called before generation
    setupGUI();                    // Create the UI panel
    generateGeometry(); // NEW CALL

    // --- Event Listeners ---\
    window.addEventListener('resize', onWindowResize, false);

    console.log("Initialization complete.");
}

/** Handles window resize */
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

/** Animation loop */
function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Required for damping
    render();
}

/** Renders the scene */
function render() {
    renderer.render(scene, camera);
}

// =============================================================================
// Main Execution
// =============================================================================

init();
animate();

// --- NEW: Multigrid Generator (Skeleton) ---
/**
 * Fast De Bruijn (multigrid) generator.
 * Complexity  O(N² · tiles)  ≈ linear in visible area.
 */
function performMultigridGeneration() {
    console.log("Starting Multigrid generation...");
    const startTime = performance.now();

    const N   = config.nSym;
    const R   = config.extent * 1.2; // view radius in world units (Adjust multiplier as needed)
    console.log(` -> N = ${N}, View Radius R ≈ ${R.toFixed(2)}`);

    // Check for N<3 (though config should handle it)
    if (N < 3) {
        console.error("Multigrid requires N >= 3.");
        // Clear visuals? Or maybe just show nothing.
        acceptedPointsData = []; generatedEdges = []; generatedFaces = [];
        updatePointsObject(); updateEdgesObject(); updateFacesObject(); updateDomeGeometry(); updateVisibility();
        return;
    }

    const dir = []; // unit directions u_k
    for (let k = 0; k < N; k++) {
        const th = 2 * Math.PI * k / N;
        dir.push(new THREE.Vector2(Math.cos(th), Math.sin(th)));
    }

    // 1) pick N random offsets (phase between neighbouring parallel lines)
    // Use goldenRatio for quasi-randomness related to Penrose?
    // const offs = Array.from({length:N}, (_, k) => (k * config.goldenRatio) % 1.0); // OLD explicit golden ratio
    // const offs = Array.from({length:N}, () => Math.random()); // OLD Simpler: random [0, 1)
    const offs = Array.from({length:N}, (_,k)=> (k+1)*config.tau % 1);   // deterministic & incommensurate
    // Scale offsets? Playbook mentions config.tau, but using [0,1) is common.
    // Let's stick to random [0,1) for now.
    console.log(" -> Random offsets:", offs.map(o => o.toFixed(3)));

    // clear result containers
    acceptedPointsData = [];
    generatedEdges     = [];
    generatedFaces     = [];

    const vMap = new Map();   // key = `${x.toFixed(5)},${y.toFixed(5)}` → pointID
    const idToPointMap = new Map(); // Need this to retrieve point data later

    // --- Data Structures for Edge/Face Finding ---
    // lineMap: Map familyIndex (k) => Map ( lineIndex (m) => [{id, t}] ) - for sorting points along lines
    const lineMap = new Map(); 
    // vertexLookupByLines: Map key=`${i}:${m}_${j}:${n}` -> id - for finding face corners (built using all families point belongs to)
    const vertexLookupByLines = new Map(); 

    // 2) iterate over unordered pairs (i,j) to find intersection vertices
    console.log("Finding vertices (intersections)... ");
    let nextId = 0;
    for (let i = 0; i < N; i++) {
        if (dir[i].lengthSq() === 0) continue;   // safety; never hit but helps V8 inline
        for (let j = i+1; j < N; j++) {

            const d  = dir[i];
            const e  = dir[j];
            const D  = d.x*e.y - d.y*e.x; // det(d, e)
            if (Math.abs(D) < 1e-6) { // Check determinant
                console.warn(`Skipping parallel directions i=${i}, j=${j}`);
                continue; // Skip parallel vectors
            }

            // Solve A * [x y]^T = b for each integer pair (m,n)
            // Where b = [m + offs_i, n + offs_j]^T (shifted grid lines)
            // A = [[d.x, d.y], [e.x, e.y]] ? NO, should be: d.x*x+d.y*y = m+offs[i] etc.
            // Let's rewrite based on dual grid lines u_k · x = m_k + gamma_k
            // u_i · x = m + offs[i]
            // u_j · x = n + offs[j]
            // [ dir[i].x  dir[i].y ] [x] = [m + offs[i]]
            // [ dir[j].x  dir[j].y ] [y]   [n + offs[j]]
            // Let M = [[di.x, di.y], [dj.x, dj.y]]
            // det(M) = di.x*dj.y - di.y*dj.x (same as D calculated above)
            const detM = D;
            const invDet = 1.0 / detM;
            // Inverse M = (1/det) * [[dj.y, -di.y], [-dj.x, di.x]]
            const M_inv_11 =  e.y * invDet;
            const M_inv_12 = -d.y * invDet;
            const M_inv_21 = -e.x * invDet;
            const M_inv_22 =  d.x * invDet;

            // Estimate index range (over-approximate)
            // Max value of m+offs[i] occurs roughly when x points along dir[i] with length R
            // So, m_max ~ R + offs[i]. Similarly for n_max.
            // A tighter bound might be possible.
            const mMin = Math.floor(-R - offs[i]); // Use floor/ceil for safety
            const mMax = Math.ceil( R - offs[i]);
            const nMin = Math.floor(-R - offs[j]);
            const nMax = Math.ceil( R - offs[j]);

            for (let m = mMin; m <= mMax; m++) {
                for (let n = nMin; n <= nMax; n++) {

                    const b1 = m + offs[i]; // Target value for line i
                    const b2 = n + offs[j]; // Target value for line j

                    // Solve [x, y]^T = M_inv * [b1, b2]^T
                    const x = M_inv_11 * b1 + M_inv_12 * b2;
                    const y = M_inv_21 * b1 + M_inv_22 * b2;

                    // Check if vertex is within the view radius
                    if (x*x + y*y > R*R * 1.01) continue; // Add tolerance

                    const key = `${x.toFixed(5)},${y.toFixed(5)}`;
                    let pointData = vMap.get(key);
                    let currentId;

                    if (pointData === undefined) {
                        currentId = nextId++;
                        const physVec = new THREE.Vector2(x, y);
                        const newPoint = {
                            id: currentId,
                            lattice: null, 
                            phys: physVec,
                            lineIndices: {} // Store { familyIndex: lineIndex } for ALL families
                        };
                        // Calculate line indices for ALL families for this new point
                        for (let k = 0; k < N; k++) {
                            // Rounding might be needed due to float precision
                            const lineVal = dir[k].dot(physVec) - offs[k]; 
                            // Use Math.round to get the nearest integer index m_k
                            newPoint.lineIndices[k] = Math.round(lineVal);
                        }
                        
                        acceptedPointsData.push(newPoint);
                        vMap.set(key, newPoint);
                        idToPointMap.set(currentId, newPoint);
                        pointData = newPoint;

                        // Populate vertexLookupByLines for all pairs involving this point
                        const p_indices = pointData.lineIndices;
                        const families = Object.keys(p_indices).map(Number);
                        for (let fi = 0; fi < families.length; fi++) {
                            for (let fj = fi + 1; fj < families.length; fj++) {
                                const f1 = families[fi];
                                const f2 = families[fj];
                                const m1 = p_indices[f1];
                                const m2 = p_indices[f2];
                                // Ensure canonical order i < j for the key
                                const keyI = Math.min(f1,f2); const keyJ = Math.max(f1,f2);
                                const keyM = (keyI === f1) ? m1 : m2;
                                const keyN = (keyJ === f2) ? m2 : m1;
                                const lookupKey = `${keyI}:${keyM}_${keyJ}:${keyN}`;
                                vertexLookupByLines.set(lookupKey, currentId);
                            }
                        }
                    } else {
                        // Point already existed (intersection of other lines)
                        currentId = pointData.id;
                        // Ensure its lineIndices are fully populated (might have been created by a different pair)
                        if (Object.keys(pointData.lineIndices).length < N) {
                            for (let k = 0; k < N; k++) {
                                if (pointData.lineIndices[k] === undefined) {
                                     const lineVal = dir[k].dot(pointData.phys) - offs[k]; 
                                     pointData.lineIndices[k] = Math.round(lineVal);
                                }
                            }
                            // Re-populate vertexLookupByLines just in case (though less critical here)
                            const p_indices = pointData.lineIndices;
                            const families = Object.keys(p_indices).map(Number);
                            for (let fi = 0; fi < families.length; fi++) {
                                for (let fj = fi + 1; fj < families.length; fj++) {
                                     const f1 = families[fi];
                                     const f2 = families[fj];
                                     const m1 = p_indices[f1];
                                     const m2 = p_indices[f2];
                                     const keyI = Math.min(f1,f2); const keyJ = Math.max(f1,f2);
                                     const keyM = (keyI === f1) ? m1 : m2;
                                     const keyN = (keyJ === f2) ? m2 : m1;
                                     const lookupKey = `${keyI}:${keyM}_${keyJ}:${keyN}`;
                                     if (!vertexLookupByLines.has(lookupKey)) {
                                         vertexLookupByLines.set(lookupKey, currentId);
                                     }
                                }
                            }
                        }
                    }

                    // Populate lineMap for edge finding (using projection orthogonal to line dir for sorting)
                    for(let k = 0; k < N; k++) {
                         // const m_k = pointData.lineIndices[k]; // OLD
                         const m_k = Number(pointData.lineIndices[k]);   // ← force numeric key
                         const t_param_k = -dir[k].y * x + dir[k].x * y; // Project onto orthogonal
                         if (!lineMap.has(k)) lineMap.set(k, new Map());
                         if (!lineMap.get(k).has(m_k)) lineMap.get(k).set(m_k, []);
                         // Avoid adding duplicate points to the same line list
                         if (!lineMap.get(k).get(m_k).some(p => p.id === currentId)) {
                             lineMap.get(k).get(m_k).push({ id: currentId, t: t_param_k });
                         }
                    }
                }
            }
        }
    }
    console.log(` -> Found ${acceptedPointsData.length} unique vertices.`);

    // --- Process lineMap to generate edges ---
    console.log("Generating edges from lineMap...");
    const edgeSet = new Set(); // To avoid duplicates
    for (const [familyIndex, lines] of lineMap.entries()) {
        for (const [lineIndex, pointsOnLine] of lines.entries()) {
            // Sort points along the line based on parameter t
            pointsOnLine.sort((a, b) => a.t - b.t);
            // Create edges between consecutive points
            for (let k = 0; k < pointsOnLine.length - 1; k++) {
                const id1 = pointsOnLine[k].id;
                const id2 = pointsOnLine[k+1].id;
                const edgeKey = id1 < id2 ? `${id1}-${id2}` : `${id2}-${id1}`;
                if (!edgeSet.has(edgeKey)) {
                    generatedEdges.push({ v1: id1, v2: id2 });
                    edgeSet.add(edgeKey);
                }
            }
        }
    }
    console.log(` -> Generated ${generatedEdges.length} unique edges.`);

    // --- Build faces from line indices ---
    console.log("Generating faces...");
    const faceSet = new Set(); // Avoid duplicates
    const maxRhombTypeIndex = Math.floor(N / 2);

    // Iterate through all pairs of families (i, j)
    for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
            // Iterate through relevant line indices m (for family i)
            if (!lineMap.has(i)) continue;
            for (const m of lineMap.get(i).keys()) {
                 // Iterate through relevant line indices n (for family j)
                 if (!lineMap.has(j)) continue; // Should not happen if i has lines
                 for (const n of lineMap.get(j).keys()) {
                     // Check if vertex (m,n) for families (i,j) exists
                     const key0 = `${i}:${m}_${j}:${n}`;
                     const id0 = vertexLookupByLines.get(key0);
                     if (id0 === undefined) continue;

                     // Look for neighbours
                     const key1 = `${i}:${m + 1}_${j}:${n}`;
                     const key2 = `${i}:${m}_${j}:${n + 1}`;
                     const key3 = `${i}:${m + 1}_${j}:${n + 1}`;

                     const id1 = vertexLookupByLines.get(key1);
                     const id2 = vertexLookupByLines.get(key2);
                     const id3 = vertexLookupByLines.get(key3);

                     if (id1 !== undefined && id2 !== undefined && id3 !== undefined) {
                         // Found a potential rhombus: (id0, id1, id3, id2)
                         const vertexIds = [id0, id1, id3, id2];
                         const canonicalKey = vertexIds.slice().sort((a,b)=>a-b).join('-');

                         if (!faceSet.has(canonicalKey)) {
                             faceSet.add(canonicalKey);
                             const typeIndex = Math.min(j - i, N - (j - i)); // Canonical type index
                             if (typeIndex > 0 && typeIndex <= maxRhombTypeIndex) {
                                 const type = `rhomb_d${typeIndex}`;
                                 generatedFaces.push({ vertices: vertexIds, type: type });
                             } else {
                                 // Should not happen for i != j and N >= 3
                                 console.warn(`Degenerate face found? i=${i}, j=${j}, N=${N}, typeIndex=${typeIndex}`);
                             }
                         }
                     }
                 }
            }
        }
    }
    console.log(` -> Generated ${generatedFaces.length} potential faces.`);

    // 4) continue rendering pipeline
    console.log("Updating visuals...");
    // Calculate r_max based on generated points
    r_max = 0;
    acceptedPointsData.forEach(pt => {
        r_max = Math.max(r_max, pt.phys.length());
    });
    console.log(` -> Multigrid Max radial extent (r_max): ${r_max.toFixed(4)}`);

    // Update Three.js objects
    updatePointsObject();
    updateEdgesObject();
    updateFacesObject(); // Will be empty if face generation skipped
    updateDomeGeometry(); // Requires faces
    updateVisibility();

    const endTime = performance.now();
    console.log(`Multigrid generation finished in ${(endTime - startTime).toFixed(2)} ms.`);
    console.info(`Multigrid summary: verts=${acceptedPointsData.length}, edges=${generatedEdges.length}, faces=${generatedFaces.length}`);
}