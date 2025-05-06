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

const config = {
    // --- Mathematical Constants ---
    goldenRatio: (1 + Math.sqrt(5)) / 2,
    tau: (1 + Math.sqrt(5)) / 2, // Explicit tau for clarity

    // --- Basis Vectors (calculated in calculateProjectionMatrices) ---
    parVecs: [], // Physical space basis vectors (2D) - ROWS of P_PHYS
    ortVecs: [], // Internal space basis vectors (3D) - ROWS of P_INT
    P_PHYS: null, // 2x5 Physical projection matrix
    P_INT: null,  // 3x5 Internal projection matrix
    windowCenterInternalPerturbed: null,        // For genericity
    internalProjectionNormals: [], // Normals for window planes (projections of 5D basis)

    // --- Generation Parameters (UI controllable) ---
    extent: 4,                  // Range [-extent, extent] for 5D lattice search
    windowShiftInternal: new THREE.Vector3(0, 0, 0), // UI shift for the window center in 3D E_int
    windowScale: 1.0,                  // NEW: Scale factor for the acceptance window size

    // --- Extrusion Parameters (UI controllable) ---
    extrudeDome: false,         // Toggle extrusion view
    domeRadius: 6.0,           // Target dome radius (TEMPORARY default, calculated later)
    profileType: 'spherical',   // 'spherical', 'eased', 'stepped'
    tierCount: 5,               // For 'stepped' profile
    stepHeight: 1.0,           // For 'stepped' profile (TEMPORARY default, calculated later)
    _stepHeightUserSet: false, // Internal flag to track if user changed stepHeight
    tiltDeg: 0.0, // NEW: For cone-tilt of rhombi roofs
    // wallThickness: 0,        // Future: for solid cells

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
    dimension: 5, // Dimension of the source lattice Z^d
    physDimension: 2, // Dimension of the physical space R^d_phys
    intDimension: 3, // Dimension of the internal space R^d_int
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
let acceptedPointsData = []; // Stores { id, lattice(5D), phys(2D), internal(3D) } records
let generatedEdges = [];    // Stores { v1: id1, v2: id2 }
let generatedFaces = [];    // Stores { vertices: [id0, id1, id2, id3], type: 'thin' | 'thick' }
let windowPlanes = [];      // Stores { normal: Vec3 (in E_int), offset: number } for the window

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

function scale(v, scalar) {
    return v.map(x => x * scalar);
}

function subtract(v1, v2) {
    const len = Math.min(v1.length, v2.length);
    const result = new Array(len);
    for (let i = 0; i < len; i++) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}


function add(v1, v2) {
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
    return scale(v, 1 / magnitude);
}

/**
 * Calculates orthonormal 2D physical (E_phys) and 3D internal (E_int) space basis vectors
 * for projecting from Z^5 lattice for Penrose tiling.
 */
function calculateProjectionMatrices() {
    const N = config.dimension; // Should be 5
    if (N !== 5) console.error("Dimension mismatch: Expected 5 for Penrose setup.");

    console.log("Setting up canonical de Bruijn projection basis...");

    // --- Use Explicit Orthogonal de Bruijn Basis (No Gram-Schmidt) ---
    const P_PHYS_matrix = [[], []]; // 2x5
    const P_INT_matrix = [[], [], []]; // 3x5
    const sqrt5 = Math.sqrt(5);
    const scaleFactor = 1 / sqrt5; // Equivalent to sqrt(2/N) * 1/sqrt(2) ??? Check scaling
                                    // Let's use 1/sqrt(5) as given in user prompt

    for (let k = 0; k < N; k++) {
        const angle1 = 2 * Math.PI * k / N; // Phys angle
        const angle2 = 4 * Math.PI * k / N; // Internal angle (double)

        // Physical space components (COLUMNS of P_PHYS^T = ROWS of P_PHYS)
        P_PHYS_matrix[0][k] = Math.cos(angle1) * scaleFactor;
        P_PHYS_matrix[1][k] = Math.sin(angle1) * scaleFactor;

        // Internal space components (COLUMNS of P_INT^T = ROWS of P_INT)
        P_INT_matrix[0][k] = Math.cos(angle2) * scaleFactor;
        P_INT_matrix[1][k] = Math.sin(angle2) * scaleFactor;
        P_INT_matrix[2][k] = 1 * scaleFactor;
    }

    config.P_PHYS = P_PHYS_matrix;
    config.P_INT = P_INT_matrix;

    // Store basis vectors as ROWS for dot product projection functions
    // config.parVecs = []; // Rows of P_PHYS^T - effectively the columns used above
    // config.ortVecs = []; // Rows of P_INT^T
    // for(let i=0; i<N; ++i) {
    //     config.parVecs.push(Object.freeze([P_PHYS_matrix[0][i], P_PHYS_matrix[1][i]]));
    //     config.ortVecs.push(Object.freeze([P_INT_matrix[0][i], P_INT_matrix[1][i], P_INT_matrix[2][i]]));
    // }
    // The projection functions need the ROWS of P_PHYS / P_INT, not columns.
    // Let's redefine parVecs/ortVecs correctly.
    config.parVecs = P_PHYS_matrix.map(row => Object.freeze([...row])); // 2 vectors of length 5
    config.ortVecs = P_INT_matrix.map(row => Object.freeze([...row])); // 3 vectors of length 5

    console.log("Using fixed orthogonal de Bruijn projection basis.");
    console.log("Phys Basis Vectors (Rows of P_PHYS):", config.parVecs);
    console.log("Int Basis Vectors (Rows of P_INT):", config.ortVecs);


    // --- Calculate perturbed window center (remains the same logic) ---\
    config.windowCenterInternalPerturbed = new THREE.Vector3(
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude,
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude,
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude
    );

    // --- Define Window Planes: Use hardcoded 15 planes for Rhombic Triacontahedron ---
    console.log("Using hardcoded 15 planes for canonical Penrose acceptance window...");
    windowPlanes = [
        { normal: new THREE.Vector3( 0.44721359549996,  0.00000000000000,  0.44721359549996), offset: 0.50000000000000 },
        { normal: new THREE.Vector3(-0.36180339887499,  0.26286555605957,  0.44721359549996), offset: 0.50000000000000 },
        { normal: new THREE.Vector3( 0.13819660112501, -0.42532540417602,  0.44721359549996), offset: 0.50000000000000 },
        { normal: new THREE.Vector3( 0.13819660112501,  0.42532540417602,  0.44721359549996), offset: 0.50000000000000 },
        { normal: new THREE.Vector3(-0.36180339887499, -0.26286555605957,  0.44721359549996), offset: 0.50000000000000 },

        { normal: new THREE.Vector3(-0.11755705045849, -0.36180339887499,  0.11755705045849), offset: 0.24898982848828 },
        { normal: new THREE.Vector3( 0.19021130325903, -0.13819660112501, -0.19021130325903), offset: 0.21266270208801 },
        { normal: new THREE.Vector3(-0.19021130325903, -0.13819660112501,  0.19021130325903), offset: 0.21266270208801 },
        { normal: new THREE.Vector3( 0.11755705045849, -0.36180339887499, -0.11755705045849), offset: 0.24898982848828 },
        { normal: new THREE.Vector3( 0.30776835371753,  0.22360679774998,  0.11755705045849), offset: 0.24898982848828 },

        { normal: new THREE.Vector3(-0.07265425280054,  0.22360679774998, -0.19021130325903), offset: 0.21266270208801 },
        { normal: new THREE.Vector3( 0.23511410091699,  0.00000000000000,  0.19021130325903), offset: 0.21266270208801 },
        { normal: new THREE.Vector3(-0.38042260651806,  0.00000000000000,  0.11755705045849), offset: 0.24898982848828 },
        { normal: new THREE.Vector3(-0.07265425280054, -0.22360679774998, -0.19021130325903), offset: 0.21266270208801 },
        { normal: new THREE.Vector3( 0.30776835371753, -0.22360679774998,  0.11755705045849), offset: 0.24898982848828 },
    ];

    if (windowPlanes.length !== 15) {
        console.warn(`Expected 15 hardcoded planes, found ${windowPlanes.length}.`);
    } else {
        console.log(` -> Loaded ${windowPlanes.length} hardcoded planes for the acceptance window.`);
    }
}


/**
 * Projects a 5D vector onto the 2D physical subspace (E_phys).
 * @param {number[]} vec5D - The input 5D vector.
 * @returns {THREE.Vector2} The resulting 2D vector in physical space.
 */
function projectToPhysical(vec5D) {
    // Dot product of vec5D with each ROW of P_PHYS
    const x = dot(vec5D, config.parVecs[0]);
    const y = dot(vec5D, config.parVecs[1]);
    return new THREE.Vector2(x, y);
}

/**
 * Projects a 5D vector onto the 3D internal subspace (E_int).
 * @param {number[]} vec5D - The input 5D vector.
 * @returns {THREE.Vector3} The resulting 3D vector in internal space.
 */
function projectToInternal(vec5D) {
    // Dot product of vec5D with each ROW of P_INT
    const x = dot(vec5D, config.ortVecs[0]);
    const y = dot(vec5D, config.ortVecs[1]);
    const z = dot(vec5D, config.ortVecs[2]);
    return new THREE.Vector3(x, y, z);
}

/**
 * Checks if the projection of a 5D point into internal space (vecInternal)
 * falls within the acceptance window (projected 5D hypercube).
 * Condition: |N_k ⋅ (p_int - p_shift)| <= d_k * scale for k=1..5
 * where N_k is the UNNORMALIZED projection Π_int(e_k) and d_k is its calculated offset.
 * @param {THREE.Vector3} vecInternal - The 3D point in internal space.
 * @returns {boolean} True if the point is within the window, false otherwise.
 */
function isInWindow(vecInternal) {
    if (windowPlanes.length === 0) {
        console.warn("isInWindow called but windowPlanes is empty. Defaulting to false.");
        return false;
    }

    // Adjust point by window center perturbation and UI shift
    const effectiveVec = vecInternal.clone()
        .sub(config.windowCenterInternalPerturbed)
        .sub(config.windowShiftInternal);

    // Check against the 5 plane conditions |N_k . x| <= d_k * scale
    // Uses UNNORMALIZED normals (plane.normal) and their calculated offsets (plane.offset)
    for (const plane of windowPlanes) {
        const scaledOffset = plane.offset * config.windowScale;
        if (Math.abs(plane.normal.dot(effectiveVec)) > scaledOffset + config.epsilonComparison) {
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


// =============================================================================
// Quasicrystal Generation Logic (Adapted for 5D -> 2D)
// =============================================================================

/**
 * Performs the main generation process:
 * 1. Iterates through points in a 5D integer lattice (Z^5).
 * 2. Projects each 5D point into 3D internal and 2D physical spaces.
 * 3. Accepts points if their internal projection is within the acceptance window.
 * 4. Stores accepted points with their 5D lattice coordinates and 2D physical coords.
 * 5. Generates edges and faces based on 5D connectivity.
 * 6. Creates/updates the Three.js objects for points, edges, and faces in the XY plane.
 */
function performGeneration() {
    console.log("Starting new generation cycle (5D -> 2D Penrose)...");
    const startTime = performance.now();

    // --- Clear previous generated data ---\
    acceptedPointsData = [];
    generatedEdges = [];
    generatedFaces = [];

    const N = config.dimension; // 5
    const maxCoord = Math.max(1, Math.round(config.extent));
    const minCoord = -maxCoord;
    let acceptedCount = 0;
    let processedCount = 0;
    let nextPointId = 0;

    console.log(`Scanning 5D integer lattice Z^5 within extent: [${minCoord}, ${maxCoord}]`);

    // Performance Tweak: Precompute max internal radius squared for sphere pre-check
    // Use a slightly generous bound based on the window planes offset
    const maxInternalNormBound = windowPlanes.length > 0
        ? windowPlanes.reduce((max, p) => Math.max(max, p.offset / Math.sqrt(p.normal.lengthSq())), 0) * config.windowScale * 1.5 // 1.5 safety factor
        : (config.extent + 0.5) * 2; // Fallback if planes not ready
    const maxInternalRadiusSq = maxInternalNormBound * maxInternalNormBound;
    let preCheckSkipped = 0;

    // --- Iterate through the 5D integer lattice ---\
    // Recursive generator for iterating through N-dimensional hypercube
    function* latticePoints(currentDim, currentPoint) {
        if (currentDim > N) { // Base case: yield the complete 5D point
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
    for (const p5D of latticePoints(1, new Array(N))) {
         // Optional: Add progress logging for very large extents
        // if (processedCount % 100000 === 0) {
        //    console.log(`... scanned ${processedCount} points`);
        // }

        const pInternal = projectToInternal(p5D);

        // Performance Tweak: Sphere pre-check
        // Skip if point's internal projection is definitely outside a bounding sphere around the window
        if (pInternal.lengthSq() > maxInternalRadiusSq) {
            preCheckSkipped++;
            continue;
        }

        if (isInWindow(pInternal)) {
            const pPhysical = projectToPhysical(p5D); // Project to 2D

            acceptedPointsData.push({
                id: nextPointId++,
                lattice: [...p5D], // Store a copy of the 5D lattice coordinates
                phys: pPhysical, // Store the 2D physical coordinates
                internal: pInternal // Store 3D internal coordinates (optional)
            });
            acceptedCount++;
        }
    }


    const scanEndTime = performance.now();
    console.log(`Lattice scan complete in ${(scanEndTime - startTime).toFixed(2)} ms.`);
    console.log(` -> Processed ${processedCount} total 5D lattice points (skipped ${preCheckSkipped} by pre-check).`);
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

// =============================================================================
// Connectivity Generation (Adapted for Penrose from Z^5)
// =============================================================================

/**
 * Generates edges and faces (Penrose rhombi) based on the 5D lattice
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
    console.log("Generating connectivity (edges and faces) for Z^5 lattice...");

    // --- 1. Build LookupMap from 5D lattice coords to point data ---\
    const lookupMap = new Map();
    acceptedPointsData.forEach(pt => {
        lookupMap.set(pt.lattice.join(','), pt);
    });
    console.log(` -> Built LookupMap with ${lookupMap.size} entries.`);

    // --- 2. Generate Edges (Z^5 rule: neighbors differ by +/- e_i) ---\
    generatedEdges = [];
    let edgeCount = 0;
    const checkedEdges = new Set(); // Avoid duplicate checks (e.g., 1-2 vs 2-1)
    const N = config.dimension; // 5
    const standardBasis5D = [];
     for(let i=0; i<N; ++i) {
         standardBasis5D.push(Object.freeze(Array(N).fill(0).map((_, idx) => idx === i ? 1 : 0)));
     }


    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;

        // Iterate through each dimension i
        for (let i = 0; i < N; i++) {
            // Check neighbor v0 + e_i
            const neighborLatticePos = add(v0_lattice, standardBasis5D[i]);
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
    console.log(` -> Generated ${edgeCount} edges (Z^5 rule).`);

    // --- 3. Generate Faces (Penrose Rhombi from Z^5 parallelograms) ---\
    // A rhombus is formed by v0, v0+e_i, v0+e_j, v0+e_i+e_j if all are accepted points.
    generatedFaces = [];
    let faceCount = 0;
    const checkedFaces = new Set(); // Avoid duplicates

    // Pre-calculate 2D projections of 5D basis vectors for angle check
    const physProjections = standardBasis5D.map(e_i => projectToPhysical(e_i));

    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;
        const p0 = pt; // Point data for v0

        // Iterate through distinct pairs of dimensions (i, j)
        for (let i = 0; i < N; i++) {
            const step_i = standardBasis5D[i];
            const v1_lattice = add(v0_lattice, step_i);
            const key1 = v1_lattice.join(',');

            // Check if v0 + e_i is accepted
            if (!lookupMap.has(key1)) continue;
            const p1 = lookupMap.get(key1); // Point data for v1

            for (let j = i + 1; j < N; j++) {
                const step_j = standardBasis5D[j];
                const v2_lattice = add(v0_lattice, step_j); // v0 + e_j
                const key2 = v2_lattice.join(',');

                const v3_lattice = add(v1_lattice, step_j); // v0 + e_i + e_j
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
                            const diff = (j - i + N) % N; // N=5. Ensures diff is 1, 2, 3, or 4
                            if (diff === 1 || diff === 4) {
                                type = 'thick'; // Adjacent basis vectors -> thick rhombus
                            } else if (diff === 2 || diff === 3) {
                                type = 'thin'; // Non-adjacent basis vectors -> thin rhombus
                            } else {
                                console.warn("Unexpected index difference in face gen:", i, j, diff);
                                type = 'degenerate';
                            }
                        } else {
                            // console.warn(`Zero length edge vector for face ${ids.join(',')}`);
                            type = 'degenerate';
                        }

                        // Only add non-degenerate faces
                        if (type === 'thin' || type === 'thick') { // Exclude degenerate
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

/** Creates/updates the THREE.Mesh object for faces (rhombi) using geometry groups */
function updateFacesObject() {
    disposeObject(facesObject);
    facesObject = null;

    if (!config.showFaces || generatedFaces.length === 0 || acceptedPointsData.length === 0) {
        // console.log("Faces hidden or no data.");
        return;
    }

    // console.log("Updating faces object...");
    const idToPointMap = new Map();
    acceptedPointsData.forEach(pt => idToPointMap.set(pt.id, pt));

    const fullPositions = []; // Holds all unique vertex positions
    const allIndices = [];    // Holds all triangle indices
    const vertexMapFull = new Map(); // Map point ID to index in fullPositions array

    // Helper to add vertex data and return its index
    function addVertex(ptData) {
        if (!vertexMapFull.has(ptData.id)) {
            const index = vertexMapFull.size;
            vertexMapFull.set(ptData.id, index);
            fullPositions.push(ptData.phys.x, ptData.phys.y, 0); // RESET to z=0
            return index;
        }
        return vertexMapFull.get(ptData.id);
    }

    // Populate vertex and index arrays first
    generatedFaces.forEach(face => {
        const p0Data = idToPointMap.get(face.vertices[0]);
        const p1Data = idToPointMap.get(face.vertices[1]);
        const p2Data = idToPointMap.get(face.vertices[2]); // v0+ei+ej
        const p3Data = idToPointMap.get(face.vertices[3]); // v0+ej

        if (p0Data && p1Data && p2Data && p3Data) {
            const i0 = addVertex(p0Data);
            const i1 = addVertex(p1Data);
            const i2 = addVertex(p2Data);
            const i3 = addVertex(p3Data);

            // Add two triangles for the rhombus: (P0, P1, P3) and (P0, P3, P2) -> indices (i0, i1, i2) and (i0, i2, i3)
            allIndices.push(i0, i1, i2);
            allIndices.push(i0, i2, i3);
        } else {
            console.warn("Face references missing point ID:", face.vertices);
        }
    });

    if (vertexMapFull.size === 0 || allIndices.length === 0) {
        // console.log(" -> No valid face vertices or indices found.");
        return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(fullPositions, 3));
    geometry.setIndex(allIndices);

    // Add groups for different materials based on face type
    geometry.clearGroups();
    let currentFaceIndex = 0;
    let thinCount = 0, thickCount = 0, unknownCount = 0;
    generatedFaces.forEach(face => {
         // Ensure the face vertices were valid (check if corresponding indices exist - though should be guaranteed if added to allIndices)
         const i0 = vertexMapFull.get(face.vertices[0]);
         if (i0 === undefined) return; // Skip if face had missing vertices

        let materialIndex = -1;
        if (face.type === 'thin') {
            materialIndex = 0;
            thinCount++;
        } else if (face.type === 'thick') {
            materialIndex = 1;
            thickCount++;
        } else { // 'unknown' or 'degenerate'
            materialIndex = 2;
            unknownCount++;
        }

        if (materialIndex !== -1) {
            // Each face corresponds to 2 triangles (6 indices)
            geometry.addGroup(currentFaceIndex * 6, 6, materialIndex);
            currentFaceIndex++;
        }
    });


    geometry.computeVertexNormals(); // Normals for lighting (all pointing up/down in XY plane)

    const thinMaterial = new THREE.MeshStandardMaterial({
        color: config.faceColor1,
        opacity: config.faceOpacity,
        transparent: config.faceOpacity < 1.0,
        side: THREE.DoubleSide,
        metalness: 0.1, roughness: 0.7, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 // Offset to avoid z-fighting with edges
    });
    const thickMaterial = new THREE.MeshStandardMaterial({
         color: config.faceColor2,
         opacity: config.faceOpacity,
         transparent: config.faceOpacity < 1.0,
         side: THREE.DoubleSide,
         metalness: 0.1, roughness: 0.7, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1
     });
   const unknownMaterial = new THREE.MeshStandardMaterial({ color: 0x808080, side: THREE.DoubleSide, opacity: 0.5, transparent: true }); // Grey for unknown/degenerate


   facesObject = new THREE.Mesh(geometry, [thinMaterial, thickMaterial, unknownMaterial]);
   scene.add(facesObject);

   // console.log(` -> Faces object updated (${thinCount} thin, ${thickCount} thick, ${unknownCount} unknown/degenerate).`);
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
    if (u * r_max > R) return 0; // Clamp height if point is beyond dome radius footprint
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
        default:
            console.warn("Unknown profile type:", config.profileType);
            return 0; // Default to flat
    }
}

/**
 * Creates or updates the extruded dome mesh.
 */
function updateDomeGeometry() {
    // Ensure r_max and base geometry is ready
    if (acceptedPointsData.length === 0 || generatedFaces.length === 0) {
        // console.log("Cannot update dome: No base data.");
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
    domeMaterials = {}; // Clear material refs

    if (!config.extrudeDome || acceptedPointsData.length === 0 || generatedFaces.length === 0) {
        console.log("Dome view disabled or no data.");
        return;
    }

    // console.log("Updating dome geometry...");
    console.log("Updating dome geometry...");
    const startTime = performance.now();

    // --- 1. Prepare Vertex Data using vertexFinalZMap ---
    const idToPointData = new Map(); // Keep: needed for new logic
    acceptedPointsData.forEach(pt => idToPointData.set(pt.id, pt)); // Keep

    const idToBaseIndex = new Map();   // Map ID to index in the base vertex array (0..N-1)
    // const baseVertexCount = acceptedPointsData.length; // Will be redefined
    // const positions = new Float32Array(baseVertexCount * 2 * 3); // Will be redefined

    // acceptedPointsData.forEach((pt, index) => { // OLD LOOP TO BE DELETED
    //     idToBaseIndex.set(pt.id, index);
    //     const finalZ = vertexFinalZMap.get(pt.id) ?? 0; // Get pre-calculated Z
    //
    //     // Base vertex (z=0)
    //     positions[index * 3 + 0] = pt.phys.x;
    //     positions[index * 3 + 1] = pt.phys.y;
    //     positions[index * 3 + 2] = 0;
    //
    //     // Top vertex (z=finalZ)
    //     const topIndexOffset = baseVertexCount * 3;
    //     positions[topIndexOffset + index * 3 + 0] = pt.phys.x;
    //     positions[topIndexOffset + index * 3 + 1] = pt.phys.y;
    //     positions[topIndexOffset + index * 3 + 2] = finalZ;
    // });

    // --- NEW: allocate separate top vertices for every rhombus -------------
    const baseVertexCount = acceptedPointsData.length;
    acceptedPointsData.forEach((pt, index) => { // Create idToBaseIndex here
        idToBaseIndex.set(pt.id, index);
    });

    const topOffsetStart  = baseVertexCount;                // first free slot
    let runningTopIndex   = 0;                              // per-rhombus

    // map  faceIndex → [t0,t1,t2,t3]  (indices into *combined* position array)
    const faceTopIndexMap = new Map();

    // We will push positions later after we know how many top vertices we need
    const topPositions = [];   // flat array of XYZ
    generatedFaces.forEach((face, fIdx) => {
        const ptIds = face.vertices;
        // compute one height for the entire rhombus
        _centroidHelper.set(0,0,0); // Reset centroid helper
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
            console.warn(`updateDomeGeometry: Face ${fIdx} has only ${validPointsForCentroid} valid points for centroid. Skipping tilt/face.`);
            // Fill with flat, zero-height vertices to maintain array structure if necessary
            // or handle by not adding this face to faceTopIndexMap and skipping in roof/wall gen.
            // For now, let's assume it might lead to issues, so skip adding its top vertices.
            // Make sure faceTopIndexMap reflects this skip if other parts rely on its entries.
            // One simple way is to push degenerate vertices and let localTopIndices be created,
            // but the face won't look right.
            for (let i = 0; i < ptIds.length; i++) { // ptIds.length should be 4
                 topPositions.push(0,0,0); // Push degenerate flat vertices
            }
            // Ensure faceTopIndexMap gets *something* if it's critical downstream, even if degenerate.
            const localTopIndicesOnError = [];
            for (let i = 0; i < ptIds.length; i++) {
                localTopIndicesOnError.push(topOffsetStart + runningTopIndex++);
            }
            faceTopIndexMap.set(fIdx, localTopIndicesOnError);
            return; // Skip this face
        }
        _centroidHelper.divideScalar(validPointsForCentroid); // Calculate average for centroid XY (z is 0)

        // --- NEW: base height coming from the radial profile ---
        const u        = r_max > 0 ? _centroidHelper.length() / r_max : 0;
        const zProfile = heightProfile(u);

        const tiltAngleRad = THREE.MathUtils.degToRad(config.tiltDeg);
        const tiltQuat = getTiltQuaternion(_centroidHelper, tiltAngleRad, _tmpQuat); // Pass _tmpQuat as output

        const localTopIndices = [];
        ptIds.forEach(id => {
            const pData = idToPointData.get(id);
            _vertexPosHelper.set(
                pData ? pData.phys.x : 0,
                pData ? pData.phys.y : 0,
                0
            );

            // 2.a  move to local frame, tilt, move back
            _vertexPosHelper.sub(_centroidHelper)
                            .applyQuaternion(tiltQuat)
                            .add(_centroidHelper);

            // 2.b  lift everything by the height profile
            _vertexPosHelper.z += zProfile;

            // 2.c  (optional) make sure roof never dips below the base plane
            // _vertexPosHelper.z = Math.max(_vertexPosHelper.z, 0);

            topPositions.push(
                _vertexPosHelper.x,
                _vertexPosHelper.y,
                _vertexPosHelper.z
            );
            localTopIndices.push(topOffsetStart + runningTopIndex++);
        });
        faceTopIndexMap.set(fIdx, localTopIndices);
    });

    const totalVerts = baseVertexCount + runningTopIndex;
    const positions  = new Float32Array(totalVerts * 3);

    // copy base layer (unchanged)
    acceptedPointsData.forEach((pt,i)=>{
        positions[3*i]   = pt.phys.x;
        positions[3*i+1] = pt.phys.y;
        positions[3*i+2] = 0;
    });
    // copy top layer
    for (let i=0;i<topPositions.length;i++) positions[baseVertexCount*3 + i] = topPositions[i];

    // --- 2. Prepare Geometry Data ---
    const indices = [];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.clearGroups(); // Clear existing groups

    // --- 3. Generate Wall Faces (per edge) ---
    const wallEdgeSeen = new Set();        // avoid double walls
    generatedEdges.forEach(e=>{
        const key = e.v1 < e.v2 ? `${e.v1}_${e.v2}` : `${e.v2}_${e.v1}`;
        if (wallEdgeSeen.has(key)) return;
        wallEdgeSeen.add(key);

        const b0 = idToBaseIndex.get(e.v1);
        const b1 = idToBaseIndex.get(e.v2);

        if (b0 === undefined || b1 === undefined) { // Check for undefined base indices
            console.warn(`updateDomeGeometry wall gen: Skipping edge ${e.v1}-${e.v2} due to missing base index.`);
            return;
        }

        // find which faces share this edge
        const facesSharing = generatedFaces
            .map((f,idx)=> ({f,idx}))
            .filter(obj=>{ const v=obj.f.vertices; return v.includes(e.v1)&&v.includes(e.v2); });

        facesSharing.forEach(obj=>{
            const idx  = obj.idx;
            const vArr = obj.f.vertices;
            const localTop = faceTopIndexMap.get(idx);

            if (!localTop) { // Check if localTop exists
                console.warn(`updateDomeGeometry wall gen: Missing faceTopIndexMap entry for face index ${idx}`);
                return;
            }

            // localTop order matches vArr order → find positions of the two vertices
            const lf0 = vArr.indexOf(e.v1);
            const lf1 = vArr.indexOf(e.v2);

            if (lf0 === -1 || lf1 === -1 || lf0 >= localTop.length || lf1 >= localTop.length) { // Bounds check
                 console.warn(`updateDomeGeometry wall gen: Vertex index out of bounds for edge ${e.v1}-${e.v2} in face ${idx}. lf0: ${lf0}, lf1: ${lf1}, localTop.length: ${localTop.length}`);
                 return;
            }
            const t0  = localTop[lf0];
            const t1  = localTop[lf1];

            indices.push(b0,b1,t1,  b0,t1,t0);
        });
    });

    const wallIndexCount = indices.length;
    if (wallIndexCount > 0) { // Only add group if there are walls
        geometry.addGroup(0, wallIndexCount, 0); // Material group 0: Walls
    }

    // --- 4. Generate Roof Faces (per rhombus) ---
    let roofStartIndex = indices.length; // Corrected: was wallIndexCount, should be current indices.length
    generatedFaces.forEach((face,fIdx)=>{
       const t = faceTopIndexMap.get(fIdx);   // [t0,t1,t2,t3] already in correct order
       if (!t || t.length !== 4) { // Check if 't' is valid
           console.warn(`updateDomeGeometry roof gen: Invalid or incomplete faceTopIndexMap for face index ${fIdx}. Skipping.`);
           return;
       }
       indices.push(t[0],t[1],t[2],  t[0],t[2],t[3]);

       const mat = (face.type==='thin')?1:(face.type==='thick')?2:3;
       geometry.addGroup(roofStartIndex, 6, mat);
       roofStartIndex += 6;
    });

    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    // --- 5. Create Mesh with Multiple Materials ---
    domeMaterials.wall = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa, // Neutral grey for walls
        side: THREE.DoubleSide,
        metalness: 0.1, roughness: 0.8,
        opacity: config.faceOpacity, transparent: config.faceOpacity < 1.0,
        polygonOffset: true, polygonOffsetFactor: 1.1, polygonOffsetUnits: 1 // Offset slightly more than faces
    });
     domeMaterials.thinRoof = new THREE.MeshStandardMaterial({
         color: config.faceColor1,
         opacity: config.faceOpacity, transparent: config.faceOpacity < 1.0,
         side: THREE.DoubleSide,
         metalness: 0.1, roughness: 0.7
     });
     domeMaterials.thickRoof = new THREE.MeshStandardMaterial({
          color: config.faceColor2,
          opacity: config.faceOpacity, transparent: config.faceOpacity < 1.0,
          side: THREE.DoubleSide,
          metalness: 0.1, roughness: 0.7
      });
     domeMaterials.unknownRoof = new THREE.MeshStandardMaterial({ color: 0x808080, side: THREE.DoubleSide, opacity: 0.5, transparent: true }); // Degenerate roof


    domeMeshObject = new THREE.Mesh(geometry, [
        domeMaterials.wall,
        domeMaterials.thinRoof,
        domeMaterials.thickRoof,
        domeMaterials.unknownRoof
    ]);
    // Visibility handled by updateVisibility
    // domeMeshObject.visible = config.extrudeDome;
    scene.add(domeMeshObject);

    // --- 6. Create Dome Edges ---
    const edgesGeom = new THREE.EdgesGeometry(geometry, 30); // Threshold angle to show sharp edges
    const edgeMaterial = new THREE.LineBasicMaterial({ color: config.edgeColor });
    domeEdgesObject = new THREE.LineSegments(edgesGeom, edgeMaterial);
    // Visibility handled by updateVisibility
    // domeEdgesObject.visible = config.extrudeDome && config.showEdges;
    scene.add(domeEdgesObject);

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
        if (domeMaterials.thinRoof) domeMaterials.thinRoof.visible = config.showFaces;
        if (domeMaterials.thickRoof) domeMaterials.thickRoof.visible = config.showFaces;
        if (domeMaterials.unknownRoof) domeMaterials.unknownRoof.visible = config.showFaces; // Also hide degenerate roofs
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

    // --- Generation Parameters Folder ---\
    const genFolder = gui.addFolder('Generation Parameters');
    genFolder.add(config, 'extent', 1, 7, 1).name('5D Search Extent').onChange(performGeneration); // Max 7 is already very slow
    genFolder.add(config, 'windowScale', 0.1, 3.0, 0.05).name('Window Scale').onChange(performGeneration); // NEW control

    // --- Internal Window Shift Sub-Folder (3D) ---\
    const shiftFolder = genFolder.addFolder('Window Shift (Internal)');
    const shiftRange = 1.0; // Adjust range based on window size (~0.5 offset)
    shiftFolder.add(config.windowShiftInternal, 'x', -shiftRange, shiftRange, 0.01).name('Shift X').onChange(performGeneration).listen();
    shiftFolder.add(config.windowShiftInternal, 'y', -shiftRange, shiftRange, 0.01).name('Shift Y').onChange(performGeneration).listen();
    shiftFolder.add(config.windowShiftInternal, 'z', -shiftRange, shiftRange, 0.01).name('Shift Z').onChange(performGeneration).listen();
    genFolder.open();

    // --- Visualization Parameters Folder ---\
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
    facesFolder.addColor(config, 'faceColor1').name('Thin Color').onChange(() => {
        if (facesObject && Array.isArray(facesObject.material) && facesObject.material[0]) facesObject.material[0].color.set(config.faceColor1);
        if (domeMaterials.thinRoof) domeMaterials.thinRoof.color.set(config.faceColor1);
    });
     facesFolder.addColor(config, 'faceColor2').name('Thick Color').onChange(() => {
         if (facesObject && Array.isArray(facesObject.material) && facesObject.material[1]) facesObject.material[1].color.set(config.faceColor2);
         if (domeMaterials.thickRoof) domeMaterials.thickRoof.color.set(config.faceColor2);
     });
    facesFolder.add(config, 'faceOpacity', 0, 1, 0.01).name('Opacity').onChange(() => {
         if (facesObject && Array.isArray(facesObject.material)) {
             facesObject.material.forEach(m => {
                 m.opacity = config.faceOpacity;
                 m.transparent = config.faceOpacity < 1.0; // Update transparency flag
             });
         }
         // Also update dome ROOF opacity
         const isTransparent = config.faceOpacity < 1.0;
         if (domeMaterials.thinRoof) {
             domeMaterials.thinRoof.opacity = config.faceOpacity;
             domeMaterials.thinRoof.transparent = isTransparent;
         }
         if (domeMaterials.thickRoof) {
             domeMaterials.thickRoof.opacity = config.faceOpacity;
             domeMaterials.thickRoof.transparent = isTransparent;
         }
         if (domeMaterials.unknownRoof) { // Keep degenerate semi-transparent maybe?
              domeMaterials.unknownRoof.opacity = config.faceOpacity * 0.7; // Slightly more transparent
              domeMaterials.unknownRoof.transparent = isTransparent || domeMaterials.unknownRoof.opacity < 1.0;
         }
         // Update WALL opacity too?
         if (domeMaterials.wall) {
              domeMaterials.wall.opacity = config.faceOpacity; // Link wall opacity
              domeMaterials.wall.transparent = isTransparent;
         }
     });
    facesFolder.open(); // Default open

    // --- Extrusion Controls ---
    const extrudeFolder = gui.addFolder('Dome Extrusion');
    guiControllers.domeToggle = extrudeFolder.add(config, 'extrudeDome').name('Enable Dome').onChange(updateVisibility);
    guiControllers.domeRadius = extrudeFolder.add(config, 'domeRadius', 0.1, 30, 0.1).name('Dome Radius (R)').onChange(updateDomeGeometry).listen(); // Use temporary wide range initially
    guiControllers.profileType = extrudeFolder.add(config, 'profileType', ['spherical', 'eased', 'stepped']).name('Profile Type').onChange(updateDomeGeometry);
    guiControllers.tierCount = extrudeFolder.add(config, 'tierCount', 1, 20, 1).name('Tier Count').onChange(updateDomeGeometry).listen(); // For stepped
    guiControllers.stepHeight = extrudeFolder.add(config, 'stepHeight', 0.01, 10, 0.01).name('Step Height') // Use temporary wide range initially
                 .onChange(v => { config._stepHeightUserSet = true; updateDomeGeometry(); })
                 .listen(); // For stepped

    const tiltFolder = extrudeFolder.addFolder('Tilt');
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

    // --- Lighting (Simple setup for 2D) ---\
    const ambientLight = new THREE.AmbientLight(0x707070); // Slightly brighter ambient
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight.position.set(1, 1, 2).normalize(); // Light from above/side
    scene.add(directionalLight);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
     directionalLight2.position.set(-1, -1, 1).normalize(); // Fill light
     scene.add(directionalLight2);


    // --- Axes Helper (Optional) ---\
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // --- Initial Calculations & Setup ---\
    calculateProjectionMatrices(); // Must be called before generation
    setupGUI();                    // Create the UI panel
    performGeneration();           // Generate initial tiling

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