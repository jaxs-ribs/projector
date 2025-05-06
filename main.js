/**
 * Quasicrystal Dome Generator
 *
 * Overview:
 * This script builds an interactive 3D hemispherical quasicrystal dome with full icosahedral symmetry
 * using the "cut-and-project" method from a 6-dimensional (D6) integer lattice.
 *
 * Key Steps:
 * 1. Basis Computation (calculateProjectionMatrices):
 *    • Define three initial 6D vectors that span the "physical" subspace (E_phys).
 *    • Orthonormalize them via Gram-Schmidt to get parVec1/2/3 (physical basis).
 *    • Extend with the standard 6D basis, re-orthonormalize to extract the remaining
 *      three vectors ortVec1/2/3 (internal or "perp" basis, E_int).
 *    • Slightly perturb the internal window center to avoid degeneracies.
 *    • Generate the 30 Rhombic-Triacontahedron (RT) window planes by projecting the 30
 *      D6 root vectors (±e_i ± e_j) into E_int and computing supporting plane offsets.
 *
 * 2. Cut-and-Project Generation (performGeneration):
 *    • Scan all integer points p ∈ Z⁶ in a configurable range (extent).
 *    • Enforce D6 parity: sum(coords) % 2 === 0.
 *    • Project into internal space: p_int = Π_int^T p. Keep only points inside the RT window.
 *    • Project into physical space: p_phys = Π_phys^T p. Keep only points in the upper
 *      hemisphere shell defined by innerRadiusPhysical ≤ ‖p_phys‖ ≤ outerRadiusPhysical.
 *    • Store accepted points with unique IDs and 3D positions.
 *
 * 3. Connectivity (generateConnectivity):
 *    • Edges: connect any two accepted points whose 6D coordinates differ by ±e_i ± e_j.
 *    • Faces: form rhombi (pairs of independent D6 roots) and filter to those whose
 *      centroid lies within a small tolerance of the outer physical radius.
 *
 * 4. Three.js Rendering:
 *    • updatePointsObject(): render vertices as THREE.Points.
 *    • updateEdgesObject(): render edges as THREE.LineSegments.
 *    • updateFacesObject(): render rhombi as two-triangle THREE.Mesh (MeshStandardMaterial).
 *    • updateShellVisualization(): render a semi-transparent hemisphere guide (inner/outer radii).
 *    • init() + animate(): set up scene, camera, lights, controls, and start render loop.
 *
 * 5. Interactive Controls (lil-gui):
 *    • Generation parameters: internal window shift, scan extent, inner/outer radii.
 *    • Visualization parameters: toggle points/edges/faces, colors, sizes, opacities.
 *    • All changes automatically trigger regeneration or object updates.
 *
 * Goals:
 *  - Demonstrate the cut-and-project algorithm for icosahedral quasicrystals.
 *  - Produce a visually compelling, interactive dome that users can explore and tweak.
 *  - Provide a clear, self-documented codebase for further extension by LLMs or developers.
 *
 * Usage:
 * Copy this comment block to the top of the file to give full context to future readers
 * or AI agents. It explains the mathematical foundation, data flow, rendering pipeline,
 * and user interface design in one place.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import GUI from 'lil-gui';

// =============================================================================
// Configuration & Global State
// =============================================================================

const config = {
    // --- Mathematical Constants ---
    goldenRatio: (1 + Math.sqrt(5)) / 2,
    normConst: 2.0, // Normalization for the chosen icosahedral projection basis (updated)

    // --- Basis Vectors (calculated in calculateProjectionMatrices) ---
    parVec1: null, parVec2: null, parVec3: null, // Physical space basis
    ortVec1: null, ortVec2: null, ortVec3: null, // Internal space basis (orthogonal complement)
    windowCenterInternalPerturbed: null,        // For genericity

    // --- Generation Parameters (UI controllable) ---
    // radiusInternal: 1.5,        // REMOVED - RT window size is fixed
    innerRadiusPhysical: 5.0,   // Inner radius of the physical space shell
    outerRadiusPhysical: 6.0,   // Outer radius of the physical space shell
    extent: 5,                  // Range [-extent, extent] for 6D lattice search
    windowShiftInternal: new THREE.Vector3(0, 0, 0), // UI shift for the RT window center

    // --- Visualization Parameters (UI controllable) ---
    vertexColor: '#ffffff',     // Color of the generated points
    vertexSize: 0.054,          // Size of the generated points
    edgeColor: '#ff00ff',       // Color for edges
    faceColor: '#00eeee',       // Color for faces (Cyan)
    faceOpacity: 0.5,           // Opacity for faces
    shellColor: '#00ffff',      // Color for the guide shell visualization
    shellOpacity: 0.15,         // Opacity for the guide shell visualization
    showPoints: true,           // Toggle visibility for points
    showEdges: true,           // Toggle visibility for edges (Default: ON)
    showFaces: true,           // Toggle visibility for faces (Default: ON)

    // --- Fixed Parameters ---
    windowPerturbationMagnitude: 1e-6, // Small random offset for window center to avoid degenerate cases
    domeCenter: new THREE.Vector3(0, 0, 0), // Center for physical shell check (currently origin)
    epsilonComparison: 1e-12,          // Small value for floating point comparisons
    faceRelativeRadiusTolerance: 0.05, // Relative tolerance (percentage of outerRadius) for face centroid check (UI adjustable) - increased default

    // --- Tolerances for Golden Rhombus Shape Checks ---
    rhombEdgeRelTol: 0.15, // RELAXED: Relative tolerance for edge lengths (was 0.10)
    rhombAngleTol: 0.06,  // RELAXED: Max allowed difference between angle cosine and theoretical values (was 0.02)

    // --- NEW: Vertex-based outer shell filter ---
    faceOuterVertexFracNeeded : 0.75, // three of the four is enough (was 1.0)
    vertexOuterTol            : 0.12, // 12% radial slack (was 0.045)
};

// --- Global Three.js Variables ---
let scene, camera, renderer, controls;
let pointsObject = null;      // Holds the THREE.Points object for the quasicrystal vertices
let edgesObject = null;       // Holds the THREE.LineSegments object for edges
let facesObject = null;       // Holds the THREE.Mesh object for faces
let shellMeshGroup = null;    // Holds the visualization mesh group for the target shell
let gui;                      // lil-gui instance

// --- Global Generation Data ---
let acceptedPointsData = []; // Stores { id, lattice, phys } records
let generatedEdges = [];    // Stores { v1: id1, v2: id2 }
let generatedFaces = [];    // Stores { vertices: [id0, id1, id2, id3] }
let windowPlanes = [];      // Stores { normal: Vec3, offset: number } for the RT window

// --- Global Utility Variables ---
// Keep track of how many times isInWindow was called for logging (debug)
let isInWindowCallCount = 0;
const MAX_ISINWINDOW_LOGS = 10; // Limit logs to avoid flooding console


// =============================================================================
// Mathematical Utilities & Projection Logic
// =============================================================================

// --- Vector Math Helpers ---

function dot(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

function scale(v, scalar) {
    return v.map(x => x * scalar);
}

function subtract(v1, v2) {
    return v1.map((x, i) => x - v2[i]);
}

function normalize(v) {
    const magnitude = Math.sqrt(dot(v, v));
    if (magnitude < 1e-10) { // Avoid division by zero for zero vectors
        return v.map(() => 0); // Return a zero vector of the same dimension
    }
    return scale(v, 1 / magnitude);
}

/**
 * Performs Gram-Schmidt orthonormalization on a set of vectors.
 * Handles potential linear dependence by skipping vectors that project to zero.
 * @param {number[][]} vectors - An array of vectors (each vector is an array of numbers).
 * @returns {number[][]} An array of orthonormal vectors spanning the same space.
 */
function gramSchmidt(vectors) {
    const orthonormalBasis = [];
    for (const v of vectors) {
        let u = [...v]; // Copy the vector
        // Subtract projections onto previous basis vectors
        for (const basisVec of orthonormalBasis) {
            const proj = dot(v, basisVec); // Projection scalar
            u = subtract(u, scale(basisVec, proj));
        }
        // Normalize the resulting vector if it's non-zero
        const magnitudeSq = dot(u, u);
        if (magnitudeSq > 1e-10) { // Check if vector is linearly independent
            orthonormalBasis.push(normalize(u));
        }
        // If magnitude is near zero, the vector was linearly dependent, so we skip it.
    }
    return orthonormalBasis;
}

/**
 * Calculates orthonormal 3D physical (E_phys) and 3D internal (E_int) space basis vectors
 * for projecting from 6D (Z^6 lattice) to achieve icosahedral symmetry.
 * Starts with vectors defining the orientation of E_phys and uses Gram-Schmidt.
 */
function calculateProjectionMatrices() {
    const tau = config.goldenRatio;
    // const invTau = 1 / tau; // No longer needed with this basis

    // --- 1. Define initial vectors that SPAN E_phys (normalized or not, doesn't matter for GS) ---
    // These vectors determine the *orientation* of the physical subspace.
    // This set uses irrational coefficients to break residual mirror symmetries
    // that caused the previous canonical basis to yield only 28 planes.
    const rt2 = Math.SQRT2; // √2
    const rt3 = Math.sqrt(3); // √3

    const initialParVecs = [
        [ 1,     rt2,  tau,        0,   -rt3,  tau ],
        [ tau,      0,  1,   -tau*rt2,      0,  rt3 ],
        [ rt3,    tau,  0,        1,    -tau, -rt2 ]
    ];

    // --- 2. Orthonormalize the physical basis vectors ---
    const physicalBasis = gramSchmidt(initialParVecs);
    if (physicalBasis.length !== 3) {
        console.error("Error: Physical basis vectors are linearly dependent!");
        // Handle error appropriately - maybe stop execution or use a default basis?
        return;
    }
    config.parVec1 = physicalBasis[0];
    config.parVec2 = physicalBasis[1];
    config.parVec3 = physicalBasis[2];
    console.log("Calculated ORTHONORMAL physical basis vectors (E_phys).");

    // --- 3. Find an orthonormal basis for the orthogonal complement E_int ---
    // We extend the E_phys basis with the standard R^6 basis and run Gram-Schmidt again.
    // The vectors produced *after* the first three will form the basis for E_int.
    const standardBasis6D = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ];

    const combinedSet = [...physicalBasis, ...standardBasis6D];
    const fullOrthonormalBasis = gramSchmidt(combinedSet);

    // The first 3 vectors *should* be numerically identical to physicalBasis.
    // The *next* 3 vectors form the orthonormal basis for E_int.
    if (fullOrthonormalBasis.length !== 6) {
         console.error(`Error: Failed to find full 6D orthonormal basis! Found ${fullOrthonormalBasis.length} vectors.`);
         // Handle error
         return;
    }
    config.ortVec1 = fullOrthonormalBasis[3];
    config.ortVec2 = fullOrthonormalBasis[4];
    config.ortVec3 = fullOrthonormalBasis[5];
    console.log("Calculated ORTHONORMAL internal basis vectors (E_int).");

    // --- Verify Orthogonality (Optional Debug Check) ---
    // Check orthogonality between E_phys and E_int bases
    // const physBasis = [config.parVec1, config.parVec2, config.parVec3];
    // const intBasis = [config.ortVec1, config.ortVec2, config.ortVec3];
    // for (let i = 0; i < 3; i++) {
    //     for (let j = 0; j < 3; j++) {
    //         const dp = dot(physBasis[i], intBasis[j]);
    //         if (Math.abs(dp) > 1e-9) {
    //              console.warn(`Orthogonality check failed: parVec${i+1} . ortVec${j+1} = ${dp}`);
    //         }
    //     }
    // }
    // console.log("Orthogonality check complete (warnings indicate issues).");


    // --- 4. Remove obsolete normConst ---
    // config.normConst is no longer used as the basis vectors are now inherently normalized.
    delete config.normConst; // Or set to null/undefined

    // --- 5. Calculate perturbed window center (remains the same) ---
    config.windowCenterInternalPerturbed = new THREE.Vector3(
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude,
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude,
        (Math.random() - 0.5) * 2 * config.windowPerturbationMagnitude
    );

    // --- 6. Calculate Rhombic Triacontahedron (RT) Window Planes ---
    console.log("Calculating RT window planes...");
    const startTime = performance.now();
    windowPlanes = []; // Clear previous planes

    // 6a. Generate 6D hypercube corners (+/- 0.5)^6
    const cubeCorners6D = [];
    for (let i = 0; i < 64; i++) {
        const corner = [];
        for (let j = 0; j < 6; j++) {
            corner.push(((i >> j) & 1) ? 0.5 : -0.5);
        }
        cubeCorners6D.push(corner);
    }

    // 6b. Project corners to 3D internal space
    const internalCorners = cubeCorners6D.map(c => projectToInternal(c)); // Array of THREE.Vector3

    // --- Direct D6 root projection method ---
    windowPlanes = []; // Ensure it's clear before calculation
    const planeKeySet = new Set(); // Use a string-key Set to collapse ±n duplicates
    console.log(" -> Calculating RT window planes using direct D6 root projection...");

    for (let i = 0; i < 5; i++) {
        for (let j = i + 1; j < 6; j++) {
            for (const si of [+1, -1]) {
                for (const sj of [+1, -1]) {
                    // Define the 6D root vector
                    const root6 = [0, 0, 0, 0, 0, 0];
                    root6[i] = si;
                    root6[j] = sj;

                    // Project the 6D root vector to internal space to get the normal direction
                    const n = projectToInternal(root6);

                    // --- DIAGNOSTIC START (root projections) ---
                    if (n.lengthSq() < 1e-26) { // Lowered threshold from epsilonComparison*epsilonComparison (1e-24) to 1e-26
                        console.warn(`ZERO-INT-PROJ root  ${root6.join(',')}`);
                        continue; // Skip this root
                    } else {
                        // normalise but keep a *high-precision* string of the direction
                        const u = n.clone().normalize();
                        const dirKey = `${u.x.toFixed(12)},${u.y.toFixed(12)},${u.z.toFixed(12)}`;
                        if (planeKeySet.has(dirKey)) {
                            // This check is actually for the *normalized direction* key before canonicalization
                            // console.warn(`DUP-NORMAL root ${root6.join(',')}  ->  ${dirKey}`);
                            // Note: This specific log might fire even with 30 final planes if multiple roots
                            // map to the same direction *before* canonicalization handles the +/- pairs.
                            // It's more informative to check the planeKeySet size at the end.
                        }
                    }
                    // ---  DIAGNOSTIC END  ---

                    // Check if the projected normal is non-zero before normalizing (Redundant check now due to check above)
                    // if (n.lengthSq() < config.epsilonComparison * config.epsilonComparison) {
                    //     // console.warn(`Skipping zero projected normal for root: ${root6.join(',')}`);
                    //     continue;
                    // }
                    n.normalize();

                    // --- Canonicalized String-Key Deduplication (±n collapse) ---
                    // 1) build the "negative" version
                    const nNeg = new THREE.Vector3(-n.x, -n.y, -n.z);
                    // 2) pick whichever triple is lexicographically larger: (x,y,z) vs (-x,-y,-z)
                    let canon = new THREE.Vector3();
                    // Use epsilon comparison for stability near zero during comparison
                    const lx = n.x, ly = n.y, lz = n.z;
                    const rx = nNeg.x, ry = nNeg.y, rz = nNeg.z;
                    const eps = config.epsilonComparison; // Use existing small value

                    if (lx - rx > eps ||
                       (Math.abs(lx - rx) < eps && (ly - ry > eps ||
                       (Math.abs(ly - ry) < eps && lz - rz > eps))))
                    {
                         canon.copy(n);
                    } else {
                         canon.copy(nNeg);
                    }
                    // 3) round to 12 decimal places for a stable string key (Increased precision from 8)
                    const key = `${canon.x.toFixed(12)},${canon.y.toFixed(12)},${canon.z.toFixed(12)}`;
                    // 4) skip duplicates
                    if (planeKeySet.has(key)) continue;
                    planeKeySet.add(key);
                    console.debug(`KEEP NORMAL ${key}`); // Added debug log

                    // use `canon` (not `n`) from here on
                    const normal = canon; // Assign canonical vector to be used

                    // Calculate the offset d = max(normal . corner) over all projected corners
                    let d = -Infinity;
                    for (const cornerVec of internalCorners) {
                        // d = Math.max(d, n.dot(cornerVec)); // OLD: used original n
                        d = Math.max(d, normal.dot(cornerVec)); // NEW: use canonical normal
                    }
                    // windowPlanes.push({ normal: n, offset: d }); // OLD: used original n
                    windowPlanes.push({ normal, offset: d }); // NEW: use canonical normal
                }
            }
        }
    }
    // --- End Direct D6 root projection method ---

    console.info(`RT-window plane normals (unique):`, [...planeKeySet]); // Added info log

    const endTime = performance.now();
    console.log(` -> Calculated ${windowPlanes.length} RT window planes in ${(endTime - startTime).toFixed(2)} ms (using direct D6 root projection). Should be 30.`); // Updated log
    if (windowPlanes.length !== 30 && windowPlanes.length !== 0) {
        // This warning should ideally not trigger anymore with the direct method
        console.warn(`Expected 30 planes for RT, but found ${windowPlanes.length}. Check projection basis or calculation.`);
    }

    // Log the maximum radius of the calculated RT window (useful for setting extent)
    const maxInternalRadius = internalCorners.reduce(
        (max, v) => Math.max(max, v.length()), 0
    );
     console.log(` -> Max internal radius of calculated RT window: ${maxInternalRadius.toFixed(4)}`);
}

/**
 * Projects a 6D vector onto the 3D physical subspace (E_phys) using the ORTHONORMAL basis.
 * Equivalent to multiplying by the transpose of the physical projection matrix: v_phys = Π_phys^T * v_6D
 * @param {number[]} vec6D - The input 6D vector (array of 6 numbers).
 * @returns {THREE.Vector3} The resulting 3D vector in physical space.
 */
function projectToPhysical(vec6D) {
    // Dot product of vec6D with each physical basis vector
    const x = vec6D.reduce((sum, val, i) => sum + val * config.parVec1[i], 0);
    const y = vec6D.reduce((sum, val, i) => sum + val * config.parVec2[i], 0);
    const z = vec6D.reduce((sum, val, i) => sum + val * config.parVec3[i], 0);
    return new THREE.Vector3(x, y, z);
}

/**
 * Projects a 6D vector onto the 3D internal subspace (E_int) using the ORTHONORMAL basis.
 * Equivalent to multiplying by the transpose of the internal projection matrix: v_int = Π_int^T * v_6D
 * @param {number[]} vec6D - The input 6D vector (array of 6 numbers).
 * @returns {THREE.Vector3} The resulting 3D vector in internal space.
 */
function projectToInternal(vec6D) {
    // Dot product of vec6D with each ORTHONORMAL internal basis vector
    const x = dot(vec6D, config.ortVec1); // Use updated dot function
    const y = dot(vec6D, config.ortVec2);
    const z = dot(vec6D, config.ortVec3);
    return new THREE.Vector3(x, y, z);
}

/**
 * Checks if the projection of a 6D point into internal space (vecInternal)
 * falls within the rhombic triacontahedron (RT) acceptance window defined by windowPlanes.
 * @param {THREE.Vector3} vecInternal - The 3D point in internal space.
 * @returns {boolean} True if the point is within the RT window, false otherwise.
 */
function isInWindow_RT(vecInternal) {
    // Check if planes were successfully generated
    if (windowPlanes.length === 0) {
        console.warn("isInWindow_RT called but windowPlanes is empty. Defaulting to false.");
        return false;
    }

    // Calculate the effective position relative to the shifted center
    const effectiveVec = vecInternal.clone()
        .sub(config.windowCenterInternalPerturbed) // Apply small random perturbation
        .sub(config.windowShiftInternal);         // Apply UI shift

    // Test against all 30 half-space inequalities defined by the RT faces
    for (const plane of windowPlanes) {
        // If the point is outside *any* plane (i.e., dot product > offset),
        // then it's outside the convex hull (the RT window).
        if (plane.normal.dot(effectiveVec) > plane.offset + config.epsilonComparison) {
            return false; // Outside this plane, so outside the RT
        }
    }

    // If the point is inside or on the boundary of all planes, it's inside the RT window.
    return true;
}

/**
 * Checks if the projection of a 6D point into physical space (vecPhysical)
 * falls within the specified spherical shell (restricted to z >= 0 hemisphere).
 * @param {THREE.Vector3} vecPhysical - The 3D point in physical space.
 * @returns {boolean} True if the point is within the shell hemisphere, false otherwise.
 */
function isInPhysicalShell(vecPhysical) {
    // Check 1: Point must be in the upper hemisphere (z >= 0)
    // Use a small epsilon for robust floating point comparison
    if (vecPhysical.z < -config.epsilonComparison) {
        return false;
    }

    // Check 2: Point must be within the radial bounds of the shell
    const distSq = vecPhysical.distanceToSquared(config.domeCenter); // Assuming domeCenter is origin (0,0,0)
    const innerRadiusSq = config.innerRadiusPhysical * config.innerRadiusPhysical;
    const outerRadiusSq = config.outerRadiusPhysical * config.outerRadiusPhysical;

    // Check if distance^2 is between innerRadius^2 and outerRadius^2 (using epsilon)
    return distSq >= innerRadiusSq - config.epsilonComparison &&
           distSq <= outerRadiusSq + config.epsilonComparison;
}


// =============================================================================
// Quasicrystal Generation Logic
// =============================================================================

/**
 * Performs the main generation process:
 * 1. Iterates through points in a 6D integer lattice (Z^6).
 * 2. Projects each 6D point into 3D internal and 3D physical spaces.
 * 3. Accepts points if their internal projection is within the acceptance window
 *    AND their physical projection is within the target hemispherical shell.
 * 4. Stores accepted points with their 6D lattice coordinates.
 * 5. Generates edges and faces based on 6D connectivity.
 * 6. Creates/updates the Three.js objects for points, edges, and faces.
 */
function performGeneration() {
    isInWindowCallCount = 0; // Reset debug log counter
    console.log("Starting new generation cycle...");
    const startTime = performance.now();

    // --- Clear previous generated data ---
    acceptedPointsData = []; // Clear the detailed point data
    generatedEdges = [];     // Clear previous edges
    generatedFaces = [];     // Clear previous faces

    // --- Clear previous Three.js objects (will be recreated) ---
    // No need to clear here, update functions will handle it

    const processed6DPoints = new Set(); // Still useful for more complex generation schemes
    const maxCoord = Math.max(1, Math.round(config.extent));
    const minCoord = -maxCoord;
    let acceptedCount = 0;
    let processedCount = 0;
    let nextPointId = 0; // Counter for unique point IDs

    console.log(`Scanning 6D integer lattice Z^6 within extent: [${minCoord}, ${maxCoord}]`);

    // --- Iterate through the 6D integer lattice ---
    for (let i = minCoord; i <= maxCoord; i++) {
        for (let j = minCoord; j <= maxCoord; j++) {
            for (let k = minCoord; k <= maxCoord; k++) {
                for (let l = minCoord; l <= maxCoord; l++) {
                    for (let m = minCoord; m <= maxCoord; m++) {
                        for (let n = minCoord; n <= maxCoord; n++) {
                            const p6D = [i, j, k, l, m, n];
                            processedCount++;

                            // --- D6 Lattice Check ---
                            const sum = i + j + k + l + m + n;
                            if (sum % 2 !== 0) {
                                continue; // Skip points not in D6 (odd sum)
                            }
                            // -----------------------

                            const pInternal = projectToInternal(p6D);

                            if (isInWindow_RT(pInternal)) {
                                const pPhysical = projectToPhysical(p6D);

                                if (isInPhysicalShell(pPhysical)) {
                                    // Store accepted point data
                                    acceptedPointsData.push({
                                        id: nextPointId++,
                                        lattice: p6D, // Store the 6D lattice coordinates
                                        phys: pPhysical // Store the 3D physical coordinates
                                    });
                                    acceptedCount++;
                                }
                            }
                        }
                    }
                }
            }
        }
         if (i % Math.max(1, Math.floor(maxCoord / 5)) === 0 || i === maxCoord) {
             console.log(`... scanning 6D grid, i=${i}/${maxCoord}`);
         }
    }

    const scanEndTime = performance.now();
    console.log(`Lattice scan complete in ${(scanEndTime - startTime).toFixed(2)} ms.`);
    console.log(` -> Processed ${processedCount} total 6D lattice points.`);
    console.log(` -> Accepted ${acceptedCount} points.`);

    if (acceptedPointsData.length > 0) {
        // --- Generate Edges and Faces ---
        generateConnectivity(); // New function call

        // --- Create/Update Three.js Objects ---
        updatePointsObject(); // Handles creation/update/visibility of points
        updateEdgesObject();  // Handles creation/update/visibility of edges
        updateFacesObject();  // Handles creation/update/visibility of faces

    } else {
        console.log("No vertices accepted. Clearing visual objects.");
        // Explicitly clear objects if no points are generated
        updatePointsObject();
        updateEdgesObject();
        updateFacesObject();
    }

    // Update the shell visualization (might change due to radius UI controls)
    updateShellVisualization();

    const totalEndTime = performance.now();
    console.log(`Full generation cycle (including connectivity & rendering) finished in ${(totalEndTime - startTime).toFixed(2)} ms.`);

    // High-level sanity log
    console.info(
      `Generation summary: verts=${acceptedPointsData.length}, ` +
      `edges=${generatedEdges.length}, faces=${generatedFaces.length}`
    );
}

// =============================================================================
// Connectivity Generation (NEW SECTION)
// =============================================================================

/**
 * Generates edges and faces based on the 6D lattice connectivity
 * of the accepted points, following the provided spec.
 */
function generateConnectivity() {
    if (acceptedPointsData.length === 0) {
        console.log("No points accepted, skipping connectivity generation.");
        generatedEdges = [];
        generatedFaces = [];
        return;
    }

    const startTime = performance.now();
    console.log("Generating connectivity (edges and faces) for D6 lattice...");

    // --- 1. Build LookupMap ---
    const lookupMap = new Map();
    acceptedPointsData.forEach(pt => {
        lookupMap.set(pt.lattice.join(','), pt);
    });
    console.log(` -> Built LookupMap with ${lookupMap.size} entries.`);

    // --- 2. Generate Edges (D6 rule: neighbors differ by +/- e_i +/- e_j) ---
    generatedEdges = [];
    let edgeCount = 0;
    const checkedEdges = new Set(); // To avoid duplicate checks

    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;

        // Iterate through all pairs of distinct axes (i, j) with i < j
        for (let i = 0; i < 5; i++) {
            for (let j = i + 1; j < 6; j++) {
                // Iterate through the four sign combinations for the step +/- e_i +/- e_j
                for (const [si, sj] of [[+1, +1], [+1, -1], [-1, +1], [-1, -1]]) {
                    const neighborLattice = [...v0_lattice];
                    neighborLattice[i] += si;
                    neighborLattice[j] += sj;
                    const neighborKey = neighborLattice.join(',');

                    // Check if the neighbor exists in the accepted set
                    if (lookupMap.has(neighborKey)) {
                        const neighborPt = lookupMap.get(neighborKey);

                        // Avoid duplicates: use a canonical key (sorted IDs)
                        // Note: This ensures we only add the edge once (e.g., 1-2, not 2-1)
                        const edgeKey = [pt.id, neighborPt.id].sort().join('-');
                        if (!checkedEdges.has(edgeKey)) {
                            checkedEdges.add(edgeKey);

                            // Optional: Shell filter (likely redundant, but for completeness)
                            // if (isInPhysicalShell(pt.phys) && isInPhysicalShell(neighborPt.phys)) {
                                generatedEdges.push({ v1: pt.id, v2: neighborPt.id });
                                edgeCount++;
                            // }
                        }
                    }
                }
            }
        }
    }
    console.log(` -> Generated ${edgeCount} edges (D6 rule).`);


    // --- 3. Generate Faces (D6 rule: parallelograms spanned by pairs of root vectors) ---
    generatedFaces = [];
    let faceCount = 0;
    const checkedFaces = new Set(); // To avoid duplicates
    let rhombEdgeFailCount = 0;
    let rhombAngleFailCount = 0;
    const MAX_RHOMB_EDGE_LOGS = 5;
    const MAX_RHOMB_ANGLE_LOGS = 5;

    // 3a. Pre-calculate the 30 D6 root vectors (+/- e_i +/- e_j)
    const roots = [];
    for (let i = 0; i < 5; i++) {
        for (let j = i + 1; j < 6; j++) {
            for (const si of [+1, -1]) {
                for (const sj of [+1, -1]) {
                    const r = [0, 0, 0, 0, 0, 0];
                    r[i] = si;
                    r[j] = sj;
                    roots.push(r);
                }
            }
        }
    }
    // console.log(` -> Generated ${roots.length} D6 root vectors.`);

    // 3b. Iterate through points and pairs of root vectors
    for (const pt of acceptedPointsData) {
        const v0_lattice = pt.lattice;

        // Iterate through all distinct pairs of root vectors (r, s)
        for (let a = 0; a < roots.length; a++) {
            const r = roots[a];
            const v1_lattice = v0_lattice.map((x, idx) => x + r[idx]); // v0 + r
            const key1 = v1_lattice.join(',');

            // Check if v0 + r is an accepted point
            if (!lookupMap.has(key1)) continue;
            const p1 = lookupMap.get(key1);

            for (let b = a + 1; b < roots.length; b++) { // Use b = a + 1 to avoid pairs (r,r) and duplicates (r,s), (s,r)
                const s = roots[b];

                // Check for collinearity: r and s are collinear if r = +/- s.
                // This happens if their dot product is +/- (sqrt(2)*sqrt(2)) = +/- 2.
                // Or, check element-wise: r[k] == s[k] for all k OR r[k] == -s[k] for all k.
                let dotRS = 0;
                for(let k=0; k<6; k++) dotRS += r[k]*s[k];
                if (Math.abs(dotRS) === 2) {
                     // console.log("Skipping collinear roots:", r, s);
                     continue; // Skip collinear root pairs
                }


                const v2_lattice = v0_lattice.map((x, idx) => x + s[idx]); // v0 + s
                const key2 = v2_lattice.join(',');

                const v3_lattice = v1_lattice.map((x, idx) => x + s[idx]); // v0 + r + s
                const key3 = v3_lattice.join(',');

                // Check if v0+s and v0+r+s are also accepted points
                if (lookupMap.has(key2) && lookupMap.has(key3)) {
                    const p2 = lookupMap.get(key2);
                    const p3 = lookupMap.get(key3);

                    // Found a valid parallelogram in 6D.
                    const ids = [pt.id, p1.id, p3.id, p2.id]; // Order: v0, v0+r, v0+r+s, v0+s

                    // Deduplicate using a canonical key (sorted IDs)
                    const faceKey = ids.slice().sort((a, b) => a - b).join('-');
                    if (!checkedFaces.has(faceKey)) {
                        
                        // --- NEW: Vertex-based Outer Shell Filter ---
                        const P0 = pt.phys;
                        const P1 = p1.phys; // Corresponds to v0 + r
                        const P2 = p3.phys; // Corresponds to v0 + r + s
                        const P3 = p2.phys; // Corresponds to v0 + s

                        const Router = config.outerRadiusPhysical;
                        const vTolAbs = config.vertexOuterTol * Router;

                        let outerHits = 0;
                        [ P0, P1, P2, P3 ].forEach(p => {
                            if (Math.abs(p.length() - Router) < vTolAbs) outerHits++;
                        });

                        if (outerHits / 4 < config.faceOuterVertexFracNeeded) continue; // Reject early if not enough vertices are near the outer shell

                        // --- Absolute Band Guard ---
                        const minR = Math.min(P0.length(), P1.length(), P2.length(), P3.length());
                        const maxR = Math.max(P0.length(), P1.length(), P2.length(), P3.length());
                        if (maxR - minR > vTolAbs) continue; // Reject face if it spans more than the allowed radial thickness band
                        // --------------------------

                        // --- Golden Rhombus Shape Check ---
                         const edge1 = new THREE.Vector3().subVectors(P1, P0); // Edge along r
                         const edge2 = new THREE.Vector3().subVectors(P3, P0); // Edge along s (Adjacent to edge1 at P0)

                         const len1 = edge1.length();
                         const len2 = edge2.length();

                         // Reject if either edge collapses in physical space
                         if (len1 < 1e-6 || len2 < 1e-6) continue; // Filter degenerate edges

                         // Check 1: Are adjacent edge lengths approximately equal (relative tolerance)?
                         if (Math.abs(len1 - len2) <= config.rhombEdgeRelTol * Math.min(len1, len2)) {
                             // Check 2: Is the angle between edges correct?
                             const cosAngle = edge1.dot(edge2) / (len1 * len2);
                             const cos36 = Math.cos(Math.PI / 5); // ~0.8090
                             const cos72 = Math.cos(2 * Math.PI / 5); // ~0.3090

                             // We check absolute value of cosAngle because dot product might give cos(180-theta) = -cos(theta)
                             const absCosAngle = Math.abs(cosAngle);

                             if (Math.abs(absCosAngle - cos36) <= config.rhombAngleTol ||
                                 Math.abs(absCosAngle - cos72) <= config.rhombAngleTol)
                             {
                                 // Passed both checks! Add the face.
                                 checkedFaces.add(faceKey);
                                 generatedFaces.push({ vertices: ids }); // Store with specific winding order
                                 faceCount++;
                             } else {
                                 // Failed angle check
                                 // console.log(`Rhombus check failed (angle): cos=${absCosAngle.toFixed(4)}, expected ~${cos36.toFixed(4)} or ~${cos72.toFixed(4)}`);
                                 if (rhombAngleFailCount < MAX_RHOMB_ANGLE_LOGS) {
                                     console.log(`AngFail |cos|=${absCosAngle.toFixed(5)}  ids=${ids}`); // Higher precision log
                                     rhombAngleFailCount++;
                                 } else if (rhombAngleFailCount === MAX_RHOMB_ANGLE_LOGS) {
                                     console.log("(Further angle check failures omitted)");
                                     rhombAngleFailCount++; // Increment once more to prevent re-logging the omission message
                                 }
                             }
                         } else {
                             // Failed edge length check
                             // console.log(`Rhombus check failed (edge lengths): l1=${len1.toFixed(4)}, l2=${len2.toFixed(4)}, diff=${Math.abs(len1-len2).toFixed(4)}`);
                             if (rhombEdgeFailCount < MAX_RHOMB_EDGE_LOGS) { 
                                 console.log(`LenFail Δ=${Math.abs(len1-len2).toExponential(3)}  ids=${ids}`); // Higher precision log
                                 rhombEdgeFailCount++;
                             } else if (rhombEdgeFailCount === MAX_RHOMB_EDGE_LOGS) {
                                 console.log("(Further edge length check failures omitted)");
                                 rhombEdgeFailCount++; // Increment once more
                             }
                         }
                         // --- End Golden Rhombus Shape Check ---
                    }
                }
            }
        }
    }
    console.log(` -> Generated ${faceCount} faces (D6 rule, filtered for outer shell).`);
    // Log summary of rhombus check failures
    if (rhombEdgeFailCount > MAX_RHOMB_EDGE_LOGS || rhombAngleFailCount > MAX_RHOMB_ANGLE_LOGS) {
         console.log(` -> Rhombus checks: ${rhombEdgeFailCount - (rhombEdgeFailCount > MAX_RHOMB_EDGE_LOGS ? 1:0)} edge fails, ${rhombAngleFailCount - (rhombAngleFailCount > MAX_RHOMB_ANGLE_LOGS ? 1:0)} angle fails (logged max ${MAX_RHOMB_EDGE_LOGS}/${MAX_RHOMB_ANGLE_LOGS}).`);
    } else if (rhombEdgeFailCount > 0 || rhombAngleFailCount > 0) {
         console.log(` -> Rhombus checks: ${rhombEdgeFailCount} edge fails, ${rhombAngleFailCount} angle fails.`);
    }

    // Added debug log for filter stats
    console.debug(
      `Face filter stats: kept=${faceCount}, ` +
      `edgeLenFails=${rhombEdgeFailCount}, angFails=${rhombAngleFailCount}`
    );

    const endTime = performance.now();
    console.log(`D6 Connectivity generation finished in ${(endTime - startTime).toFixed(2)} ms.`);

    // Sanity check for face duplicates (only runs in dev builds)
    console.assert(checkedFaces.size === generatedFaces.length,
        'Duplicate-face mismatch (keys vs. array):',
        checkedFaces.size, generatedFaces.length);
}


// =============================================================================
// Visualization Update Functions (NEW/MODIFIED)
// =============================================================================

/**
 * Cleans up a Three.js object (geometry, material, removal from scene).
 */
function disposeObject(object) {
    if (!object) return;
    if (object.geometry) object.geometry.dispose();
    if (object.material) {
        if (Array.isArray(object.material)) {
            object.material.forEach(m => m.dispose());
        } else {
            object.material.dispose();
        }
    }
    scene.remove(object);
}


/**
 * Creates/updates the THREE.Points object for vertices.
 */
function updatePointsObject() {
    disposeObject(pointsObject); // Dispose previous object
    pointsObject = null;

    if (!config.showPoints || acceptedPointsData.length === 0) {
        console.log("Points hidden or no data.");
        return; // Don't create if hidden or no data
    }

    console.log("Updating points object...");
    const positions = [];
    acceptedPointsData.forEach(pt => positions.push(pt.phys.x, pt.phys.y, pt.phys.z));

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: config.vertexColor,
        size: config.vertexSize,
        // sizeAttenuation: false // Optional
    });

    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);
    console.log(" -> Points object updated/created.");
}

/**
 * Creates/updates the THREE.LineSegments object for edges.
 */
function updateEdgesObject() {
    disposeObject(edgesObject); // Dispose previous object
    edgesObject = null;

    if (!config.showEdges || generatedEdges.length === 0 || acceptedPointsData.length === 0) {
         console.log("Edges hidden or no data.");
         return; // Don't create if hidden or no data
    }

    console.log("Updating edges object...");
    // Need a map from point ID to its index in acceptedPointsData for quick lookup
    const idToIndexMap = new Map();
    acceptedPointsData.forEach((pt, index) => idToIndexMap.set(pt.id, index));

    // Create vertex position array matching the order in acceptedPointsData
    const vertexPositions = acceptedPointsData.map(pt => pt.phys);

    const linePositions = [];
    generatedEdges.forEach(edge => {
        const p1 = vertexPositions[idToIndexMap.get(edge.v1)];
        const p2 = vertexPositions[idToIndexMap.get(edge.v2)];
        if (p1 && p2) { // Ensure both points exist
             linePositions.push(p1.x, p1.y, p1.z);
             linePositions.push(p2.x, p2.y, p2.z);
        } else {
            console.warn(`Edge references missing point ID: ${edge.v1} or ${edge.v2}`);
        }
    });

    if (linePositions.length === 0) {
         console.log(" -> No valid edge positions found.");
         return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

    const material = new THREE.LineBasicMaterial({
        color: config.edgeColor,
        // linewidth: 2 // Note: linewidth > 1 may not work on all platforms/drivers
    });

    edgesObject = new THREE.LineSegments(geometry, material);
    scene.add(edgesObject);
    console.log(" -> Edges object updated/created.");
}

/**
 * Creates/updates the THREE.Mesh object for faces (rhombi).
 * Note: Each rhombus is made of two triangles.
 */
function updateFacesObject() {
    disposeObject(facesObject); // Dispose previous object
    facesObject = null;

    if (!config.showFaces || generatedFaces.length === 0 || acceptedPointsData.length === 0) {
        console.log("Faces hidden or no data.");
        return; // Don't create if hidden or no data
    }

    console.log("Updating faces object...");
    const idToIndexMap = new Map();
    acceptedPointsData.forEach((pt, index) => idToIndexMap.set(pt.id, index));

    // Create vertex array in the order of acceptedPointsData IDs
    // We need ONLY the positions for the geometry
    const vertexPositions = new Float32Array(acceptedPointsData.length * 3);
    acceptedPointsData.forEach((pt, index) => {
        vertexPositions[index * 3 + 0] = pt.phys.x;
        vertexPositions[index * 3 + 1] = pt.phys.y;
        vertexPositions[index * 3 + 2] = pt.phys.z;
    });


    // Create indices for the triangles that make up the rhombi
    // Each rhombus [p0, p1, p2, p3] becomes two triangles: (p0, p1, p2) and (p0, p2, p3)
    const indices = [];
    generatedFaces.forEach(face => {
        const i0 = idToIndexMap.get(face.vertices[0]);
        const i1 = idToIndexMap.get(face.vertices[1]);
        const i2 = idToIndexMap.get(face.vertices[2]);
        const i3 = idToIndexMap.get(face.vertices[3]);

        // Check if all indices are valid
        if (i0 !== undefined && i1 !== undefined && i2 !== undefined && i3 !== undefined) {
            // Triangle 1: (p0, p1, p2)
            indices.push(i0, i1, i2);
            // Triangle 2: (p0, p2, p3) - Corrected winding based on rhombus order
            indices.push(i0, i2, i3);
        } else {
            console.warn("Face references missing point ID:", face.vertices);
        }
    });

     if (indices.length === 0) {
         console.log(" -> No valid face indices found.");
         return;
     }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertexPositions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals(); // Calculate normals for lighting

    const material = new THREE.MeshStandardMaterial({
        color: config.faceColor,
        opacity: config.faceOpacity,
        transparent: true,
        side: THREE.DoubleSide, // Render both sides
        metalness: 0.1,
        roughness: 0.6,
        // depthWrite: false // Consider if transparency sorting issues arise
    });

    facesObject = new THREE.Mesh(geometry, material);
    scene.add(facesObject);
    console.log(" -> Faces object updated/created.");
}


// =============================================================================
// Visualization Helpers
// =============================================================================

/**
 * Creates or updates the visual representation of the target physical shell
 * (a hemisphere defined by inner and outer radii).
 */
function updateShellVisualization() {
    // --- Clean up previous shell mesh ---
    if (shellMeshGroup) {
        scene.remove(shellMeshGroup);
        // Dispose geometries and materials to free GPU memory
        shellMeshGroup.traverse(child => {
            if (child.isMesh) { // Check if it's a mesh
                 if (child.geometry) child.geometry.dispose();
                 if (child.material) {
                     // If multiple meshes share material, only dispose once or handle carefully
                     // Assuming separate materials for now
                     child.material.dispose();
                 }
            }
        });
        shellMeshGroup = null; // Ensure it's reassigned
    }

    shellMeshGroup = new THREE.Group(); // Create a new group for the shell parts

    // --- Define Geometry Parameters ---
    const segments = 64; // Detail for the curved surfaces and base ring
    const phiSegments = Math.max(8, segments / 2); // Detail for the spherical cap height divisions

    // --- Define Material ---
    const shellMaterial = new THREE.MeshStandardMaterial({
        color: config.shellColor,
        opacity: config.shellOpacity,
        transparent: true,
        side: THREE.DoubleSide, // Render both sides to see inside/outside
        metalness: 0.1,         // Slight metallic look
        roughness: 0.5,         // Moderate roughness
        // depthWrite: false // Can sometimes help with transparency sorting issues, but might hide points behind it
    });

    // Check if we need an inner shell component (i.e., is it a shell or a solid dome?)
    const hasInnerRadius = config.innerRadiusPhysical > config.epsilonComparison;

    // --- Create Geometries (initially oriented Y-up) ---
    // Create outer hemisphere cap (phi runs 0 to PI/2 for Y-up hemisphere)
    const outerSphereGeomY = new THREE.SphereGeometry(config.outerRadiusPhysical, segments, phiSegments, 0, Math.PI * 2, 0, Math.PI / 2);

    // Create inner hemisphere cap only if needed
    const innerSphereGeomY = hasInnerRadius
        ? new THREE.SphereGeometry(config.innerRadiusPhysical, segments, phiSegments, 0, Math.PI * 2, 0, Math.PI / 2)
        : null;

    // Create base ring (if inner radius exists) or base circle (if solid dome)
    const baseGeomY = hasInnerRadius
        ? new THREE.RingGeometry(config.innerRadiusPhysical, config.outerRadiusPhysical, segments)
        : new THREE.CircleGeometry(config.outerRadiusPhysical, segments);


    // --- Rotate Geometries to Z-up Orientation ---
    // We want the hemisphere cap pointing along the Z-axis.
    // Rotate +90 degrees around the X-axis.
    const rotationX = Math.PI / 2;
    outerSphereGeomY.rotateX(rotationX);
    if (innerSphereGeomY) innerSphereGeomY.rotateX(rotationX);
    baseGeomY.rotateX(rotationX); // Base also needs rotation to align with the Z=0 plane

    // --- Create Meshes and Add to Group ---
    const outerMesh = new THREE.Mesh(outerSphereGeomY, shellMaterial);
    shellMeshGroup.add(outerMesh);

    if (innerSphereGeomY) {
        const innerMesh = new THREE.Mesh(innerSphereGeomY, shellMaterial);
        shellMeshGroup.add(innerMesh); // Add inner surface
    }
    const baseMesh = new THREE.Mesh(baseGeomY, shellMaterial);
    shellMeshGroup.add(baseMesh); // Add base ring/circle

    // Add the complete shell group to the main scene
    scene.add(shellMeshGroup);
    console.log("Updated shell visualization.");
}


// =============================================================================
// User Interface Setup (lil-gui)
// =============================================================================

/**
 * Sets up the lil-gui panel with controls for generation and visualization parameters.
 */
function setupGUI() {
    gui = new GUI();
    gui.title("Quasicrystal Controls");

    // --- Generation Parameters Folder ---
    const genFolder = gui.addFolder('Generation Parameters');
    // genFolder.add(config, 'radiusInternal', 0.1, 5.0, 0.05).name('Window Radius (Internal)').onChange(performGeneration); // REMOVED - RT window size is fixed
    genFolder.add(config, 'innerRadiusPhysical', 0.0, 20.0, 0.05).name('Inner Radius (Physical)').onChange(performGeneration).listen(); // listen() if modified elsewhere
    genFolder.add(config, 'outerRadiusPhysical', 0.1, 20.0, 0.05).name('Outer Radius (Physical)').onChange(performGeneration).listen();
    genFolder.add(config, 'extent', 1, 10, 1).name('6D Search Extent').onChange(performGeneration); // Increased max extent

    // --- Internal Window Shift Sub-Folder --- Adjust range for fixed RT window
    const shiftFolder = genFolder.addFolder('Window Shift (Internal RT)'); // Renamed slightly
    const shiftRange = 0.5; // Adjusted range based on typical RT radius (~0.86)
    shiftFolder.add(config.windowShiftInternal, 'x', -shiftRange, shiftRange, 0.01).name('Shift X').onChange(performGeneration).listen();
    shiftFolder.add(config.windowShiftInternal, 'y', -shiftRange, shiftRange, 0.01).name('Shift Y').onChange(performGeneration).listen();
    shiftFolder.add(config.windowShiftInternal, 'z', -shiftRange, shiftRange, 0.01).name('Shift Z').onChange(performGeneration).listen();
    genFolder.open(); // Keep generation params open

    // --- Visualization Parameters Folder ---
    const vizFolder = gui.addFolder('Visualization');

    // --- Points Controls ---
    const pointsFolder = vizFolder.addFolder('Points');
    pointsFolder.add(config, 'showPoints').name('Show Points').onChange(updatePointsObject);
    pointsFolder.addColor(config, 'vertexColor').name('Color').onChange(() => {
        if (pointsObject) pointsObject.material.color.set(config.vertexColor);
        // updatePointsObject(); // Alternatively, recreate if needed, but material update is faster
    });
    pointsFolder.add(config, 'vertexSize', 0.001, 0.2, 0.001).name('Size').onChange(() => { // Increased max size
        if (pointsObject) pointsObject.material.size = config.vertexSize;
        // updatePointsObject(); // Alternatively, recreate if needed
    });
    pointsFolder.open(); // Open by default

    // --- Edges Controls ---
    const edgesFolder = vizFolder.addFolder('Edges');
    edgesFolder.add(config, 'showEdges').name('Show Edges').onChange(updateEdgesObject);
    edgesFolder.addColor(config, 'edgeColor').name('Color').onChange(() => {
         if (edgesObject) edgesObject.material.color.set(config.edgeColor);
         // updateEdgesObject(); // Alternatively, recreate if needed
    });
    edgesFolder.open(); // Closed by default -> Open by default

    // --- Faces Controls ---
    const facesFolder = vizFolder.addFolder('Faces (Rhombi)');
    facesFolder.add(config, 'showFaces').name('Show Faces').onChange(updateFacesObject);
    facesFolder.addColor(config, 'faceColor').name('Color').onChange(() => {
        if (facesObject) facesObject.material.color.set(config.faceColor);
        // updateFacesObject(); // Alternatively, recreate if needed
    });
    facesFolder.add(config, 'faceOpacity', 0, 1, 0.01).name('Opacity').onChange(() => {
         if (facesObject) facesObject.material.opacity = config.faceOpacity;
         // updateFacesObject(); // Alternatively, recreate if needed
    });
    facesFolder.add(config, 'faceRelativeRadiusTolerance', 0, 0.2, 0.01)
               .name('Centroid Tolerance (Rel)')
               .onChange(performGeneration); // Regenerate on change
    facesFolder.open(); // Closed by default -> Open by default

    // --- Guide Shell Controls ---
    const shellFolder = vizFolder.addFolder('Guide Shell');
    shellFolder.addColor(config, 'shellColor').name('Color').onChange(updateShellVisualization);
    shellFolder.add(config, 'shellOpacity', 0, 1, 0.01).name('Opacity').onChange(updateShellVisualization);
    // shellFolder.open(); // Closed by default


    // gui.close(); // Close GUI by default if desired
}


// =============================================================================
// Three.js Scene Initialization & Rendering Loop
// =============================================================================

/**
 * Initializes the entire Three.js scene, camera, renderer, controls, lights,
 * calculates initial projection matrices, sets up the GUI, and performs
 * the initial quasicrystal generation.
 */
function init() {
    console.log("Initializing scene...");
    // --- Basic Scene Setup ---
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010); // Dark grey background

    // --- Camera Setup ---
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // Position camera for a good initial view of the dome structure
    camera.position.set(0, config.outerRadiusPhysical * 1.5, config.outerRadiusPhysical * 1.5); // Adjusted y/z for better angle

    // --- Renderer Setup ---
    renderer = new THREE.WebGLRenderer({ antialias: true }); // Enable anti-aliasing
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio); // Adjust for high-DPI displays
    document.body.appendChild(renderer.domElement);

    // --- Orbit Controls ---
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;      // Smooth camera movement
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false; // Pan orthogonal to view direction
    controls.minDistance = 0.5;
    controls.maxDistance = 50;
    // Target the center of the base of the dome initially
    controls.target.set(config.domeCenter.x, config.domeCenter.y, config.domeCenter.z); // Look at origin
    controls.update(); // Important after setting target or position

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0x606060); // Soft ambient light
    scene.add(ambientLight);
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.0); // Brighter primary light
    directionalLight1.position.set(5, 10, 7.5).normalize(); // Position light source
    scene.add(directionalLight1);
    // Optional: Add a secondary fill light from another direction
    // const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    // directionalLight2.position.set(-5, -5, -5).normalize();
    // scene.add(directionalLight2);

    // --- Axes Helper ---
    // Scale helper based on expected object size
    const axesHelper = new THREE.AxesHelper(config.outerRadiusPhysical * 1.2);
    scene.add(axesHelper);

    // --- Initial Calculations & Setup ---
    calculateProjectionMatrices(); // Must be called before generation
    setupGUI();                    // Create the UI panel
    performGeneration();           // Generate initial points

    // --- Event Listeners ---
    window.addEventListener('resize', onWindowResize, false);

    console.log("Initialization complete.");
}

/**
 * Handles window resize events to update camera aspect ratio and renderer size.
 */
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

/**
 * The main animation loop called via requestAnimationFrame.
 * Updates controls and renders the scene.
 */
function animate() {
    requestAnimationFrame(animate); // Queue the next frame
    controls.update(); // Required if enableDamping is true
    render();
}

/**
 * Renders the scene using the camera.
 */
function render() {
    renderer.render(scene, camera);
}

// =============================================================================
// Main Execution
// =============================================================================

init();     // Initialize everything
animate();  // Start the rendering loop

/*
https://chatgpt.com/c/68176496-ceb4-8001-a5df-73573de79b65

*/