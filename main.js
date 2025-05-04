import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
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
    radiusInternal: 1.5,        // Acceptance window radius in internal space
    innerRadiusPhysical: 5.0,   // Inner radius of the physical space shell
    outerRadiusPhysical: 6.0,   // Outer radius of the physical space shell
    extent: 5,                  // Range [-extent, extent] for 6D lattice search
    windowShiftInternal: new THREE.Vector3(0, 0, 0), // UI shift for the acceptance window center

    // --- Visualization Parameters (UI controllable) ---
    vertexColor: '#ffffff',     // Color of the generated points
    vertexSize: 0.054,          // Size of the generated points
    edgeColor: '#ff00ff',       // Color for edges
    faceColor: '#ffff00',       // Color for faces
    faceOpacity: 0.5,           // Opacity for faces
    shellColor: '#00ffff',      // Color for the guide shell visualization
    shellOpacity: 0.15,         // Opacity for the guide shell visualization
    showPoints: true,           // Toggle visibility for points
    showEdges: false,           // Toggle visibility for edges
    showFaces: false,           // Toggle visibility for faces

    // --- Fixed Parameters ---
    windowPerturbationMagnitude: 1e-6, // Small random offset for window center to avoid degenerate cases
    domeCenter: new THREE.Vector3(0, 0, 0), // Center for physical shell check (currently origin)
    epsilonComparison: 1e-12,          // Small value for floating point comparisons
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
    const invTau = 1 / tau;

    // --- 1. Define initial vectors that SPAN E_phys (normalized or not, doesn't matter for GS) ---
    // These vectors determine the *orientation* of the physical subspace.
    // Using the same structure as before, but unnormalized for clarity.
    const initialParVecs = [
        [1, 0, tau, invTau, 0, 0],
        [0, 1, 0, tau, invTau, 0],
        [0, 0, 1, 0, tau, invTau]
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
 * falls within the spherical acceptance window.
 * @param {THREE.Vector3} vecInternal - The 3D point in internal space.
 * @returns {boolean} True if the point is within the window, false otherwise.
 */
function isInWindow(vecInternal) {
    // Calculate the effective window center (base perturbed center + UI shift)
    const effectiveCenter = config.windowCenterInternalPerturbed.clone().add(config.windowShiftInternal);

    // Debug Log: Print effective center for the first few calls (optional)
    // if (isInWindowCallCount < MAX_ISINWINDOW_LOGS) {
    //     console.log(`isInWindow Call ${isInWindowCallCount}: Effective Center = (${effectiveCenter.x.toFixed(3)}, ${effectiveCenter.y.toFixed(3)}, ${effectiveCenter.z.toFixed(3)})`);
    //     isInWindowCallCount++;
    // }

    // Check if squared distance to effective center is within squared radius
    const distanceSq = vecInternal.distanceToSquared(effectiveCenter);
    return distanceSq <= config.radiusInternal * config.radiusInternal;
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

                            const pInternal = projectToInternal(p6D);

                            if (isInWindow(pInternal)) {
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
    console.log("Generating connectivity (edges and faces)...");

    // --- 1. Build LookupMap ---
    const lookupMap = new Map();
    acceptedPointsData.forEach(pt => {
        lookupMap.set(pt.lattice.join(','), pt);
    });
    console.log(` -> Built LookupMap with ${lookupMap.size} entries.`);

    // --- 2. Generate Edges ---
    generatedEdges = []; // Clear previous edges
    let edgeCount = 0;
    const checkedEdges = new Set(); // To avoid duplicate checks A->B vs B->A

    for (const pt of acceptedPointsData) {
        for (let i = 0; i < 6; i++) { // Iterate through 6 basis vector directions
            const neighborLattice = [...pt.lattice];
            neighborLattice[i]++; // Check neighbor in positive direction

            const neighborKey = neighborLattice.join(',');

            // Check if neighbor exists in the accepted set
            if (lookupMap.has(neighborKey)) {
                const neighborPt = lookupMap.get(neighborKey);

                // Avoid duplicates: use a canonical key (e.g., sorted IDs)
                const edgeKey = [pt.id, neighborPt.id].sort().join('-');
                if (!checkedEdges.has(edgeKey)) {
                     checkedEdges.add(edgeKey);

                     // Optional: Shell filter - both points must be in shell (already guaranteed by generation?)
                     // This check is technically redundant if points were already filtered,
                     // but kept for clarity matching the spec's option.
                     if (isInPhysicalShell(pt.phys) && isInPhysicalShell(neighborPt.phys)) {
                         generatedEdges.push({ v1: pt.id, v2: neighborPt.id });
                         edgeCount++;
                     }
                }
            }
        }
    }
    console.log(` -> Generated ${edgeCount} edges.`);

    // --- 3. Generate Faces (Rhombi) ---
    generatedFaces = []; // Clear previous faces
    let faceCount = 0;
    const checkedFaces = new Set(); // To avoid duplicates

    for (const pt of acceptedPointsData) {
        const v00_lattice = pt.lattice;
        const p00 = pt; // Use pt directly

        for (let i = 0; i < 5; i++) {       // First basis direction index
            for (let j = i + 1; j < 6; j++) { // Second basis direction index

                // Calculate lattice coordinates of the other 3 corners
                const v10_lattice = [...v00_lattice]; v10_lattice[i]++;
                const v11_lattice = [...v10_lattice]; v11_lattice[j]++;
                const v01_lattice = [...v00_lattice]; v01_lattice[j]++;

                // Serialize keys
                const key10 = v10_lattice.join(',');
                const key11 = v11_lattice.join(',');
                const key01 = v01_lattice.join(',');

                // Check if all four corners were accepted points
                if (lookupMap.has(key10) && lookupMap.has(key11) && lookupMap.has(key01)) {
                    const p10 = lookupMap.get(key10);
                    const p11 = lookupMap.get(key11);
                    const p01 = lookupMap.get(key01);

                    // Avoid duplicates: Use a canonical key based on sorted IDs
                    const faceVertexIds = [p00.id, p10.id, p11.id, p01.id].sort();
                    const faceKey = faceVertexIds.join('-');

                    if (!checkedFaces.has(faceKey)) {
                         checkedFaces.add(faceKey);

                         // Optional: Shell filter face-wise (again, likely redundant but matches spec)
                         if ([p00, p10, p11, p01].every(p => isInPhysicalShell(p.phys))) {
                             // Store face with consistent vertex order (matches spec)
                             generatedFaces.push({
                                 vertices: [p00.id, p10.id, p11.id, p01.id]
                             });
                             faceCount++;
                         }
                    }
                }
            }
        }
    }
    console.log(` -> Generated ${faceCount} faces (rhombi).`);

    const endTime = performance.now();
    console.log(`Connectivity generation finished in ${(endTime - startTime).toFixed(2)} ms.`);
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
    genFolder.add(config, 'radiusInternal', 0.1, 5.0, 0.05).name('Window Radius (Internal)').onChange(performGeneration);
    genFolder.add(config, 'innerRadiusPhysical', 0.0, 20.0, 0.05).name('Inner Radius (Physical)').onChange(performGeneration).listen(); // listen() if modified elsewhere
    genFolder.add(config, 'outerRadiusPhysical', 0.1, 20.0, 0.05).name('Outer Radius (Physical)').onChange(performGeneration).listen();
    genFolder.add(config, 'extent', 1, 10, 1).name('6D Search Extent').onChange(performGeneration); // Increased max extent

    // --- Internal Window Shift Sub-Folder ---
    const shiftFolder = genFolder.addFolder('Window Shift (Internal)'); // Open by default now
    shiftFolder.add(config.windowShiftInternal, 'x', -config.radiusInternal*2, config.radiusInternal*2, 0.01).name('Shift X').onChange(performGeneration).listen(); // Range relative to radius
    shiftFolder.add(config.windowShiftInternal, 'y', -config.radiusInternal*2, config.radiusInternal*2, 0.01).name('Shift Y').onChange(performGeneration).listen();
    shiftFolder.add(config.windowShiftInternal, 'z', -config.radiusInternal*2, config.radiusInternal*2, 0.01).name('Shift Z').onChange(performGeneration).listen();

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
    // edgesFolder.open(); // Closed by default

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
    // facesFolder.open(); // Closed by default

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
