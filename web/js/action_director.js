import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/** * YEDP ACTION DIRECTOR - V9.11 (Full PNG Export)
 * - Added: Support for up to 4 characters in the same scene.
 * - Added: TransformControls for moving and rotating characters individually.
 * - Added: Sidebar UI for independent character animation selection and looping.
 * - Added: Camera Keyframing (Start/End) with linear & ease interpolations.
 * - Updated: Optimized BVH offsets and clone instancing using SkeletonUtils.
 * - Fix (9.1): Resolved resolution gate collapse (blue line) and WebGL baking crash caused by null dimensions.
 * - Fix (9.1): Applied modulo time wrapping to ensure characters correctly loop when global frame count exceeds their animation duration.
 * - Fix (9.2): Eradicated QuotaExceededError and UI stutter by uploading payload to Python memory cache.
 * - Update (9.2): Removed strict character limitation (capped at 16 for WebGL safety).
 * - Update (9.2): Added progress indicator to Bake button.
 * - Update (9.3): Added dynamic Male/Female Mesh Toggle matching 'Geo_Depth_F' vs 'Geo_Depth' conventions.
 * - Update (9.4): Added Scale mode to Gizmo Tools for resizing individual characters.
 * - Fix (9.5): Unrecognized meshes (props) automatically attach to both genders' depth passes for future prop support.
 * - Fix (9.6): Made Mesh naming parser completely immune to GLTF exporter renaming.
 * - Fix (9.7): Implemented Full Ancestry String Parsing to solve Blender's "Object vs Mesh Data" naming conflict. 
 * - Update (9.7): Added tiny Mesh Counter [M:1 | F:1 | Pose:1] to UI for immediate parsing feedback.
 * - Fix (9.8): Added timestamp Cache-Buster to Rig/Animation loaders to prevent browser from loading stale GLB files.
 * - Update (9.8): Added verbose console logging for mesh categorization transparency.
 * - Fix (9.9): Removed destructive position overrides to restore full Root Motion (hips moving through space).
 * - Fix (9.9): Decoupled internal engine time from integer UI slider to fix playback freezing on non-30 FPS values.
 * - Update (9.10): Upgraded all rendering passes (Depth, Canny, Normal) to 100% lossless pure PNG now that payload size limits are removed.
 * - Fix (Camera): Added OrbitControls target to keyframes to fix panning and zooming translation issues.
 * - Update (Offline): Pointed all imports to the same local directory.
 * - Fix (9.11): Ensured "none" is always available in animation lists to fix UI default-selection mismatches.
 * - Fix (9.11): Forcefully hide the skeleton during all baking passes to prevent contamination of the OpenPose render.
 * - Update (9.12): Overhauled UI with floating Gizmo icons, Blender shortcuts, collapsible right menus, and high-visibility Depth inputs.
 */

const loadThreeJS = async () => {
    if (window._YEDP_THREE_CACHE) return window._YEDP_THREE_CACHE;

    return window._YEDP_THREE_CACHE = new Promise(async (resolve, reject) => {
        const baseUrl = new URL(".", import.meta.url).href;
        try {
            console.log("[Yedp] Initializing Engine V9.11 (Offline Mode)...");
            
            // Loaded locally from the same folder
            const THREE = await import(new URL("./three.module.js", baseUrl).href);
            const { OrbitControls } = await import(new URL("./OrbitControls.js", baseUrl).href);
            const { TransformControls } = await import(new URL("./TransformControls.js", baseUrl).href);
            const { GLTFLoader } = await import(new URL("./GLTFLoader.js", baseUrl).href);
            await import(new URL("./fflate.module.js", baseUrl).href); 
            const { FBXLoader } = await import(new URL("./FBXLoader.js", baseUrl).href);
            const { BVHLoader } = await import(new URL("./BVHLoader.js", baseUrl).href);
            const { clone } = await import(new URL("./SkeletonUtils.js", baseUrl).href);

            resolve({ THREE, OrbitControls, TransformControls, GLTFLoader, FBXLoader, BVHLoader, SkeletonUtils: { clone } });
        } catch (e) {
            console.error("[Yedp] Critical Engine Load Failure:", e);
            reject(e);
        }
    });
};

// --- UNIVERSAL BONE MAPPING DICTIONARY ---
const { BONE_MAP, BONE_KEYS_SORTED } = (() => {
    const map = {};
    const synonyms = {
        "hips": ["hips", "pelvis", "root", "cg", "center"],
        "spine": ["spine", "spine0", "spine00", "abdomen", "waist", "lowerback"],
        "spine1": ["spine1", "spine01", "spine001", "chest", "chest1", "torso1", "middleback"],
        "spine2": ["spine2", "spine02", "spine002", "upperchest", "chest2", "torso2", "upperback"],
        "spine3": ["spine3", "spine03", "spine003", "chest3", "torso3"],
        "neck": ["neck", "neck0", "neck00", "cervical"],
        "head": ["head"],
        "leftshoulder": ["leftshoulder", "lshoulder", "shoulderl", "lclavicle", "claviclel", "leftcollar", "lcollar", "collarl"],
        "leftarm": ["leftarm", "larm", "arml", "leftuparm", "luparm", "uparml", "leftupperarm", "lupperarm", "upperarml", "lshldr", "leftbicep"],
        "leftforearm": ["leftforearm", "lforearm", "forearml", "leftelbow", "lelbow", "elbowl", "leftlowerarm", "llowerarm", "lowerarml"],
        "lefthand": ["lefthand", "lhand", "handl", "leftwrist", "lwrist", "wristl"],
        "rightshoulder": ["rightshoulder", "rshoulder", "shoulderr", "rclavicle", "clavicler", "rightcollar", "rcollar", "collarr"],
        "rightarm": ["rightarm", "rarm", "armr", "rightuparm", "ruparm", "uparmr", "rightupperarm", "rupperarm", "upperarmr", "rshldr", "rightbicep"],
        "rightforearm": ["rightforearm", "rforearm", "forearmr", "rightelbow", "relbow", "elbowr", "rightlowerarm", "rlowerarm", "lowerarmr"],
        "righthand": ["righthand", "rhand", "handr", "rightwrist", "rwrist", "wristr"],
        "leftupleg": ["leftupleg", "lupleg", "uplegl", "leftthigh", "lthigh", "thighl", "leftupperleg", "lupperleg", "upperlegl", "lefthip", "lhip", "hip_l"],
        "leftleg": ["leftleg", "lleg", "legl", "leftcalf", "lcalf", "calfl", "leftknee", "lknee", "kneel", "leftlowerleg", "llowerleg", "lowerlegl", "lshin", "shinl"],
        "leftfoot": ["leftfoot", "lfoot", "footl", "leftankle", "lankle", "anklel"],
        "lefttoebase": ["lefttoebase", "ltoebase", "toebasel", "lefttoe", "ltoe", "toel", "lefttoes", "ltoes", "toesl", "leftfootball", "lfootball", "footballl"],
        "rightupleg": ["rightupleg", "rupleg", "uplegr", "rightthigh", "rthigh", "thighr", "rightupperleg", "rupperleg", "upperlegr", "righthip", "rhip", "hip_r"],
        "rightleg": ["rightleg", "rleg", "legr", "rightcalf", "rcalf", "calfr", "rightknee", "rknee", "kneer", "rightlowerleg", "rlowerleg", "lowerlegr", "rshin", "shinr"],
        "rightfoot": ["rightfoot", "rfoot", "footr", "rightankle", "rankle", "ankler"],
        "righttoebase": ["righttoebase", "rtoebase", "toebaser", "righttoe", "rtoe", "toer", "righttoes", "rtoes", "toesr", "rightfootball", "rfootball", "footballr"]
    };

    for (const [canonical, synList] of Object.entries(synonyms)) {
        map[canonical] = canonical;
        for (const syn of synList) map[syn] = canonical;
    }

    const fingers = ["thumb", "index", "middle", "ring", "pinky"];
    const sides = [ { canon: "lefthand", shorts: ["l", "left"] }, { canon: "righthand", shorts: ["r", "right"] } ];

    sides.forEach(side => {
        fingers.forEach(finger => {
            for (let i = 1; i <= 4; i++) {
                const canonical = `${side.canon}${finger}${i}`;
                map[canonical] = canonical;
                side.shorts.forEach(s => {
                    map[`${s}${finger}${i}`] = canonical;
                    map[`${finger}${i}${s}`] = canonical;
                    map[`${s}${finger}0${i}`] = canonical;
                    if (finger === "pinky") {
                        map[`${s}little${i}`] = canonical;
                        map[`little${i}${s}`] = canonical;
                        map[`${s}pinkie${i}`] = canonical;
                    }
                });
            }
        });
    });
    return { BONE_MAP: map, BONE_KEYS_SORTED: Object.keys(map).sort((a, b) => b.length - a.length) };
})();

const semanticNormalize = (name) => {
    if (!name) return "";
    let clean = name.split(/[:/|]/).pop();
    const lower = clean.toLowerCase();
    
    if (lower === "l_foot" || lower === "left_foot") return BONE_MAP["lefttoebase"];
    if (lower === "r_foot" || lower === "right_foot") return BONE_MAP["righttoebase"];
    if (lower === "l_ankle" || lower === "left_ankle") return BONE_MAP["leftfoot"];
    if (lower === "r_ankle" || lower === "right_ankle") return BONE_MAP["rightfoot"];
    if (lower === "l_hip" || lower === "left_hip") return BONE_MAP["leftupleg"];
    if (lower === "r_hip" || lower === "right_hip") return BONE_MAP["rightupleg"];
    if (lower === "l_collar" || lower === "left_collar") return BONE_MAP["leftshoulder"];
    if (lower === "r_collar" || lower === "right_collar") return BONE_MAP["rightshoulder"];
    if (lower === "l_shoulder" || lower === "left_shoulder") return BONE_MAP["leftarm"];
    if (lower === "r_shoulder" || lower === "right_shoulder") return BONE_MAP["rightarm"];

    clean = clean.replace(/^(b_|j_bip_|bip_|cc_base_|def_|org_|mch_|mixamorig\d*_?|mixamo_?)/i, "")
                 .replace(/(ik|fk|nub|end|twist\d*)$/i, "")
                 .replace(/[\s\-_.[\]]+/g, "")
                 .toLowerCase();

    if (BONE_MAP[clean]) return BONE_MAP[clean];
    for (const key of BONE_KEYS_SORTED) {
        if (clean.endsWith(key)) return BONE_MAP[key];
    }
    return clean;
};

// Extracted Character Class
class CharacterInstance {
    constructor(id, baseRig, THREE) {
        this.id = id;
        this.scene = window._YEDP_SKEL_UTILS.clone(baseRig);
        this.mixer = new THREE.AnimationMixer(this.scene);
        this.action = null;
        this.duration = 0; // in seconds
        this.loop = true;
        this.gender = 'M'; // Default
        this.hasFemaleMesh = false; // Track if a female mesh was successfully parsed

        this.poseMeshes = [];
        this.depthMeshesM = [];
        this.depthMeshesF = [];
        
        this.skeletonHelper = new THREE.SkeletonHelper(this.scene);
        this.skeletonHelper.visible = true;
        this.animFile = "none";

        // Spread them out slightly upon spawn
        this.scene.position.set((id - 1) * 1.0, 0, 0);

        this.scene.traverse((child) => {
            if(child.isMesh) {
                child.visible = true; 
                child.frustumCulled = false; 
                
                // V9.7 FIX: Ancestry String Builder
                // This solves Blender's 'Object vs Mesh Data' naming conflict 
                // by flattening the entire parent chain into a single string.
                let fullPath = "";
                let curr = child;
                while (curr && curr !== this.scene && curr !== null) {
                    if (curr.name) fullPath += curr.name.toLowerCase() + "|";
                    curr = curr.parent;
                }

                // Strip spaces and underscores, leaving pipes for node boundaries
                const n = fullPath.replace(/[\s_]/g, '');
                let category = "";

                if (n.includes("openpose") || n.includes("pose")) {
                    this.poseMeshes.push(child);
                    category = "Pose";
                } else if (n.includes("depthf") || n.includes("female") || n.includes("woman") || n.includes("|f|") || n.endsWith("f|")) {
                    this.hasFemaleMesh = true;
                    this.depthMeshesF.push(child);
                    child.visible = false; 
                    category = "Female Depth";
                } else if (n.includes("depth") || n.includes("male") || n.includes("man")) {
                    this.depthMeshesM.push(child);
                    child.visible = false; 
                    category = "Male Depth";
                } else {
                    // Prop Fallback: Any unclassified mesh (like a sword/hat) gets added to BOTH genders automatically
                    this.depthMeshesM.push(child);
                    this.depthMeshesF.push(child);
                    child.visible = false;
                    category = "Prop (Fallback)";
                }
                
                // Verbose logging for transparent debugging of the exporter output
                if(this.id === 1) {
                    console.log(`[Yedp] Parsed Mesh: "${child.name}" -> categorized as [${category}]`);
                }
            }
        });
    }

    get activeDepthMeshes() {
        return (this.gender === 'F' && this.hasFemaleMesh) ? this.depthMeshesF : this.depthMeshesM;
    }

    get inactiveDepthMeshes() {
        return (this.gender === 'F' && this.hasFemaleMesh) ? this.depthMeshesM : this.depthMeshesF;
    }

    destroy(scene) {
        scene.remove(this.scene);
        scene.remove(this.skeletonHelper);
        this.mixer.stopAllAction();
    }
}

class YedpViewport {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.baseUrl = new URL(".", import.meta.url).href;
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.transformControls = null;
        this.clock = null;
        
        this.baseRig = null;
        this.characters = []; // Array of CharacterInstance
        this.activeCharId = null; 
        this.charCounter = 0;

        this.gridHelper = null;
        this.axesHelper = null;
        this.semanticMap = new Map(); 

        // Camera Keys
        this.camKeys = { start: null, end: null, ease: 'linear' };
        
        // Materials
        this.depthMat = null;
        this.cannyMat = null; 
        this.normalMat = null; 
        this.originalMaterials = new Map();
        
        // Control States
        this.isDepthMode = false;
        this.userNear = 0.1;
        this.userFar = 10.0;
        this.defaultNear = 0.1;
        this.defaultFar = 100.0;

        this.isPlaying = false;
        this.isBaking = false; 
        
        // V9.9 FIX: Fully decouple global internal time from UI slider to avoid fraction truncation freezes
        this.globalTime = 0;
        
        this.renderWidth = 512;
        this.renderHeight = 512;
        this.availableAnimations = ["none"];

        // UI references
        this.uiSidebar = null;
        this.uiCharList = null;

        this.isHovered = false;
        this._handleKeyDown = this.handleKeyDown.bind(this);
        window.addEventListener('keydown', this._handleKeyDown);

        this.init();
    }

    async init() {
        try {
            const libs = await loadThreeJS();
            this.THREE = libs.THREE;
            this.OrbitControls = libs.OrbitControls;
            this.TransformControls = libs.TransformControls;
            this.GLTFLoaderClass = libs.GLTFLoader;
            this.FBXLoader = libs.FBXLoader; 
            this.BVHLoader = libs.BVHLoader;
            window._YEDP_SKEL_UTILS = libs.SkeletonUtils;

            this.depthMat = new this.THREE.MeshDepthMaterial({ depthPacking: this.THREE.BasicDepthPacking, skinning: true });
            this.cannyMat = new this.THREE.MeshMatcapMaterial({ matcap: this.createRimTexture(), skinning: true });
            this.normalMat = new this.THREE.MeshNormalMaterial({ skinning: true });

            // --- LAYOUT ---
            this.container.innerHTML = "";
            Object.assign(this.container.style, {
                display: "flex", flexDirection: "row", background: "#111",
                width: "100%", height: "100%", overflow: "hidden",
                border: "1px solid #333", borderRadius: "4px"
            });

            // MAIN VIEW AREA
            const mainCol = document.createElement("div");
            Object.assign(mainCol.style, { display: "flex", flexDirection: "column", flex: "1", minWidth: 0 });
            this.container.appendChild(mainCol);

            // SIDEBAR
            this.uiSidebar = document.createElement("div");
            Object.assign(this.uiSidebar.style, {
                width: "240px", flex: "0 0 240px", background: "#1a1a1a", borderLeft: "1px solid #333",
                display: "flex", flexDirection: "column", overflowY: "auto", padding: "8px", boxSizing: "border-box",
                gap: "10px", fontSize: "11px", color: "#ccc"
            });
            this.container.appendChild(this.uiSidebar);

            // INSIDE MAIN COL: Header, Viewport, Timeline
            const headerDiv = document.createElement("div");
            Object.assign(headerDiv.style, {
                height: "36px", flex: "0 0 36px", background: "#222", borderBottom: "1px solid #333",
                display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px"
            });
            mainCol.appendChild(headerDiv);

            const viewportDiv = document.createElement("div");
            viewportDiv.className = "yedp-vp-area";
            Object.assign(viewportDiv.style, { flex: "1 1 0", position: "relative", overflow: "hidden", background: "#000" });
            mainCol.appendChild(viewportDiv);

            const timelineDiv = document.createElement("div");
            Object.assign(timelineDiv.style, {
                height: "30px", flex: "0 0 30px", background: "#1a1a1a", borderTop: "1px solid #333",
                display: "flex", alignItems: "center", padding: "0 8px", gap: "8px"
            });
            mainCol.appendChild(timelineDiv);

            // --- 3D ENGINE SETUP ---
            this.clock = new this.THREE.Clock();
            this.scene = new this.THREE.Scene();
            this.scene.background = new this.THREE.Color(0x1a1a1a); 
            
            this.scene.add(new this.THREE.AmbientLight(0xffffff, 1.2));
            const dirLight = new this.THREE.DirectionalLight(0x00d2ff, 1.5);
            dirLight.position.set(5, 10, 7);
            this.scene.add(dirLight);

            this.gridHelper = new this.THREE.GridHelper(10, 10, 0x444444, 0x222222);
            this.scene.add(this.gridHelper);
            this.axesHelper = new this.THREE.AxesHelper(1);
            this.scene.add(this.axesHelper);

            this.camera = new this.THREE.PerspectiveCamera(45, 1, 0.01, 2000); 
            this.camera.position.set(0, 1.2, 4);
            
            this.renderer = new this.THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
            if (this.renderer.outputColorSpace) this.renderer.outputColorSpace = this.THREE.SRGBColorSpace;
            else this.renderer.outputEncoding = this.THREE.sRGBEncoding;
            
            viewportDiv.appendChild(this.renderer.domElement);
            Object.assign(this.renderer.domElement.style, { width: "100%", height: "100%", display: "block" });

            this.gate = document.createElement("div");
            this.gate.className = "yedp-resolution-gate";
            Object.assign(this.gate.style, {
                position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
                border: "2px solid #00d2ff", boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.65)", pointerEvents: "none", zIndex: "10",
                boxSizing: "content-box" // Critical fix for preventing collapse
            });
            viewportDiv.appendChild(this.gate);

            // Controls
            this.controls = new this.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.target.set(0, 1, 0);
            this.controls.enableDamping = true;

            this.transformControls = new this.TransformControls(this.camera, this.renderer.domElement);
            this.transformControls.addEventListener('dragging-changed', (event) => {
                this.controls.enabled = !event.value;
            });
            this.scene.add(this.transformControls);

            this.setupHeader(headerDiv);
            this.setupTimeline(timelineDiv);
            this.buildGizmoPanel(viewportDiv);
            this.buildSidebar();
            
            await this.fetchAnimations();
            await this.loadBaseRig();
            
            this.hookNodeWidgets();

            const resizeObserver = new ResizeObserver(() => this.onResize(viewportDiv));
            resizeObserver.observe(viewportDiv);

            // Start loop
            this.animate();

        } catch (e) {
            this.container.innerHTML = `<div style="color:red; padding:20px;">Init Error: ${e.message}</div>`;
        }
    }

    createRimTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 256;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, 256, 256);
        const grad = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
        grad.addColorStop(0.0, '#000000'); grad.addColorStop(0.75, '#000000'); 
        grad.addColorStop(0.85, '#666666'); grad.addColorStop(1.0, '#ffffff'); 
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(128, 128, 128, 0, Math.PI * 2); ctx.fill();
        const tex = new this.THREE.CanvasTexture(canvas);
        tex.colorSpace = this.THREE.SRGBColorSpace;
        return tex;
    }

    // --- UI SETUP ---
    setupHeader(div) {
        div.innerHTML = `
            <div style="display:flex; align-items:center; gap:6px;">
                <label style="color:#ccc; font-size:11px; cursor:pointer;"><input type="checkbox" id="chk-depth"> Depth</label>
                <div id="depth-ctrls" style="display:flex; align-items:center; gap:4px; opacity:0.5; transition:opacity 0.2s;">
                    <span style="color:#4ade80; font-size:10px; font-weight:bold;">NEAR DEPTH:</span>
                    <input id="inp-near" type="number" step="0.1" value="0.1" style="width:40px; background:#111; color:#4ade80; border:1px solid #4ade80; font-size:10px; padding:1px 2px; border-radius:2px; font-weight:bold;">
                    <span style="color:#4ade80; font-size:10px; font-weight:bold; margin-left:4px;">FAR DEPTH:</span>
                    <input id="inp-far" type="number" step="0.5" value="10.0" style="width:40px; background:#111; color:#4ade80; border:1px solid #4ade80; font-size:10px; padding:1px 2px; border-radius:2px; font-weight:bold;">
                </div>
                <label style="color:#666; font-size:11px; cursor:pointer;"><input type="checkbox" id="chk-skel" checked> Skel</label>
            </div>
            <div style="display:flex; gap:4px;">
                <span id="lbl-res" style="color:#00d2ff; font-family:monospace; font-size:10px; margin-right:5px; align-self:center;">512x512</span>
                <button id="btn-bake" style="border:1px solid #ff0055; color:#ff0055; background:transparent; padding:0px 6px; font-size:10px; cursor:pointer; border-radius:3px;">BAKE V9.10</button>
            </div>
        `;

        div.querySelector("#inp-near").onchange = (e) => { this.userNear = parseFloat(e.target.value); if(this.isDepthMode) this.updateCameraBounds(); };
        div.querySelector("#inp-far").onchange = (e) => { this.userFar = parseFloat(e.target.value); if(this.isDepthMode) this.updateCameraBounds(); };
        div.querySelector("#chk-depth").onchange = (e) => {
            const act = e.target.checked;
            div.querySelector("#depth-ctrls").style.opacity = act ? "1.0" : "0.5";
            this.toggleDepthMode(act);
        };
        div.querySelector("#chk-skel").onchange = (e) => {
            this.characters.forEach(c => { if(c.skeletonHelper) c.skeletonHelper.visible = e.target.checked; });
        };
        div.querySelector("#btn-bake").onclick = () => this.performBatchRender();
    }

    setupTimeline(div) {
        div.innerHTML = `
            <div style="display:flex; width:100%; align-items:center; gap:5px;">
                <button id="btn-play" style="background:none;border:none;color:#fff;cursor:pointer;width:20px;">▶</button>
                <input type="range" id="t-slider" min="0" max="100" value="0" step="1" style="flex:1;cursor:pointer;">
                <span id="t-time" style="font-family:monospace;font-size:10px;color:#888;min-width:70px;text-align:right;">0 / 0</span>
            </div>`;
        
        const btn = div.querySelector("#btn-play");
        const slider = div.querySelector("#t-slider");
        const lbl = div.querySelector("#t-time");
        
        btn.onclick = () => { this.isPlaying = !this.isPlaying; btn.innerText = this.isPlaying ? "⏸" : "▶"; };
        slider.onmousedown = () => { this.isDraggingSlider = true; this.isPlaying = false; btn.innerText = "▶"; };
        slider.onmouseup = () => { this.isDraggingSlider = false; };
        
        slider.oninput = (e) => {
            const frame = parseInt(e.target.value);
            const totalFrames = this.getWidgetValue("frame_count", 48);
            const fps = this.getWidgetValue("fps", 24);
            
            // V9.9 FIX: Set the exact float time from the scrubber
            this.globalTime = frame / fps;
            
            this.characters.forEach(c => {
                if(c.action && c.duration > 0) { 
                    c.action.time = c.loop ? (this.globalTime % c.duration) : Math.min(this.globalTime, c.duration);
                    c.mixer.update(0); 
                }
            });
            lbl.innerText = `${frame} / ${totalFrames}`;
            
            // Preview camera if scrubbing
            this.applyCameraKeyframes(frame / Math.max(1, totalFrames - 1));
        };
    }

    handleKeyDown(e) {
        if (!this.isHovered || !this.activeCharId || this.isBaking || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const k = e.key.toLowerCase();
        if (k === 'g') { this.transformControls.setMode("translate"); this.updateGizmoUI("translate"); }
        if (k === 'r') { this.transformControls.setMode("rotate"); this.updateGizmoUI("rotate"); }
        if (k === 's') { this.transformControls.setMode("scale"); this.updateGizmoUI("scale"); }
    }

    buildGizmoPanel(vpDiv) {
        this.container.addEventListener('mouseenter', () => this.isHovered = true);
        this.container.addEventListener('mouseleave', () => this.isHovered = false);

        this.gizmoBtns = {};
        const panel = document.createElement("div");
        Object.assign(panel.style, {
            position: "absolute", top: "10px", left: "10px", zIndex: "100",
            display: "flex", flexDirection: "column", gap: "6px",
            background: "rgba(20,20,20,0.8)", padding: "6px", borderRadius: "6px", border: "1px solid #333"
        });

        const createIconBtn = (id, svgPath, tooltip, onClick) => {
            const b = document.createElement("button");
            b.title = tooltip;
            Object.assign(b.style, {
                width: "32px", height: "32px", background: "#333", color: "#ccc",
                border: "1px solid #555", borderRadius: "4px", cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center", padding: "4px",
                transition: "all 0.1s"
            });
            b.innerHTML = `<svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${svgPath}</svg>`;
            b.onmouseover = () => { if(b.dataset.active !== "true") b.style.background = "#555"; };
            b.onmouseout = () => { if(b.dataset.active !== "true") b.style.background = "#333"; };
            b.onclick = () => { onClick(); };
            this.gizmoBtns[id] = b;
            return b;
        };

        const pathMove = `<path d="M5 9l-3 3 3 3M9 5l3-3 3 3M9 19l3 3 3-3M19 9l3 3-3 3M2 12h20M12 2v20"/>`;
        const pathRot = `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`;
        const pathScale = `<path d="M21 3l-6 6"/><path d="M21 3v6"/><path d="M21 3h-6"/><path d="M3 21l6-6"/><path d="M3 21v-6"/><path d="M3 21h6"/><path d="M14 10l-4 4"/>`;
        const pathDeselect = `<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>`;

        panel.append(
            createIconBtn("translate", pathMove, "Move (G)", () => { this.transformControls.setMode("translate"); this.updateGizmoUI("translate"); }),
            createIconBtn("rotate", pathRot, "Rotate (R)", () => { this.transformControls.setMode("rotate"); this.updateGizmoUI("rotate"); }),
            createIconBtn("scale", pathScale, "Scale (S)", () => { this.transformControls.setMode("scale"); this.updateGizmoUI("scale"); }),
            createIconBtn("deselect", pathDeselect, "Deselect", () => { this.transformControls.detach(); this.activeCharId = null; this.refreshSidebarHighlights(); this.updateGizmoUI("none"); })
        );
        vpDiv.appendChild(panel);
        this.updateGizmoUI("translate");
    }

    updateGizmoUI(mode) {
        Object.keys(this.gizmoBtns).forEach(k => {
            const b = this.gizmoBtns[k];
            if (k === mode) {
                b.dataset.active = "true";
                b.style.background = "#00d2ff";
                b.style.color = "#000";
                b.style.borderColor = "#00d2ff";
            } else {
                b.dataset.active = "false";
                b.style.background = "#333";
                b.style.color = "#ccc";
                b.style.borderColor = "#555";
            }
        });
    }

    buildSidebar() {
        const createBtn = (text, color="#444", hover="#555") => {
            const b = document.createElement("button");
            b.innerText = text;
            Object.assign(b.style, { background: color, color: "#fff", border: "1px solid #555", borderRadius: "3px", cursor: "pointer", padding: "4px", fontSize: "10px", flex: "1" });
            b.onmouseover = () => b.style.background = hover;
            b.onmouseout = () => b.style.background = color;
            return b;
        };

        const createCollapsible = (titleText, defaultOpen) => {
            const wrap = document.createElement("div");
            Object.assign(wrap.style, { background: "#222", borderRadius: "4px", marginBottom: "8px", border: "1px solid #333" });
            
            const head = document.createElement("div");
            Object.assign(head.style, { background: "#1a1a1a", padding: "8px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: defaultOpen ? "1px solid #333" : "none", borderTopLeftRadius: "4px", borderTopRightRadius: "4px" });
            
            const titleSpan = document.createElement("span");
            Object.assign(titleSpan.style, { fontWeight: "bold", color: "#aaa", fontSize: "11px", display: "flex", alignItems: "center", gap: "4px" });
            titleSpan.innerHTML = `<span style="color:#666; font-size:9px;">${defaultOpen ? '▼' : '▶'}</span> ${titleText}`;

            head.appendChild(titleSpan);

            const content = document.createElement("div");
            Object.assign(content.style, { padding: "8px", display: defaultOpen ? "flex" : "none", flexDirection: "column", gap: "6px" });

            head.onclick = (e) => {
                if (e.target.tagName === 'BUTTON') return; 
                const isOpen = content.style.display !== "none";
                content.style.display = isOpen ? "none" : "flex";
                head.style.borderBottom = isOpen ? "none" : "1px solid #333";
                titleSpan.innerHTML = `<span style="color:#666; font-size:9px;">${isOpen ? '▶' : '▼'}</span> ${titleText}`;
            };

            wrap.append(head, content);
            return { wrap, head, content };
        };

        // --- CAMERA SEQUENCE (Collapsible) ---
        const camCol = createCollapsible("Camera Sequence", false);
        
        const camRow1 = document.createElement("div"); camRow1.style.display = "flex"; camRow1.style.gap = "4px"; camRow1.style.marginBottom = "4px";
        const btnSetStart = createBtn("Set Start");
        const btnSetEnd = createBtn("Set End");
        
        btnSetStart.onclick = () => {
            this.camKeys.start = { 
                pos: this.camera.position.clone(), 
                quat: this.camera.quaternion.clone(),
                target: this.controls.target.clone()
            };
            btnSetStart.innerText = "Start Set ✓"; btnSetStart.style.borderColor = "#0f0";
        };
        btnSetEnd.onclick = () => {
            this.camKeys.end = { 
                pos: this.camera.position.clone(), 
                quat: this.camera.quaternion.clone(),
                target: this.controls.target.clone()
            };
            btnSetEnd.innerText = "End Set ✓"; btnSetEnd.style.borderColor = "#0f0";
        };
        
        const camRow2 = document.createElement("div"); camRow2.style.display = "flex"; camRow2.style.gap = "4px";
        const selEase = document.createElement("select");
        Object.assign(selEase.style, { flex: "1", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px" });
        ['linear', 'easeIn', 'easeOut', 'easeInOut'].forEach(e => selEase.add(new Option(e, e)));
        selEase.onchange = (e) => this.camKeys.ease = e.target.value;
        
        const btnClearCam = createBtn("Clear", "#522", "#733");
        btnClearCam.onclick = () => {
            this.camKeys.start = null; this.camKeys.end = null;
            btnSetStart.innerText = "Set Start"; btnSetStart.style.borderColor = "#555";
            btnSetEnd.innerText = "Set End"; btnSetEnd.style.borderColor = "#555";
        };

        camRow1.append(btnSetStart, btnSetEnd);
        camRow2.append(selEase, btnClearCam);
        camCol.content.append(camRow1, camRow2);
        this.uiSidebar.appendChild(camCol.wrap);

        // --- CHARACTERS (Collapsible) ---
        const charCol = createCollapsible("Characters", true);
        
        const btnAddChar = createBtn("+ Add Char", "#252", "#373");
        btnAddChar.style.flex = "none"; btnAddChar.style.padding = "2px 6px"; btnAddChar.style.fontSize = "9px";
        btnAddChar.onclick = (e) => { e.stopPropagation(); this.addCharacter(); };
        charCol.head.appendChild(btnAddChar);
        
        this.uiCharList = document.createElement("div");
        this.uiCharList.style.display = "flex"; this.uiCharList.style.flexDirection = "column"; this.uiCharList.style.gap = "6px";
        charCol.content.appendChild(this.uiCharList);

        this.uiSidebar.appendChild(charCol.wrap);
    }

    refreshSidebarHighlights() {
        Array.from(this.uiCharList.children).forEach(card => {
            const isActive = card.dataset.id == this.activeCharId;
            card.style.borderColor = isActive ? "#00d2ff" : "#444";
        });
    }

    renderCharacterCards() {
        this.uiCharList.innerHTML = "";
        this.characters.forEach(c => {
            const card = document.createElement("div");
            card.dataset.id = c.id;
            Object.assign(card.style, { background: "#222", border: "1px solid #444", borderRadius: "4px", padding: "6px" });
            
            const head = document.createElement("div");
            head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.marginBottom = "2px";
            head.innerHTML = `<span style="font-weight:bold; font-size:12px;">Char ${c.id}</span>`;
            
            const actBox = document.createElement("div"); actBox.style.display = "flex"; actBox.style.gap = "4px";
            const btnSel = document.createElement("button"); btnSel.innerText = "Select";
            Object.assign(btnSel.style, { background: "#444", color: "#fff", border: "1px solid #555", borderRadius: "2px", cursor: "pointer", fontSize: "9px" });
            btnSel.onclick = () => { 
                this.activeCharId = c.id; 
                this.transformControls.attach(c.scene); 
                this.refreshSidebarHighlights(); 
                if (this.updateGizmoUI) this.updateGizmoUI(this.transformControls.getMode());
            };
            
            const btnDel = document.createElement("button"); btnDel.innerText = "X";
            Object.assign(btnDel.style, { background: "#622", color: "#fff", border: "1px solid #555", borderRadius: "2px", cursor: "pointer", fontSize: "9px" });
            btnDel.onclick = () => this.removeCharacter(c.id);
            
            actBox.append(btnSel, btnDel);
            head.appendChild(actBox);
            
            // V9.7 Debug Info to show exactly what meshes got parsed successfully
            const meshInfo = document.createElement("div");
            meshInfo.style.fontSize = "9px";
            meshInfo.style.color = "#888";
            meshInfo.style.marginBottom = "4px";
            meshInfo.innerText = `[M:${c.depthMeshesM.length} | F:${c.depthMeshesF.length} | Pose:${c.poseMeshes.length}]`;
            
            const selAnim = document.createElement("select");
            Object.assign(selAnim.style, { width: "100%", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px", marginBottom: "4px" });
            this.availableAnimations.forEach(anim => selAnim.add(new Option(anim, anim)));
            selAnim.value = c.animFile;
            selAnim.onchange = (e) => this.loadAnimationForChar(c, e.target.value);
            
            const foot = document.createElement("div"); foot.style.display = "flex"; foot.style.justifyContent = "space-between"; foot.style.alignItems = "center";
            
            // Loop & Gender Box
            const loopBox = document.createElement("div");
            loopBox.style.display = "flex"; loopBox.style.alignItems = "center"; loopBox.style.gap = "6px";
            
            const btnGender = document.createElement("button");
            btnGender.innerText = c.gender;
            Object.assign(btnGender.style, {
                background: "#111", border: "1px solid #444", borderRadius: "3px", 
                cursor: "pointer", fontSize: "10px", padding: "1px 6px", fontWeight: "bold",
                color: c.gender === 'F' ? '#ff66b2' : '#66b2ff'
            });
            btnGender.onclick = () => {
                c.gender = c.gender === 'M' ? 'F' : 'M';
                btnGender.innerText = c.gender;
                btnGender.style.color = c.gender === 'F' ? '#ff66b2' : '#66b2ff';
                if (this.isDepthMode) this.toggleDepthMode(true);
            };

            const lblLoop = document.createElement("label"); lblLoop.style.cursor = "pointer"; lblLoop.style.display = "flex"; lblLoop.style.gap = "2px";
            const chkLoop = document.createElement("input"); chkLoop.type = "checkbox"; chkLoop.checked = c.loop;
            chkLoop.onchange = (e) => { c.loop = e.target.checked; if(c.action) { c.action.setLoop(c.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce); c.action.clampWhenFinished = !c.loop; }};
            lblLoop.append(chkLoop, "Loop");
            
            loopBox.append(btnGender, lblLoop);

            const lblDur = document.createElement("span");
            const fps = this.getWidgetValue("fps", 24);
            lblDur.innerText = c.duration > 0 ? `${Math.floor(c.duration * fps)}f` : "--";
            lblDur.id = `dur-${c.id}`;
            lblDur.style.color = "#888"; lblDur.style.fontFamily = "monospace";
            
            foot.append(loopBox, lblDur);
            card.append(head, meshInfo, selAnim, foot);
            this.uiCharList.appendChild(card);
        });
        this.refreshSidebarHighlights();
    }

    // --- CORE LOGIC ---
    async fetchAnimations() {
        try {
            const res = await api.fetchApi("/yedp/get_animations");
            const data = await res.json();
            if (data.files && data.files.length > 0) {
                // Ensure 'none' is always available to prevent UI default mismatches
                this.availableAnimations = ["none", ...data.files.filter(f => f !== "none")];
            }
        } catch(e) { console.error("Failed to fetch animations."); }
    }

    async loadBaseRig() {
        const loader = new this.GLTFLoaderClass();
        // V9.8 FIX: Cache Buster. Appending the current timestamp forces the browser 
        // to bypass the local cache and download the absolute newest version of your GLB.
        const rigUrl = new URL(`../Yedp_Rig.glb?t=${Date.now()}`, this.baseUrl).href;
        console.log("[Yedp] Loading Base Rig from:", rigUrl);
        const gltf = await loader.loadAsync(rigUrl);
        this.baseRig = gltf.scene;
        
        this.baseRig.traverse((child) => {
            if(child.isBone || child.type === "Bone" || child.isObject3D) {
                const normalized = semanticNormalize(child.name);
                if (normalized) this.semanticMap.set(normalized, child.name);
            }
        });
        
        // Spawn first character automatically
        this.addCharacter();
    }

    addCharacter() {
        if (this.characters.length >= 16) { alert("Maximum 16 characters recommended for WebGL performance."); return; }
        this.charCounter++;
        const newChar = new CharacterInstance(this.charCounter, this.baseRig, this.THREE);
        this.scene.add(newChar.scene);
        this.scene.add(newChar.skeletonHelper);
        this.characters.push(newChar);
        this.renderCharacterCards();
    }

    removeCharacter(id) {
        const idx = this.characters.findIndex(c => c.id === id);
        if (idx === -1) return;
        const c = this.characters[idx];
        if (this.activeCharId === id) { this.transformControls.detach(); this.activeCharId = null; }
        c.destroy(this.scene);
        this.characters.splice(idx, 1);
        this.renderCharacterCards();
    }

    async loadAnimationForChar(charObj, filename) {
        if(!filename || filename === "none") return;
        charObj.animFile = filename;
        const isFBX = filename.toLowerCase().endsWith(".fbx");
        const isBVH = filename.toLowerCase().endsWith(".bvh");
        // V9.8 FIX: Cache Buster for animations too, just in case you overwrite an FBX.
        const url = `/view?filename=${filename}&type=input&subfolder=yedp_anims&t=${Date.now()}`;
        
        try {
            let model;
            if(isFBX) model = await new this.FBXLoader().loadAsync(url);
            else if (isBVH) model = await new this.BVHLoader().loadAsync(url);
            else model = await new this.GLTFLoaderClass().loadAsync(url);

            let clip = isBVH ? model.clip : (model.animations?.[0] || model.scene?.animations?.[0] || model.asset?.animations?.[0]);

            if(clip) {
                const tracks = [];
                clip.tracks.forEach(t => {
                    const lastDot = t.name.lastIndexOf(".");
                    const prop = t.name.substring(lastDot + 1);
                    const fullBonePath = t.name.substring(0, lastDot);
                    const normalizedTrackBone = semanticNormalize(fullBonePath);
                    
                    if (prop === "scale") return; 

                    if(this.semanticMap.has(normalizedTrackBone)) {
                        const targetRealName = this.semanticMap.get(normalizedTrackBone);
                        const tc = t.clone(); 
                        
                        // V9.9 FIX: Removed completely the aggressive positional override.
                        // Root motions, jumps, and world-space paths are now perfectly preserved!
                        tc.name = `${targetRealName}.${prop}`;
                        tracks.push(tc);
                    }
                });

                const cleanClip = new this.THREE.AnimationClip(clip.name, clip.duration, tracks);
                charObj.mixer.stopAllAction();
                charObj.mixer.uncacheRoot(charObj.scene);
                
                charObj.action = charObj.mixer.clipAction(cleanClip);
                charObj.action.setLoop(charObj.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce);
                charObj.action.clampWhenFinished = !charObj.loop;
                charObj.action.reset().setEffectiveWeight(1).play();
                charObj.mixer.update(0); 

                charObj.duration = cleanClip.duration;
                
                // Update label dynamically using the container scope to ensure accuracy
                const lbl = this.container.querySelector(`#dur-${charObj.id}`);
                if (lbl) {
                    const fps = this.getWidgetValue("fps", 24);
                    lbl.innerText = `${Math.floor(charObj.duration * fps)}f`;
                }
                
                this.isPlaying = true;
                const btn = this.container.querySelector("#btn-play");
                if(btn) btn.innerText = "⏸";
            }
        } catch(e) { console.error("Anim Load Error:", e); }
    }

    applyCameraKeyframes(timeRatio) {
        if (!this.camKeys.start || !this.camKeys.end) return;
        let t = timeRatio;
        // Easing functions
        if (this.camKeys.ease === 'easeIn') t = t * t;
        else if (this.camKeys.ease === 'easeOut') t = t * (2 - t);
        else if (this.camKeys.ease === 'easeInOut') t = t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

        this.camera.position.lerpVectors(this.camKeys.start.pos, this.camKeys.end.pos, t);
        this.camera.quaternion.slerpQuaternions(this.camKeys.start.quat, this.camKeys.end.quat, t);
        
        // Fixed: Lerp OrbitControls target for panning and zooming support
        if (this.controls && this.camKeys.start.target && this.camKeys.end.target) {
            this.controls.target.lerpVectors(this.camKeys.start.target, this.camKeys.end.target, t);
        }
    }

    animate() {
        if (!this.renderer) return;
        requestAnimationFrame(() => this.animate());
        if (this.isBaking) return;

        const delta = this.clock.getDelta();
        const totalFrames = this.getWidgetValue("frame_count", 48);
        const fps = this.getWidgetValue("fps", 24);

        if (this.isPlaying) {
            // V9.9 FIX: Use decoupled internal time to prevent slider truncation logic.
            // This allows the animation to play perfectly smoothly regardless of fractional FPS steps.
            this.globalTime += delta;
            const totalDuration = totalFrames / fps;
            
            // Global Loop
            if (this.globalTime >= totalDuration) {
                this.globalTime = this.globalTime % totalDuration; 
            }
            
            const currentFrame = Math.floor(this.globalTime * fps);
            
            // Only update the slider visuals without triggering a fractional math error
            const slider = this.container.querySelector("#t-slider");
            if (slider && !this.isDraggingSlider) {
                slider.value = currentFrame;
            }
            
            const timeLabel = this.container.querySelector("#t-time");
            if (timeLabel) {
                timeLabel.innerText = `${currentFrame} / ${totalFrames}`;
            }

            const t = this.globalTime;
            
            this.characters.forEach(c => {
                if(c.action && c.duration > 0) {
                    c.action.time = c.loop ? (t % c.duration) : Math.min(t, c.duration);
                    c.mixer.update(0); // strict evaluate
                }
            });

            this.applyCameraKeyframes(currentFrame / Math.max(1, totalFrames - 1));
        }
        
        if (this.controls) this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    toggleDepthMode(active) {
        this.isDepthMode = active;
        this.characters.forEach(c => {
            // Keep inactive meshes hidden at all times
            c.inactiveDepthMeshes.forEach(m => m.visible = false);
            
            c.activeDepthMeshes.forEach(m => m.visible = active);
            c.poseMeshes.forEach(m => m.visible = !active);
            
            if (active) {
                 c.activeDepthMeshes.forEach(m => {
                     if(m.isMesh && !this.originalMaterials.has(m)) this.originalMaterials.set(m, m.material);
                     m.material = this.depthMat;
                 });
            } else {
                // Restore all materials unconditionally just to be safe
                c.depthMeshesM.forEach(m => { if(this.originalMaterials.has(m)) m.material = this.originalMaterials.get(m); });
                c.depthMeshesF.forEach(m => { if(this.originalMaterials.has(m)) m.material = this.originalMaterials.get(m); });
            }
        });

        if (active) this.updateCameraBounds();
        else this.resetCamera();
    }

    updateCameraBounds() {
        if(!this.camera) return;
        this.camera.near = Math.max(0.01, this.userNear);
        this.camera.far = Math.max(0.1, this.userFar);
        this.camera.updateProjectionMatrix();
    }
    resetCamera() {
        if(!this.camera) return;
        this.camera.near = this.defaultNear;
        this.camera.far = this.defaultFar;
        this.camera.updateProjectionMatrix();
    }

    hookNodeWidgets() {
        const updateDim = (w, val) => {
            if(w && w.name === "width") this.renderWidth = val; 
            else if(w && w.name === "height") this.renderHeight = val;
            
            const lbl = this.container.querySelector("#lbl-res");
            if(lbl) lbl.innerText = `${this.renderWidth}x${this.renderHeight}`;
            this.onResize(this.container.querySelector(".yedp-vp-area"));
        };

        const wWidget = this.node.widgets?.find(w => w.name === "width");
        const hWidget = this.node.widgets?.find(w => w.name === "height");
        
        if(wWidget) { 
            this.renderWidth = wWidget.value; 
            const orig = wWidget.callback; 
            wWidget.callback = v => { updateDim(wWidget, v); if(orig) orig(v); };
        }
        
        if(hWidget) { 
            this.renderHeight = hWidget.value; 
            const orig = hWidget.callback; 
            hWidget.callback = v => { updateDim(hWidget, v); if(orig) orig(v); };
        }
        
        updateDim();

        // Frame count & FPS updates max slider
        const slider = this.container.querySelector("#t-slider");
        const fWidget = this.node.widgets?.find(w => w.name === "frame_count");
        if(fWidget && slider) {
            slider.max = fWidget.value;
            const orig = fWidget.callback;
            fWidget.callback = v => { slider.max = v; if(orig) orig(v); };
        }
    }

    onResize(vpDiv) {
        if (this.isBaking || !this.renderer || !vpDiv || !this.camera) return;
        const w = vpDiv.clientWidth;
        const h = vpDiv.clientHeight;
        if (w && h) {
            this.renderer.setSize(w, h);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            
            const aspectContainer = w / h;
            const aspectTarget = this.renderWidth / this.renderHeight;
            let gw, gh;
            if (aspectContainer > aspectTarget) { gh = h - 20; gw = gh * aspectTarget; } 
            else { gw = w - 20; gh = gw / aspectTarget; }
            if(this.gate) { this.gate.style.width = `${gw}px`; this.gate.style.height = `${gh}px`; }
        }
    }

    getWidgetValue(name, defaultVal) {
        const w = this.node.widgets?.find(x => x.name === name);
        return w ? w.value : defaultVal;
    }

    async performBatchRender() {
        if (this.characters.length === 0) { alert("No characters added!"); return; }
        const THREE = this.THREE;
        const btn = this.container.querySelector('#btn-bake');
        btn.innerText = "PREPARING...";
        
        this.isBaking = true;
        this.isPlaying = false;
        
        // Hide gizmo
        this.transformControls.detach();
        this.activeCharId = null;
        this.refreshSidebarHighlights();

        const originalSize = new THREE.Vector2();
        this.renderer.getSize(originalSize);
        const originalAspect = this.camera.aspect;
        const originalZoom = this.camera.zoom;
        const originalBg = this.scene.background;
        
        const vpArea = this.container.querySelector(".yedp-vp-area");
        if (vpArea) {
            const vpW = vpArea.clientWidth; const vpH = vpArea.clientHeight;
            const vpAspect = vpW / vpH; const targetAspect = this.renderWidth / this.renderHeight;
            if (vpAspect < targetAspect) this.camera.zoom = originalZoom * (targetAspect / vpAspect);
            else this.camera.zoom = originalZoom;
        }

        this.renderer.setSize(this.renderWidth, this.renderHeight);
        this.camera.aspect = this.renderWidth / this.renderHeight;
        this.camera.updateProjectionMatrix();

        const frames = this.getWidgetValue("frame_count", 48);
        const fps = this.getWidgetValue("fps", 24);
        const step = 1.0 / fps;
        
        const results = { pose: [], depth: [], canny: [], normal: [] };

        // Determine current visibility intent
        const visSkel = this.container.querySelector("#chk-skel").checked;
        const toggleHelpers = (vis) => {
            if(this.gridHelper) this.gridHelper.visible = vis;
            if(this.axesHelper) this.axesHelper.visible = vis;
        };
        toggleHelpers(false);

        const setVisibility = (mode) => {
            const showPose = mode === 'pose';
            const showDepth = mode === 'depth' || mode === 'canny' || mode === 'normal';
            this.characters.forEach(c => {
                c.poseMeshes.forEach(m => m.visible = showPose);
                c.inactiveDepthMeshes.forEach(m => m.visible = false); // Force inactive meshes to stay hidden
                c.activeDepthMeshes.forEach(m => m.visible = showDepth);
                
                // Force hide skeleton helper during all bake passes so it never contaminates the OpenPose result
                c.skeletonHelper.visible = false; 
            });
        };

        const swapPoseToUnlit = () => {
            const originalMats = new Map();
            this.characters.forEach(c => {
                c.poseMeshes.forEach(child => {
                    if (child.isMesh && child.material) {
                        originalMats.set(child, child.material);
                        const oldColor = child.material.color || new THREE.Color(0xffffff);
                        const newMat = new THREE.MeshBasicMaterial({ color: oldColor, skinning: true });
                        if (child.material.map) { newMat.map = child.material.map; newMat.color.setHex(0xffffff); }
                        child.material = newMat;
                    }
                });
            });
            return () => {
                this.characters.forEach(c => {
                    c.poseMeshes.forEach(child => { if (originalMats.has(child)) child.material = originalMats.get(child); });
                });
            };
        };

        const compressCanvas = document.createElement("canvas");
        compressCanvas.width = this.renderWidth; compressCanvas.height = this.renderHeight;
        const compressCtx = compressCanvas.getContext("2d");

        const captureFrame = (array, mimeType = "image/png", quality = undefined) => {
            this.renderer.render(this.scene, this.camera);
            this.renderer.getContext().finish(); 
            if (mimeType === "image/jpeg") {
                compressCtx.fillStyle = "#000000";
                compressCtx.fillRect(0, 0, this.renderWidth, this.renderHeight);
                compressCtx.drawImage(this.renderer.domElement, 0, 0);
                array.push(compressCanvas.toDataURL(mimeType, quality));
            } else {
                array.push(this.renderer.domElement.toDataURL(mimeType));
            }
        };

        // RENDER LOOP
        for (let i = 0; i < frames; i++) {
            const time = i * step;
            const timeRatio = frames > 1 ? i / (frames - 1) : 0;
            
            // UI Progress Feedback
            btn.innerText = `BAKING ${i+1}/${frames}`;
            
            this.applyCameraKeyframes(timeRatio);

            this.characters.forEach(c => {
                if(c.action && c.duration > 0) { 
                    c.action.time = c.loop ? (time % c.duration) : Math.min(time, c.duration);
                    c.mixer.update(0); 
                }
                c.scene.updateMatrixWorld(true);
            });

            // PASS 1: OPENPOSE
            this.scene.background = new THREE.Color(0x000000); 
            setVisibility('pose');
            this.resetCamera(); 
            const restoreMaterials = swapPoseToUnlit(); 
            captureFrame(results.pose, "image/png"); 
            restoreMaterials(); 

            // PASS 2: DEPTH
            this.scene.background = new THREE.Color(0x000000);
            setVisibility('depth');
            this.camera.near = Math.max(0.01, this.userNear);
            this.camera.far = Math.max(0.1, this.userFar);
            this.camera.updateProjectionMatrix();

            const depthRestores = [];
            this.characters.forEach(c => {
                c.activeDepthMeshes.forEach(m => { depthRestores.push({mesh: m, mat: m.material}); m.material = this.depthMat; });
            });
            captureFrame(results.depth, "image/png");
            depthRestores.forEach(o => o.mesh.material = o.mat);

            // PASS 3: CANNY
            this.scene.background = new THREE.Color(0x000000); 
            setVisibility('canny'); 
            this.resetCamera();
            const cannyRestores = [];
            this.characters.forEach(c => {
                c.activeDepthMeshes.forEach(m => { cannyRestores.push({mesh: m, mat: m.material}); m.material = this.cannyMat; });
            });
            captureFrame(results.canny, "image/png");
            cannyRestores.forEach(o => o.mesh.material = o.mat);

            // PASS 4: NORMAL
            this.scene.background = new THREE.Color(0x000000); 
            setVisibility('normal'); 
            this.resetCamera();
            const normalRestores = [];
            this.characters.forEach(c => {
                c.activeDepthMeshes.forEach(m => { normalRestores.push({mesh: m, mat: m.material}); m.material = this.normalMat; });
            });
            captureFrame(results.normal, "image/png");
            normalRestores.forEach(o => o.mesh.material = o.mat);
            
            await new Promise(r => setTimeout(r, 10)); // tiny yield
        }
        
        // Restoration
        this.renderer.setSize(originalSize.width, originalSize.height);
        this.camera.aspect = originalAspect;
        this.camera.zoom = originalZoom; 
        
        if(this.isDepthMode) { this.camera.near = this.userNear; this.camera.far = this.userFar; } 
        else this.resetCamera();
        this.camera.updateProjectionMatrix();

        toggleHelpers(true); 
        this.scene.background = originalBg;
        this.isBaking = false;
        
        // Restore standard view
        const chkDepth = this.container.querySelector("#chk-depth").checked;
        this.toggleDepthMode(chkDepth);
        this.characters.forEach(c => { c.skeletonHelper.visible = visSkel; });
        
        // ---- CRITICAL NEW UPLOAD LOGIC TO PREVENT QUOTA CRASHES ----
        btn.innerText = "UPLOADING TO CACHE...";
        const clientDataWidget = this.node.widgets.find(w => w.name === "client_data");
        if (clientDataWidget) {
            try {
                // Upload massive payload to python memory to avoid localStorage serialization crash
                const response = await api.fetchApi("/yedp/upload_payload", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(results)
                });
                
                if (!response.ok) throw new Error("Upload failed");
                const resData = await response.json();
                
                // Store only the tiny ID string inside the widget (safe for localStorage!)
                clientDataWidget.value = resData.payload_id;
                console.log(`[Yedp] Batch Render Complete. Cached payload ID: ${resData.payload_id}`);
            } catch (err) {
                console.error("[Yedp] Memory cache upload failed, falling back to local string:", err);
                clientDataWidget.value = JSON.stringify(results);
            }
        }
        
        btn.innerText = "BAKE (DONE)";
        setTimeout(() => { btn.innerText = "BAKE V9.10"; }, 2000);
    }
}

app.registerExtension({
    name: "Yedp.ActionDirector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YedpActionDirector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const container = document.createElement("div");
                container.classList.add("yedp-container");
                container.style.width = "100%";
                container.style.height = "100%"; 
                
                const widget = this.addDOMWidget("3d_viewport", "vp", container, { serialize: false, hideOnZoom: false });
                widget.computeSize = (w) => [w, 0];
                
                setTimeout(() => {
                    const vp = new YedpViewport(this, container);
                    this.vp = vp; // Store reference for cleanup
                    const onResizeOrig = this.onResize;
                    this.onResize = function(size) {
                        if (onResizeOrig) onResizeOrig.call(this, size);
                        let usedHeight = 30; 
                        if (this.widgets) {
                            for (const w of this.widgets) {
                                if (w === widget) break;
                                usedHeight += w.last_h || 26; 
                            }
                        }
                        const safeHeight = Math.max(10, size[1] - usedHeight - 35);
                        container.style.height = safeHeight + "px";
                        container.style.maxHeight = "none";
                        vp.onResize(container.querySelector(".yedp-vp-area"));
                    };
                    
                    // Hide the raw JSON text widget like in the webcam node
                    const w = this.widgets?.find(w => w.name === "client_data");
                    if (w?.inputEl) w.inputEl.style.display = "none";
                }, 100);
                
                // Made the default UI slightly wider to comfortably fit the viewport + sidebar
                this.setSize([720, 600]);
                
                // Cleanup memory and animation loop when node is deleted
                this.onRemoved = function() {
                    if (this.vp) {
                        this.vp.isBaking = false;
                        this.vp.isPlaying = false;
                        if (this.vp._handleKeyDown) window.removeEventListener('keydown', this.vp._handleKeyDown);
                        if (this.vp.renderer) {
                            this.vp.renderer.dispose();
                            this.vp.renderer = null;
                        }
                    }
                };

                return r;
            };
        }
    }
});
