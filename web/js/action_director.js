import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

/** * YEDP ACTION DIRECTOR - V9.21 (Auto-Save & Hot-Reload Update)
 * - Added: ComfyUI Native Serialization. The entire 3D scene state (Camera, Keyframes, Chars, Envs, Lights, Settings) 
 * is now automatically saved into the ComfyUI workflow .json file.
 * - Added: "SYNC FOLDERS" button to dynamically hot-reload newly added files into dropdowns without reloading the browser.
 * - Added: Environments Support. Load GLTF/FBX props and scenes from `yedp_envs`.
 * - Added: Alembic-Style Physics Support via GLTF Morph Targets and Skeletal Animations natively supported in Environments.
 * - Added: Black and White Alpha Mask generation pass.
 * - Added: Custom Animated Camera import override (FBX/GLB support) from `yedp_cams`.
 */

const loadThreeJS = async () => {
    if (window._YEDP_THREE_CACHE) return window._YEDP_THREE_CACHE;

    return window._YEDP_THREE_CACHE = new Promise(async (resolve, reject) => {
        const baseUrl = new URL(".", import.meta.url).href;
        try {
            console.log("[Yedp] 正在初始化引擎 V9.21 (离线模式)...");
            
            const THREE = await import(new URL("./three.module.js", baseUrl).href);

            if (THREE.ColorManagement && typeof THREE.ColorManagement.colorSpaceToWorking !== 'function') {
                THREE.ColorManagement.colorSpaceToWorking = function (color, colorSpace) {
                    if (colorSpace === THREE.SRGBColorSpace || colorSpace === 'srgb') return color.convertSRGBToLinear();
                    return color;
                };
            } else if (!THREE.ColorManagement) {
                THREE.ColorManagement = { enabled: false, colorSpaceToWorking: function (color) { return color; } };
            }

            const { OrbitControls } = await import(new URL("./OrbitControls.js", baseUrl).href);
            const { TransformControls } = await import(new URL("./TransformControls.js", baseUrl).href);
            const { GLTFLoader } = await import(new URL("./GLTFLoader.js", baseUrl).href);
            await import(new URL("./fflate.module.js", baseUrl).href); 
            const { FBXLoader } = await import(new URL("./FBXLoader.js", baseUrl).href);
            const { BVHLoader } = await import(new URL("./BVHLoader.js", baseUrl).href);
            const { clone } = await import(new URL("./SkeletonUtils.js", baseUrl).href);

            resolve({ THREE, OrbitControls, TransformControls, GLTFLoader, FBXLoader, BVHLoader, SkeletonUtils: { clone } });
        } catch (e) {
            console.error("[Yedp] 引擎加载严重失败:", e);
            reject(e);
        }
    });
};

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

// --- CLASSES ---
class CharacterInstance {
    constructor(id, baseRig, THREE) {
        this.id = id;
        this.scene = window._YEDP_SKEL_UTILS.clone(baseRig);
        this.mixer = new THREE.AnimationMixer(this.scene);
        this.action = null;
        this.duration = 0; 
        this.loop = true;
        this.gender = 'M'; 
        this.hasFemaleMesh = false; 

        this.poseMeshes = [];
        this.depthMeshesM = [];
        this.depthMeshesF = [];
        
        this.skeletonHelper = new THREE.SkeletonHelper(this.scene);
        this.skeletonHelper.visible = true;
        this.animFile = "none";

        this.scene.position.set((id - 1) * 1.0, 0, 0);

        this.scene.traverse((child) => {
            if(child.isMesh || child.isSkinnedMesh) {
                child.visible = true; 
                child.frustumCulled = false; 
                child.castShadow = true;
                child.receiveShadow = true;
                
                let fullPath = "";
                let curr = child;
                while (curr && curr !== this.scene && curr !== null) {
                    if (curr.name) fullPath += curr.name.toLowerCase() + "|";
                    curr = curr.parent;
                }

                const n = fullPath.replace(/[\s_]/g, '');
                let category = "";

                if (n.includes("openpose") || n.includes("pose")) {
                    this.poseMeshes.push(child);
                    category = "Pose";
                    if (child.material) {
                        const processMat = (mat) => {
                            const oldColor = mat.color || new THREE.Color(0xffffff);
                            const newMat = new THREE.MeshBasicMaterial({ color: oldColor });
                            if (mat.map) { newMat.map = mat.map; newMat.color.setHex(0xffffff); }
                            return newMat;
                        };
                        if (Array.isArray(child.material)) child.material = child.material.map(processMat);
                        else child.material = processMat(child.material);
                    }

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
                    this.depthMeshesM.push(child);
                    this.depthMeshesF.push(child);
                    child.visible = false;
                    category = "Prop (Fallback)";
                }
            }
        });
    }

    get activeDepthMeshes() { return (this.gender === 'F' && this.hasFemaleMesh) ? this.depthMeshesF : this.depthMeshesM; }
    get inactiveDepthMeshes() { return (this.gender === 'F' && this.hasFemaleMesh) ? this.depthMeshesM : this.depthMeshesF; }

    destroy(scene) {
        scene.remove(this.scene);
        scene.remove(this.skeletonHelper);
        this.mixer.stopAllAction();
    }
}

class EnvironmentInstance {
    constructor(id, THREE) {
        this.id = id;
        this.THREE = THREE;
        this.group = new THREE.Group();
        this.mixer = new THREE.AnimationMixer(this.group);
        this.action = null;
        this.duration = 0;
        this.loop = true;
        this.envFile = "none";
        this.meshes = [];
    }

    destroy(scene) {
        scene.remove(this.group);
        this.mixer.stopAllAction();
    }
}

// --- MAIN VIEWPORT ---
class YedpViewport {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.baseUrl = new URL(".", import.meta.url).href;
        
        this.isInitialized = false;

        this.scene = null;
        this.camera = null;
        this.perspCam = null;
        this.orthoCam = null;
        this.isOrthographic = false;

        this.renderer = null;
        this.controls = null;
        this.transformControls = null;
        this.clock = null;
        
        this.baseRig = null;
        
        this.characters = []; 
        this.charCounter = 0;
        
        this.environments = [];
        this.envCounter = 0;

        this.lights = [];
        this.lightCounter = 0;

        this.selected = { obj: null, type: null, id: null };

        this.gridHelper = null;
        this.axesHelper = null;
        this.semanticMap = new Map(); 

        this.camKeys = { start: null, end: null, ease: 'linear' };
        this.cameraAnimGroup = null;
        this.cameraMixer = null;
        this.cameraAction = null;
        this.cameraAnimNode = null;
        this.importedCamProxy = null; 
        this.isCameraOverride = false;
        
        this.camOverrideOffset = { rx: 0, ry: 0, rz: 0 };
        this.camOverrideScale = 1.0;
        
        this.matsSkinned = null;
        this.matsStatic = null;
        this.originalMaterials = new Map();
        
        this.isShadedMode = false;
        this.isDepthMode = false;
        this.userNear = 0.1;
        this.userFar = 10.0;
        this.defaultNear = 0.1;
        this.defaultFar = 100.0;

        this.isPlaying = false;
        this.isBaking = false; 
        this.globalTime = 0;
        
        this.renderWidth = 512;
        this.renderHeight = 512;
        this.availableAnimations = ["none"];
        this.availableEnvs = ["none"];
        this.availableCams = ["none"];

        this.uiSidebar = null;
        this.uiCharList = null;
        this.uiEnvList = null;
        this.uiLightList = null;
        this.uiTransformInputs = {};

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

            const createMats = () => ({
                shaded: new this.THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.6, metalness: 0.1 }),
                depth: new this.THREE.MeshDepthMaterial({ depthPacking: this.THREE.BasicDepthPacking }),
                canny: new this.THREE.MeshMatcapMaterial({ matcap: this.createRimTexture() }),
                normal: new this.THREE.MeshNormalMaterial(),
                alpha: new this.THREE.MeshBasicMaterial({ color: 0xffffff })
            });
            this.matsSkinned = createMats();
            this.matsStatic = createMats();

            // --- LAYOUT ---
            this.container.innerHTML = "";
            Object.assign(this.container.style, {
                display: "flex", flexDirection: "row", background: "#111",
                width: "100%", height: "100%", overflow: "hidden",
                border: "1px solid #333", borderRadius: "4px",
                boxSizing: "border-box"
            });

            // MAIN VIEW AREA
            const mainCol = document.createElement("div");
            Object.assign(mainCol.style, { display: "flex", flexDirection: "column", flex: "1", minWidth: 0, overflow: "hidden" });
            this.container.appendChild(mainCol);

            // SIDEBAR
            this.uiSidebar = document.createElement("div");
            Object.assign(this.uiSidebar.style, {
                width: "240px", flex: "0 0 240px", background: "#1a1a1a", borderLeft: "1px solid #333",
                display: "flex", flexDirection: "column", overflowY: "auto", padding: "8px", boxSizing: "border-box",
                gap: "8px", fontSize: "11px", color: "#ccc"
            });
            this.container.appendChild(this.uiSidebar);

            // INSIDE MAIN COL
            const headerDiv = document.createElement("div");
            Object.assign(headerDiv.style, {
                height: "36px", flex: "0 0 36px", background: "#222", borderBottom: "1px solid #333",
                display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px",
                boxSizing: "border-box" 
            });
            mainCol.appendChild(headerDiv);

            const viewportDiv = document.createElement("div");
            viewportDiv.className = "yedp-vp-area";
            Object.assign(viewportDiv.style, { flex: "1 1 0", position: "relative", overflow: "hidden", background: "#000" });
            mainCol.appendChild(viewportDiv);

            const timelineDiv = document.createElement("div");
            Object.assign(timelineDiv.style, {
                height: "30px", flex: "0 0 30px", background: "#1a1a1a", borderTop: "1px solid #333",
                display: "flex", alignItems: "center", padding: "0 8px", gap: "8px",
                boxSizing: "border-box" 
            });
            mainCol.appendChild(timelineDiv);

            // --- 3D ENGINE SETUP ---
            this.clock = new this.THREE.Clock();
            this.scene = new this.THREE.Scene();
            this.scene.background = new this.THREE.Color(0x1a1a1a); 

            this.gridHelper = new this.THREE.GridHelper(10, 10, 0x444444, 0x222222);
            this.scene.add(this.gridHelper);
            this.axesHelper = new this.THREE.AxesHelper(1);
            this.scene.add(this.axesHelper);

            const floorGeo = new this.THREE.PlaneGeometry(50, 50);
            const floorMat = new this.THREE.ShadowMaterial({ opacity: 0.6 });
            this.floor = new this.THREE.Mesh(floorGeo, floorMat);
            this.floor.rotation.x = -Math.PI / 2;
            this.floor.receiveShadow = true;
            this.scene.add(this.floor);

            // Setup Cameras
            const aspect = this.renderWidth / this.renderHeight || 1;
            this.perspCam = new this.THREE.PerspectiveCamera(45, aspect, 0.01, 2000);
            this.perspCam.position.set(0, 1.2, 4);

            const d = 2.0; 
            this.orthoCam = new this.THREE.OrthographicCamera(-d * aspect, d * aspect, d, -d, 0.01, 2000);
            this.orthoCam.position.set(0, 1.2, 4);

            this.camera = this.perspCam;
            this.isOrthographic = false;
            
            this.cameraAnimGroup = new this.THREE.Group();
            this.cameraAnimGroup.visible = false;
            this.scene.add(this.cameraAnimGroup);

            this.importedCamProxy = new this.THREE.Group();
            const boxG = new this.THREE.BoxGeometry(0.2, 0.2, 0.4);
            const boxM = new this.THREE.MeshBasicMaterial({ color: 0x00d2ff, wireframe: true });
            const pBox = new this.THREE.Mesh(boxG, boxM);
            const coneG = new this.THREE.ConeGeometry(0.3, 0.5, 4);
            const coneM = new this.THREE.MeshBasicMaterial({ color: 0x00d2ff, wireframe: true });
            const pCone = new this.THREE.Mesh(coneG, coneM);
            pCone.rotation.x = -Math.PI / 2; pCone.rotation.y = Math.PI / 4; pCone.position.z = -0.4;
            this.importedCamProxy.add(pBox); this.importedCamProxy.add(pCone);
            this.scene.add(this.importedCamProxy);
            this.importedCamProxy.visible = false;
            
            this.renderer = new this.THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
            if (this.renderer.outputColorSpace) this.renderer.outputColorSpace = this.THREE.SRGBColorSpace;
            else this.renderer.outputEncoding = this.THREE.sRGBEncoding;
            
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = this.THREE.PCFSoftShadowMap;

            viewportDiv.appendChild(this.renderer.domElement);
            Object.assign(this.renderer.domElement.style, { width: "100%", height: "100%", display: "block" });

            this.gate = document.createElement("div");
            this.gate.className = "yedp-resolution-gate";
            Object.assign(this.gate.style, {
                position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
                border: "2px solid #00d2ff", boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.65)", pointerEvents: "none", zIndex: "10",
                boxSizing: "content-box" 
            });
            viewportDiv.appendChild(this.gate);

            // Controls
            this.controls = new this.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.target.set(0, 1, 0);
            this.controls.enableDamping = true;
            
            this.controls.addEventListener('change', () => {
                if (this.selected && this.selected.type === 'camera' && this.uiTransformInputs) {
                    const isTyping = Object.values(this.uiTransformInputs).some(inp => inp === document.activeElement);
                    if (!isTyping) this.updateTransformUIFromObject();
                }
            });

            this.transformControls = new this.TransformControls(this.camera, this.renderer.domElement);
            this.transformControls.addEventListener('dragging-changed', (event) => {
                this.controls.enabled = !event.value && !this.isCameraOverride;
            });
            this.transformControls.addEventListener('change', () => {
                this.updateTransformUIFromObject();
            });
            this.scene.add(this.transformControls);

            this.raycaster = new this.THREE.Raycaster();
            this.mouse = new this.THREE.Vector2();
            this.pointerDownPos = new this.THREE.Vector2();

            viewportDiv.addEventListener('pointerdown', (e) => {
                this.pointerDownPos.set(e.clientX, e.clientY);
            });

            viewportDiv.addEventListener('pointerup', (e) => {
                if (e.target !== this.renderer.domElement) return;
                if (this.transformControls.dragging || this.transformControls.axis) return;
                
                const dist = Math.hypot(e.clientX - this.pointerDownPos.x, e.clientY - this.pointerDownPos.y);
                if (dist > 5) return;

                const rect = viewportDiv.getBoundingClientRect();
                this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

                this.raycaster.setFromCamera(this.mouse, this.camera);

                const interactables = [];
                const objMap = new Map();

                this.characters.forEach(c => {
                    if (c.scene.visible) {
                        interactables.push(c.scene);
                        c.scene.traverse(child => { if(child.isMesh || child.isSkinnedMesh) objMap.set(child, { obj: c, type: 'character', id: c.id }); });
                    }
                });
                
                this.environments.forEach(env => {
                    if (env.group.visible) {
                        interactables.push(env.group);
                        env.group.traverse(child => { if(child.isMesh || child.isSkinnedMesh) objMap.set(child, { obj: env, type: 'environment', id: env.id }); });
                    }
                });
                
                this.lights.forEach(l => {
                    if (l.helper && l.helper.visible) {
                        interactables.push(l.helper);
                        l.helper.traverse(child => { objMap.set(child, { obj: l, type: 'light', id: l.id }); });
                    }
                });

                const intersects = this.raycaster.intersectObjects(interactables, true);
                const hit = intersects.find(i => i.object.visible);

                if (hit && objMap.has(hit.object)) {
                    const match = objMap.get(hit.object);
                    this.selectObject(match.obj, match.type, match.id);
                } else {
                    this.selectObject(null, null, null);
                }
            });

            await this.fetchAnimations();
            await this.fetchEnvs();
            await this.fetchCams(); 

            this.setupHeader(headerDiv);
            this.setupTimeline(timelineDiv);
            this.buildGizmoPanel(viewportDiv);
            this.buildViewNav(viewportDiv);
            this.buildSidebar();
            
            this.addLight("ambient");
            this.addLight("directional");
            const dl = this.lights[1].group;
            dl.position.set(2, 4, 3);
            dl.lookAt(0, 0, 0); 
            
            await this.loadBaseRig();
            
            this.hookNodeWidgets();

            const resizeObserver = new ResizeObserver(() => this.onResize(viewportDiv));
            resizeObserver.observe(viewportDiv);

            // Signal initialization is complete. If ComfyUI saved a state, load it now!
            this.isInitialized = true;
            if (this.node.saved_scene_state) {
                await this.loadScene(this.node.saved_scene_state);
            }

            this.animate();

        } catch (e) {
            this.container.innerHTML = `<div style="color:red; padding:20px;">初始化错误: ${e.message}</div>`;
        }
    }

    createRimTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 256;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, 256, 256);
        const grad = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
        
        grad.addColorStop(0.0, '#000000'); 
        grad.addColorStop(0.88, '#000000'); 
        grad.addColorStop(0.90, '#ffffff'); 
        grad.addColorStop(1.0, '#ffffff'); 
        
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(128, 128, 128, 0, Math.PI * 2); ctx.fill();
        const tex = new this.THREE.CanvasTexture(canvas);
        tex.colorSpace = this.THREE.SRGBColorSpace;
        return tex;
    }

// --- COMfyUI STATE SERIALIZATION ---
    serializeScene() {
        // 【关键修复】：加上安全保护。如果相机还没加载出来，直接返回空状态，防止报错卡死！
        if (!this.camera || !this.controls || !this.isInitialized) {
            return JSON.stringify({ version: 1, note: "initializing" });
        }

        return JSON.stringify({
            version: 1,
            camera: {
                pos: this.camera.position ? this.camera.position.toArray() : [0, 1.2, 4],
                rot: this.camera.rotation ? this.camera.rotation.toArray() : [0, 0, 0],
                target: this.controls.target ? this.controls.target.toArray() : [0, 1, 0],
                fov: this.perspCam ? this.perspCam.fov : 45,
                isOrtho: this.isOrthographic,
                camOverrideOffset: this.camOverrideOffset,
                camOverrideScale: this.camOverrideScale,
                isCameraOverride: this.isCameraOverride,
                camKeys: {
                    start: this.camKeys.start ? { pos: this.camKeys.start.pos.toArray(), quat: this.camKeys.start.quat.toArray(), zoom: this.camKeys.start.zoom } : null,
                    end: this.camKeys.end ? { pos: this.camKeys.end.pos.toArray(), quat: this.camKeys.end.quat.toArray(), zoom: this.camKeys.end.zoom } : null,
                    ease: this.camKeys.ease
                }
            },
            lights: this.lights.map(l => ({
                type: l.type, color: l.color, intensity: l.intensity,
                range: l.range, angle: l.angle, castShadow: l.castShadow,
                pos: l.group.position.toArray(), rot: l.group.rotation.toArray()
            })),
            characters: this.characters.map(c => ({
                pos: c.scene.position.toArray(), rot: c.scene.rotation.toArray(), scl: c.scene.scale.toArray(),
                gender: c.gender, loop: c.loop, animFile: c.animFile
            })),
            environments: this.environments.map(e => ({
                pos: e.group.position.toArray(), rot: e.group.rotation.toArray(), scl: e.group.scale.toArray(),
                loop: e.loop, envFile: e.envFile
            })),
            settings: {
                isShadedMode: this.isShadedMode,
                isDepthMode: this.isDepthMode,
                userNear: this.userNear,
                userFar: this.userFar,
                customCamAnim: this.container.querySelector("#sel-cam-anim")?.value || "none"
            }
        });
    }

    async loadScene(stateStr) {
        if (!stateStr) return;
        try {
            console.log("[Yedp] 正在从 ComfyUI 工作流加载已保存的场景状态...");
            const state = JSON.parse(stateStr);
            
            // 1. Wipe Defaults
            const oldChars = [...this.characters]; oldChars.forEach(c => this.removeCharacter(c.id));
            const oldEnvs = [...this.environments]; oldEnvs.forEach(e => this.removeEnvironment(e.id));
            const oldLights = [...this.lights]; oldLights.forEach(l => this.removeLight(l.id));

            // 2. Load General Settings
            if (state.settings) {
                this.isShadedMode = state.settings.isShadedMode || false;
                this.isDepthMode = state.settings.isDepthMode || false;
                this.userNear = state.settings.userNear || 0.1;
                this.userFar = state.settings.userFar || 10.0;
                
                const chkS = this.container.querySelector("#chk-shaded"); if(chkS) chkS.checked = this.isShadedMode;
                const chkD = this.container.querySelector("#chk-depth"); if(chkD) chkD.checked = this.isDepthMode;
                const inpN = this.container.querySelector("#inp-near"); if(inpN) inpN.value = this.userNear;
                const inpF = this.container.querySelector("#inp-far"); if(inpF) inpF.value = this.userFar;
                if (this.isDepthMode) this.container.querySelector("#depth-ctrls").style.opacity = "1.0";
            }

            // 3. Load Camera
            if (state.camera) {
                this.isOrthographic = state.camera.isOrtho || false;
                const chkOrtho = this.container.querySelector("#chk-ortho"); if(chkOrtho) chkOrtho.checked = this.isOrthographic;
                
                this.perspCam.fov = state.camera.fov || 45;
                const fovSld = this.container.querySelector("#inp-cam-fov-sld"); if(fovSld) fovSld.value = this.perspCam.fov;
                const fovVal = this.container.querySelector("#inp-cam-fov-val"); if(fovVal) fovVal.value = this.perspCam.fov;
                this.perspCam.updateProjectionMatrix();

                this.camera = this.isOrthographic ? this.orthoCam : this.perspCam;
                this.camera.position.fromArray(state.camera.pos || [0, 1.2, 4]);
                this.camera.rotation.fromArray(state.camera.rot || [0, 0, 0]);
                
                if (state.camera.target) this.controls.target.fromArray(state.camera.target);
                else {
                    const forward = new this.THREE.Vector3(0, 0, -1).applyEuler(this.camera.rotation);
                    this.controls.target.copy(this.camera.position).add(forward.multiplyScalar(4));
                }
                
                this.camOverrideOffset = state.camera.camOverrideOffset || { rx:0, ry:0, rz:0 };
                this.camOverrideScale = state.camera.camOverrideScale || 1.0;
                this.cameraAnimGroup.scale.setScalar(this.camOverrideScale);
                
                const inpRx = this.container.querySelector("#inp-cam-rx"); if(inpRx) inpRx.value = this.camOverrideOffset.rx;
                const inpRy = this.container.querySelector("#inp-cam-ry"); if(inpRy) inpRy.value = this.camOverrideOffset.ry;
                const inpRz = this.container.querySelector("#inp-cam-rz"); if(inpRz) inpRz.value = this.camOverrideOffset.rz;
                const inpSc = this.container.querySelector("#inp-cam-scl"); if(inpSc) inpSc.value = this.camOverrideScale;

                this.isCameraOverride = state.camera.isCameraOverride || false;
                const chkOvr = this.container.querySelector("#cam-override-chk"); if(chkOvr) chkOvr.checked = this.isCameraOverride;
                this.controls.enabled = !this.isCameraOverride;
                
                if (state.camera.camKeys) {
                    this.camKeys.ease = state.camera.camKeys.ease || 'linear';
                    if (state.camera.camKeys.start) {
                        this.camKeys.start = { 
                            pos: new this.THREE.Vector3().fromArray(state.camera.camKeys.start.pos), 
                            quat: new this.THREE.Quaternion().fromArray(state.camera.camKeys.start.quat), 
                            zoom: state.camera.camKeys.start.zoom || 1 
                        };
                    }
                    if (state.camera.camKeys.end) {
                        this.camKeys.end = { 
                            pos: new this.THREE.Vector3().fromArray(state.camera.camKeys.end.pos), 
                            quat: new this.THREE.Quaternion().fromArray(state.camera.camKeys.end.quat), 
                            zoom: state.camera.camKeys.end.zoom || 1 
                        };
                    }
                }
                this.controls.object = this.camera;
                this.controls.update();
            }

            if (state.settings && state.settings.customCamAnim && state.settings.customCamAnim !== "none") {
                const selCam = this.container.querySelector("#sel-cam-anim");
                if (selCam) selCam.value = state.settings.customCamAnim;
                await this.loadCameraAnim(state.settings.customCamAnim);
            }

            // 4. Load Lights
            if (state.lights) {
                state.lights.forEach(l => {
                    this.addLight(l.type);
                    const newL = this.lights[this.lights.length - 1];
                    newL.color = l.color; newL.intensity = l.intensity; newL.range = l.range; newL.angle = l.angle; newL.castShadow = l.castShadow;
                    newL.group.position.fromArray(l.pos); newL.group.rotation.fromArray(l.rot);
                    this.updateLightType(newL);
                });
            }

            // 5. Load Characters
            if (state.characters) {
                for (const cData of state.characters) {
                    this.addCharacter();
                    const newC = this.characters[this.characters.length - 1];
                    newC.scene.position.fromArray(cData.pos);
                    newC.scene.rotation.fromArray(cData.rot);
                    newC.scene.scale.fromArray(cData.scl);
                    newC.gender = cData.gender || 'M';
                    newC.loop = cData.loop !== false;
                    if (cData.animFile && cData.animFile !== "none") await this.loadAnimationForChar(newC, cData.animFile);
                }
            }

            // 6. Load Environments
            if (state.environments) {
                for (const eData of state.environments) {
                    this.addEnvironment();
                    const newE = this.environments[this.environments.length - 1];
                    newE.group.position.fromArray(eData.pos);
                    newE.group.rotation.fromArray(eData.rot);
                    newE.group.scale.fromArray(eData.scl);
                    newE.loop = eData.loop !== false;
                    if (eData.envFile && eData.envFile !== "none") await this.loadEnvironmentFile(newE, eData.envFile);
                }
            }
            
            this.updateVisibilities();
            this.renderCharacterCards();
            this.renderEnvironmentCards();
            this.renderLightCards();
            this.forceUpdateFrame();

        } catch (e) {
            console.error("[Yedp] 加载保存的场景状态失败:", e);
        }
    }

    forceUpdateFrame() {
        const totalFrames = this.getWidgetValue("frame_count", 48);
        const slider = this.container.querySelector("#t-slider");
        const currentFrame = slider ? parseInt(slider.value) : 0;
        
        const timeLabel = this.container.querySelector("#t-time");
        if (timeLabel) timeLabel.innerText = `${currentFrame} / ${totalFrames}`;

        this.evaluateAnimations(this.globalTime);
        this.applyCameraKeyframes(currentFrame / Math.max(1, totalFrames - 1));
        
        this.characters.forEach(c => c.scene.updateMatrixWorld(true));
        this.environments.forEach(e => e.group.updateMatrixWorld(true));
        
        if (this.controls) this.controls.update();
    }

    // --- UI SETUP ---
    setupHeader(div) {
        div.innerHTML = `
            <div style="display:flex; align-items:center; gap:8px;">
                <label style="color:#ccc; font-size:11px; cursor:pointer;"><input type="checkbox" id="chk-shaded"> 着色 (Shaded)</label>
                <div style="width:1px; height:16px; background:#444;"></div>
                <label style="color:#ccc; font-size:11px; cursor:pointer;"><input type="checkbox" id="chk-depth"> 深度 (Depth)</label>
                <div id="depth-ctrls" style="display:flex; align-items:center; gap:4px; opacity:0.5; transition:opacity 0.2s;">
                    <span style="color:#4ade80; font-size:10px; font-weight:bold;">近端:</span>
                    <input id="inp-near" type="number" step="0.1" value="0.1" style="width:40px; background:#111; color:#4ade80; border:1px solid #4ade80; font-size:10px; padding:1px 2px; border-radius:2px; font-weight:bold;">
                    <span style="color:#4ade80; font-size:10px; font-weight:bold; margin-left:4px;">远端:</span>
                    <input id="inp-far" type="number" step="0.5" value="10.0" style="width:40px; background:#111; color:#4ade80; border:1px solid #4ade80; font-size:10px; padding:1px 2px; border-radius:2px; font-weight:bold;">
                </div>
                <div style="width:1px; height:16px; background:#444;"></div>
                <label style="color:#666; font-size:11px; cursor:pointer;"><input type="checkbox" id="chk-skel" checked> 骨骼 (Skel)</label>
            </div>
            <div style="display:flex; gap:4px; align-items:center;">
                <span id="lbl-res" style="color:#00d2ff; font-family:monospace; font-size:10px; margin-right:5px; align-self:center;">512x512</span>
                <button id="btn-refresh" style="border:1px solid #4ade80; color:#4ade80; background:transparent; padding:0px 6px; font-size:10px; cursor:pointer; border-radius:3px;" title="刷新文件列表">↻ 同步文件夹</button>
                <button id="btn-bake-frame" style="border:1px solid #ffaa00; color:#ffaa00; background:transparent; padding:0px 6px; font-size:10px; cursor:pointer; border-radius:3px;">单帧烘焙</button>
                <button id="btn-bake" style="border:1px solid #ff0055; color:#ff0055; background:transparent; padding:0px 6px; font-size:10px; cursor:pointer; border-radius:3px;">全部烘焙 V9.21</button>
            </div>
        `;

        const chkShaded = div.querySelector("#chk-shaded");
        const chkDepth = div.querySelector("#chk-depth");

        chkShaded.onchange = (e) => { 
            if (e.target.checked && chkDepth.checked) {
                chkDepth.checked = false;
                this.isDepthMode = false;
                div.querySelector("#depth-ctrls").style.opacity = "0.5";
            }
            this.isShadedMode = e.target.checked; 
            this.updateVisibilities(); 
            this.forceUpdateFrame();
        };
        
        chkDepth.onchange = (e) => {
            if (e.target.checked && chkShaded.checked) {
                chkShaded.checked = false;
                this.isShadedMode = false;
            }
            this.isDepthMode = e.target.checked;
            div.querySelector("#depth-ctrls").style.opacity = this.isDepthMode ? "1.0" : "0.5";
            this.updateVisibilities();
            this.forceUpdateFrame();
        };

        div.querySelector("#inp-near").onchange = (e) => { this.userNear = parseFloat(e.target.value); if(this.isDepthMode) { this.updateCameraBounds(); this.forceUpdateFrame(); } };
        div.querySelector("#inp-far").onchange = (e) => { this.userFar = parseFloat(e.target.value); if(this.isDepthMode) { this.updateCameraBounds(); this.forceUpdateFrame(); } };
        
        div.querySelector("#chk-skel").onchange = (e) => {
            this.characters.forEach(c => { if(c.skeletonHelper) c.skeletonHelper.visible = e.target.checked; });
            this.forceUpdateFrame();
        };
        
        // NEW SYNC BUTTON LOGIC
        div.querySelector("#btn-refresh").onclick = async () => {
            const btn = div.querySelector("#btn-refresh");
            btn.innerText = "同步中...";
            await this.fetchAnimations();
            await this.fetchEnvs();
            await this.fetchCams();
            
            this.renderCharacterCards();
            this.renderEnvironmentCards();
            
            const selCam = this.container.querySelector("#sel-cam-anim");
            if (selCam) {
                const currentVal = selCam.value;
                selCam.innerHTML = "";
                this.availableCams.forEach(anim => selCam.add(new Option(anim, anim)));
                selCam.value = this.availableCams.includes(currentVal) ? currentVal : "none";
            }
            btn.innerText = "↻ 同步文件夹";
        };

        div.querySelector("#btn-bake-frame").onclick = () => this.performBake(true);
        div.querySelector("#btn-bake").onclick = () => this.performBake(false);
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
        
        btn.onclick = () => { this.isPlaying = !this.isPlaying; btn.innerText = this.isPlaying ? "⏸" : "▶"; };
        slider.onmousedown = () => { this.isDraggingSlider = true; this.isPlaying = false; btn.innerText = "▶"; };
        slider.onmouseup = () => { this.isDraggingSlider = false; };
        
        slider.oninput = (e) => {
            const frame = parseInt(e.target.value);
            const fps = this.getWidgetValue("fps", 24);
            this.globalTime = frame / fps;
            this.forceUpdateFrame();
        };
    }

    evaluateAnimations(t) {
        this.characters.forEach(c => {
            if(c.action && c.duration > 0) { 
                c.action.time = c.loop ? (t % c.duration) : Math.min(t, c.duration);
                c.mixer.update(0); 
            }
        });
        this.environments.forEach(e => {
            if(e.action && e.duration > 0) { 
                e.action.time = e.loop ? (t % e.duration) : Math.min(t, e.duration);
                e.mixer.update(0); 
            }
        });
        if (this.cameraAction && this.cameraMixer) {
            const d = this.cameraAction.getClip().duration;
            this.cameraAction.time = (t % d);
            this.cameraMixer.update(0);
        }
    }

    handleKeyDown(e) {
        if (!this.isHovered || !this.selected.obj || this.isBaking || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const k = e.key.toLowerCase();
        if (k === 'g') { this.transformControls.setMode("translate"); this.updateGizmoUI("translate"); }
        if (k === 'r') { this.transformControls.setMode("rotate"); this.updateGizmoUI("rotate"); }
        if (k === 's' && (this.selected.type === 'character' || this.selected.type === 'environment')) { 
            this.transformControls.setMode("scale"); this.updateGizmoUI("scale"); 
        }
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
            createIconBtn("translate", pathMove, "移动 (G)", () => { this.transformControls.setMode("translate"); this.updateGizmoUI("translate"); }),
            createIconBtn("rotate", pathRot, "旋转 (R)", () => { this.transformControls.setMode("rotate"); this.updateGizmoUI("rotate"); }),
            createIconBtn("scale", pathScale, "缩放 (S)", () => { if(['character', 'environment'].includes(this.selected.type)) { this.transformControls.setMode("scale"); this.updateGizmoUI("scale"); } }),
            createIconBtn("deselect", pathDeselect, "取消选择", () => { this.selectObject(null, null, null); })
        );
        vpDiv.appendChild(panel);
        this.updateGizmoUI("translate");
    }

    updateGizmoUI(mode) {
        Object.keys(this.gizmoBtns).forEach(k => {
            const b = this.gizmoBtns[k];
            if (k === mode) {
                b.dataset.active = "true"; b.style.background = "#00d2ff"; b.style.color = "#000"; b.style.borderColor = "#00d2ff";
            } else {
                b.dataset.active = "false"; b.style.background = "#333"; b.style.color = "#ccc"; b.style.borderColor = "#555";
            }
            if (k === 'scale' && !['character', 'environment'].includes(this.selected.type)) {
                b.style.opacity = "0.3"; b.style.pointerEvents = "none";
            } else if (k === 'scale') {
                b.style.opacity = "1.0"; b.style.pointerEvents = "auto";
            }
        });
    }

    buildViewNav(vpDiv) {
        const nav = document.createElement("div");
        Object.assign(nav.style, {
            position: "absolute", top: "10px", right: "10px", zIndex: "100",
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px",
            background: "rgba(20,20,20,0.8)", padding: "6px", borderRadius: "6px", border: "1px solid #333"
        });

        const createViewBtn = (lbl, dir) => {
            const b = document.createElement("button");
            b.innerText = lbl;
            Object.assign(b.style, { background: "#222", color: "#ccc", border: "1px solid #444", borderRadius: "3px", cursor: "pointer", fontSize: "9px", padding: "4px 8px", fontWeight: "bold", transition: "all 0.1s" });
            b.onmouseover = () => { b.style.background = "#00d2ff"; b.style.color = "#000"; b.style.borderColor = "#00d2ff"; };
            b.onmouseout = () => { b.style.background = "#222"; b.style.color = "#ccc"; b.style.borderColor = "#444"; };
            b.onclick = () => {
                const dist = this.camera.position.distanceTo(this.controls.target);
                const tgt = this.controls.target;
                const p = this.camera.position;
                switch(dir) {
                    case 'top': p.set(tgt.x, tgt.y + dist, tgt.z); break;
                    case 'bottom': p.set(tgt.x, tgt.y - dist, tgt.z); break;
                    case 'front': p.set(tgt.x, tgt.y, tgt.z + dist); break;
                    case 'back': p.set(tgt.x, tgt.y, tgt.z - dist); break;
                    case 'left': p.set(tgt.x - dist, tgt.y, tgt.z); break;
                    case 'right': p.set(tgt.x + dist, tgt.y, tgt.z); break;
                    case 'reset': 
                        p.set(0, 1.2, 4);
                        tgt.set(0, 1, 0);
                        this.camera.zoom = 1;
                        if(this.isOrthographic) this.orthoCam.updateProjectionMatrix();
                        else this.perspCam.updateProjectionMatrix();
                        
                        const ovrChk = this.container.querySelector("#cam-override-chk");
                        if (this.isCameraOverride && ovrChk) {
                            ovrChk.checked = false;
                            this.isCameraOverride = false;
                            this.controls.enabled = true;
                        }
                        break;
                }
                this.controls.update(); 
                this.updateTransformUIFromObject();
                this.forceUpdateFrame();
            };
            return b;
        };
        nav.append(
            createViewBtn("顶视图", "top"), createViewBtn("底视图", "bottom"),
            createViewBtn("左视图", "left"), createViewBtn("右视图", "right"), 
            createViewBtn("前视图", "front"), createViewBtn("后视图", "back"),
            createViewBtn("重置视角", "reset")
        );
        nav.lastChild.style.gridColumn = "1 / span 2";
        nav.lastChild.style.borderColor = "#555";
        
        vpDiv.appendChild(nav);
    }

    buildSidebar() {
        const createBtn = (text, color="#444", hover="#555") => {
            const b = document.createElement("button"); b.innerText = text;
            Object.assign(b.style, { background: color, color: "#fff", border: "1px solid #555", borderRadius: "3px", cursor: "pointer", padding: "4px", fontSize: "10px", flex: "1" });
            b.onmouseover = () => b.style.background = hover;
            b.onmouseout = () => b.style.background = color;
            return b;
        };

        const createCollapsible = (titleText, defaultOpen) => {
            const wrap = document.createElement("div");
            Object.assign(wrap.style, { background: "#222", borderRadius: "4px", border: "1px solid #333" });
            const head = document.createElement("div");
            Object.assign(head.style, { 
                background: "#16252d", borderLeft: "3px solid #00d2ff", padding: "8px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", 
                borderBottom: defaultOpen ? "1px solid #333" : "none", borderTopRightRadius: "4px", borderBottomRightRadius: defaultOpen ? "0px" : "4px" 
            });
            const titleSpan = document.createElement("span");
            Object.assign(titleSpan.style, { fontWeight: "bold", color: "#fff", fontSize: "11px", display: "flex", alignItems: "center", gap: "6px" });
            titleSpan.innerHTML = `<span style="color:#00d2ff; font-size:9px; width:10px; display:inline-block; text-align:center;">${defaultOpen ? '▼' : '▶'}</span> <span style="display:flex; align-items:center; gap:4px;">${titleText}</span>`;
            head.appendChild(titleSpan);
            const content = document.createElement("div");
            Object.assign(content.style, { padding: "8px", display: defaultOpen ? "flex" : "none", flexDirection: "column", gap: "6px" });
            head.onclick = (e) => {
                if (['BUTTON', 'INPUT', 'SELECT', 'OPTION'].includes(e.target.tagName)) return; 
                const isOpen = content.style.display !== "none";
                content.style.display = isOpen ? "none" : "flex";
                head.style.borderBottom = isOpen ? "none" : "1px solid #333";
                head.style.borderBottomRightRadius = isOpen ? "4px" : "0px";
                titleSpan.innerHTML = `<span style="color:#00d2ff; font-size:9px; width:10px; display:inline-block; text-align:center;">${isOpen ? '▶' : '▼'}</span> <span style="display:flex; align-items:center; gap:4px;">${titleText}</span>`;
            };
            wrap.append(head, content);
            return { wrap, head, content };
        };

        // UI SVG Icons
        const iconTransform = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`;
        const iconCamera = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>`;
        const iconLighting = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
        const iconChars = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
        const iconEnv = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>`;

        // TRANSFORM
        const transCol = createCollapsible(`${iconTransform} 变换 (Transform)`, true);
        const tRow = (lbl, keys) => {
            const r = document.createElement("div"); r.style.display = "flex"; r.style.gap = "4px"; r.style.alignItems = "center";
            const l = document.createElement("span"); l.innerText = lbl; l.style.width = "20px"; l.style.color = "#888"; r.appendChild(l);
            keys.forEach(k => {
                const inp = document.createElement("input"); inp.type = "number"; inp.step = "0.1";
                Object.assign(inp.style, { flex: "1", width: "0", background: "#111", color: "#fff", border: "1px solid #444", fontSize: "10px", padding: "2px" });
                inp.onchange = () => { this.updateObjectFromTransformUI(); this.forceUpdateFrame(); };
                this.uiTransformInputs[k] = inp; r.appendChild(inp);
            });
            return r;
        };
        transCol.content.append(tRow("位置", ['px', 'py', 'pz']), tRow("旋转", ['rx', 'ry', 'rz']), tRow("缩放", ['sx', 'sy', 'sz']));
        this.uiSidebar.appendChild(transCol.wrap);

        // CAMERA
        const camCol = createCollapsible(`${iconCamera} 相机 (Camera)`, false);
        const camSettingsRow = document.createElement("div");
        Object.assign(camSettingsRow.style, { display: "flex", gap: "6px", alignItems: "center", marginBottom: "4px", flexWrap: "wrap" });
        const btnSelCam = createBtn("选择相机", "#334", "#445");
        btnSelCam.style.flex = "0 0 auto"; btnSelCam.onclick = () => this.selectObject(this.camera, 'camera', 'main');

        const lblOrtho = document.createElement("label");
        Object.assign(lblOrtho.style, { cursor: "pointer", display: "flex", gap: "4px", color: "#ccc", fontSize: "10px", alignItems: "center" });
        const chkOrtho = document.createElement("input"); chkOrtho.type = "checkbox"; chkOrtho.checked = this.isOrthographic;
        chkOrtho.id = "chk-ortho";
        lblOrtho.append(chkOrtho, "正交透视");

        const fovContainer = document.createElement("div");
        Object.assign(fovContainer.style, { display: "flex", gap: "4px", alignItems: "center", flex: "1", transition: "opacity 0.2s", minWidth: "0" });
        const lblFov = document.createElement("span"); lblFov.innerText = "视野(FOV)"; Object.assign(lblFov.style, {fontSize: "10px", color: "#888"});
        const sldFov = document.createElement("input"); sldFov.type = "range"; sldFov.min = 10; sldFov.max = 120; sldFov.value = this.perspCam.fov; 
        sldFov.id = "inp-cam-fov-sld";
        Object.assign(sldFov.style, { flex: "1", width: "0", minWidth: "30px" });
        const valFov = document.createElement("input"); valFov.type = "number"; valFov.min = 10; valFov.max = 120; valFov.value = this.perspCam.fov; 
        valFov.id = "inp-cam-fov-val";
        Object.assign(valFov.style, { fontSize: "10px", color: "#00d2ff", width: "36px", background: "#111", border: "1px solid #444", padding: "1px 2px", borderRadius: "2px", textAlign: "right" });
        
        fovContainer.append(lblFov, sldFov, valFov);
        camSettingsRow.append(btnSelCam, lblOrtho, fovContainer);

        chkOrtho.onchange = (e) => {
            this.isOrthographic = e.target.checked;
            fovContainer.style.opacity = this.isOrthographic ? "0.3" : "1.0";
            sldFov.disabled = this.isOrthographic; valFov.disabled = this.isOrthographic;

            const oldCam = this.isOrthographic ? this.perspCam : this.orthoCam;
            this.camera = this.isOrthographic ? this.orthoCam : this.perspCam;
            this.camera.position.copy(oldCam.position); this.camera.quaternion.copy(oldCam.quaternion); this.camera.zoom = oldCam.zoom;

            this.controls.object = this.camera;
            if (this.selected.type !== 'camera') this.transformControls.camera = this.camera;
            if (this.selected.type === 'camera') this.selectObject(this.camera, 'camera', 'main');
            const vpArea = this.container.querySelector(".yedp-vp-area");
            if (vpArea) this.onResize(vpArea); 
            this.forceUpdateFrame();
        };
        sldFov.oninput = (e) => { valFov.value = e.target.value; this.perspCam.fov = parseFloat(e.target.value); this.perspCam.updateProjectionMatrix(); this.forceUpdateFrame(); };
        valFov.oninput = (e) => { sldFov.value = e.target.value; this.perspCam.fov = parseFloat(e.target.value); this.perspCam.updateProjectionMatrix(); this.forceUpdateFrame(); };

        const camRow1 = document.createElement("div"); camRow1.style.display = "flex"; camRow1.style.gap = "4px"; camRow1.style.marginBottom = "4px";
        const btnSetStart = createBtn("设为起点"); const btnSetEnd = createBtn("设为终点");
        btnSetStart.onclick = () => {
            this.camKeys.start = { pos: this.camera.position.clone(), quat: this.camera.quaternion.clone(), target: this.controls.target.clone(), zoom: this.camera.zoom };
            btnSetStart.innerText = "起点已设 ✓"; btnSetStart.style.borderColor = "#0f0";
        };
        btnSetEnd.onclick = () => {
            this.camKeys.end = { pos: this.camera.position.clone(), quat: this.camera.quaternion.clone(), target: this.controls.target.clone(), zoom: this.camera.zoom };
            btnSetEnd.innerText = "终点已设 ✓"; btnSetEnd.style.borderColor = "#0f0";
        };
        
        const camRow2 = document.createElement("div"); camRow2.style.display = "flex"; camRow2.style.gap = "4px";
        const selEase = document.createElement("select");
        Object.assign(selEase.style, { flex: "1", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px" });
        const easeTexts = {'linear':'线性', 'easeIn':'缓入', 'easeOut':'缓出', 'easeInOut':'平滑'};
        ['linear', 'easeIn', 'easeOut', 'easeInOut'].forEach(e => selEase.add(new Option(easeTexts[e], e)));
        selEase.onchange = (e) => { this.camKeys.ease = e.target.value; this.forceUpdateFrame(); };
        const btnClearCam = createBtn("清除关键帧", "#522", "#733");
        btnClearCam.onclick = () => {
            this.camKeys.start = null; this.camKeys.end = null;
            btnSetStart.innerText = "设为起点"; btnSetStart.style.borderColor = "#555";
            btnSetEnd.innerText = "设为终点"; btnSetEnd.style.borderColor = "#555";
        };
        camRow1.append(btnSetStart, btnSetEnd); camRow2.append(selEase, btnClearCam);

        // Custom Camera Import Block
        const camImportRow = document.createElement("div");
        Object.assign(camImportRow.style, { display: "flex", gap: "4px", alignItems: "center", marginTop: "8px", borderTop: "1px solid #333", paddingTop: "8px" });
        
        const lblOverride = document.createElement("label");
        Object.assign(lblOverride.style, { cursor: "pointer", display: "flex", gap: "2px", color: "#ccc", fontSize: "10px", alignItems: "center" });
        const chkOverride = document.createElement("input"); 
        chkOverride.type = "checkbox"; 
        chkOverride.id = "cam-override-chk";
        chkOverride.checked = this.isCameraOverride;
        chkOverride.onchange = (e) => { 
            this.isCameraOverride = e.target.checked; 
            this.controls.enabled = !this.isCameraOverride;
            this.forceUpdateFrame();
        };
        lblOverride.append(chkOverride, "覆盖控制");

        const selCamAnim = document.createElement("select");
        selCamAnim.id = "sel-cam-anim";
        Object.assign(selCamAnim.style, { flex: "1", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px" });
        this.availableCams.forEach(anim => selCamAnim.add(new Option(anim, anim))); 
        selCamAnim.onchange = (e) => { this.loadCameraAnim(e.target.value); };
        camImportRow.append(lblOverride, selCamAnim);

        // FIX: FBX Maya Camera Local Offset logic
        const camImportFixRow = document.createElement("div");
        Object.assign(camImportFixRow.style, { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px", marginTop: "4px" });
        
        const makeFixRow = (lbl, def, callback, id) => {
            const wrap = document.createElement("div");
            Object.assign(wrap.style, { display: "flex", gap: "2px", alignItems: "center", color: "#888", fontSize: "9px" });
            const inp = document.createElement("input"); inp.type = "number"; inp.step = "0.1"; inp.value = def;
            inp.id = id;
            Object.assign(inp.style, { width: "100%", background: "#111", color: "#fff", border: "1px solid #444", fontSize: "9px", padding: "1px" });
            inp.onchange = callback;
            const label = document.createElement("span"); label.innerText = lbl; label.style.width = "20px";
            wrap.append(label, inp);
            return wrap;
        };

        const rxWrap = makeFixRow("Rx", "0", (e) => { this.camOverrideOffset.rx = parseFloat(e.target.value)||0; this.forceUpdateFrame(); }, "inp-cam-rx");
        const ryWrap = makeFixRow("Ry", "0", (e) => { this.camOverrideOffset.ry = parseFloat(e.target.value)||0; this.forceUpdateFrame(); }, "inp-cam-ry");
        const rzWrap = makeFixRow("Rz", "0", (e) => { this.camOverrideOffset.rz = parseFloat(e.target.value)||0; this.forceUpdateFrame(); }, "inp-cam-rz");
        const scWrap = makeFixRow("Scl", "1.0", (e) => { this.camOverrideScale = parseFloat(e.target.value)||1.0; this.cameraAnimGroup.scale.setScalar(this.camOverrideScale); this.cameraAnimGroup.updateMatrixWorld(true); this.forceUpdateFrame(); }, "inp-cam-scl");
        scWrap.querySelector('input').step = "0.01";

        camImportFixRow.append(rxWrap, ryWrap, rzWrap, scWrap);

        camCol.content.append(camSettingsRow, camRow1, camRow2, camImportRow, camImportFixRow);
        this.uiSidebar.appendChild(camCol.wrap);

        // LIGHTING
        const lightCol = createCollapsible(`${iconLighting} 灯光 (Lighting)`, false);
        const btnAddLight = createBtn("+ 添加灯光", "#542", "#653");
        btnAddLight.style.flex = "none"; btnAddLight.style.padding = "2px 6px"; btnAddLight.style.fontSize = "9px";
        btnAddLight.onclick = (e) => { e.stopPropagation(); this.addLight(); };
        lightCol.head.appendChild(btnAddLight);
        this.uiLightList = document.createElement("div");
        this.uiLightList.style.display = "flex"; this.uiLightList.style.flexDirection = "column"; this.uiLightList.style.gap = "6px";
        lightCol.content.appendChild(this.uiLightList);
        this.uiSidebar.appendChild(lightCol.wrap);

        // CHARACTERS
        const charCol = createCollapsible(`${iconChars} 角色 (Characters)`, true);
        const btnAddChar = createBtn("+ 添加角色", "#252", "#373");
        btnAddChar.style.flex = "none"; btnAddChar.style.padding = "2px 6px"; btnAddChar.style.fontSize = "9px";
        btnAddChar.onclick = (e) => { e.stopPropagation(); this.addCharacter(); };
        charCol.head.appendChild(btnAddChar);
        this.uiCharList = document.createElement("div");
        this.uiCharList.style.display = "flex"; this.uiCharList.style.flexDirection = "column"; this.uiCharList.style.gap = "6px";
        charCol.content.appendChild(this.uiCharList);
        this.uiSidebar.appendChild(charCol.wrap);

        // ENVIRONMENTS
        const envCol = createCollapsible(`${iconEnv} 环境 (Environments)`, true);
        const btnAddEnv = createBtn("+ 添加环境", "#353", "#474");
        btnAddEnv.style.flex = "none"; btnAddEnv.style.padding = "2px 6px"; btnAddEnv.style.fontSize = "9px";
        btnAddEnv.onclick = (e) => { e.stopPropagation(); this.addEnvironment(); };
        envCol.head.appendChild(btnAddEnv);
        this.uiEnvList = document.createElement("div");
        this.uiEnvList.style.display = "flex"; this.uiEnvList.style.flexDirection = "column"; this.uiEnvList.style.gap = "6px";
        envCol.content.appendChild(this.uiEnvList);
        this.uiSidebar.appendChild(envCol.wrap);
        
        this.updateTransformUIFromObject();
    }

    // --- SELECTION & TRANSFORM ---
    selectObject(obj, type, id) {
        this.selected = { obj, type, id };
        
        if (!obj) {
            this.transformControls.detach();
        } else if (type === 'camera') {
            this.transformControls.detach(); 
        } else {
            this.transformControls.attach(type === 'character' ? obj.scene : obj.group);
            if (type === 'light' && this.transformControls.getMode() === 'scale') {
                this.transformControls.setMode('translate'); 
            }
        }
        
        this.updateGizmoUI(this.transformControls.getMode());
        this.refreshSidebarHighlights();
        this.updateTransformUIFromObject();
    }

    updateTransformUIFromObject() {
        const ui = this.uiTransformInputs;
        const s = this.selected;
        if (!s.obj) {
            Object.values(ui).forEach(inp => { inp.value = "0.0"; inp.disabled = true; inp.style.opacity = "0.3"; });
            return;
        }
        
        const tgt = s.type === 'character' ? s.obj.scene : (s.type === 'light' || s.type === 'environment' ? s.obj.group : s.obj);
        
        Object.values(ui).forEach(inp => { inp.disabled = false; inp.style.opacity = "1.0"; });
        
        ui.px.value = tgt.position.x.toFixed(2); ui.py.value = tgt.position.y.toFixed(2); ui.pz.value = tgt.position.z.toFixed(2);
        ui.rx.value = this.THREE.MathUtils.radToDeg(tgt.rotation.x).toFixed(1); 
        ui.ry.value = this.THREE.MathUtils.radToDeg(tgt.rotation.y).toFixed(1); 
        ui.rz.value = this.THREE.MathUtils.radToDeg(tgt.rotation.z).toFixed(1);
        
        if (s.type === 'camera' || s.type === 'light') {
            ui.sx.disabled = ui.sy.disabled = ui.sz.disabled = true;
            ui.sx.style.opacity = ui.sy.style.opacity = ui.sz.style.opacity = "0.3";
            ui.sx.value = ui.sy.value = ui.sz.value = "1.0";
        } else {
            ui.sx.value = tgt.scale.x.toFixed(2); ui.sy.value = tgt.scale.y.toFixed(2); ui.sz.value = tgt.scale.z.toFixed(2);
        }
    }

    updateObjectFromTransformUI() {
        const s = this.selected;
        if (!s.obj) return;
        const tgt = s.type === 'character' ? s.obj.scene : (s.type === 'light' || s.type === 'environment' ? s.obj.group : s.obj);
        const ui = this.uiTransformInputs;
        
        const px = parseFloat(ui.px.value)||0; const py = parseFloat(ui.py.value)||0; const pz = parseFloat(ui.pz.value)||0;
        const rx = this.THREE.MathUtils.degToRad(parseFloat(ui.rx.value)||0);
        const ry = this.THREE.MathUtils.degToRad(parseFloat(ui.ry.value)||0);
        const rz = this.THREE.MathUtils.degToRad(parseFloat(ui.rz.value)||0);

        if (s.type === 'camera') {
            const deltaPos = new this.THREE.Vector3(px, py, pz).sub(this.camera.position);
            this.camera.position.add(deltaPos);
            this.controls.target.add(deltaPos);
            const dist = this.camera.position.distanceTo(this.controls.target) || 1.0;
            this.camera.rotation.set(rx, ry, rz);
            const forward = new this.THREE.Vector3(0, 0, -1).applyEuler(this.camera.rotation);
            this.controls.target.copy(this.camera.position).add(forward.multiplyScalar(dist));
            this.controls.update();
        } else {
            tgt.position.set(px, py, pz); tgt.rotation.set(rx, ry, rz);
            if (s.type === 'character' || s.type === 'environment') tgt.scale.set(parseFloat(ui.sx.value)||1, parseFloat(ui.sy.value)||1, parseFloat(ui.sz.value)||1);
        }
    }

    refreshSidebarHighlights() {
        const resetStyles = (list, type) => {
            if(list) Array.from(list.children).forEach(card => {
                const isActive = this.selected.type === type && card.dataset.id == this.selected.id;
                card.style.borderColor = isActive ? "#00d2ff" : "#444";
            });
        };
        resetStyles(this.uiCharList, 'character');
        resetStyles(this.uiEnvList, 'environment');
        resetStyles(this.uiLightList, 'light');
    }

    // --- LOGIC: DATA FETCHING ---
    async fetchAnimations() {
        try {
            const res = await api.fetchApi("/yedp/get_animations");
            const data = await res.json();
            if (data.files && data.files.length > 0) this.availableAnimations = ["none", ...data.files.filter(f => f !== "none")];
        } catch(e) { console.error("Failed to fetch animations."); }
    }

    async fetchEnvs() {
        try {
            const res = await api.fetchApi("/yedp/get_envs");
            const data = await res.json();
            if (data.files && data.files.length > 0) this.availableEnvs = ["none", ...data.files.filter(f => f !== "none")];
        } catch(e) { console.error("Failed to fetch envs."); }
    }

    async fetchCams() {
        try {
            const res = await api.fetchApi("/yedp/get_cams");
            const data = await res.json();
            if (data.files && data.files.length > 0) this.availableCams = ["none", ...data.files.filter(f => f !== "none")];
        } catch(e) { console.error("Failed to fetch cams."); }
    }

    async loadBaseRig() {
        const loader = new this.GLTFLoaderClass();
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
        
        // Add defaults only if we aren't loading a saved scene
        if (!this.node.saved_scene_state) {
            this.addCharacter();
        }
    }

    // --- LOGIC: CAMERA ANIM ---
    async loadCameraAnim(filename) {
        if(!filename || filename === "none") {
            this.cameraMixer?.stopAllAction();
            this.cameraAction = null;
            this.cameraAnimNode = null;
            this.forceUpdateFrame();
            return;
        }
        const isFBX = filename.toLowerCase().endsWith(".fbx");
        const url = `/view?filename=${filename}&type=input&subfolder=yedp_cams&t=${Date.now()}`;
        try {
            const model = isFBX ? await new this.FBXLoader().loadAsync(url) : await new this.GLTFLoaderClass().loadAsync(url);
            let clip = isFBX ? model.animations?.[0] : (model.animations?.[0] || model.scene?.animations?.[0] || model.asset?.animations?.[0]);
            
            if (clip) {
                this.cameraAnimGroup.clear();
                let animTarget = model.scene || model;
                
                this.cameraAnimGroup.add(animTarget);
                this.cameraMixer = new this.THREE.AnimationMixer(animTarget);
                this.cameraAction = this.cameraMixer.clipAction(clip);
                this.cameraAction.play();
                
                this.cameraAnimNode = animTarget.getObjectByProperty('isCamera', true) || animTarget;
                console.log("[Yedp] 自定义相机动画已绑定至:", this.cameraAnimNode.name);
                this.forceUpdateFrame();
            }
        } catch(e) { console.error("Camera Anim Load Error:", e); }
    }

    // --- LOGIC: LIGHTING ---
    addLight(presetType = 'point') {
        this.lightCounter++;
        const id = this.lightCounter;
        const group = new this.THREE.Group(); group.position.set(0, 2, 2);
        const lObj = { id, group, helper: null, light: null, type: presetType, color: '#ffffff', intensity: 1.0, range: 10, angle: 45, castShadow: true };
        this.lights.push(lObj);
        this.scene.add(group);
        this.updateLightType(lObj);
        this.renderLightCards();
    }
    removeLight(id) {
        const idx = this.lights.findIndex(l => l.id === id);
        if (idx === -1) return;
        const l = this.lights[idx];
        if (this.selected.type === 'light' && this.selected.id === id) this.selectObject(null, null, null);
        this.scene.remove(l.group);
        this.lights.splice(idx, 1);
        this.renderLightCards();
    }
    updateLightType(lObj) {
        if (lObj.light) lObj.group.remove(lObj.light);
        if (lObj.helper) lObj.group.remove(lObj.helper);
        const c = new this.THREE.Color(lObj.color);
        switch(lObj.type) {
            case 'ambient': lObj.light = new this.THREE.AmbientLight(c, lObj.intensity); break;
            case 'directional': lObj.light = new this.THREE.DirectionalLight(c, lObj.intensity); break;
            case 'point': lObj.light = new this.THREE.PointLight(c, lObj.intensity, lObj.range); break;
            case 'spot': lObj.light = new this.THREE.SpotLight(c, lObj.intensity, lObj.range, this.THREE.MathUtils.degToRad(lObj.angle), 0.5, 1); break;
        }
        if (lObj.type === 'directional' || lObj.type === 'spot') {
            lObj.light.target.position.set(0, 0, -1); lObj.group.add(lObj.light.target);
        }
        const helperMat = new this.THREE.MeshBasicMaterial({color:0xffaa00, wireframe:true});
        if (lObj.type === 'point') lObj.helper = new this.THREE.Mesh(new this.THREE.SphereGeometry(0.2, 8, 8), helperMat);
        else if (lObj.type === 'spot') {
            lObj.helper = new this.THREE.Mesh(new this.THREE.ConeGeometry(0.3, 0.6, 8, 1, true), helperMat);
            lObj.helper.rotation.x = Math.PI / 2; lObj.helper.position.z = -0.3;
        } else if (lObj.type === 'directional') {
            lObj.helper = new this.THREE.Group();
            const shaft = new this.THREE.Mesh(new this.THREE.CylinderGeometry(0.02, 0.02, 0.4), helperMat);
            shaft.rotation.x = Math.PI / 2; shaft.position.z = -0.2;
            const head = new this.THREE.Mesh(new this.THREE.ConeGeometry(0.1, 0.2), helperMat);
            head.rotation.x = -Math.PI / 2; head.position.z = -0.5;
            lObj.helper.add(shaft, head);
        } else lObj.helper = new this.THREE.Group();
        
        if (lObj.type !== 'ambient') {
            lObj.light.castShadow = lObj.castShadow;
            if (lObj.light.shadow) { lObj.light.shadow.mapSize.width = 1024; lObj.light.shadow.mapSize.height = 1024; }
        }
        lObj.group.add(lObj.light); lObj.group.add(lObj.helper);
        lObj.helper.visible = lObj.type !== 'ambient'; 
        this.forceUpdateFrame();
    }
    renderLightCards() {
        this.uiLightList.innerHTML = "";
        this.lights.forEach(l => {
            const card = document.createElement("div"); card.dataset.id = l.id;
            Object.assign(card.style, { background: "#222", border: "1px solid #444", borderRadius: "4px", padding: "6px", cursor: "pointer", transition: "border-color 0.2s" });
            
            card.onclick = (e) => {
                if (['button', 'input', 'select', 'option'].includes(e.target.tagName.toLowerCase())) return;
                this.selectObject(l, 'light', l.id);
            };

            const head = document.createElement("div"); head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.marginBottom = "4px";
            const selType = document.createElement("select");
            Object.assign(selType.style, { background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "1px" });
            const lightTexts = {'ambient':'环境光', 'directional':'平行光', 'point':'点光源', 'spot':'聚光灯'};
            ['ambient', 'directional', 'point', 'spot'].forEach(t => selType.add(new Option(lightTexts[t], t)));
            selType.value = l.type; selType.onchange = (e) => { l.type = e.target.value; this.updateLightType(l); this.renderLightCards(); };
            
            const btnDel = document.createElement("button"); btnDel.innerText = "X"; Object.assign(btnDel.style, { background: "#622", color: "#fff", border: "1px solid #555", borderRadius: "2px", cursor: "pointer", fontSize: "9px" }); btnDel.onclick = () => this.removeLight(l.id);
            head.append(selType, btnDel);

            const ctrls = document.createElement("div"); ctrls.style.display = "grid"; ctrls.style.gridTemplateColumns = "1fr 1fr"; ctrls.style.gap = "4px";
            const wrap = (lbl, elem) => { const d = document.createElement("div"); d.style.display="flex"; d.style.alignItems="center"; d.style.gap="4px"; const s = document.createElement("span"); s.innerText = lbl; s.style.fontSize="9px"; s.style.color="#888"; s.style.width="20px"; d.append(s, elem); return d; };

            const inpCol = document.createElement("input"); inpCol.type = "color"; inpCol.value = l.color; inpCol.style.width = "100%"; inpCol.style.height = "16px"; inpCol.style.padding = "0"; inpCol.style.border = "none"; inpCol.onchange = (e) => { l.color = e.target.value; this.updateLightType(l); };
            const inpInt = document.createElement("input"); inpInt.type = "number"; inpInt.step = "0.1"; inpInt.value = l.intensity; Object.assign(inpInt.style, { width:"100%", background:"#111", color:"#fff", border:"1px solid #444", fontSize:"10px" }); inpInt.onchange = (e) => { l.intensity = parseFloat(e.target.value); this.updateLightType(l); };
            ctrls.append(wrap("颜色", inpCol), wrap("强度", inpInt));

            if (l.type === 'point' || l.type === 'spot') {
                const inpRng = document.createElement("input"); inpRng.type = "number"; inpRng.step = "1"; inpRng.value = l.range; Object.assign(inpRng.style, { width:"100%", background:"#111", color:"#fff", border:"1px solid #444", fontSize:"10px" }); inpRng.onchange = (e) => { l.range = parseFloat(e.target.value); this.updateLightType(l); }; ctrls.append(wrap("范围", inpRng));
            }
            if (l.type === 'spot') {
                const inpAng = document.createElement("input"); inpAng.type = "number"; inpAng.step = "1"; inpAng.value = l.angle; Object.assign(inpAng.style, { width:"100%", background:"#111", color:"#fff", border:"1px solid #444", fontSize:"10px" }); inpAng.onchange = (e) => { l.angle = parseFloat(e.target.value); this.updateLightType(l); }; ctrls.append(wrap("角度", inpAng));
            }
            if (l.type !== 'ambient') {
                const lblShad = document.createElement("label"); lblShad.style.fontSize="9px"; lblShad.style.color="#ccc"; lblShad.style.display="flex"; lblShad.style.gap="2px"; lblShad.style.alignItems="center";
                const chkShad = document.createElement("input"); chkShad.type = "checkbox"; chkShad.checked = l.castShadow; chkShad.onchange = (e) => { l.castShadow = e.target.checked; this.updateLightType(l); };
                lblShad.append(chkShad, "阴影"); ctrls.append(lblShad);
            }
            card.append(head, ctrls); this.uiLightList.appendChild(card);
        });
        this.refreshSidebarHighlights();
    }

    // --- LOGIC: ENVIRONMENTS ---
    addEnvironment() {
        this.envCounter++;
        const newEnv = new EnvironmentInstance(this.envCounter, this.THREE);
        this.scene.add(newEnv.group);
        this.environments.push(newEnv);
        this.renderEnvironmentCards();
    }

    removeEnvironment(id) {
        const idx = this.environments.findIndex(e => e.id === id);
        if (idx === -1) return;
        const e = this.environments[idx];
        if (this.selected.type === 'environment' && this.selected.id === id) this.selectObject(null, null, null);
        e.destroy(this.scene);
        this.environments.splice(idx, 1);
        this.renderEnvironmentCards();
    }

    async loadEnvironmentFile(envObj, filename) {
        const info = this.container.querySelector(`#env-mesh-info-${envObj.id}`);
        if(info) info.innerText = `[加载中...]`;

        if(!filename || filename === "none") {
            envObj.group.clear();
            envObj.meshes = [];
            envObj.mixer.stopAllAction();
            envObj.action = null;
            if(info) info.innerText = `[网格: 0]`;
            this.forceUpdateFrame();
            return;
        }
        envObj.envFile = filename;
        const isFBX = filename.toLowerCase().endsWith(".fbx");
        const url = `/view?filename=${filename}&type=input&subfolder=yedp_envs&t=${Date.now()}`;
        try {
            const model = isFBX ? await new this.FBXLoader().loadAsync(url) : await new this.GLTFLoaderClass().loadAsync(url);
            
            envObj.group.clear();
            envObj.meshes = [];
            
            let targetObj = isFBX ? model : model.scene;
            envObj.group.add(targetObj);
            
            targetObj.traverse((child) => {
                child.visible = true; // Force all nested groups to be visible
                if(child.isMesh || child.isSkinnedMesh || child.type === 'Mesh' || child.type === 'SkinnedMesh') {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    child.frustumCulled = false; 
                    
                    if (child.material) {
                        const fixMat = (mat) => {
                            if (mat.transparent && mat.opacity < 0.01) {
                                mat.transparent = false;
                                mat.opacity = 1.0;
                            }
                            mat.side = this.THREE.DoubleSide; 
                        };
                        if (Array.isArray(child.material)) child.material.forEach(fixMat);
                        else fixMat(child.material);
                    }
                    
                    envObj.meshes.push(child);
                    if(!this.originalMaterials.has(child)) this.originalMaterials.set(child, child.material);
                }
            });
            
            if(info) info.innerText = `[网格: ${envObj.meshes.length}]`;

            let clip = isFBX ? model.animations?.[0] : (model.animations?.[0] || model.scene?.animations?.[0] || model.asset?.animations?.[0]);
            if(clip) {
                envObj.mixer = new this.THREE.AnimationMixer(targetObj);
                envObj.action = envObj.mixer.clipAction(clip);
                envObj.action.setLoop(envObj.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce);
                envObj.action.clampWhenFinished = !envObj.loop;
                envObj.action.reset().setEffectiveWeight(1).play();
                envObj.duration = clip.duration;
            } else {
                envObj.action = null;
                envObj.duration = 0;
            }
            
            const lbl = this.container.querySelector(`#env-dur-${envObj.id}`);
            if (lbl) {
                const fps = this.getWidgetValue("fps", 24);
                lbl.innerText = envObj.duration > 0 ? `${Math.floor(envObj.duration * fps)}f` : "静止";
            }
            
            this.updateVisibilities();
            this.forceUpdateFrame();

        } catch(e) { 
            console.error("Env Load Error:", e); 
            envObj.group.clear();
            if(info) info.innerHTML = `<span style="color:red;">[加载错误]</span>`;
        }
    }

    renderEnvironmentCards() {
        this.uiEnvList.innerHTML = "";
        this.environments.forEach(e => {
            const card = document.createElement("div"); card.dataset.id = e.id;
            Object.assign(card.style, { background: "#222", border: "1px solid #444", borderRadius: "4px", padding: "6px", cursor: "pointer", transition: "border-color 0.2s" });
            
            card.onclick = (evt) => {
                if (['button', 'input', 'select', 'option'].includes(evt.target.tagName.toLowerCase())) return;
                this.selectObject(e, 'environment', e.id);
            };

            const head = document.createElement("div"); head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.marginBottom = "2px";
            head.innerHTML = `<span style="font-weight:bold; font-size:12px;">环境 ${e.id}</span>`;
            
            const btnDel = document.createElement("button"); btnDel.innerText = "X"; Object.assign(btnDel.style, { background: "#622", color: "#fff", border: "1px solid #555", borderRadius: "2px", cursor: "pointer", fontSize: "9px" }); btnDel.onclick = () => this.removeEnvironment(e.id);
            head.appendChild(btnDel);
            
            const meshInfo = document.createElement("div");
            meshInfo.style.fontSize = "9px";
            meshInfo.style.color = "#888";
            meshInfo.style.marginBottom = "4px";
            meshInfo.id = `env-mesh-info-${e.id}`;
            meshInfo.innerText = `[网格: ${e.meshes.length}]`;
            
            const selEnv = document.createElement("select");
            Object.assign(selEnv.style, { width: "100%", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px", marginBottom: "4px" });
            this.availableEnvs.forEach(env => selEnv.add(new Option(env, env)));
            selEnv.value = e.envFile;
            selEnv.onchange = (evt) => this.loadEnvironmentFile(e, evt.target.value);

            const foot = document.createElement("div"); foot.style.display = "flex"; foot.style.justifyContent = "space-between"; foot.style.alignItems = "center";
            const lblLoop = document.createElement("label"); lblLoop.style.cursor = "pointer"; lblLoop.style.display = "flex"; lblLoop.style.gap = "2px";
            const chkLoop = document.createElement("input"); chkLoop.type = "checkbox"; chkLoop.checked = e.loop;
            chkLoop.onchange = (evt) => { e.loop = evt.target.checked; if(e.action) { e.action.setLoop(e.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce); e.action.clampWhenFinished = !e.loop; }};
            lblLoop.append(chkLoop, "循环 (动画)");
            
            const lblDur = document.createElement("span");
            const fps = this.getWidgetValue("fps", 24);
            lblDur.innerText = e.duration > 0 ? `${Math.floor(e.duration * fps)}f` : "静止";
            lblDur.id = `env-dur-${e.id}`; lblDur.style.color = "#888"; lblDur.style.fontFamily = "monospace";
            
            foot.append(lblLoop, lblDur);
            card.append(head, meshInfo, selEnv, foot);
            this.uiEnvList.appendChild(card);
        });
        this.refreshSidebarHighlights();
    }

    // --- LOGIC: CHARACTERS ---
    renderCharacterCards() {
        this.uiCharList.innerHTML = "";
        this.characters.forEach(c => {
            const card = document.createElement("div"); card.dataset.id = c.id;
            Object.assign(card.style, { background: "#222", border: "1px solid #444", borderRadius: "4px", padding: "6px", cursor: "pointer", transition: "border-color 0.2s" });
            
            card.onclick = (evt) => {
                if (['button', 'input', 'select', 'option'].includes(evt.target.tagName.toLowerCase())) return;
                this.selectObject(c, 'character', c.id);
            };

            const head = document.createElement("div"); head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.marginBottom = "2px";
            head.innerHTML = `<span style="font-weight:bold; font-size:12px;">角色 ${c.id}</span>`;
            
            const btnDel = document.createElement("button"); btnDel.innerText = "X"; Object.assign(btnDel.style, { background: "#622", color: "#fff", border: "1px solid #555", borderRadius: "2px", cursor: "pointer", fontSize: "9px" }); btnDel.onclick = () => this.removeCharacter(c.id);
            head.appendChild(btnDel);
            
            const meshInfo = document.createElement("div");
            meshInfo.style.fontSize = "9px";
            meshInfo.style.color = "#888";
            meshInfo.style.marginBottom = "4px";
            meshInfo.innerText = `[男:${c.depthMeshesM.length} | 女:${c.depthMeshesF.length} | 姿势:${c.poseMeshes.length}]`;
            
            const selAnim = document.createElement("select");
            Object.assign(selAnim.style, { width: "100%", background: "#111", color: "#fff", border: "1px solid #444", borderRadius: "3px", fontSize: "10px", padding: "2px", marginBottom: "4px" });
            this.availableAnimations.forEach(anim => selAnim.add(new Option(anim, anim)));
            selAnim.value = c.animFile; selAnim.onchange = (e) => this.loadAnimationForChar(c, e.target.value);
            
            const foot = document.createElement("div"); foot.style.display = "flex"; foot.style.justifyContent = "space-between"; foot.style.alignItems = "center";
            const loopBox = document.createElement("div"); loopBox.style.display = "flex"; loopBox.style.alignItems = "center"; loopBox.style.gap = "6px";
            
            const btnGender = document.createElement("button"); btnGender.innerText = c.gender;
            Object.assign(btnGender.style, { background: "#111", border: "1px solid #444", borderRadius: "3px", cursor: "pointer", fontSize: "10px", padding: "1px 6px", fontWeight: "bold", color: c.gender === 'F' ? '#ff66b2' : '#66b2ff' });
            btnGender.onclick = () => { c.gender = c.gender === 'M' ? 'F' : 'M'; btnGender.innerText = c.gender; btnGender.style.color = c.gender === 'F' ? '#ff66b2' : '#66b2ff'; this.updateVisibilities(); this.forceUpdateFrame(); };

            const lblLoop = document.createElement("label"); lblLoop.style.cursor = "pointer"; lblLoop.style.display = "flex"; lblLoop.style.gap = "2px";
            const chkLoop = document.createElement("input"); chkLoop.type = "checkbox"; chkLoop.checked = c.loop;
            chkLoop.onchange = (e) => { c.loop = e.target.checked; if(c.action) { c.action.setLoop(c.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce); c.action.clampWhenFinished = !c.loop; }};
            lblLoop.append(chkLoop, "循环"); loopBox.append(btnGender, lblLoop);

            const lblDur = document.createElement("span");
            const fps = this.getWidgetValue("fps", 24);
            lblDur.innerText = c.duration > 0 ? `${Math.floor(c.duration * fps)}f` : "--";
            lblDur.id = `dur-${c.id}`; lblDur.style.color = "#888"; lblDur.style.fontFamily = "monospace";
            
            foot.append(loopBox, lblDur); card.append(head, meshInfo, selAnim, foot); this.uiCharList.appendChild(card);
        });
        this.refreshSidebarHighlights();
    }

    addCharacter() {
        if (this.characters.length >= 16) { alert("为了保证 WebGL 性能，建议最多添加 16 个角色。"); return; }
        this.charCounter++;
        const newChar = new CharacterInstance(this.charCounter, this.baseRig, this.THREE);
        this.scene.add(newChar.scene); this.scene.add(newChar.skeletonHelper);
        this.characters.push(newChar);

        newChar.depthMeshesM.forEach(m => { if(m.isMesh && !this.originalMaterials.has(m)) this.originalMaterials.set(m, m.material); });
        newChar.depthMeshesF.forEach(m => { if(m.isMesh && !this.originalMaterials.has(m)) this.originalMaterials.set(m, m.material); });

        this.updateVisibilities(); this.renderCharacterCards();
    }

    removeCharacter(id) {
        const idx = this.characters.findIndex(c => c.id === id);
        if (idx === -1) return;
        const c = this.characters[idx];
        if (this.selected.type === 'character' && this.selected.id === id) this.selectObject(null, null, null);
        c.destroy(this.scene); this.characters.splice(idx, 1);
        this.renderCharacterCards();
    }

    async loadAnimationForChar(charObj, filename) {
        const lbl = this.container.querySelector(`#dur-${charObj.id}`);
        if (lbl) lbl.innerText = "加载中...";

        if(!filename || filename === "none") {
            charObj.animFile = filename;
            charObj.mixer.stopAllAction();
            charObj.action = null;
            if (lbl) lbl.innerText = "--";
            this.forceUpdateFrame();
            return;
        }
        
        charObj.animFile = filename;
        const isFBX = filename.toLowerCase().endsWith(".fbx");
        const isBVH = filename.toLowerCase().endsWith(".bvh");
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
                        const tc = t.clone(); 
                        tc.name = `${this.semanticMap.get(normalizedTrackBone)}.${prop}`;
                        tracks.push(tc);
                    }
                });

                const cleanClip = new this.THREE.AnimationClip(clip.name, clip.duration, tracks);
                charObj.mixer.stopAllAction(); charObj.mixer.uncacheRoot(charObj.scene);
                
                charObj.action = charObj.mixer.clipAction(cleanClip);
                charObj.action.setLoop(charObj.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce);
                charObj.action.clampWhenFinished = !charObj.loop;
                charObj.action.reset().setEffectiveWeight(1).play();
                charObj.duration = cleanClip.duration;
                
                if (lbl) lbl.innerText = `${Math.floor(charObj.duration * this.getWidgetValue("fps", 24))}f`;
                
                this.isPlaying = true; const btn = this.container.querySelector("#btn-play"); if(btn) btn.innerText = "⏸";
                this.forceUpdateFrame();
            }
        } catch(e) { 
            console.error("Anim Load Error:", e); 
            if (lbl) lbl.innerHTML = `<span style="color:red;">错误</span>`;
        }
    }

    applyCameraKeyframes(timeRatio) {
        if (this.cameraAnimNode && this.cameraAction) {
            this.cameraAnimGroup.updateMatrixWorld(true);
            
            const worldPos = new this.THREE.Vector3();
            const worldQuat = new this.THREE.Quaternion();
            this.cameraAnimNode.getWorldPosition(worldPos);
            this.cameraAnimNode.getWorldQuaternion(worldQuat);

            const eulerOffset = new this.THREE.Euler(
                this.THREE.MathUtils.degToRad(this.camOverrideOffset.rx),
                this.THREE.MathUtils.degToRad(this.camOverrideOffset.ry),
                this.THREE.MathUtils.degToRad(this.camOverrideOffset.rz),
                'XYZ'
            );
            const quatOffset = new this.THREE.Quaternion().setFromEuler(eulerOffset);
            worldQuat.multiply(quatOffset);

            this.importedCamProxy.visible = !this.isCameraOverride && !this.isBaking;
            if (!this.isCameraOverride) {
                this.importedCamProxy.position.copy(worldPos);
                this.importedCamProxy.quaternion.copy(worldQuat);
            }

            if (this.isCameraOverride) {
                this.camera.position.copy(worldPos);
                this.camera.quaternion.copy(worldQuat);
                
                const forward = new this.THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
                this.controls.target.copy(this.camera.position).add(forward.multiplyScalar(5.0));
                
                if (this.selected.type === 'camera') this.updateTransformUIFromObject();
                return;
            }
        }

        if (!this.camKeys.start || !this.camKeys.end) return;
        let t = timeRatio;
        if (this.camKeys.ease === 'easeIn') t = t * t;
        else if (this.camKeys.ease === 'easeOut') t = t * (2 - t);
        else if (this.camKeys.ease === 'easeInOut') t = t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

        this.camera.position.lerpVectors(this.camKeys.start.pos, this.camKeys.end.pos, t);
        this.camera.quaternion.slerpQuaternions(this.camKeys.start.quat, this.camKeys.end.quat, t);
        
        if (this.camKeys.start.zoom !== undefined && this.camKeys.end.zoom !== undefined) {
            this.camera.zoom = this.camKeys.start.zoom + (this.camKeys.end.zoom - this.camKeys.start.zoom) * t;
            this.camera.updateProjectionMatrix();
        }
        if (this.controls && this.camKeys.start.target && this.camKeys.end.target) {
            this.controls.target.lerpVectors(this.camKeys.start.target, this.camKeys.end.target, t);
        }
        
        if (this.selected.type === 'camera') this.updateTransformUIFromObject();
    }

    animate() {
        if (!this.renderer) return;
        requestAnimationFrame(() => this.animate());
        if (this.isBaking) return;

        const delta = this.clock.getDelta();
        const totalFrames = this.getWidgetValue("frame_count", 48);
        const fps = this.getWidgetValue("fps", 24);

        if (this.isPlaying) {
            this.globalTime += delta;
            const totalDuration = totalFrames / fps;
            if (this.globalTime >= totalDuration) this.globalTime = this.globalTime % totalDuration; 
            
            const currentFrame = Math.floor(this.globalTime * fps);
            
            const slider = this.container.querySelector("#t-slider");
            if (slider && !this.isDraggingSlider) slider.value = currentFrame;
            
            const timeLabel = this.container.querySelector("#t-time");
            if (timeLabel) timeLabel.innerText = `${currentFrame} / ${totalFrames}`;

            this.evaluateAnimations(this.globalTime);
            this.applyCameraKeyframes(currentFrame / Math.max(1, totalFrames - 1));
        }
        
        if (this.controls) this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    updateVisibilities() {
        const isDepth = this.isDepthMode;
        const isShaded = this.isShadedMode;

        this.characters.forEach(c => {
            c.inactiveDepthMeshes.forEach(m => m.visible = false);
            const showDepthMeshes = isDepth || isShaded;
            c.activeDepthMeshes.forEach(m => {
                m.visible = showDepthMeshes;
                if (isDepth) m.material = m.isSkinnedMesh ? this.matsSkinned.depth : this.matsStatic.depth;
                else if (isShaded) m.material = m.isSkinnedMesh ? this.matsSkinned.shaded : this.matsStatic.shaded;
                else m.material = this.originalMaterials.get(m) || m.material;
            });
            c.poseMeshes.forEach(m => m.visible = !showDepthMeshes);
        });

        this.environments.forEach(e => {
            e.meshes.forEach(m => {
                m.visible = true; 
                if (isDepth) m.material = m.isSkinnedMesh ? this.matsSkinned.depth : this.matsStatic.depth;
                else if (isShaded) m.material = m.isSkinnedMesh ? this.matsSkinned.shaded : this.matsStatic.shaded;
                else m.material = this.originalMaterials.get(m) || m.material;
            });
        });

        if (isDepth) this.updateCameraBounds();
        else this.resetCamera();
    }

    updateCameraBounds() {
        if(!this.camera) return;
        this.camera.near = Math.max(0.01, this.userNear); this.camera.far = Math.max(0.1, this.userFar);
        this.camera.updateProjectionMatrix();
    }
    resetCamera() {
        if(!this.camera) return;
        this.camera.near = this.defaultNear; this.camera.far = this.defaultFar;
        this.camera.updateProjectionMatrix();
    }

    // 关键修改点 2：将尺寸钩子中的组件名称映射为中文，保持画布实时同步
    hookNodeWidgets() {
        const updateDim = (w, val) => {
            if(w && w.name === "宽度") this.renderWidth = val; 
            else if(w && w.name === "高度") this.renderHeight = val;
            const lbl = this.container.querySelector("#lbl-res");
            if(lbl) lbl.innerText = `${this.renderWidth}x${this.renderHeight}`;
            this.onResize(this.container.querySelector(".yedp-vp-area"));
        };
        const wWidget = this.node.widgets?.find(w => w.name === "宽度");
        const hWidget = this.node.widgets?.find(w => w.name === "高度");
        if(wWidget) { this.renderWidth = wWidget.value; const orig = wWidget.callback; wWidget.callback = v => { updateDim(wWidget, v); if(orig) orig(v); }; }
        if(hWidget) { this.renderHeight = hWidget.value; const orig = hWidget.callback; hWidget.callback = v => { updateDim(hWidget, v); if(orig) orig(v); }; }
        updateDim();
        const slider = this.container.querySelector("#t-slider");
        const fWidget = this.node.widgets?.find(w => w.name === "总帧数");
        if(fWidget && slider) {
            slider.max = fWidget.value;
            const orig = fWidget.callback; fWidget.callback = v => { slider.max = v; if(orig) orig(v); };
        }
    }

    onResize(vpDiv) {
        if (this.isBaking || !this.renderer || !vpDiv || !this.camera) return;
        const w = vpDiv.clientWidth; const h = vpDiv.clientHeight;
        if (w && h) {
            this.renderer.setSize(w, h);
            const aspect = w / h;
            if (this.perspCam) { this.perspCam.aspect = aspect; this.perspCam.updateProjectionMatrix(); }
            if (this.orthoCam) {
                const frustumSize = 4.0;
                this.orthoCam.left = -frustumSize * aspect / 2; this.orthoCam.right = frustumSize * aspect / 2;
                this.orthoCam.top = frustumSize / 2; this.orthoCam.bottom = -frustumSize / 2;
                this.orthoCam.updateProjectionMatrix();
            }
            const aspectContainer = w / h; const aspectTarget = this.renderWidth / this.renderHeight;
            let gw, gh;
            if (aspectContainer > aspectTarget) { gh = h - 20; gw = gh * aspectTarget; } 
            else { gw = w - 20; gh = gw / aspectTarget; }
            if(this.gate) { this.gate.style.width = `${gw}px`; this.gate.style.height = `${gh}px`; }
        }
    }

    // 关键修改点 1：中英文名称自动映射
    getWidgetValue(name, defaultVal) {
        const nameMap = {
            "width": "宽度",
            "height": "高度",
            "frame_count": "总帧数",
            "fps": "帧率",
            "client_data": "客户端数据"
        };
        const targetName = nameMap[name] || name;
        const w = this.node.widgets?.find(x => x.name === targetName); 
        return w ? w.value : defaultVal;
    }

    async performBake(isSingleFrame = false) {
        if (this.characters.length === 0 && this.environments.length === 0) { alert("场景为空！"); return; }
        const THREE = this.THREE;
        
        const btnId = isSingleFrame ? '#btn-bake-frame' : '#btn-bake';
        const originalBtnText = isSingleFrame ? '单帧烘焙' : '全部烘焙 V9.21';
        const btn = this.container.querySelector(btnId); btn.innerText = "准备中...";
        
        this.isBaking = true; this.isPlaying = false;
        
        this.transformControls.detach();
        this.lights.forEach(l => l.helper.visible = false);

        const originalSize = new THREE.Vector2(); this.renderer.getSize(originalSize);
        const originalAspect = this.camera.aspect || (originalSize.width / originalSize.height);
        const originalZoom = this.camera.zoom; const originalBg = this.scene.background;
        
        const vpArea = this.container.querySelector(".yedp-vp-area");
        if (vpArea) {
            const vpW = vpArea.clientWidth; const vpH = vpArea.clientHeight;
            const vpAspect = vpW / vpH; const targetAspect = this.renderWidth / this.renderHeight;
            if (vpAspect < targetAspect) this.camera.zoom = originalZoom * (targetAspect / vpAspect);
            else this.camera.zoom = originalZoom;
        }

        this.renderer.setSize(this.renderWidth, this.renderHeight);
        const targetRenderAspect = this.renderWidth / this.renderHeight;
        this.perspCam.aspect = targetRenderAspect; this.perspCam.updateProjectionMatrix();

        const frustumSize = 4.0;
        this.orthoCam.left = -frustumSize * targetRenderAspect / 2;
        this.orthoCam.right = frustumSize * targetRenderAspect / 2;
        this.orthoCam.top = frustumSize / 2;
        this.orthoCam.bottom = -frustumSize / 2;
        this.orthoCam.updateProjectionMatrix();

        const totalNodeFrames = this.getWidgetValue("frame_count", 48);
        const fps = this.getWidgetValue("fps", 24);
        const step = 1.0 / fps;
        
        const framesToRender = isSingleFrame ? 1 : totalNodeFrames;
        const currentUIFrame = parseInt(this.container.querySelector("#t-slider").value) || 0;
        
        const results = { pose: [], depth: [], canny: [], normal: [], shaded: [], alpha: [] };

        const visSkel = this.container.querySelector("#chk-skel").checked;
        const toggleHelpers = (vis) => { 
            if(this.gridHelper) this.gridHelper.visible = vis; 
            if(this.axesHelper) this.axesHelper.visible = vis; 
            if(this.floor) this.floor.visible = vis; 
            if(this.importedCamProxy) this.importedCamProxy.visible = false; 
        };
        toggleHelpers(false);

        const setVisibility = (mode) => {
            const showPose = mode === 'pose';
            const showDepth = ['depth', 'canny', 'normal', 'shaded', 'alpha'].includes(mode);
            const showEnv = ['depth', 'normal', 'shaded'].includes(mode);

            this.characters.forEach(c => {
                c.poseMeshes.forEach(m => m.visible = showPose);
                c.inactiveDepthMeshes.forEach(m => m.visible = false); 
                c.activeDepthMeshes.forEach(m => m.visible = showDepth);
                c.skeletonHelper.visible = false; 
            });
            this.environments.forEach(e => e.meshes.forEach(m => m.visible = showEnv));
            
            this.scene.background = new THREE.Color(0x000000); 
        };

        const compressCanvas = document.createElement("canvas");
        compressCanvas.width = this.renderWidth; compressCanvas.height = this.renderHeight;
        const compressCtx = compressCanvas.getContext("2d");

        const captureFrame = (array, mimeType = "image/png", quality = undefined) => {
            this.renderer.render(this.scene, this.camera);
            this.renderer.getContext().finish(); 
            if (mimeType === "image/jpeg") {
                compressCtx.fillStyle = "#000000"; compressCtx.fillRect(0, 0, this.renderWidth, this.renderHeight);
                compressCtx.drawImage(this.renderer.domElement, 0, 0); array.push(compressCanvas.toDataURL(mimeType, quality));
            } else array.push(this.renderer.domElement.toDataURL(mimeType));
        };

        const executeMaterialPass = (matKey, arrayObj) => {
            const restores = [];
            this.characters.forEach(c => c.activeDepthMeshes.forEach(m => { 
                restores.push({mesh: m, mat: m.material}); 
                m.material = m.isSkinnedMesh ? this.matsSkinned[matKey] : this.matsStatic[matKey]; 
            }));
            this.environments.forEach(e => e.meshes.forEach(m => { 
                restores.push({mesh: m, mat: m.material}); 
                m.material = m.isSkinnedMesh ? this.matsSkinned[matKey] : this.matsStatic[matKey]; 
            }));
            captureFrame(arrayObj, "image/png");
            restores.forEach(o => o.mesh.material = o.mat);
        };

        // RENDER LOOP
        for (let idx = 0; idx < framesToRender; idx++) {
            const actualFrame = isSingleFrame ? currentUIFrame : idx;
            const time = actualFrame * step;
            const timeRatio = totalNodeFrames > 1 ? actualFrame / (totalNodeFrames - 1) : 0;
            
            btn.innerText = `烘焙中 ${idx+1}/${framesToRender}`;
            
            this.evaluateAnimations(time);
            this.characters.forEach(c => c.scene.updateMatrixWorld(true));
            this.environments.forEach(e => e.group.updateMatrixWorld(true));
            this.applyCameraKeyframes(timeRatio);

            setVisibility('pose'); this.resetCamera(); 
            captureFrame(results.pose, "image/png"); 

            setVisibility('depth');
            this.camera.near = Math.max(0.01, this.userNear); this.camera.far = Math.max(0.1, this.userFar); this.camera.updateProjectionMatrix();
            executeMaterialPass('depth', results.depth);

            setVisibility('canny'); this.resetCamera();
            executeMaterialPass('canny', results.canny);

            setVisibility('normal'); this.resetCamera();
            executeMaterialPass('normal', results.normal);
            
            setVisibility('shaded'); this.resetCamera();
            executeMaterialPass('shaded', results.shaded);

            setVisibility('alpha'); this.resetCamera();
            executeMaterialPass('alpha', results.alpha);

            await new Promise(r => setTimeout(r, 10)); 
        }
        
        // Restoration
        this.renderer.setSize(originalSize.width, originalSize.height);
        if (this.perspCam) this.perspCam.aspect = originalAspect;
        if (this.orthoCam) {
            const aspectTarget = originalSize.width / originalSize.height;
            const frustumSize = 4.0;
            this.orthoCam.left = -frustumSize * aspectTarget / 2; this.orthoCam.right = frustumSize * aspectTarget / 2;
            this.orthoCam.top = frustumSize / 2; this.orthoCam.bottom = -frustumSize / 2;
        }
        
        this.camera.zoom = originalZoom; 
        if(this.isDepthMode) { this.camera.near = this.userNear; this.camera.far = this.userFar; } 
        else this.resetCamera();
        this.camera.updateProjectionMatrix();

        toggleHelpers(true); this.scene.background = originalBg; this.isBaking = false;
        
        this.lights.forEach(l => l.helper.visible = l.type !== 'ambient');
        if (this.selected.obj) this.selectObject(this.selected.obj, this.selected.type, this.selected.id);
        
        this.updateVisibilities();
        this.characters.forEach(c => { c.skeletonHelper.visible = visSkel; });
        
        // 关键修改点 3：寻找包含最终数据的中文字段
        btn.innerText = "上传数据中...";
        const clientDataWidget = this.node.widgets.find(w => w.name === "客户端数据");
        if (clientDataWidget) {
            try {
                const response = await api.fetchApi("/yedp/upload_payload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(results) });
                if (!response.ok) throw new Error("Upload failed");
                const resData = await response.json();
                clientDataWidget.value = resData.payload_id;
            } catch (err) {
                console.error("[Yedp] 内存缓存上传失败，回退到本地字符串:", err);
                clientDataWidget.value = JSON.stringify(results);
            }
        }
        
        btn.innerText = "烘焙完成";
        setTimeout(() => { btn.innerText = originalBtnText; }, 2000);
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
                // 宽度保持 100% 动态填充，内部基础高度增加 100，改为 600px
                container.style.width = "100%"; 
                container.style.height = "600px"; 
                
                const widget = this.addDOMWidget("3d_viewport", "vp", container, { serialize: false, hideOnZoom: false });
                
                // 【核心修复：打破死循环，且还原动态宽度 w】
                // 向引擎上报原始宽度 w，并将画布最小高度锁定为 600
                widget.computeSize = function(w) {
                    return [w, 600]; 
                };
                
                setTimeout(() => {
                    const vp = new YedpViewport(this, container);
                    this.vp = vp; 
                    const onResizeOrig = this.onResize;
                    this.onResize = function(size) {
                        if (onResizeOrig) onResizeOrig.call(this, size);
                        let usedHeight = 30; 
                        if (this.widgets) for (const w of this.widgets) { if (w === widget) break; usedHeight += w.last_h || 26; }
                        const safeHeight = Math.max(100, size[1] - usedHeight - 35);
                        container.style.height = safeHeight + "px"; 
                        container.style.maxHeight = "none";
                        vp.onResize(container.querySelector(".yedp-vp-area"));
                    };
                    
                    const w = this.widgets?.find(w => w.name === "客户端数据");
                    if (w?.inputEl) {
                        w.inputEl.style.display = "none";
                        w.computeSize = () => [0, -4]; // 抹平隐藏输入框留下的白边
                    }
                    
                    // 只在节点刚创建（高度不足）时，赋予初始尺寸
                    // 宽度保持原本的 720，高度增加 100 变为 700
                    if (this.size[1] < 650) {
                        this.setSize([720, 700]);
                        if (app.graph) app.graph.setDirtyCanvas(true, true);
                    }
                }, 100);
                
                this.onRemoved = function() {
                    if (this.vp) {
                        this.vp.isBaking = false; this.vp.isPlaying = false;
                        if (this.vp._handleKeyDown) window.removeEventListener('keydown', this.vp._handleKeyDown);
                        if (this.vp.renderer) { this.vp.renderer.dispose(); this.vp.renderer = null; }
                    }
                };
                return r;
            };

            const onSerializeOrig = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (o) {
                if (onSerializeOrig) onSerializeOrig.apply(this, arguments);
                if (this.vp && this.vp.isInitialized) {
                    o.scene_state = this.vp.serializeScene();
                }
            };

            const onConfigureOrig = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (o) {
                if (onConfigureOrig) onConfigureOrig.apply(this, arguments);
                if (o.scene_state) {
                    this.saved_scene_state = o.scene_state;
                    if (this.vp && this.vp.isInitialized) {
                        this.vp.loadScene(this.saved_scene_state);
                    }
                }
            };
        }
    }
});