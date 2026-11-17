// The scenery around the drones: sky, sun, fog, and either a plain grid or a pine
// forest. It's only scenery: the simulator's ground is the plane z = 0 whichever
// world is drawn, and drones fly straight through the forest's trees.

import { useGLTF, useTexture } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { Component, type ReactNode, Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import forestUrl from "../assets/pine-forest.glb?url";
import grassUrl from "../assets/forest-grass.webp?url";
import rockUrl from "../assets/forest-rock.webp?url";
import type { TimeOfDay, WorldName } from "../store";
import { SCENE, type SceneLook } from "../theme";

// Fog distances (m). The forest's fog reaches further so the trees read as a
// forest, and still hides where its ground ends, about 110 m out.
const FOG: Record<WorldName, [number, number]> = { grid: [16, 55], forest: [20, 95] };

export function World({ kind, time }: { kind: WorldName; time: TimeOfDay }) {
  const look = SCENE[time];
  const sun = useMemo(() => new THREE.Vector3(...look.sunDirection).normalize(), [look]);
  const [near, far] = FOG[kind];
  return (
    <>
      <color attach="background" args={[look.horizon]} />
      <fog attach="fog" args={[look.horizon, near, far]} />
      <Sky look={look} sun={sun} />
      <hemisphereLight args={[look.skyLight, look.groundLight, look.ambientIntensity]} position={[0, 0, 1]} />
      <directionalLight position={sun.clone().multiplyScalar(20)} color={look.sunLight} intensity={look.sunIntensity} />
      {kind === "forest" ? (
        // Show the grid while the forest downloads, and keep it if the download fails.
        <FallbackOnError fallback={<GridGround look={look} />}>
          <Suspense fallback={<GridGround look={look} />}>
            <Forest />
          </Suspense>
        </FallbackOnError>
      ) : (
        <GridGround look={look} />
      )}
      {/* Launch pad at the origin, where drone0 starts. Raised a little in the
          forest, whose ground undulates a few centimeters around z = 0 there. */}
      <mesh position-z={kind === "forest" ? 0.03 : 0.003}>
        <ringGeometry args={[0.42, 0.5, 4, 1, Math.PI / 4]} />
        <meshBasicMaterial color={look.pad} />
      </mesh>
    </>
  );
}

function GridGround({ look }: { look: SceneLook }) {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(80, 80, look.gridMajor, look.grid);
    g.rotation.x = Math.PI / 2; // GridHelper lies in XZ; the sim's ground is XY
    g.position.z = 0.002;
    return g;
  }, [look.grid, look.gridMajor]);
  useEffect(() => () => grid.dispose(), [grid]);
  return (
    <>
      <mesh position-z={-0.002}>
        <planeGeometry args={[400, 400]} />
        <meshStandardMaterial color={look.ground} roughness={1} />
      </mesh>
      <primitive object={grid} />
    </>
  );
}

// --- Sky ----------------------------------------------------------------------------
// A dome that follows the camera: a gradient from the horizon up to the zenith,
// with the sun and its glow. Colors go straight to the screen (no tone mapping),
// so the horizon matches the fog exactly and distant ground fades into the sky.

const SKY_VERTEX = /* glsl */ `
varying vec3 vDirection;
void main() {
  vDirection = position; // the dome is centered on the camera, so this is the view direction
  vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = clip.xyww; // on the far plane: behind everything, and never clipped
}
`;

const SKY_FRAGMENT = /* glsl */ `
uniform vec3 zenith;
uniform vec3 horizon;
uniform vec3 sunGlow;
uniform vec3 sunDisk;
uniform vec3 sunDirection;
varying vec3 vDirection;
void main() {
  vec3 d = normalize(vDirection);
  vec3 color = mix(horizon, zenith, pow(clamp(d.z, 0.0, 1.0), 0.6));
  float toSun = max(dot(d, sunDirection), 0.0);
  float glow = 0.55 * pow(toSun, 6.0) + 0.35 * pow(toSun, 48.0);
  color = mix(color, sunGlow, clamp(glow, 0.0, 1.0));
  color = mix(color, sunDisk, smoothstep(0.9992, 0.9996, toSun));
  gl_FragColor = vec4(color, 1.0);
}
`;

/** "#rrggbb" as 0..1 sRGB components, untouched by three's color management. */
function srgb(hex: string, out = new THREE.Vector3()) {
  const n = parseInt(hex.slice(1), 16);
  return out.set(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
}

function Sky({ look, sun }: { look: SceneLook; sun: THREE.Vector3 }) {
  const mesh = useRef<THREE.Mesh>(null!);
  // Built whole, uniforms and all: a compiled material keeps reading the uniforms
  // object it started with, so the colors are changed in place below.
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          zenith: { value: new THREE.Vector3() },
          horizon: { value: new THREE.Vector3() },
          sunGlow: { value: new THREE.Vector3() },
          sunDisk: { value: new THREE.Vector3() },
          sunDirection: { value: new THREE.Vector3() },
        },
        vertexShader: SKY_VERTEX,
        fragmentShader: SKY_FRAGMENT,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
      }),
    [],
  );
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    const u = material.uniforms;
    srgb(look.zenith, u.zenith.value);
    srgb(look.horizon, u.horizon.value);
    srgb(look.sunGlow, u.sunGlow.value);
    srgb(look.sunDisk, u.sunDisk.value);
    u.sunDirection.value.copy(sun);
  }, [look, sun, material]);
  useFrame(({ camera }) => mesh.current.position.copy(camera.position));
  return (
    <mesh ref={mesh} material={material} renderOrder={-1} frustumCulled={false}>
      <sphereGeometry args={[10, 48, 24]} />
    </mesh>
  );
}

// --- Pine forest --------------------------------------------------------------------
// A stand of ponderosa pines, 260 m across (src/assets/pine-forest.glb; see the
// README for where it comes from). glTF is y-up, so the scene is turned z-up and
// moved to put a flat clearing under the launch pad. The spot is given in the
// file's own coordinates (x, height, z), and `heading` turns the forest about it:
// together they keep trees out of the area the built-in scenarios fly through
// and away from where the Fly and Scenarios cameras start, and keep the ground
// there within 5 cm of the pad's height.
const FOREST_PAD = { x: 19, height: 1.83, z: -3, heading: 45 };

const GRASS_TILE = 4; // m of ground per repeat of the grass texture
const ROCK_TILE = 1.5;
const LOG_TILE = 0.8;
const PAD_MOWED = 1.3; // m of short grass around the launch pad

function Forest() {
  // The file is meshopt-compressed; the decoder ships with three-stdlib, so no Draco (and no CDN).
  const { scene } = useGLTF(forestUrl, false, true);
  const [grass, rock] = useTexture([grassUrl, rockUrl]);
  useMemo(() => dressForest(scene, grass, rock), [scene, grass, rock]);
  const { x, height, z, heading } = FOREST_PAD;
  return (
    <group rotation-z={(-heading * Math.PI) / 180}>
      <group rotation-x={Math.PI / 2}>
        <primitive object={scene} position={[-x, -height, -z]} />
      </group>
    </group>
  );
}

/** Put the forest's loose textures on it: grass over the ground (the file's own
 * ground texture is a flat green), moss on the rocks, and the pines' bark on the
 * fallen logs. Rocks and logs have no texture coordinates, so they get theirs
 * from a triplanar shader. Runs once per loaded scene. */
function dressForest(scene: THREE.Object3D, grass: THREE.Texture, rock: THREE.Texture) {
  if (scene.userData.dressed) return;
  scene.userData.dressed = true;

  const materials = new Map<string, THREE.MeshStandardMaterial>();
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const m of [mesh.material].flat()) materials.set(m.name, m as THREE.MeshStandardMaterial);
  });

  const ground = materials.get("ground");
  if (ground) {
    grass.colorSpace = THREE.SRGBColorSpace;
    grass.wrapS = grass.wrapT = THREE.RepeatWrapping;
    const repeats = 260 / GRASS_TILE; // the ground's texture coordinates span its 260 m once
    grass.repeat.set(repeats, repeats);
    grass.anisotropy = 8; // stays sharp at the low angles you see the ground from
    ground.map = grass;
    ground.color.set("#ffffff");
    ground.needsUpdate = true;
  }

  const rocks = materials.get("rock");
  if (rocks) {
    rock.colorSpace = THREE.SRGBColorSpace;
    rock.wrapS = rock.wrapT = THREE.RepeatWrapping;
    triplanar(rocks, rock, ROCK_TILE, "#a6a69c");
  }

  const bark = [...materials.values()].find((m) => m.name.startsWith("FS_branches"))?.map;
  const logs = materials.get("log");
  if (logs && bark) triplanar(logs, bark, LOG_TILE, "#9a8a78");

  // Mow the grass around the launch pad, so a landed drone isn't buried in it.
  const pad = new THREE.Vector3(FOREST_PAD.x, FOREST_PAD.height, FOREST_PAD.z);
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh && [mesh.material].flat().some((m) => m.name.startsWith("grass"))) mow(mesh, scene, pad, PAD_MOWED);
  });
}

/** Drop the triangles of `mesh` within `radius` (horizontally) of `spot`, both in `root`'s coordinates. */
function mow(mesh: THREE.Mesh, root: THREE.Object3D, spot: THREE.Vector3, radius: number) {
  const index = mesh.geometry.index;
  if (!index) return;
  root.updateWorldMatrix(true, true);
  const toRoot = root.matrixWorld.clone().invert().multiply(mesh.matrixWorld);
  const position = mesh.geometry.getAttribute("position");
  const v = new THREE.Vector3();
  const far = new Uint8Array(position.count);
  for (let i = 0; i < position.count; i++) {
    v.fromBufferAttribute(position, i).applyMatrix4(toRoot);
    far[i] = Math.hypot(v.x - spot.x, v.z - spot.z) > radius ? 1 : 0;
  }
  const kept: number[] = [];
  for (let t = 0; t < index.count; t += 3) {
    const a = index.getX(t);
    const b = index.getX(t + 1);
    const c = index.getX(t + 2);
    if (far[a] && far[b] && far[c]) kept.push(a, b, c);
  }
  mesh.geometry.setIndex(kept);
}

/** Texture a material by projecting `map` along the three world axes and
 * blending by the surface normal, for meshes without texture coordinates. */
function triplanar(material: THREE.MeshStandardMaterial, map: THREE.Texture, tile: number, tint: string) {
  material.color.set(tint);
  material.onBeforeCompile = (shader) => {
    shader.uniforms.triMap = { value: map };
    shader.uniforms.triScale = { value: 1 / tile };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vTriPos;\nvarying vec3 vTriNormal;")
      .replace(
        "#include <begin_vertex>",
        /* glsl */ `#include <begin_vertex>
        vec4 triPos = vec4(transformed, 1.0);
        vec3 triNormal = normal;
        #ifdef USE_INSTANCING
          triPos = instanceMatrix * triPos;
          triNormal = mat3(instanceMatrix) * triNormal;
        #endif
        vTriPos = (modelMatrix * triPos).xyz;
        vTriNormal = mat3(modelMatrix) * triNormal;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nuniform sampler2D triMap;\nuniform float triScale;\nvarying vec3 vTriPos;\nvarying vec3 vTriNormal;",
      )
      .replace(
        "#include <map_fragment>",
        /* glsl */ `
        vec3 triWeight = pow(abs(normalize(vTriNormal)), vec3(4.0));
        triWeight /= triWeight.x + triWeight.y + triWeight.z;
        diffuseColor *= texture2D(triMap, vTriPos.yz * triScale) * triWeight.x
          + texture2D(triMap, vTriPos.zx * triScale) * triWeight.y
          + texture2D(triMap, vTriPos.xy * triScale) * triWeight.z;`,
      );
  };
  material.customProgramCacheKey = () => "triplanar";
  material.needsUpdate = true;
}

class FallbackOnError extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    console.warn("Couldn't load the forest; showing the grid instead.", error);
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
