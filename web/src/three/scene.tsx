// three.js pieces shared by the Fly and Scenarios views. The scene is z-up,
// matching drone_sim's world frame, so sim coordinates are used directly.
// Everything that moves is updated imperatively in useFrame from a getter, so
// 60 fps motion never re-renders React.

import { Billboard, Line } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import * as THREE from "three";
import { LineMaterial, LineSegments2, LineSegmentsGeometry } from "three-stdlib";
import type { DroneParams } from "../sim/protocol";
import { HIVIS, SCENE, SIGNAL_RED } from "../theme";

export const DRAW_SCALE = 2; // drawn bigger than life so it reads at a distance, as in the matplotlib views

export interface Pose {
  pos: ArrayLike<number>;
  quat: ArrayLike<number>; // (w, x, y, z), body -> world
  thrust?: ArrayLike<number>;
}

// --- Ground, sky, light ------------------------------------------------------------

export function World({ dark }: { dark: boolean }) {
  const c = SCENE[dark ? "dark" : "light"];
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(80, 80, c.gridMajor, c.grid);
    g.rotation.x = Math.PI / 2; // GridHelper lies in XZ; the sim's ground is XY
    g.position.z = 0.002;
    return g;
  }, [c.grid, c.gridMajor]);
  useEffect(() => () => grid.dispose(), [grid]);

  return (
    <>
      <color attach="background" args={[c.sky]} />
      <fog attach="fog" args={[c.sky, 16, 55]} />
      <hemisphereLight args={[c.skyLight, c.groundLight, dark ? 1.1 : 1.4]} position={[0, 0, 1]} />
      <directionalLight position={[5, -7, 12]} intensity={dark ? 1.4 : 1.8} />
      <mesh position-z={-0.002}>
        <planeGeometry args={[400, 400]} />
        <meshStandardMaterial color={c.ground} roughness={1} />
      </mesh>
      <primitive object={grid} />
      {/* Launch pad at the origin, where drone0 starts. */}
      <mesh position-z={0.003}>
        <ringGeometry args={[0.42, 0.5, 4, 1, Math.PI / 4]} />
        <meshBasicMaterial color={c.pad} />
      </mesh>
    </>
  );
}

// --- The drone ----------------------------------------------------------------------

export function DroneModel({
  params,
  getPose,
  body = HIVIS,
}: {
  params: DroneParams;
  getPose: () => Pose | null;
  body?: string;
}) {
  const S = DRAW_SCALE;
  const group = useRef<THREE.Group>(null!);
  const blades = useRef<(THREE.Object3D | null)[]>([]);
  const rotors = useMemo(
    () => params.rotor_positions.map(([x, y, z]) => [x * S, y * S, z * S] as [number, number, number]),
    [params, S],
  );
  const ringR = 0.35 * params.arm_length * S;
  const armLen = 2 * params.arm_length * S;

  useFrame((_, dt) => {
    const pose = getPose();
    const g = group.current;
    if (!pose) {
      g.visible = false;
      return;
    }
    g.visible = true;
    g.position.set(pose.pos[0], pose.pos[1], pose.pos[2]);
    g.quaternion.set(pose.quat[1], pose.quat[2], pose.quat[3], pose.quat[0]);
    blades.current.forEach((blade, i) => {
      if (!blade) return;
      const load = pose.thrust ? pose.thrust[i] / params.max_thrust : 0;
      // Props spin against their reaction torque.
      blade.rotation.z -= params.yaw_signs[i] * (4 + 36 * load) * dt;
    });
  });

  return (
    <group ref={group}>
      {[Math.PI / 4, -Math.PI / 4].map((angle) => (
        <mesh key={angle} rotation-z={angle}>
          <boxGeometry args={[armLen, 0.022 * S, 0.016 * S]} />
          <meshStandardMaterial color="#2b323a" roughness={0.6} />
        </mesh>
      ))}
      <mesh>
        <boxGeometry args={[0.1 * S, 0.065 * S, 0.035 * S]} />
        <meshStandardMaterial color={body} roughness={0.45} />
      </mesh>
      {rotors.map((pos, i) => (
        <group key={i} position={pos}>
          <mesh>
            <torusGeometry args={[ringR, 0.006 * S, 8, 40]} />
            <meshStandardMaterial color={i < 2 ? SIGNAL_RED : "#4b5560"} roughness={0.5} />
          </mesh>
          <mesh rotation-x={Math.PI / 2} position-z={0.008 * S}>
            <cylinderGeometry args={[0.012 * S, 0.012 * S, 0.02 * S, 12]} />
            <meshStandardMaterial color="#2b323a" />
          </mesh>
          <group ref={(el) => void (blades.current[i] = el)} position-z={0.02 * S}>
            <mesh>
              <boxGeometry args={[ringR * 1.8, 0.012 * S, 0.002 * S]} />
              <meshStandardMaterial color="#59636e" transparent opacity={0.85} />
            </mesh>
          </group>
        </group>
      ))}
    </group>
  );
}

// --- Markers ------------------------------------------------------------------------

/** The red X the drone flies to, with a pole and ring marking the spot below. */
export function TargetMarker({ getTarget }: { getTarget: () => ArrayLike<number> | null }) {
  const group = useRef<THREE.Group>(null!);
  const pole = useRef<THREE.Mesh>(null!);
  const ring = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    const t = getTarget();
    if (!t) {
      group.current.visible = false;
      return;
    }
    const z = Math.max(t[2], 0.001);
    group.current.visible = true;
    group.current.position.set(t[0], t[1], z);
    pole.current.scale.z = z;
    pole.current.position.z = -z / 2;
    ring.current.position.z = -z + 0.004;
  });
  return (
    <group ref={group}>
      <Billboard>
        {[Math.PI / 4, -Math.PI / 4].map((a) => (
          <mesh key={a} rotation-z={a}>
            <boxGeometry args={[0.42, 0.055, 0.01]} />
            <meshBasicMaterial color={SIGNAL_RED} />
          </mesh>
        ))}
      </Billboard>
      <mesh ref={pole}>
        <boxGeometry args={[0.012, 0.012, 1]} />
        <meshBasicMaterial color={SIGNAL_RED} transparent opacity={0.35} />
      </mesh>
      <mesh ref={ring}>
        <ringGeometry args={[0.2, 0.25, 40]} />
        <meshBasicMaterial color={SIGNAL_RED} transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

/** A soft spot on the ground under the drone, fading with altitude. */
export function GroundShadow({ getPos, dark }: { getPos: () => ArrayLike<number> | null; dark: boolean }) {
  const mesh = useRef<THREE.Mesh>(null!);
  const material = useRef<THREE.MeshBasicMaterial>(null!);
  useFrame(() => {
    const p = getPos();
    if (!p) {
      mesh.current.visible = false;
      return;
    }
    mesh.current.visible = true;
    mesh.current.position.set(p[0], p[1], 0.004);
    const fade = Math.max(0.15, 1 - p[2] / 10);
    material.current.opacity = (dark ? 0.5 : 0.28) * fade;
    mesh.current.scale.setScalar(1 + p[2] * 0.04);
  });
  return (
    <mesh ref={mesh}>
      <circleGeometry args={[0.32, 32]} />
      <meshBasicMaterial ref={material} color={SCENE[dark ? "dark" : "light"].shadow} transparent depthWrite={false} />
    </mesh>
  );
}

// --- Lines --------------------------------------------------------------------------
// Plain WebGL lines are always 1 device pixel, which nearly vanishes on a
// high-DPI screen, so trails use three's "fat" lines with a width in CSS pixels.
// Segments live in one preallocated buffer that is updated in place.

function makeFatLine(maxSegments: number, color: string, opacity: number, width: number) {
  const segments = new Float32Array(Math.max(maxSegments, 1) * 6); // start xyz, end xyz
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(segments);
  const buffer = (geometry.attributes.instanceStart as THREE.InterleavedBufferAttribute).data;
  buffer.setUsage(THREE.DynamicDrawUsage);
  geometry.instanceCount = 0;
  const material = new LineMaterial({
    color: new THREE.Color(color).getHex(),
    linewidth: width,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 1,
  });
  const line = new LineSegments2(geometry, material);
  line.frustumCulled = false;
  return { line, geometry, buffer, segments, material, maxSegments: Math.max(maxSegments, 1) };
}

type FatLine = ReturnType<typeof makeFatLine>;

/** Keep the line material's idea of the canvas size current (it sets the pixel width). */
function useFatLine(parts: FatLine, color: string) {
  const size = useThree((s) => s.size);
  useEffect(() => void parts.material.resolution.set(size.width, size.height), [parts, size]);
  useEffect(() => void parts.material.color.set(color), [parts, color]);
  useEffect(() => () => (parts.geometry.dispose(), parts.material.dispose()), [parts]);
}

export interface TrailHandle {
  push(x: number, y: number, z: number): void;
  clear(): void;
}

/** A trail that grows point by point, keeping the most recent `capacity` points. */
export const Trail = forwardRef<
  TrailHandle,
  { capacity: number; color: string; opacity?: number; width?: number }
>(function Trail({ capacity, color, opacity = 0.85, width = 2.5 }, ref) {
  // Built once per capacity; color changes are applied in useFatLine without rebuilding.
  const parts = useMemo(() => makeFatLine(capacity - 1, color, opacity, width), [capacity]);
  useFatLine(parts, color);
  const last = useRef<[number, number, number] | null>(null);
  const count = useRef(0); // segments drawn
  useImperativeHandle(
    ref,
    () => ({
      push(x, y, z) {
        const prev = last.current;
        last.current = [x, y, z];
        if (!prev) return;
        let n = count.current;
        if (n === parts.maxSegments) {
          parts.segments.copyWithin(0, 6);
          n -= 1;
        }
        const s = parts.segments;
        const o = n * 6;
        s[o] = prev[0];
        s[o + 1] = prev[1];
        s[o + 2] = prev[2];
        s[o + 3] = x;
        s[o + 4] = y;
        s[o + 5] = z;
        count.current = n + 1;
        parts.geometry.instanceCount = n + 1;
        parts.buffer.needsUpdate = true;
      },
      clear() {
        last.current = null;
        count.current = 0;
        parts.geometry.instanceCount = 0;
      },
    }),
    [parts],
  );
  return <primitive object={parts.line} />;
});

/** A precomputed path drawn up to `getCount()` points (playback trails). */
export function PathLine({
  points,
  getCount,
  color,
  opacity = 0.85,
  width = 2.5,
  flat = false,
}: {
  points: Float64Array; // N x 3
  getCount: () => number;
  color: string;
  opacity?: number;
  width?: number;
  flat?: boolean; // project onto the ground
}) {
  const parts = useMemo(() => {
    const n = points.length / 3;
    const p = makeFatLine(n - 1, color, opacity, width);
    const z = (i: number) => (flat ? 0.006 : points[i * 3 + 2]);
    for (let i = 0; i < n - 1; i++) {
      const o = i * 6;
      p.segments[o] = points[i * 3];
      p.segments[o + 1] = points[i * 3 + 1];
      p.segments[o + 2] = z(i);
      p.segments[o + 3] = points[i * 3 + 3];
      p.segments[o + 4] = points[i * 3 + 4];
      p.segments[o + 5] = z(i + 1);
    }
    p.buffer.needsUpdate = true;
    return p;
  }, [points, flat]);
  useFatLine(parts, color);
  useFrame(() => {
    parts.geometry.instanceCount = Math.max(0, Math.min(getCount() - 1, parts.maxSegments));
  });
  return <primitive object={parts.line} />;
}

/** The whole target path as a static dashed line. */
export function TargetPath({ target }: { target: Float64Array }) {
  const points = useMemo(() => {
    const n = target.length / 4;
    const step = Math.max(1, Math.floor(n / 1500));
    const out: [number, number, number][] = [];
    for (let i = 0; i < n; i += step) out.push([target[i * 4], target[i * 4 + 1], target[i * 4 + 2]]);
    return out;
  }, [target]);
  if (points.length < 2) return null;
  return (
    <Line points={points} color={SIGNAL_RED} lineWidth={1.2} dashed dashSize={0.12} gapSize={0.1} transparent opacity={0.55} />
  );
}
