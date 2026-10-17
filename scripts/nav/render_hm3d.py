#!/usr/bin/env python
"""Render navigation episodes in HM3D (semantic scenes) for multimodal PISCO: RGB frames + per-frame
visible-object ground truth from the SEMANTIC sensor.

Runs under the habitat-sim 0.3.3 env (python 3.9!) -> no pisco imports, stdlib + numpy + PIL only.
Adapted from silver-nav/data/render_clips.py (make_sim / roll_out / pick_start).

Per scene: --episodes random-walk trajectories of --steps actions (forward with occasional turns, turning
away when blocked). Per frame: RGB JPEG (resized so the long side is --long_side px) and, from the semantic
sensor, the visible object categories with pixel share and horizontal position (left / center / right).
Output: <out>/<scene>__e<k>/frames/NNN.jpg + meta.json {scene, split, actions, poses, frames:[{file, objects:
[{category, area, pos}]}]}. Objects are HM3D-Sem categories mapped from instance ids via <scene>.semantic.txt.

  pixi run --manifest-path .../habitat_env/pixi.toml python scripts/nav/render_hm3d.py \
      --split train --n_scenes 2 --episodes 2 --steps 30 --out data/nav_hm3d_smoke
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import habitat_sim

SCENES_ROOT = Path(os.environ.get("HABITAT_SCENES", "/nfs/data/habitat/scene_datasets")) / "hm3d"
W, H = 832, 480
HFOV = 90.0
EYE = 1.0
STEP_FWD, STEP_TURN = 0.25, 15.0
IGNORE = {"floor", "wall", "ceiling", "unknown", "", "wall panel", "floor mat", "door frame", "window frame",
          "stair", "stairs", "beam", "pillar", "column", "baseboard", "molding", "trim", "roof"}


def semantic_scenes(split):
    out = []
    for d in sorted((SCENES_ROOT / split).iterdir()):
        sem = list(d.glob("*.semantic.txt"))
        glb = list(d.glob("*.basis.glb"))
        if sem and glb:
            out.append((d.name, glb[0], (d / glb[0].name.replace(".glb", ".navmesh")), sem[0]))
    return out


def load_categories(sem_txt):
    cats = {}
    with open(sem_txt) as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].isdigit():
                cats[int(row[0])] = row[2].strip().strip('"').lower()
    return cats


def make_sim(glb, navmesh, seed):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(SCENES_ROOT / "hm3d_annotated_basis.scene_dataset_config.json")
    backend.gpu_device_id = 0
    backend.enable_physics = False
    backend.random_seed = seed
    rgb = habitat_sim.CameraSensorSpec(); rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [H, W]; rgb.position = [0.0, EYE, 0.0]
    sem = habitat_sim.CameraSensorSpec(); sem.uuid = "sem"; sem.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem.resolution = [H, W]; sem.position = [0.0, EYE, 0.0]
    for s in (rgb, sem):
        try:
            s.hfov = HFOV
        except TypeError:
            pass
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [rgb, sem]
    agent.height, agent.radius = 0.88, 0.18
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=STEP_FWD)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=STEP_TURN)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=STEP_TURN)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    if navmesh is not None and Path(navmesh).exists():
        sim.pathfinder.load_nav_mesh(str(navmesh))
    if not sim.pathfinder.is_loaded:
        st = habitat_sim.NavMeshSettings(); st.set_defaults(); st.agent_radius = 0.18; st.agent_height = 0.88
        sim.recompute_navmesh(sim.pathfinder, st)
    return sim


def objects_in(sem_img, cats, min_share=0.004):
    """Visible object categories from a semantic frame: [{category, area, pos}] sorted by area."""
    ids, counts = np.unique(sem_img, return_counts=True)
    total = sem_img.size
    cols = np.arange(sem_img.shape[1])[None, :]
    out = {}
    for i, c in zip(ids, counts):
        cat = cats.get(int(i), "")
        if cat in IGNORE or c / total < min_share:
            continue
        mask = sem_img == i
        xmean = float((mask * cols).sum() / max(1, mask.sum())) / sem_img.shape[1]
        pos = "left" if xmean < 0.38 else ("right" if xmean > 0.62 else "center")
        o = out.setdefault(cat, {"category": cat, "area": 0.0, "pos": pos, "count": 0})
        o["area"] += float(c / total); o["count"] += 1
    return sorted(out.values(), key=lambda o: -o["area"])


def random_walk(rng, steps):
    acts = []
    while len(acts) < steps:
        r = rng.random()
        if r < 0.7:
            acts.append("move_forward")
        else:
            acts.extend([rng.choice(["turn_left", "turn_right"])] * rng.randint(1, 3))
    return acts[:steps]


def pose(agent):
    s = agent.get_state()
    return {"position": [float(v) for v in s.position],
            "rotation": [float(s.rotation.w), float(s.rotation.x), float(s.rotation.y), float(s.rotation.z)]}


def run_episode(sim, cats, start_pos, start_rot, actions, out_dir, long_side, quality):
    import quaternion  # noqa: F401  (habitat-sim dependency)
    agent = sim.get_agent(0)
    st = agent.get_state(); st.position = start_pos; st.rotation = start_rot; agent.set_state(st)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    frames, poses, done_actions = [], [], []
    obs = sim.get_sensor_observations()
    rng = random.Random(len(actions))

    def save(i, obs):
        rgb = Image.fromarray(obs["rgb"][..., :3]); rgb.thumbnail((long_side, long_side))
        fn = f"{i:03d}.jpg"; rgb.save(out_dir / "frames" / fn, quality=quality, optimize=True)
        frames.append({"file": f"frames/{fn}", "objects": objects_in(obs["sem"], cats)})
        poses.append(pose(agent))

    save(0, obs)
    last = np.array(agent.get_state().position)
    for a in actions:
        obs = sim.step(a)
        cur = np.array(agent.get_state().position)
        if a == "move_forward" and np.linalg.norm(cur - last) < 0.05:   # blocked: turn away instead
            a = rng.choice(["turn_left", "turn_right"]); obs = sim.step(a)
        last = np.array(agent.get_state().position)
        done_actions.append(a)
        save(len(frames), obs)
    return frames, poses, done_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_scenes", type=int, default=2)
    ap.add_argument("--scene_offset", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2, help="per scene")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--long_side", type=int, default=832)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scenes = semantic_scenes(args.split)[args.scene_offset: args.scene_offset + args.n_scenes]
    print(f"[render] {len(scenes)} semantic scenes from {args.split} (of {len(semantic_scenes(args.split))})", flush=True)
    import quaternion
    manifest = []
    for name, glb, navmesh, sem_txt in scenes:
        cats = load_categories(sem_txt)
        sim = make_sim(glb, navmesh, args.seed)
        try:
            for k in range(args.episodes):
                ep = f"{name}__e{k}"
                pos = sim.pathfinder.get_random_navigable_point()
                if not np.isfinite(pos).all():
                    continue
                rot = quaternion.from_rotation_vector([0.0, rng.uniform(0, 2 * np.pi), 0.0])
                actions = random_walk(rng, args.steps)
                ep_dir = out / ep
                frames, poses, done = run_episode(sim, cats, pos, rot, actions, ep_dir, args.long_side, args.quality)
                meta = {"episode": ep, "scene": name, "split": args.split, "scene_glb": str(glb), "actions": done,
                        "poses": poses, "frames": frames, "hfov_deg": HFOV, "width": W, "height": H,
                        "n_categories_in_scene": len(set(cats.values()))}
                json.dump(meta, open(ep_dir / "meta.json", "w"))
                n_obj = sum(len(f["objects"]) for f in frames) / max(1, len(frames))
                manifest.append({"episode": ep, "n_frames": len(frames), "objects_per_frame": round(n_obj, 1)})
                print(f"[render] {ep}: {len(frames)} frames, {n_obj:.1f} objects/frame, "
                      f"e.g. {[o['category'] for o in frames[0]['objects'][:5]]}", flush=True)
        finally:
            sim.close()
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"[render] wrote {len(manifest)} episodes -> {out}")


if __name__ == "__main__":
    main()
