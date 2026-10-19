"""Benchmark do Stenciled Shadow Volumes (z-fail).

Varia duas coisas:
  - nº de luzes (padrão: 1, 2, 4, 8)
  - complexidade da malha (esferas com subdivisão 0..5 = 20 a 20480 triângulos cada)
e mede, com sombras (z-fail) e sem sombras (só a iluminação multi-pass):
  - FPS e tempo de quadro
  - CPU: silhueta, extrusão (montagem do volume) e envio ao buffer
  - GPU (timer queries): passe ambiente, passe de stencil (volumes) e passe de luz

Uso:
    python benchmark.py            # completo (~10 min)
    python benchmark.py --quick    # teste rápido (~1-2 min)

Resultados em ./resultados/ (benchmark.csv, info.txt e duas capturas de tela).
Deixe a janela visível e não mexa no computador durante a medição.
"""
import argparse
import csv
import datetime
import math
import os
import platform
import sys
import time

import numpy as np
import glfw
import moderngl
import trimesh
from PIL import Image

import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL import GL

from shadow import ShadowCaster

pc = time.perf_counter
MAX_LIGHTS = 8

LIGHT_COLORS = [(1.0, 0.95, 0.8), (0.5, 0.7, 1.0), (1.0, 0.5, 0.5), (0.6, 1.0, 0.6),
                (1.0, 0.8, 0.3), (0.8, 0.5, 1.0), (0.4, 1.0, 1.0), (1.0, 0.6, 0.9)]

# três esferas projetam sombra; o chão só recebe
SPHERE_POS = [(-3.2, 1.0, 0.0), (0.0, 1.0, -2.5), (3.2, 1.0, 0.0)]


# ---------------------------------------------------------------- matemática
def light_position(i, t):
    a = t * (0.5 + 0.12 * i) + i * 2.4
    r = 3.5 + 0.4 * (i % 3)
    return np.array([r * math.cos(a), 3.5 + 0.5 * (i % 3), r * math.sin(a)])


def perspective_infinite(fovy, aspect, near):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    eps = 1e-6
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = -1.0 + eps
    m[2, 3] = (-2.0 + eps) * near
    m[3, 2] = -1
    return m


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def translate(x, y, z):
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    return m


def to_gl(m):
    return m.T.astype("f4").tobytes()


def flat_vertices(mesh):
    tris = mesh.triangles.reshape(-1, 3)
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    return np.hstack([tris, normals]).astype("f4")


# ---------------------------------------------------------------- shaders
VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_pos;
out vec3 v_norm;
void main() {
    v_pos = (model * vec4(in_position, 1.0)).xyz;
    v_norm = mat3(model) * in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

FRAG = """
#version 330
uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 cam_pos;
uniform vec3 color;
uniform float k_amb;
uniform float k_lit;
uniform float intensity;
in vec3 v_pos;
in vec3 v_norm;
out vec4 f_color;
void main() {
    vec3 N = normalize(v_norm);
    vec3 L = normalize(light_pos - v_pos);
    vec3 V = normalize(cam_pos - v_pos);
    vec3 R = reflect(-L, N);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(R, V), 0.0), 32.0) * 0.4;
    float d = length(light_pos - v_pos);
    float att = 1.0 / (1.0 + 0.05 * d + 0.01 * d * d);
    vec3 amb = color * 0.15;
    vec3 lit = (color * diff + vec3(spec)) * att * intensity * light_color;
    f_color = vec4(amb * k_amb + lit * k_lit, 1.0);
}
"""

VOL_VERT = """
#version 330
uniform mat4 vp;
in vec4 in_pos;
void main() { gl_Position = vp * in_pos; }
"""

VOL_FRAG = """
#version 330
out vec4 f_color;
void main() { f_color = vec4(0.0); }
"""


# ---------------------------------------------------------------- argumentos
def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark de shadow volumes (z-fail)")
    ap.add_argument("--quick", action="store_true", help="teste rápido (1 s por medição, 1 repetição)")
    ap.add_argument("--seconds", type=float, default=3.0, help="duração mínima de cada medição (s)")
    ap.add_argument("--repeats", type=int, default=3, help="repetições por configuração")
    ap.add_argument("--warmup", type=int, default=20, help="quadros de aquecimento (descartados)")
    ap.add_argument("--min-frames", type=int, default=30, help="mínimo de quadros por medição")
    ap.add_argument("--levels", default="0,1,2,3,4,5", help="subdivisões da esfera (nível 0 = 20 tri)")
    ap.add_argument("--lights", default="1,2,4,8", help="números de luzes a testar")
    ap.add_argument("--size", default="1280x720", help="resolução, ex.: 1280x720")
    ap.add_argument("--out", default="resultados", help="pasta de saída")
    a = ap.parse_args()
    if a.quick:
        a.seconds, a.repeats, a.warmup, a.min_frames = 1.0, 1, 10, 15
    a.levels = [int(x) for x in a.levels.split(",")]
    a.lights = [int(x) for x in a.lights.split(",")]
    a.width, a.height = [int(x) for x in a.size.lower().split("x")]
    if max(a.lights) > MAX_LIGHTS:
        sys.exit(f"No máximo {MAX_LIGHTS} luzes.")
    return a


# ---------------------------------------------------------------- renderizador
class Bench:
    def __init__(self, args):
        self.args = args
        if not glfw.init():
            sys.exit("Falha ao iniciar GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.STENCIL_BITS, 8)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.window = glfw.create_window(args.width, args.height, "Benchmark", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit("Falha ao criar janela (driver sem OpenGL 3.3?)")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.ctx = ctx = moderngl.create_context()
        self.gl_version = ctx.info["GL_VERSION"]
        self.gl_renderer = ctx.info["GL_RENDERER"]
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)

        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vol_prog = ctx.program(vertex_shader=VOL_VERT, fragment_shader=VOL_FRAG)

        floor = trimesh.creation.box(extents=(20, 0.2, 20))
        floor.apply_translation([0, -0.1, 0])
        self.floor_vao = self._make_vao(floor)
        self.floor_tris = len(floor.faces)

        # buffer dinâmico do volume (cresce sob demanda)
        self.vol_cap = 0
        self.vol_vbo = None
        self.vol_vao = None

        self.queries = self._init_timers()

    # -- infraestrutura ------------------------------------------------------
    def _make_vao(self, mesh):
        vbo = self.ctx.buffer(flat_vertices(mesh).tobytes())
        return self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_position", "in_normal")])

    def _init_timers(self):
        """Timer queries da GPU: 1 ambiente + (stencil, luz) por luz. None se indisponível."""
        try:
            qs = [int(np.asarray(GL.glGenQueries(1)).ravel()[0]) for _ in range(1 + 2 * MAX_LIGHTS)]
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, qs[0])
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
            self.ctx.finish()
            self._read_query(qs[0])
            return qs
        except Exception as e:   # noqa: BLE001
            print(f"[aviso] timers de GPU indisponíveis ({e}); coluna gpu_* ficará vazia")
            return None

    @staticmethod
    def _read_query(q):
        v = GL.glGetQueryObjectuiv(q, GL.GL_QUERY_RESULT)
        return float(np.asarray(v).ravel()[0]) / 1e6   # ns -> ms

    def _vol_write(self, arr):
        nbytes = arr.nbytes
        if nbytes > self.vol_cap:
            if self.vol_vbo is not None:
                self.vol_vao.release()
                self.vol_vbo.release()
            self.vol_cap = max(int(nbytes * 1.5), 1 << 20)
            self.vol_vbo = self.ctx.buffer(reserve=self.vol_cap)
            self.vol_vao = self.ctx.vertex_array(self.vol_prog, [(self.vol_vbo, "4f", "in_pos")])
        self.vol_vbo.write(arr.tobytes())

    # -- cena ----------------------------------------------------------------
    def build_scene(self, subdiv):
        sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
        vao = self._make_vao(sphere)
        colors = [(0.8, 0.3, 0.3), (0.3, 0.5, 0.9), (0.3, 0.8, 0.4)]
        objects = [(self.floor_vao, translate(0, 0, 0), (0.6, 0.6, 0.6))]
        casters = []
        for pos, col in zip(SPHERE_POS, colors):
            m = translate(*pos)
            objects.append((vao, m, col))
            casters.append(ShadowCaster(sphere, m))
        tris_obj = len(sphere.faces)
        return {"objects": objects, "casters": casters, "vao": vao,
                "tris_obj": tris_obj, "tris_total": tris_obj * len(SPHERE_POS)}

    def free_scene(self, scene):
        scene["vao"].release()

    def _draw_scene(self, scene, pv, k_amb, k_lit):
        p = self.prog
        p["k_amb"].value = k_amb
        p["k_lit"].value = k_lit
        for vao, model, color in scene["objects"]:
            p["model"].write(to_gl(model))
            p["mvp"].write(to_gl(pv @ model))
            p["color"].value = color
            vao.render()

    # -- um quadro -----------------------------------------------------------
    def frame(self, scene, idx, n_lights, shadows, shot_path=None):
        ctx, prog, vol_prog = self.ctx, self.prog, self.vol_prog
        t_start = pc()

        # câmera e luzes: função determinística do índice do quadro
        a = idx * 0.004
        cam_pos = np.array([10.0 * math.sin(a), 4.5, 10.0 * math.cos(a)])
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        ctx.viewport = (0, 0, fb_w, fb_h)
        pv = perspective_infinite(60.0, fb_w / max(fb_h, 1), 0.1) @ \
            look_at(cam_pos, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        lights = [light_position(i, idx / 60.0) for i in range(n_lights)]
        casters = scene["casters"]

        r = {"sil": 0.0, "ext": 0.0, "up": 0.0, "vol_tris": 0}
        spans = []

        # ---- CPU: silhueta -> extrusão -> envio ao buffer
        if shadows:
            t0 = pc()
            sils = [[c.silhouette(L) for c in casters] for L in lights]
            t1 = pc()
            chunks, off = [], 0
            for L, per_caster in zip(lights, sils):
                v = np.concatenate([c.assemble_volume(L, e, lit)
                                    for c, (e, lit) in zip(casters, per_caster)])
                chunks.append(v)
                spans.append((off, len(v)))
                off += len(v)
            all_vol = np.concatenate(chunks)
            t2 = pc()
            self._vol_write(all_vol)
            t3 = pc()
            r["sil"], r["ext"], r["up"] = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
            r["vol_tris"] = len(all_vol) // 3

        q = self.queries
        use_q = q is not None

        def qbegin(i):
            if use_q:
                GL.glBeginQuery(GL.GL_TIME_ELAPSED, q[i])

        def qend():
            if use_q:
                GL.glEndQuery(GL.GL_TIME_ELAPSED)

        prog["cam_pos"].value = tuple(cam_pos)
        prog["intensity"].value = 1.5 / math.sqrt(n_lights)

        # ---- GPU passe 1: ambiente
        GL.glColorMask(True, True, True, True)
        GL.glDepthMask(True)
        qbegin(0)
        ctx.clear(0.02, 0.02, 0.05, depth=1.0)
        GL.glStencilMask(0xFF)
        GL.glClearStencil(0)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        ctx.enable(moderngl.DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        ctx.enable(moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)
        prog["light_pos"].value = (0.0, 0.0, 0.0)
        prog["light_color"].value = (0.0, 0.0, 0.0)
        self._draw_scene(scene, pv, 1.0, 0.0)
        qend()

        # ---- GPU passes por luz
        GL.glDepthMask(False)
        if shadows:
            GL.glEnable(GL.GL_STENCIL_TEST)
            vol_prog["vp"].write(to_gl(pv))
        for i, L in enumerate(lights):
            if shadows:
                first, count = spans[i]
                GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
                qbegin(1 + 2 * i)
                GL.glColorMask(False, False, False, False)
                ctx.disable(moderngl.CULL_FACE)
                ctx.disable(moderngl.BLEND)
                GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
                GL.glStencilOpSeparate(GL.GL_BACK, GL.GL_KEEP, GL.GL_INCR_WRAP, GL.GL_KEEP)
                GL.glStencilOpSeparate(GL.GL_FRONT, GL.GL_KEEP, GL.GL_DECR_WRAP, GL.GL_KEEP)
                if count:
                    self.vol_vao.render(moderngl.TRIANGLES, vertices=count, first=first)
                GL.glColorMask(True, True, True, True)
                GL.glStencilFunc(GL.GL_EQUAL, 0, 0xFF)
                GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_KEEP)
                qend()
            qbegin(2 + 2 * i)
            ctx.enable(moderngl.CULL_FACE)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            prog["light_pos"].value = tuple(L)
            prog["light_color"].value = LIGHT_COLORS[i]
            self._draw_scene(scene, pv, 0.0, 1.0)
            qend()
        GL.glDisable(GL.GL_STENCIL_TEST)
        ctx.disable(moderngl.BLEND)
        GL.glDepthMask(True)

        if shot_path:
            data = ctx.screen.read(components=3, alignment=1)
            img = Image.frombytes("RGB", (fb_w, fb_h), data).transpose(Image.FLIP_TOP_BOTTOM)
            img.save(shot_path)

        glfw.swap_buffers(self.window)
        ctx.finish()   # serializa CPU+GPU: o tempo de quadro inclui tudo
        r["frame"] = (pc() - t_start) * 1e3

        r["gpu_amb"] = r["gpu_sten"] = r["gpu_lit"] = float("nan")
        if use_q:
            r["gpu_amb"] = self._read_query(q[0])
            r["gpu_sten"] = sum(self._read_query(q[1 + 2 * i]) for i in range(n_lights)) if shadows else 0.0
            r["gpu_lit"] = sum(self._read_query(q[2 + 2 * i]) for i in range(n_lights))
        glfw.poll_events()
        return r

    # -- uma medição ---------------------------------------------------------
    def measure(self, scene, n_lights, shadows):
        a = self.args
        samples = {k: [] for k in ("frame", "sil", "ext", "up", "gpu_amb", "gpu_sten", "gpu_lit", "vol_tris")}
        repeat_fps = []
        idx = 0
        for _ in range(a.warmup):
            self.frame(scene, idx, n_lights, shadows)
            idx += 1
        for _ in range(a.repeats):
            t0, n, rep_ms = pc(), 0, []
            while (pc() - t0) < a.seconds or n < a.min_frames:
                if glfw.window_should_close(self.window):
                    raise KeyboardInterrupt
                r = self.frame(scene, idx, n_lights, shadows)
                idx += 1
                n += 1
                rep_ms.append(r["frame"])
                for k in samples:
                    samples[k].append(r[k])
            repeat_fps.append(1000.0 / float(np.mean(rep_ms)))
        m = {k: float(np.mean(v)) for k, v in samples.items()}
        m["frame_std"] = float(np.std(samples["frame"]))
        m["frames"] = len(samples["frame"])
        m["fps"] = 1000.0 / m["frame"]
        m["fps_repeat_std"] = float(np.std(repeat_fps))
        return m


# ---------------------------------------------------------------- main
FIELDS = ["subdiv", "tris_per_object", "tris_total", "lights", "shadows", "frames",
          "fps", "fps_repeat_std", "frame_ms", "frame_ms_std",
          "cpu_sil_ms", "cpu_ext_ms", "cpu_upload_ms", "cpu_total_ms",
          "gpu_ambient_ms", "gpu_stencil_ms", "gpu_light_ms", "gpu_total_ms",
          "volume_tris"]


def write_info(path, bench, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"data: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"GPU (GL_RENDERER): {bench.gl_renderer}\n")
        f.write(f"OpenGL (GL_VERSION): {bench.gl_version}\n")
        f.write(f"CPU: {platform.processor() or 'desconhecido'}\n")
        f.write(f"SO: {platform.platform()}\n")
        f.write(f"Python: {sys.version.split()[0]} | numpy {np.__version__} | trimesh {trimesh.__version__} "
                f"| moderngl {moderngl.__version__} | glfw {glfw.__version__ if hasattr(glfw, '__version__') else '?'}\n")
        f.write(f"resolução: {args.width}x{args.height} | vsync desligado\n")
        f.write(f"medição: >= {args.seconds}s e >= {args.min_frames} quadros por repetição, "
                f"{args.repeats} repetições, {args.warmup} quadros de aquecimento\n")
        f.write(f"níveis de subdivisão: {args.levels} | luzes: {args.lights}\n")
        f.write("cena: 3 esferas (projetam sombra) + chão (só recebe); câmera e luzes determinísticas por quadro\n")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    bench = Bench(args)
    print(f"GPU: {bench.gl_renderer} | OpenGL {bench.gl_version}")
    write_info(os.path.join(args.out, "info.txt"), bench, args)

    # capturas de tela para o artigo (nível intermediário, 4 luzes)
    lvl = args.levels[min(2, len(args.levels) - 1)]
    scene = bench.build_scene(lvl)
    nl = min(4, max(args.lights))
    for sh, name in ((False, "cena_sem_sombras.png"), (True, "cena_com_sombras.png")):
        for k in range(3):
            bench.frame(scene, 120, nl, sh, shot_path=os.path.join(args.out, name) if k == 2 else None)
    bench.free_scene(scene)

    total = len(args.levels) * 2 * len(args.lights)
    done = 0
    csv_path = os.path.join(args.out, "benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        try:
            for lvl in args.levels:
                scene = bench.build_scene(lvl)
                for shadows in (False, True):
                    for n in args.lights:
                        done += 1
                        glfw.set_window_title(bench.window, f"Benchmark {done}/{total}")
                        m = bench.measure(scene, n, shadows)
                        cpu = m["sil"] + m["ext"] + m["up"]
                        gpu = m["gpu_amb"] + m["gpu_sten"] + m["gpu_lit"]
                        w.writerow({
                            "subdiv": lvl, "tris_per_object": scene["tris_obj"],
                            "tris_total": scene["tris_total"], "lights": n, "shadows": int(shadows),
                            "frames": m["frames"], "fps": round(m["fps"], 2),
                            "fps_repeat_std": round(m["fps_repeat_std"], 2),
                            "frame_ms": round(m["frame"], 3), "frame_ms_std": round(m["frame_std"], 3),
                            "cpu_sil_ms": round(m["sil"], 3), "cpu_ext_ms": round(m["ext"], 3),
                            "cpu_upload_ms": round(m["up"], 3), "cpu_total_ms": round(cpu, 3),
                            "gpu_ambient_ms": round(m["gpu_amb"], 3), "gpu_stencil_ms": round(m["gpu_sten"], 3),
                            "gpu_light_ms": round(m["gpu_lit"], 3), "gpu_total_ms": round(gpu, 3),
                            "volume_tris": int(m["vol_tris"]),
                        })
                        fh.flush()
                        print(f"[{done:2d}/{total}] {scene['tris_total']:6d} tri | {n} luz(es) | "
                              f"sombras {'ON ' if shadows else 'OFF'} -> {m['fps']:7.1f} FPS | "
                              f"quadro {m['frame']:7.2f} ms | CPU {cpu:7.2f} ms | GPU {gpu:7.2f} ms")
                bench.free_scene(scene)
        except KeyboardInterrupt:
            print("\nInterrompido. Resultados parciais salvos em", csv_path)
    glfw.terminate()
    print("\nPronto! Arquivos em:", os.path.abspath(args.out))
    print("Para gerar os gráficos: python plot_benchmark.py")


if __name__ == "__main__":
    main()