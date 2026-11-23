import argparse
import math
import time

import numpy as np
import glfw
import moderngl
import trimesh
from PIL import Image

import OpenGL
OpenGL.ERROR_CHECKING = False   # evita overhead nas chamadas (importante p/ benchmark)
from OpenGL import GL

from shadow import ShadowCaster
from hud import Hud, PANEL_X, PANEL_Y, PW, PH

W, H = 1280, 720
MAX_LIGHTS = 8

LIGHT_COLORS = [(1.0, 0.95, 0.8), (0.5, 0.7, 1.0), (1.0, 0.5, 0.5), (0.6, 1.0, 0.6),
                (1.0, 0.8, 0.3), (0.8, 0.5, 1.0), (0.4, 1.0, 1.0), (1.0, 0.6, 0.9)]


# ---------- matrizes (convenção OpenGL, vetor coluna) ----------
def perspective_infinite(fovy, aspect, near):
    """Projeção com plano far no infinito (necessária para z-fail)."""
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    eps = 1e-6   # mantém pontos w=0 um fio antes do far, sem serem cortados
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


def scale_m(s):
    return np.diag([s, s, s, 1.0])


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def to_gl(m):
    # numpy é row-major, o GLSL espera column-major -> transpõe
    return m.T.astype("f4").tobytes()


# ---------- cenas ----------
# Cada parte é (malha em coordenadas de mundo, cor, projeta_sombra).
# Todas as partes que projetam sombra viram UMA malha fechada só (um ShadowCaster),
# então o custo em Python não cresce com o número de objetos, só com os triângulos.
def box(w, h, d):
    return trimesh.creation.box(extents=(w, h, d))


def place(mesh, color, M):
    m = mesh.copy()
    m.apply_transform(M)
    return (m, color, True)


def boneco(parent, shirt, pants, skin):
    """Personagem simples (~2 unidades de altura) feito de blocos e uma esfera."""
    out = []
    leg = box(0.22, 0.85, 0.22)
    for sx in (-0.15, 0.15):
        out.append(place(leg, pants, parent @ translate(sx, 0.425, 0)))
    out.append(place(box(0.6, 0.8, 0.32), shirt, parent @ translate(0, 1.2, 0)))
    arm = box(0.16, 0.75, 0.16)
    for sx, ang in ((-0.42, -12), (0.42, 12)):
        out.append(place(arm, shirt, parent @ translate(sx, 1.2, 0) @ rot_z(ang)))
    head = trimesh.creation.icosphere(subdivisions=1, radius=0.22)
    out.append(place(head, skin, parent @ translate(0, 1.84, 0)))
    return out


def poste(x, z):
    cyl = trimesh.creation.cylinder(radius=0.12, height=4.5, sections=12)
    cyl.apply_transform(rot_x(90))
    lamp = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
    return [place(cyl, (0.25, 0.25, 0.28), translate(x, 2.25, z)),
            place(lamp, (0.9, 0.85, 0.6), translate(x, 4.6, z))]


def carro(x, z, ry, color):
    P = translate(x, 0, z) @ rot_y(ry)
    return [place(box(2.0, 0.6, 0.95), color, P @ translate(0, 0.3, 0)),
            place(box(1.0, 0.5, 0.85), (0.75, 0.8, 0.9), P @ translate(-0.1, 0.8, 0))]


SKIN = (0.9, 0.72, 0.55)


def scene_padrao():
    return [place(box(1.5, 1.5, 1.5), (0.8, 0.3, 0.3), translate(-2, 0.75, 0)),
            place(trimesh.creation.icosphere(subdivisions=2, radius=1.0), (0.3, 0.5, 0.9), translate(2, 1.0, 0))]


def scene_predios():
    specs = [(-8, -4, 3.0, 3.0, 7.0, (0.62, 0.62, 0.68)), (-3, -6, 2.5, 2.5, 5.0, (0.75, 0.6, 0.5)),
             (3, -5, 3.5, 3.0, 9.0, (0.55, 0.65, 0.75)), (8, -1, 2.5, 4.0, 4.5, (0.7, 0.7, 0.55)),
             (-6, 3, 3.0, 2.5, 3.5, (0.65, 0.5, 0.55)), (0, 3, 2.0, 2.0, 6.0, (0.6, 0.7, 0.6)),
             (6, 5, 3.0, 3.0, 5.5, (0.75, 0.65, 0.7))]
    return [place(box(w, h, d), col, translate(x, h / 2, z)) for x, z, w, d, h, col in specs]


def scene_bonecos():
    return (boneco(translate(-1.6, 0, 0.6) @ rot_y(25), (0.2, 0.35, 0.8), (0.15, 0.15, 0.2), SKIN)
            + boneco(translate(1.8, 0, -0.6) @ rot_y(-35) @ scale_m(1.2), (0.8, 0.2, 0.2), (0.25, 0.2, 0.15), SKIN))


def scene_cidade():
    out = []
    for x, z, h, col in ((-6, -6, 8, (0.6, 0.62, 0.7)), (-6, 6, 6, (0.75, 0.6, 0.5)),
                         (6, -6, 10, (0.5, 0.62, 0.78)), (6, 6, 7, (0.7, 0.68, 0.55))):
        out.append(place(box(4, h, 4), col, translate(x, h / 2, z)))
    for x in (-11, 11):
        out.append(place(box(3, 3, 5), (0.7, 0.55, 0.55), translate(x, 1.5, 0)))
    out += boneco(translate(-1.2, 0, 2.5) @ rot_y(25), (0.2, 0.35, 0.8), (0.15, 0.15, 0.2), SKIN)
    out += boneco(translate(1.5, 0, -3) @ rot_y(-35) @ scale_m(1.2), (0.8, 0.2, 0.2), (0.25, 0.2, 0.15), SKIN)
    for x, z in ((3.2, 3.2), (-3.2, 3.2), (3.2, -3.2), (-3.2, -3.2)):
        out += poste(x, z)
    out += carro(-2.3, -1.0, 90, (0.8, 0.25, 0.25))
    out += carro(2.5, 1.5, 90, (0.25, 0.5, 0.8))
    return out


# floor: lado do chão | light: (raio, variação do raio, altura, variação da altura) da órbita
# gain: intensidade base das luzes | cam: ((x, y, z), yaw, pitch) inicial
SCENES = {
    "padrao":  {"label": "Padrão",  "builder": scene_padrao,  "floor": 20, "floor_color": (0.6, 0.6, 0.6),
                "light": (3.5, 0.4, 3.5, 0.5), "gain": 1.5, "cam": ((0.0, 3.0, 9.0), -90.0, -15.0)},
    "predios": {"label": "Prédios", "builder": scene_predios, "floor": 34, "floor_color": (0.42, 0.42, 0.45),
                "light": (11.0, 1.5, 11.0, 2.0), "gain": 4.0, "cam": ((0.0, 12.0, 26.0), -90.0, -26.0)},
    "bonecos": {"label": "Bonecos", "builder": scene_bonecos, "floor": 20, "floor_color": (0.55, 0.58, 0.55),
                "light": (4.5, 0.5, 4.5, 0.7), "gain": 1.8, "cam": ((0.0, 2.8, 8.0), -90.0, -12.0)},
    "cidade":  {"label": "Cidade",  "builder": scene_cidade,  "floor": 36, "floor_color": (0.35, 0.35, 0.38),
                "light": (12.0, 1.5, 12.0, 2.0), "gain": 4.0, "cam": ((0.0, 14.0, 29.0), -90.0, -30.0)},
}
SCENE_ORDER = ["padrao", "predios", "bonecos", "cidade"]


def light_position(i, t, cfg):
    r0, dr, h0, dh = cfg["light"]
    a = t * (0.5 + 0.12 * i) + i * 2.4
    r = r0 + dr * (i % 3)
    return np.array([r * math.cos(a), h0 + dh * (i % 3), r * math.sin(a)])


# ---------- shaders ----------
VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;     // já em coordenadas de mundo
in vec3 in_normal;
in vec3 in_color;
out vec3 v_pos;
out vec3 v_norm;
out vec3 v_color;
void main() {
    v_pos = in_position;
    v_norm = in_normal;
    v_color = in_color;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

# k_amb / k_lit escolhem o passe: ambiente (1,0) ou luz de uma lâmpada (0,1)
FRAG = """
#version 330
uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 cam_pos;
uniform float k_amb;
uniform float k_lit;
uniform float intensity;
in vec3 v_pos;
in vec3 v_norm;
in vec3 v_color;
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
    vec3 amb = v_color * 0.15;
    vec3 lit = (v_color * diff + vec3(spec)) * att * intensity * light_color;
    f_color = vec4(amb * k_amb + lit * k_lit, 1.0);
}
"""

VOL_VERT = """
#version 330
uniform mat4 vp;
in vec4 in_pos;          // homogêneo: w = 0 -> ponto no infinito
void main() { gl_Position = vp * in_pos; }
"""

VOL_FRAG = """
#version 330
uniform vec4 dbg;
out vec4 f_color;
void main() { f_color = dbg; }
"""

FLAT_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
void main() { gl_Position = mvp * vec4(in_position, 1.0); }
"""

FLAT_FRAG = """
#version 330
uniform vec3 color;
out vec4 f_color;
void main() { f_color = vec4(color, 1.0); }
"""

# Painel (HUD): um retângulo em pixels com a textura desenhada pelo Pillow
HUD_VERT = """
#version 330
uniform vec2 screen;     // tamanho do framebuffer em px
uniform vec4 rect;       // x, y, largura, altura em px (origem no canto superior esquerdo)
in vec2 in_vert;         // 0..1
out vec2 uv;
void main() {
    vec2 px = rect.xy + in_vert * rect.zw;
    gl_Position = vec4(px.x / screen.x * 2.0 - 1.0, 1.0 - px.y / screen.y * 2.0, 0.0, 1.0);
    uv = in_vert;
}
"""

HUD_FRAG = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 f_color;
void main() { f_color = texture(tex, uv); }
"""

# ---------- argumentos (opcionais; sem argumentos abre normalmente) ----------
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="padrao", choices=SCENE_ORDER)
ap.add_argument("--lights", type=int, default=2)
ap.add_argument("--shot", default=None, help="salva uma captura de tela nesse arquivo e sai")
args = ap.parse_args()

# Estado controlado pelos botões e pelo teclado
state = {
    "lights": max(1, min(MAX_LIGHTS, args.lights)),  # botões -/+ ou teclas 1..8
    "scene": args.scene,     # botões de cena ou tecla C
    "shadows": True,         # X: liga/desliga as sombras (útil para comparar custo)
    "volume": False,         # V: mostra o volume de sombra da luz 0
    "silhouette": False,     # E: mostra as arestas de silhueta da luz 0
    "paused": False,         # P: pausa o movimento das luzes
    "markers": False,        # L: mostra uma bolinha em cada luz
    "ui": False,             # TAB: mouse livre para clicar nos botões
}

# ---------- janela ----------
if not glfw.init():
    raise SystemExit("Falha ao iniciar GLFW")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
glfw.window_hint(glfw.STENCIL_BITS, 8)   # essencial para shadow volumes
glfw.window_hint(glfw.DEPTH_BITS, 24)

window = glfw.create_window(W, H, "Shadow Volumes", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Falha ao criar janela (driver sem OpenGL 3.3?)")

glfw.make_context_current(window)
glfw.swap_interval(0)  # vsync desligado (importante para medir FPS)
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

ctx = moderngl.create_context()
print("OpenGL:", ctx.info["GL_VERSION"], "|", ctx.info["GL_RENDERER"])
ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.CULL_FACE)

prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
vol_prog = ctx.program(vertex_shader=VOL_VERT, fragment_shader=VOL_FRAG)
flat_prog = ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)

# Bolinha usada como marcador de luz
_mark = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
mark_vbo = ctx.buffer(_mark.triangles.reshape(-1, 3).astype("f4").tobytes())
mark_vao = ctx.vertex_array(flat_prog, [(mark_vbo, "3f", "in_position")])
mark_count = len(_mark.faces) * 3

# Painel
hud = Hud()
hud_tex = ctx.texture((PW, PH), 4)
hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
hud_prog["tex"].value = 0
hud_quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4").tobytes())
hud_vao = ctx.vertex_array(hud_prog, [(hud_quad, "2f", "in_vert")])

# ---------- câmera ----------
cam_pos = np.array([0.0, 3.0, 9.0])
cam = {"yaw": -90.0, "pitch": -15.0, "last": None}

# ---------- cena atual ----------
S = {}                # vbo, vao, caster, line_vbo, line_vao, cfg
vol = {"cap": 0, "vbo": None, "vao": None}   # buffer dos volumes (cresce sob demanda)


def vol_write(arr):
    nbytes = arr.nbytes
    if nbytes == 0:
        return
    if nbytes > vol["cap"]:
        if vol["vbo"] is not None:
            vol["vao"].release()
            vol["vbo"].release()
        vol["cap"] = max(int(nbytes * 1.5), 1 << 20)
        vol["vbo"] = ctx.buffer(reserve=vol["cap"])
        vol["vao"] = ctx.vertex_array(vol_prog, [(vol["vbo"], "4f", "in_pos")])
    vol["vbo"].write(arr.tobytes())


def load_scene(name):
    cfg = SCENES[name]
    for k in ("vao", "vbo", "line_vao", "line_vbo"):
        if S.get(k) is not None:
            S[k].release()

    floor = box(cfg["floor"], 0.2, cfg["floor"])
    floor.apply_translation([0, -0.1, 0])
    parts = [(floor, cfg["floor_color"], False)] + cfg["builder"]()

    chunks = []
    for mesh, color, _ in parts:
        tris = mesh.triangles.reshape(-1, 3)
        normals = np.repeat(mesh.face_normals, 3, axis=0)
        colors = np.tile(np.array(color, dtype="f4"), (len(tris), 1))
        chunks.append(np.hstack([tris, normals, colors]).astype("f4"))
    data = np.concatenate(chunks)
    S["vbo"] = ctx.buffer(data.tobytes())
    S["vao"] = ctx.vertex_array(prog, [(S["vbo"], "3f 3f 3f", "in_position", "in_normal", "in_color")])

    caster_mesh = trimesh.util.concatenate([m for m, _, casts in parts if casts])
    S["caster"] = ShadowCaster(caster_mesh, np.identity(4))
    S["cfg"] = cfg
    S["marker_size"] = 0.05 * cfg["light"][0]

    S["line_vbo"] = ctx.buffer(reserve=max(len(S["caster"].fa), 1) * 2 * 3 * 4)
    S["line_vao"] = ctx.vertex_array(flat_prog, [(S["line_vbo"], "3f", "in_position")])

    pos, yaw, pitch = cfg["cam"]
    cam_pos[:] = pos
    cam["yaw"], cam["pitch"], cam["last"] = yaw, pitch, None
    state["scene"] = name
    glfw.set_window_title(window, f"Shadow Volumes | Cena: {cfg['label']}")


# ---------- entrada (teclado e mouse) ----------
prev_keys = {}
clicks = []


def on_mouse_move(win, x, y):
    if state["ui"]:
        cam["last"] = None
        return
    if cam["last"] is None:
        cam["last"] = (x, y)
    dx, dy = x - cam["last"][0], cam["last"][1] - y
    cam["last"] = (x, y)
    cam["yaw"] += dx * 0.1
    cam["pitch"] = max(-89.0, min(89.0, cam["pitch"] + dy * 0.1))


def on_mouse_button(win, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and state["ui"]:
        clicks.append(glfw.get_cursor_pos(win))


glfw.set_cursor_pos_callback(window, on_mouse_move)
glfw.set_mouse_button_callback(window, on_mouse_button)


def key_once(key):
    """True só no quadro em que a tecla foi apertada (interruptor, não 'segurar')."""
    down = glfw.get_key(window, key) == glfw.PRESS
    was = prev_keys.get(key, False)
    prev_keys[key] = down
    return down and not was


def set_ui_mode(on):
    state["ui"] = on
    cam["last"] = None
    glfw.set_input_mode(window, glfw.CURSOR,
                        glfw.CURSOR_NORMAL if on else glfw.CURSOR_DISABLED)


def apply_action(name):
    if name == "minus":
        state["lights"] = max(1, state["lights"] - 1)
    elif name == "plus":
        state["lights"] = min(MAX_LIGHTS, state["lights"] + 1)
    elif name.startswith("scene_"):
        if name[len("scene_"):] != state["scene"]:
            load_scene(name[len("scene_"):])
    elif name in ("shadows", "volume", "silhouette", "paused", "markers"):
        state[name] = not state[name]


def front_vector():
    yaw, pitch = math.radians(cam["yaw"]), math.radians(cam["pitch"])
    return np.array([math.cos(yaw) * math.cos(pitch),
                     math.sin(pitch),
                     math.sin(yaw) * math.cos(pitch)])


def draw_scene(proj_view, k_amb, k_lit):
    prog["k_amb"].value = k_amb
    prog["k_lit"].value = k_lit
    prog["mvp"].write(to_gl(proj_view))
    S["vao"].render()


def draw_hud(fb_w, fb_h):
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    hud_tex.use(0)
    hud_prog["screen"].value = (fb_w, fb_h)
    hud_prog["rect"].value = (PANEL_X, PANEL_Y, PW, PH)
    hud_vao.render(moderngl.TRIANGLE_STRIP)
    ctx.disable(moderngl.BLEND)
    ctx.enable(moderngl.CULL_FACE)
    ctx.enable(moderngl.DEPTH_TEST)


load_scene(state["scene"])

# ---------- loop principal ----------
last_time = glfw.get_time()
fps_time, frames = last_time, 0
cpu_acc = 0.0
light_t = 0.0
shown_fps, shown_cpu = 0.0, 0.0
hud_sig = None
frame_count = 0

while not glfw.window_should_close(window):
    now = glfw.get_time()
    dt = now - last_time
    last_time = now
    if args.shot:
        dt = 1.0 / 60.0   # captura determinística

    glfw.poll_events()
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break

    # interruptores de teclado
    if key_once(glfw.KEY_TAB): set_ui_mode(not state["ui"])
    if key_once(glfw.KEY_X): apply_action("shadows")
    if key_once(glfw.KEY_V): apply_action("volume")
    if key_once(glfw.KEY_E): apply_action("silhouette")
    if key_once(glfw.KEY_P): apply_action("paused")
    if key_once(glfw.KEY_L): apply_action("markers")
    if key_once(glfw.KEY_C):
        nxt = SCENE_ORDER[(SCENE_ORDER.index(state["scene"]) + 1) % len(SCENE_ORDER)]
        apply_action("scene_" + nxt)
    for k in range(MAX_LIGHTS):
        if key_once(glfw.KEY_1 + k):
            state["lights"] = k + 1

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    win_w, win_h = glfw.get_window_size(window)

    # cliques nos botões (coordenadas da janela -> pixels do framebuffer)
    while clicks:
        cx, cy = clicks.pop(0)
        name = hud.hit_test(cx * fb_w / max(win_w, 1), cy * fb_h / max(win_h, 1))
        if name:
            apply_action(name)

    cfg = S["cfg"]
    caster = S["caster"]

    front = front_vector()
    right = np.cross(front, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    speed = 5.0 * dt * (2.0 if cfg["floor"] > 20 else 1.0)
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: cam_pos += front * speed
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: cam_pos -= front * speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: cam_pos += right * speed
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: cam_pos -= right * speed

    ctx.viewport = (0, 0, fb_w, fb_h)
    proj = perspective_infinite(60.0, fb_w / max(fb_h, 1), 0.1)
    view = look_at(cam_pos, cam_pos + front, np.array([0.0, 1.0, 0.0]))
    pv = proj @ view

    if not state["paused"]:
        light_t += dt
    n_lights = state["lights"]
    shadows = state["shadows"]
    lights = [light_position(i, light_t, cfg) for i in range(n_lights)]

    # --- CPU: silhueta + volume de sombra de cada luz (a etapa que vamos medir) ---
    spans = []
    if shadows:
        t0 = time.perf_counter()
        chunks, offset = [], 0
        for L in lights:
            v = caster.build_volume(L)
            chunks.append(v)
            spans.append((offset, len(v)))
            offset += len(v)
        vol_write(np.concatenate(chunks))
        cpu_acc += (time.perf_counter() - t0) * 1000

    prog["cam_pos"].value = tuple(cam_pos)

    # --- Passe 1: ambiente (escreve cor ambiente e profundidade) ---
    GL.glColorMask(True, True, True, True)
    GL.glDepthMask(True)
    ctx.clear(0.02, 0.02, 0.05, depth=1.0)
    GL.glStencilMask(0xFF)
    GL.glClearStencil(0)
    GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
    ctx.enable(moderngl.DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LEQUAL)   # os passes de luz redesenham na MESMA profundidade
    ctx.enable(moderngl.CULL_FACE)
    ctx.disable(moderngl.BLEND)
    prog["light_pos"].value = (0.0, 0.0, 0.0)
    prog["light_color"].value = (0.0, 0.0, 0.0)
    draw_scene(pv, 1.0, 0.0)

    # --- Passes por luz: z-fail no stencil + iluminação onde stencil == 0 ---
    prog["intensity"].value = cfg["gain"] / math.sqrt(n_lights)
    GL.glDepthMask(False)
    if shadows:
        GL.glEnable(GL.GL_STENCIL_TEST)
        vol_prog["vp"].write(to_gl(pv))
    for i, L in enumerate(lights):
        if shadows:
            first, count = spans[i]
            GL.glClear(GL.GL_STENCIL_BUFFER_BIT)

            # (a) marca o stencil com o volume (sem desenhar cor/profundidade)
            GL.glColorMask(False, False, False, False)
            ctx.disable(moderngl.CULL_FACE)          # precisamos das duas faces
            ctx.disable(moderngl.BLEND)
            GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
            # z-fail: face de trás falha no depth -> +1 ; face da frente falha -> -1
            GL.glStencilOpSeparate(GL.GL_BACK,  GL.GL_KEEP, GL.GL_INCR_WRAP, GL.GL_KEEP)
            GL.glStencilOpSeparate(GL.GL_FRONT, GL.GL_KEEP, GL.GL_DECR_WRAP, GL.GL_KEEP)
            vol_prog["dbg"].value = (0.0, 0.0, 0.0, 0.0)
            if count:
                vol["vao"].render(moderngl.TRIANGLES, vertices=count, first=first)

            GL.glColorMask(True, True, True, True)
            GL.glStencilFunc(GL.GL_EQUAL, 0, 0xFF)   # só onde NÃO há sombra
            GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_KEEP)

        # (b) soma a luz desta lâmpada
        ctx.enable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        prog["light_pos"].value = tuple(L)
        prog["light_color"].value = LIGHT_COLORS[i]
        draw_scene(pv, 0.0, 1.0)

    GL.glDisable(GL.GL_STENCIL_TEST)
    ctx.disable(moderngl.BLEND)

    # --- Marcadores: uma bolinha colorida em cada luz ---
    if state["markers"]:
        ctx.enable(moderngl.CULL_FACE)
        for i, L in enumerate(lights):
            flat_prog["mvp"].write(to_gl(pv @ translate(*L) @ scale_m(S["marker_size"])))
            flat_prog["color"].value = LIGHT_COLORS[i]
            mark_vao.render(moderngl.TRIANGLES, vertices=mark_count)

    # --- Debug: volume da luz 0 (magenta translúcido) ---
    if state["volume"] and shadows and spans:
        first, count = spans[0]
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        vol_prog["dbg"].value = (1.0, 0.0, 1.0, 0.18)
        if count:
            vol["vao"].render(moderngl.TRIANGLES, vertices=count, first=first)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)

    # --- Debug: arestas de silhueta da luz 0 ---
    if state["silhouette"]:
        ctx.disable(moderngl.DEPTH_TEST)
        flat_prog["mvp"].write(to_gl(pv))
        flat_prog["color"].value = (1.0, 0.9, 0.1)
        edges, _ = caster.silhouette(lights[0])
        if len(edges):
            pts = caster.verts[edges.ravel()].astype("f4")
            S["line_vbo"].write(pts.tobytes())
            S["line_vao"].render(moderngl.LINES, vertices=len(pts))
        ctx.enable(moderngl.DEPTH_TEST)

    # --- Painel: só redesenha a textura quando algo muda ---
    sig = (tuple(state.values()), round(shown_fps), round(shown_cpu, 2))
    if sig != hud_sig:
        hud_tex.write(hud.render(state, shown_fps, shown_cpu))
        hud_sig = sig
    draw_hud(fb_w, fb_h)

    frame_count += 1
    if args.shot and frame_count >= 8:
        data = ctx.screen.read(components=3, alignment=1)
        Image.frombytes("RGB", (fb_w, fb_h), data).transpose(Image.FLIP_TOP_BOTTOM).save(args.shot)
        break

    glfw.swap_buffers(window)

    frames += 1
    if now - fps_time >= 1.0:
        shown_fps = frames / (now - fps_time)
        shown_cpu = cpu_acc / frames
        glfw.set_window_title(
            window,
            f"Shadow Volumes | Cena: {cfg['label']} | {n_lights} luz(es) | {shown_fps:.0f} FPS | "
            f"CPU (silhueta+volume): {shown_cpu:.2f} ms")
        fps_time, frames, cpu_acc = now, 0, 0.0

glfw.terminate()