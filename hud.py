from PIL import Image, ImageDraw, ImageFont

PANEL_X, PANEL_Y = 12, 12      # posição do painel na tela (px, canto superior esquerdo)
PW, PH = 330, 448              # tamanho do painel

# (nome, (x, y, largura, altura) dentro do painel, rótulo)
BUTTONS = [
    ("minus",         (214, 82, 44, 34),   "-"),
    ("plus",          (266, 82, 44, 34),   "+"),
    ("scene_padrao",  (14, 146, 70, 32),   "Padrão"),
    ("scene_predios", (88, 146, 70, 32),   "Prédios"),
    ("scene_bonecos", (162, 146, 70, 32),  "Bonecos"),
    ("scene_cidade",  (236, 146, 70, 32),  "Cidade"),
    ("shadows",       (14, 188, 302, 34),  "Sombras (X)"),
    ("volume",        (14, 230, 302, 34),  "Volume de sombra (V)"),
    ("silhouette",    (14, 272, 302, 34),  "Silhueta (E)"),
    ("paused",        (14, 314, 302, 34),  "Pausar luzes (P)"),
    ("markers",       (14, 356, 302, 34),  "Marcadores das luzes (L)"),
]
TOGGLES = ("shadows", "volume", "silhouette", "paused", "markers")

WHITE = (255, 255, 255, 255)
GREY = (160, 168, 190, 255)
BLUE = (150, 195, 255, 255)
C_PANEL = (18, 20, 28, 225)
C_ON = (44, 150, 88, 255)
C_OFF = (74, 78, 96, 255)
C_STEP = (58, 92, 165, 255)


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class Hud:
    def __init__(self):
        self.f_title = _font(19)
        self.f_text = _font(16)
        self.f_small = _font(13)

    def render(self, state, fps, cpu_ms):
        """Desenha o painel e devolve os bytes RGBA (linha 0 = topo)."""
        img = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, PW - 1, PH - 1), radius=14, fill=C_PANEL)

        d.text((14, 10), "Shadow Volumes (z-fail)", font=self.f_title, fill=WHITE)
        d.text((14, 40), f"{fps:.0f} FPS   |   CPU: {cpu_ms:.2f} ms", font=self.f_text, fill=BLUE)
        d.text((14, 90), f"Luzes: {state['lights']}", font=self.f_title, fill=WHITE)
        d.text((14, 126), "Cena (tecla C)", font=self.f_small, fill=GREY)

        for name, (x, y, w, h), label in BUTTONS:
            if name in ("minus", "plus"):
                d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=C_STEP)
                d.text((x + w // 2 - 5, y + 6), label, font=self.f_title, fill=WHITE)
            elif name.startswith("scene_"):
                on = state["scene"] == name[len("scene_"):]
                d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=C_ON if on else C_OFF)
                tw = d.textlength(label, font=self.f_text)
                d.text((x + (w - tw) / 2, y + 7), label, font=self.f_text, fill=WHITE)
            else:
                on = state[name]
                d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=C_ON if on else C_OFF)
                d.text((x + 12, y + 8), label, font=self.f_text, fill=WHITE)
                d.text((x + w - 44, y + 8), "ON" if on else "OFF", font=self.f_text, fill=WHITE)

        if state["ui"]:
            hint = ["Mouse livre: clique nos botões.", "TAB volta para a câmera."]
        else:
            hint = ["TAB libera o mouse para clicar.", "WASD + mouse | C: cena | 1-8: luzes"]
        d.text((14, 404), hint[0], font=self.f_small, fill=GREY)
        d.text((14, 424), hint[1], font=self.f_small, fill=GREY)
        return img.tobytes()

    def hit_test(self, x, y):
        """x, y em pixels da tela. Devolve o nome do botão clicado ou None."""
        px, py = x - PANEL_X, y - PANEL_Y
        for name, (bx, by, bw, bh), _ in BUTTONS:
            if bx <= px <= bx + bw and by <= py <= by + bh:
                return name
        return None