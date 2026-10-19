import numpy as np


class ShadowCaster:
    """Malha que projeta sombra.

    Pré-calcula a adjacência de arestas uma vez e, a cada quadro, acha a
    silhueta e monta o volume de sombra para a posição atual da luz.
    """

    def __init__(self, mesh, model):
        mesh = mesh.copy()
        mesh.merge_vertices()
        if not (mesh.is_watertight and mesh.is_winding_consistent):
            raise ValueError("A malha precisa ser fechada (watertight)")

        # Objetos estáticos: levamos tudo para coordenadas de mundo uma vez só
        v = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        self.verts = (v @ model.T)[:, :3]
        self.faces = mesh.faces

        # Normal e plano de cada triângulo (em mundo)
        tri = self.verts[self.faces]                          # (F, 3, 3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = n
        self.plane_d = (n * tri.mean(axis=1)).sum(axis=1)

        # Adjacência: cada aresta compartilhada por exatamente 2 triângulos (A e B)
        adj = mesh.face_adjacency                             # (E, 2)
        u, v_ = mesh.face_adjacency_edges.T                   # vértices da aresta
        self.fa, self.fb = adj[:, 0], adj[:, 1]

        # Direção da aresta conforme o sentido (winding) do triângulo A
        f = self.faces[self.fa]
        fwd = (((f[:, 0] == u) & (f[:, 1] == v_)) |
               ((f[:, 1] == u) & (f[:, 2] == v_)) |
               ((f[:, 2] == u) & (f[:, 0] == v_)))
        self.edge_a = np.where(fwd, u, v_)
        self.edge_b = np.where(fwd, v_, u)

    # ------------------------------------------------------------------
    def silhouette(self, light):
        """Retorna (arestas, lit).

        arestas: (S, 2) índices de vértices, orientados como no triângulo
                 iluminado (necessário para o volume sair com a face certa).
        lit: (F,) bool, True se o triângulo está virado para a luz.
        """
        lit = (self.normals @ light - self.plane_d) > 0
        la, lb = lit[self.fa], lit[self.fb]
        mask = la != lb                       # um lado na luz, outro no escuro
        a = np.where(la[mask], self.edge_a[mask], self.edge_b[mask])
        b = np.where(la[mask], self.edge_b[mask], self.edge_a[mask])
        return np.stack([a, b], axis=1), lit

    # ------------------------------------------------------------------
    def max_volume_vertices(self):
        """Limite superior de vértices do volume (para reservar o buffer)."""
        return 6 * len(self.fa) + 6 * len(self.faces)

    def build_volume(self, light):
        """Silhueta + volume de sombra numa chamada só (usado pelo main.py)."""
        edges, lit = self.silhouette(light)
        return self.assemble_volume(light, edges, lit)

    def assemble_volume(self, light, edges, lit):
        """Extrusão: monta o volume de sombra a partir da silhueta já calculada.

        Volume fechado, em coordenadas homogêneas (x, y, z, w).

        Vértices com w = 1 estão na posição original; w = 0 são pontos no
        infinito (direção vértice - luz). Todos os triângulos saem com a
        face "para fora" do volume em sentido anti-horário (CCW).

          - laterais: um quad por aresta de silhueta, esticado até o infinito
          - tampa próxima: triângulos de costas para a luz, na posição original
          - tampa distante: os mesmos triângulos, projetados no infinito

        Usar os triângulos de costas para a luz como tampa próxima evita
        z-fighting com a superfície iluminada do próprio objeto.
        """
        parts = []

        if len(edges):
            A = self.verts[edges[:, 0]]
            B = self.verts[edges[:, 1]]
            one = np.ones((len(edges), 1))
            zero = np.zeros((len(edges), 1))
            Ah, Bh = np.hstack([A, one]), np.hstack([B, one])
            Ai, Bi = np.hstack([A - light, zero]), np.hstack([B - light, zero])
            quads = np.stack([Ah, Ai, Bi, Ah, Bi, Bh], axis=1)   # (S, 6, 4)
            parts.append(quads.reshape(-1, 4))

        dark = self.faces[~lit]
        if len(dark):
            tri = self.verts[dark]                                # (D, 3, 3)
            d = len(dark)
            near = np.concatenate([tri, np.ones((d, 3, 1))], axis=2)
            near = near[:, [0, 2, 1], :]            # winding invertido
            far = np.concatenate([tri - light, np.zeros((d, 3, 1))], axis=2)
            parts.append(near.reshape(-1, 4))
            parts.append(far.reshape(-1, 4))

        if not parts:
            return np.empty((0, 4), dtype="f4")
        return np.concatenate(parts).astype("f4")