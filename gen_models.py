"""Gera modelos 3D (GLB) estilizados de pratos para o protótipo do cardápio.
Unidades em metros, eixo Y para cima (padrão glTF), em escala real para a RA."""
import json, struct, math
import numpy as np

rng = np.random.default_rng(7)

# ---------- geometria ----------
def smooth_normals(V, F):
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    l = np.linalg.norm(N, axis=1, keepdims=True)
    l[l == 0] = 1
    return N / l

def lathe(profile, seg=48, warp=None):
    """Gira um perfil [(r,y),...] em torno do eixo Y. warp(theta, i, r, y) -> (r, y)."""
    P = len(profile)
    V = []
    for i, (r, y) in enumerate(profile):
        for s in range(seg):
            t = 2 * math.pi * s / seg
            rr, yy = (warp(t, i, r, y) if warp else (r, y))
            V.append((rr * math.cos(t), yy, rr * math.sin(t)))
    F = []
    for i in range(P - 1):
        for s in range(seg):
            a = i * seg + s
            b = i * seg + (s + 1) % seg
            c = (i + 1) * seg + s
            d = (i + 1) * seg + (s + 1) % seg
            F += [(a, c, b), (b, c, d)]
    V = np.array(V, float); F = np.array(F, np.int64)
    # garante normais para fora: perfil percorrido de baixo p/ cima por fora
    return V, F

def sphere(r, ry=None, seg=20, rings=12):
    ry = ry or r
    prof = [(r * math.sin(math.pi * k / rings), -ry * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(prof, seg)

def box(sx, sy, sz):
    V, F = [], []
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [((1,0,0),(0,1,0),(0,0,1)),((-1,0,0),(0,1,0),(0,0,-1)),((0,1,0),(0,0,1),(1,0,0)),
             ((0,-1,0),(0,0,-1),(1,0,0)),((0,0,1),(1,0,0),(0,1,0)),((0,0,-1),(-1,0,0),(0,1,0))]
    for n, u, v in faces:
        n, u, v = map(np.array, (n, u, v))
        c = n * np.array([hx, hy, hz])
        ue = u * np.array([hx, hy, hz]); ve = v * np.array([hx, hy, hz])
        base = len(V)
        for a, b in ((-1,-1),(1,-1),(1,1),(-1,1)):
            V.append(c + a * ue + b * ve)
        tri = [(base, base+1, base+2), (base, base+2, base+3)]
        p = np.array(V[-4:])
        if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), n) < 0:
            tri = [(t[0], t[2], t[1]) for t in tri]
        F += tri
    return np.array(V, float), np.array(F, np.int64)

def torus(R, r, seg=64, sides=24, v0=0.0, v1=2*math.pi, warp=None, closed=True):
    n_v = sides if closed else sides + 1
    V = []
    for s in range(seg):
        u = 2 * math.pi * s / seg
        for k in range(n_v):
            v = v0 + (v1 - v0) * k / sides
            rr = r * (warp(u, v) if warp else 1)
            x = (R + rr * math.cos(v)) * math.cos(u)
            z = (R + rr * math.cos(v)) * math.sin(u)
            y = rr * math.sin(v)
            V.append((x, y, z))
    F = []
    for s in range(seg):
        for k in range(sides):
            a = s * n_v + k
            b = s * n_v + (k + 1) % n_v
            c = ((s + 1) % seg) * n_v + k
            d = ((s + 1) % seg) * n_v + (k + 1) % n_v
            F += [(a, b, c), (b, d, c)]
    return np.array(V, float), np.array(F, np.int64)

def rot_y(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_x(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_z(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def xf(mesh, R=None, t=(0,0,0)):
    V, F = mesh
    if R is not None: V = V @ R.T
    return V + np.array(t), F

# ---------- cena ----------
class Scene:
    def __init__(self): self.parts = {}
    def add(self, mat, mesh):
        self.parts.setdefault(mat, []).append(mesh)

def hex_rgb(h):
    h = h.lstrip('#'); c = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c] + [1.0]

def write_glb(scene, mats, path):
    bin_ = bytearray(); bviews = []; accs = []; meshes = []; nodes = []; materials = []
    def push(arr, target):
        nonlocal bin_
        while len(bin_) % 4: bin_ += b'\0'
        off = len(bin_); data = arr.tobytes(); bin_ += data
        bviews.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(bviews) - 1
    for mi, (name, parts) in enumerate(scene.parts.items()):
        Vs, Ns, Fs, o = [], [], [], 0
        for V, F in parts:
            vol = np.einsum('ij,ij->i', V[F[:,0]], np.cross(V[F[:,1]], V[F[:,2]])).sum()
            if vol < 0: F = F[:, ::-1]
            Vs.append(V); Ns.append(smooth_normals(V, F)); Fs.append(F + o); o += len(V)
        V = np.vstack(Vs).astype(np.float32); N = np.vstack(Ns).astype(np.float32)
        F = np.vstack(Fs).astype(np.uint32).ravel()
        m = mats[name]
        materials.append({"name": name, "doubleSided": m.get("double", False),
                          "pbrMetallicRoughness": {"baseColorFactor": hex_rgb(m["c"]),
                                                   "metallicFactor": 0.0, "roughnessFactor": m.get("r", 0.7)}})
        bp = push(V, 34962); accs.append({"bufferView": bp, "componentType": 5126, "count": len(V), "type": "VEC3",
                                          "min": V.min(0).tolist(), "max": V.max(0).tolist()}); ap = len(accs) - 1
        bn = push(N, 34962); accs.append({"bufferView": bn, "componentType": 5126, "count": len(N), "type": "VEC3"}); an = len(accs) - 1
        bi = push(F, 34963); accs.append({"bufferView": bi, "componentType": 5125, "count": len(F), "type": "SCALAR"}); ai = len(accs) - 1
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": ap, "NORMAL": an}, "indices": ai, "material": mi}]})
        nodes.append({"name": name, "mesh": mi})
    while len(bin_) % 4: bin_ += b'\0'
    gltf = {"asset": {"version": "2.0", "generator": "Mesa3D proto"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
            "materials": materials, "accessors": accs, "bufferViews": bviews,
            "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4: js += b' '
    out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_))
    out += struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(bin_), 0x004E4942) + bytes(bin_)
    open(path, 'wb').write(out)
    print(path, round(len(out) / 1024), "KB")

# ---------- pratos ----------
def burger():
    S = Scene()
    S.add("pao", lathe([(0,0),(0,0),(0.049,0),(0.054,0.004),(0.055,0.012),(0.053,0.018),(0.053,0.018),(0,0.018)]))
    S.add("carne", lathe([(0,0.018),(0.056,0.018),(0.0595,0.026),(0.056,0.0345),(0,0.0345)], 64,
          warp=lambda t,i,r,y: (r + (0.0012*math.sin(11*t) + 0.0008*math.sin(23*t+1) if r > 0 else 0), y)))
    S.add("queijo", xf(lathe([(0,0.0355),(0.074,0.029),(0.074,0.031),(0,0.0385)], 4), rot_y(0.3)))
    S.add("alface", lathe([(0,0.038),(0.061,0.037),(0.063,0.0395),(0,0.0415)], 72,
          warp=lambda t,i,r,y: (r + (0.004*math.sin(7*t+0.5) if r > 0 else 0), y + (0.0028*math.sin(9*t) if r > 0 else 0))))
    S.add("tomate", lathe([(0,0.0415),(0.046,0.0415),(0.047,0.0445),(0.046,0.0475),(0,0.0475)], 40))
    dome = [(0,0.047),(0.053,0.047),(0.056,0.050)]
    for k in range(1, 17):
        a = (math.pi/2) * k / 16
        dome.append((0.057*math.cos(a)*(1 if k < 16 else 0), 0.050 + 0.034*math.sin(a)))
    S.add("pao", lathe(dome, 56))
    for _ in range(38):
        a = rng.uniform(0.25, 1.35); t = rng.uniform(0, 2*math.pi)
        p = (0.0575*math.cos(a)*math.cos(t), 0.0502 + 0.0343*math.sin(a), 0.0575*math.cos(a)*math.sin(t))
        S.add("gergelim", xf(sphere(0.0028, 0.001, 8, 5), rot_y(rng.uniform(0, 6.28)) @ rot_x(0.9*(1.57-a)), p))
    return S, {"pao": {"c": "#d08a3c", "r": 0.55}, "carne": {"c": "#4a2617", "r": 0.85},
               "queijo": {"c": "#f2b632", "r": 0.4}, "alface": {"c": "#6aa437", "r": 0.6},
               "tomate": {"c": "#d63a2a", "r": 0.35}, "gergelim": {"c": "#f4e6c2", "r": 0.6}}

def donut():
    S = Scene(); R, r, cy = 0.040, 0.018, 0.018
    S.add("massa", xf(torus(R, r), t=(0, cy, 0)))
    glaze = torus(R, r*1.09, 72, 30, v0=-0.35, v1=math.pi+0.35, closed=False,
                  warp=lambda u, v: 1.0)
    # bordas irregulares da cobertura
    V, F = glaze; V = V.copy()
    S.add("cobertura", xf((V, F), t=(0, cy, 0)))
    cols = ["confeito_a", "confeito_b", "confeito_c", "confeito_d"]
    for _ in range(70):
        u = rng.uniform(0, 2*math.pi); v = rng.uniform(0.35, math.pi-0.35); rr = r*1.09 + 0.0008
        p = ((R + rr*math.cos(v))*math.cos(u), cy + rr*math.sin(v), (R + rr*math.cos(v))*math.sin(u))
        tilt = (math.pi/2 - v)
        Rm = rot_y(-u) @ rot_z(tilt) @ rot_y(rng.uniform(0, 6.28))
        S.add(cols[rng.integers(4)], xf(box(0.0075, 0.0018, 0.0018), Rm, p))
    return S, {"massa": {"c": "#d99a52", "r": 0.7}, "cobertura": {"c": "#e86f9f", "r": 0.22, "double": True},
               "confeito_a": {"c": "#ffffff", "r": 0.5}, "confeito_b": {"c": "#3db7e4", "r": 0.5},
               "confeito_c": {"c": "#ffd23f", "r": 0.5}, "confeito_d": {"c": "#5cc46b", "r": 0.5}}

def acai():
    S = Scene()
    S.add("tigela", lathe([(0,0),(0.038,0),(0.043,0.004),(0.062,0.032),(0.071,0.054),(0.071,0.057),(0.067,0.057),
                           (0.066,0.054),(0.058,0.034),(0.040,0.012),(0,0.012)], 72))
    S.add("acai", lathe([(0,0.050),(0.0665,0.050),(0.060,0.054),(0.035,0.0565),(0,0.057)], 64,
          warp=lambda t,i,r,y: (r, y + (0.0012*math.sin(5*t)+0.0008*math.sin(13*t) if 0 < r < 0.066 else 0))))
    for k in range(6):
        a = -0.9 + k*0.36; d = 0.036
        S.add("banana", xf(lathe([(0,0),(0.0105,0),(0.011,0.002),(0.0105,0.004),(0,0.004)], 28),
              rot_x(0.12) @ rot_z(0.08), (d*math.cos(a), 0.0555, d*math.sin(a))))
    for _ in range(90):
        a = rng.uniform(2.1, 3.8); d = rng.uniform(0.015, 0.058)
        R = rot_x(rng.uniform(0, 3)) @ rot_y(rng.uniform(0, 3)) @ rot_z(rng.uniform(0, 3))
        s = rng.uniform(0.003, 0.006)
        S.add("granola", xf(box(s, s*0.6, s*0.8), R, (d*math.cos(a), 0.057 + rng.uniform(0, 0.003), d*math.sin(a))))
    berry = [(0.0, -0.0)] + [(0.011*math.sin(math.pi*k/10)*(1-0.35*k/10), 0.016*k/10) for k in range(1, 11)] + [(0, 0.016)]
    for a, d in ((4.3, 0.03), (4.9, 0.022), (5.4, 0.034)):
        S.add("morango", xf(lathe(berry, 24), rot_z(1.25) @ rot_y(a), (d*math.cos(a), 0.058, d*math.sin(a))))
    S.add("folha", xf(sphere(0.007, 0.0012, 12, 6), rot_y(0.6), (0.0, 0.0605, -0.005)))
    S.add("folha", xf(sphere(0.006, 0.0012, 12, 6), rot_y(-0.5), (0.008, 0.0605, 0.0)))
    return S, {"tigela": {"c": "#e9e4da", "r": 0.35}, "acai": {"c": "#3b1234", "r": 0.3},
               "banana": {"c": "#f3e3a2", "r": 0.55}, "granola": {"c": "#a8682c", "r": 0.85},
               "morango": {"c": "#d1293a", "r": 0.3}, "folha": {"c": "#3d8b3d", "r": 0.5}}

def brigadeiro():
    S = Scene()
    S.add("tabua", xf(box(0.11, 0.01, 0.11), t=(0, 0.005, 0)))
    cup = [(0,0.010),(0.013,0.010),(0.0135,0.010),(0.017,0.024),(0.0165,0.024),(0.0125,0.0115),(0,0.0115)]
    for x in (-0.026, 0.026):
        for z in (-0.026, 0.026):
            S.add("forminha", xf(lathe(cup, 48, warp=lambda t,i,r,y: (r*(1+0.06*math.sin(24*t)) if r > 0 else r, y)), t=(x, 0, z)))
            S.add("chocolate", xf(sphere(0.0135, 0.0125, 28, 16), t=(x, 0.0245, z)))
            for _ in range(140):
                th = rng.uniform(0, 2*math.pi); ph = rng.uniform(0.15, 2.0)
                n = np.array([math.sin(ph)*math.cos(th), math.cos(ph), math.sin(ph)*math.sin(th)])
                p = np.array([x, 0.0245, z]) + n*np.array([0.0137, 0.0127, 0.0137])
                Rm = rot_y(-th) @ rot_z(-ph) @ rot_y(rng.uniform(0, 6.28))
                S.add("granulado", xf(box(0.0032, 0.0011, 0.0011), Rm, p))
    return S, {"tabua": {"c": "#2f3438", "r": 0.8}, "forminha": {"c": "#b88a3a", "r": 0.45, "double": True},
               "chocolate": {"c": "#3d2418", "r": 0.75}, "granulado": {"c": "#24130c", "r": 0.6}}

if __name__ == "__main__":
    import os, sys
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    os.makedirs(out, exist_ok=True)
    for name, fn in (("hamburguer", burger), ("donut", donut), ("acai", acai), ("brigadeiro", brigadeiro)):
        S, M = fn(); write_glb(S, M, f"{out}/{name}.glb")
