"""Generate docs/assets/hil-setup.svg: isometric Jetson AGX Orin + laptop HIL illustration.

python docs/gen_hero_svg.py   (pure stdlib; writes docs/assets/hil-setup.svg)
"""
import math
from pathlib import Path

S = 22.0                      # px per unit
C30, S30 = math.cos(math.pi / 6), math.sin(math.pi / 6)
OX, OY = 420.0, 120.0         # screen origin


def P(x, y, z):
    return (OX + (x - y) * C30 * S, OY + (x + y) * S30 * S - z * S)


def poly(pts, fill, stroke="none", sw=0, extra=""):
    d = " ".join(f"{a:.1f},{b:.1f}" for a, b in (P(*p) for p in pts))
    st = f' stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"' if stroke != "none" else ""
    return f'<polygon points="{d}" fill="{fill}"{st} {extra}/>'


def line(a, b, stroke, sw=1.0, extra=""):
    (x1, y1), (x2, y2) = P(*a), P(*b)
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" {extra}/>'


def box(x0, y0, z0, x1, y1, z1, top, left, right, stroke="#2b2f36", sw=0.8):
    """Visible faces for this projection: top (z1), left (y = y1), right (x = x1)."""
    return [
        poly([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], left, stroke, sw),
        poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], right, stroke, sw),
        poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top, stroke, sw),
    ]


def tile(x0, y0, x1, y1, h=0.7):
    return box(x0, y0, -h, x1, y1, 0, "#7be04a", "#4fb52c", "#3d9622", stroke="#2f7a1a", sw=1.0)


def ellipse_iso(cx, cy, z, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def jetson(x0, y0):
    out = []
    w = 6.0
    x1, y1 = x0 + w, y0 + w
    zb, zm, zt = 0.0, 1.0, 4.2
    # dark base band
    out += box(x0, y0, zb, x1, y1, zm, "#5b6068", "#6b7079", "#555a62")
    # upper shell
    out += box(x0 + 0.05, y0 + 0.05, zm, x1 - 0.05, y1 - 0.05, zt, "#e3e6ea", "#b9bec5", "#cfd3d8")
    # diagonal vents on both upper sides
    for k in range(9):
        t = 0.45 + k * 0.62
        if t + 1.2 > w - 0.3:
            break
        out.append(line((x0 + t, y1 - 0.05, zm + 0.35), (x0 + t + 1.1, y1 - 0.05, zt - 0.35), "#8d939b", 2.2))
        out.append(line((x1 - 0.05, y0 + t, zm + 0.35), (x1 - 0.05, y0 + t + 1.1, zt - 0.35), "#a3a9b0", 2.2))
    # fan grille on top
    cx, cy = x0 + w / 2, y0 + w / 2
    out.append(poly(ellipse_iso(cx, cy, zt, 2.35), "#c9cdd2", "#8d939b", 1.2))
    for r in (2.0, 1.6, 1.2, 0.8):
        out.append(poly(ellipse_iso(cx, cy, zt, r), "none", "#8d939b", 1.1))
    for k in range(12):
        a = 2 * math.pi * k / 12
        out.append(line((cx + 0.35 * math.cos(a), cy + 0.35 * math.sin(a), zt),
                        (cx + 2.3 * math.cos(a + 0.35), cy + 2.3 * math.sin(a + 0.35), zt), "#9aa0a7", 1.0))
    out.append(poly(ellipse_iso(cx, cy, zt, 0.35), "#7d838b"))
    # ports on the front (y = y1) base band
    ports = [(0.5, 1.1, 0.25, 0.75, "#1f2329"), (1.4, 2.4, 0.3, 0.7, "#1f2329"), (2.7, 3.3, 0.3, 0.7, "#2a6fdb"),
             (3.6, 4.2, 0.3, 0.7, "#2a6fdb"), (4.6, 5.4, 0.25, 0.75, "#1f2329")]
    for a, b, za, zb2, col in ports:
        out.append(poly([(x0 + a, y1 + 0.01, za), (x0 + b, y1 + 0.01, za), (x0 + b, y1 + 0.01, zb2), (x0 + a, y1 + 0.01, zb2)], col))
    # power LED + ethernet on the right base band
    out.append(poly([(x1 + 0.01, y0 + 1.0, 0.3), (x1 + 0.01, y0 + 2.0, 0.3), (x1 + 0.01, y0 + 2.0, 0.75), (x1 + 0.01, y0 + 1.0, 0.75)], "#1f2329"))
    out.append(poly(ellipse_iso(x1 + 0.01, y0 + 4.6, 0.5, 0.12), "#7be04a"))
    return out


def laptop(x0, y0):
    out = []
    w, d = 7.0, 5.0
    x1, y1 = x0 + w, y0 + d
    # base
    out += box(x0, y0, 0.0, x1, y1, 0.35, "#d6dade", "#9ea4ab", "#b8bdc3")
    # keyboard
    kx0, kx1, ky0, ky1 = x0 + 0.6, x1 - 0.6, y0 + 0.5, y0 + 3.0
    out.append(poly([(kx0, ky0, 0.36), (kx1, ky0, 0.36), (kx1, ky1, 0.36), (kx0, ky1, 0.36)], "#3a3f46"))
    for r in range(4):
        for c in range(12):
            a = kx0 + 0.15 + c * (kx1 - kx0 - 0.3) / 12
            b = ky0 + 0.15 + r * (ky1 - ky0 - 0.3) / 4
            out.append(poly([(a, b, 0.37), (a + 0.38, b, 0.37), (a + 0.38, b + 0.42, 0.37), (a, b + 0.42, 0.37)], "#5a6069"))
    # trackpad
    out.append(poly([(x0 + 2.6, y0 + 3.4, 0.36), (x1 - 2.6, y0 + 3.4, 0.36), (x1 - 2.6, y1 - 0.35, 0.36), (x0 + 2.6, y1 - 0.35, 0.36)], "#c4c9cf"))
    # screen: panel standing on the back edge (y = y0), leaning back slightly, display facing +y
    lean = 0.6
    sh = 4.6
    back = [(x0, y0, 0.35), (x1, y0, 0.35), (x1, y0 - lean, 0.35 + sh), (x0, y0 - lean, 0.35 + sh)]
    out.append(poly(back, "#2b2f36", "#1d2026", 1.0))
    # right edge thickness
    out.append(poly([(x1, y0, 0.35), (x1, y0 - 0.25, 0.35), (x1, y0 - lean - 0.25, 0.35 + sh), (x1, y0 - lean, 0.35 + sh)], "#9ea4ab", "#2b2f36", 0.8))

    def sp(u, v):  # point on the display: u along x (0..1), v up the screen (0..1)
        m = 0.35
        return (x0 + m + u * (w - 2 * m), y0 + 0.01 - lean * (0.08 + v * 0.84), 0.35 + sh * (0.08 + v * 0.84))

    out.append(poly([sp(0, 0), sp(1, 0), sp(1, 1), sp(0, 1)], "#0f1a24"))
    # sim on the display: table, cube, bowl, little arm
    out.append(poly([sp(0.08, 0.08), sp(0.92, 0.08), sp(0.78, 0.42), sp(0.22, 0.42)], "#c79a5b"))
    out.append(poly([sp(0.58, 0.18), sp(0.66, 0.18), sp(0.66, 0.28), sp(0.58, 0.28)], "#e2463c"))
    out.append(poly([sp(0.36, 0.2), sp(0.48, 0.2), sp(0.46, 0.27), sp(0.38, 0.27)], "#b8302a"))
    arm = [sp(0.16, 0.30), sp(0.20, 0.62), sp(0.42, 0.74), sp(0.55, 0.50)]
    for a, b in zip(arm, arm[1:]):
        (xa, ya, za), (xb, yb, zb) = a, b
        out.append(line((xa, ya, za), (xb, yb, zb), "#f2f4f6", 3.2))
    out.append(line(arm[-1], (arm[-1][0] + 0.25, arm[-1][1], arm[-1][2] - 0.25), "#f2f4f6", 2.4))
    out.append(poly([sp(0.03, 0.9), sp(0.3, 0.9), sp(0.3, 0.96), sp(0.03, 0.96)], "#7be04a"))
    return out


def cable(a, b, lift=2.2):
    (x1, y1), (x2, y2) = P(*a), P(*b)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + lift * S
    return (f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="#2a6fdb" '
            f'stroke-width="3" stroke-dasharray="8 6" stroke-linecap="round"><animate attributeName="stroke-dashoffset" '
            f'from="0" to="-28" dur="1.2s" repeatCount="indefinite"/></path>')


def label(x, y, z, title, sub, anchor="middle"):
    px, py = P(x, y, z)
    return (f'<g font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif" text-anchor="{anchor}">'
            f'<text x="{px:.1f}" y="{py:.1f}" font-size="15" font-weight="700" fill="#16202a">{title}</text>'
            f'<text x="{px:.1f}" y="{py + 18:.1f}" font-size="12.5" fill="#4a5664">{sub}</text></g>')


def main():
    parts = []
    # tiles: laptop (left/back) and Jetson (right/front)
    lt = (0.0, 9.5, 10.0, 19.5)
    jt = (10.5, -0.5, 20.5, 9.5)
    parts += tile(*lt)
    parts += tile(*jt)
    parts += laptop(1.5, 12.5)
    parts += jetson(12.5, 1.5)
    parts.append(cable((15.5, 7.6, 0.4), (8.6, 14.5, 0.3), lift=3.0))
    cx, cy = P(12.0, 11.0, 0.0)
    parts.append(f'<g font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif" text-anchor="middle">'
                 f'<rect x="{cx - 132:.1f}" y="{cy + 34:.1f}" width="264" height="42" rx="9" fill="#ffffff" stroke="#2a6fdb" stroke-width="1.5"/>'
                 f'<text x="{cx:.1f}" y="{cy + 52:.1f}" font-size="12.5" font-weight="700" fill="#2a6fdb">Ethernet · websocket :5000</text>'
                 f'<text x="{cx:.1f}" y="{cy + 68:.1f}" font-size="11.5" fill="#4a5664">3 cameras + joints out · 24-step actions back</text></g>')
    parts.append(label(15.5, 4.5, 10.2, "Jetson AGX Orin 64GB", "DreamZero-DROID 14B · policy server"))
    parts.append(label(4.5, 13.5, 11.2, "Laptop · Isaac Sim 6", "DROID Franka sim · HIL client"))
    # crop to the drawing: tile corners horizontally, labels on top, the link label at the bottom
    xs = [P(0.0, 19.5, 0)[0], P(20.5, -0.5, 0)[0]]
    top = min(P(15.5, 4.5, 10.2)[1], P(4.5, 13.5, 11.2)[1]) - 22
    bottom = max(P(10.0, 19.5, -0.7)[1], P(20.5, 9.5, -0.7)[1], cy + 76) + 12
    pad = 16
    vb = (min(xs) - pad, top, max(xs) - min(xs) + 2 * pad, bottom - top)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]:.0f} {vb[1]:.0f} {vb[2]:.0f} {vb[3]:.0f}" '
           f'role="img" aria-label="Isometric illustration: laptop running Isaac Sim connected over Ethernet to a Jetson AGX Orin running DreamZero">'
           + "".join(parts) + "</svg>")
    out = Path(__file__).resolve().parent / "assets" / "hil-setup.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print("wrote", out, f"{len(svg) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
