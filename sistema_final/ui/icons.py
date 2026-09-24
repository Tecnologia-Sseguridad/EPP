"""Render our small SVG primitive icon set without native Cairo dependencies."""
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageTk


def render_icon(name, size=20, color="#303030"):
    root = ET.parse(Path(__file__).with_name("assets") / "icons.svg").getroot()
    group = next((node for node in root if node.get("id") == name), None)
    if group is None:
        raise ValueError(f"Unknown icon: {name}")
    scale = size * 4 / 24
    image = Image.new("RGBA", (size * 4, size * 4))
    draw = ImageDraw.Draw(image)
    ink = color
    stroke = max(1, round(1.6 * scale))
    for node in group:
        kind = node.tag.split("}")[-1]
        def n(key, default=0):
            return float(node.get(key, default)) * scale
        if kind == "line":
            draw.line((n("x1"), n("y1"), n("x2"), n("y2")), fill=ink, width=stroke)
        elif kind in ("polyline", "polygon"):
            points = [tuple(float(v) * scale for v in pair.split(","))
                      for pair in node.get("points").split()]
            if kind == "polygon":
                points.append(points[0])
            draw.line(points, fill=ink, width=stroke, joint="curve")
        elif kind == "rect":
            draw.rounded_rectangle((n("x"), n("y"), n("x") + n("width"), n("y") + n("height")),
                                   radius=n("rx"), outline=ink, width=stroke)
        elif kind in ("circle", "ellipse"):
            rx, ry = (n("r"), n("r")) if kind == "circle" else (n("rx"), n("ry"))
            draw.ellipse((n("cx") - rx, n("cy") - ry, n("cx") + rx, n("cy") + ry),
                         outline=ink, width=stroke)
    return image.resize((size, size), Image.Resampling.LANCZOS)


class Icons:
    def __init__(self, master):
        self.master, self.cache = master, {}

    def get(self, name, size=20, color="#303030"):
        if (name, size, color) not in self.cache:
            self.cache[name, size, color] = ImageTk.PhotoImage(
                render_icon(name, size, color), master=self.master)
        return self.cache[name, size, color]
