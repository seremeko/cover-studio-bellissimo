import os
from PIL import Image, ImageDraw

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PALETTE = {
    ".": None,
    "K": (26, 24, 34, 255),       # Black outline
    "S": (255, 228, 214, 255),    # Skin
    "P": (255, 148, 184, 255),    # Blush pink
    "W": (255, 255, 255, 255),    # White
    # Miku
    "M": (57, 197, 187, 255),
    "m": (30, 142, 136, 255),
    "H": (255, 56, 136, 255),
    "G": (214, 222, 228, 255),
    "D": (42, 46, 60, 255),
    "L": (92, 196, 78, 255),
    # Teto
    "R": (229, 57, 53, 255),
    "r": (166, 28, 36, 255),
    "U": (58, 62, 78, 255),
    "B": (240, 174, 74, 255),
    "b": (176, 102, 28, 255),
    # Rin & Len
    "Y": (255, 224, 72, 255),
    "y": (218, 172, 28, 255),
    "O": (255, 160, 24, 255),
    # Luka
    "F": (255, 176, 206, 255),
    "f": (222, 128, 168, 255),
    "A": (245, 204, 88, 255),
    # KAITO
    "C": (48, 108, 214, 255),
    "c": (28, 68, 156, 255),
    # GUMI
    "V": (124, 214, 76, 255),
    "v": (78, 160, 44, 255),
}


def render_grid(grid: list[str], scale: int = 2) -> Image.Image:
    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            col = PALETTE.get(ch)
            if col is not None:
                px[x, y] = col
    return img.resize((w * scale, h * scale), Image.Resampling.NEAREST)


def shift_leg_frame(grid: list[str]) -> list[str]:
    empty_row = "." * len(grid[0])
    return [empty_row] + grid[:-1]


MIKU_F1 = [
    "........KKKKKK..........",
    "......KKMMMMMMKK........",
    ".....KMMMMMMMMMMK.......",
    "....KMMMMMMMMMMMMK......",
    "...KHHMMMMMMMMMMHHK.....",
    "..KMmKMMKMMMMKMMKmMK....",
    ".KMmKSSSKMMMMKSSSKmMK...",
    ".KMmKSMMKSSSSKMMSKmmK...",
    ".KMmKSWWKSSSSKWWKSmmK...",
    ".KMmKSPSSSSSSSSSPSmmK...",
    ".KMmKKSSSSSSSSSSKKmmK...",
    ".KMmK.KKGGGGGGKK.KmmK...",
    ".KmMK.KGGMMMMGGK.KmMK...",
    ".KmMK.KGGGGGGGGK.KmMK.LK",
    "..KmK.KGGGGGGGGK.KmK.KLK",
    "..KK..KDDDDDDDDK..KK.KWK",
    ".....KDDDDDDDDDDK...KWK.",
    ".....KDDmDDDDmDDK...KK..",
    "......KKKKKKKKKK........",
    ".......KDDK.KDDK........",
    ".......KDDK..KDDK.......",
    ".......KKKK..KKKK.......",
]

TETO_F1 = [
    "........KKKKKK..........",
    "......KKRRRRRRKK........",
    ".....KRRRRRRRRRRK.......",
    "...KKRRRRRRRRRRRRKK.....",
    "..KRRKRRRRRRRRRRKRRK....",
    ".KRrKKRRKRRRRKRRKKrRK...",
    "..KKKSSSKRRRRKSSSKKK....",
    ".KRrKSRRKSSSSKRRKSrRK...",
    "..KKKSWWKSSSSKWWKSKK....",
    ".KRrKSPSSSSSSSSSPSrRK...",
    "..KK.KSSSSSSSSSSK.KK....",
    ".KRrK.KKUUUUUUKK.KrRK...",
    "..KK..KUURRRRUUK..KK..BB",
    ".....KUUUUUUUUUUK....BbB",
    ".....KUUUUUUUUUUK...BbB.",
    ".....KDDDDDDDDDDK..BbB..",
    ".....KDDRDDDDRDDK..BB...",
    "......KKKKKKKKKK........",
    ".......KDDK.KDDK........",
    ".......KDDK..KDDK.......",
    ".......KKKK..KKKK.......",
    "........................",
]

RIN_F1 = [
    "..........KKK.KKK.......",
    "........KKWWWKWWWKK.....",
    "......KKYYYYYYYYYYKK....",
    ".....KYYYYYYYYYYYYYYK...",
    "....KYYYYYYYYYYYYYYYYK..",
    "....KYyKYYKYYYYKYYKyYK..",
    "....KYyKSSSKYYKSSSKyYK..",
    "....KYyKSMMKSSKMMSKyYK..",
    "....KYyKSWWKSSKWWKSyYK..",
    "....KYyKSPSSSSSSPSKyYK..",
    ".....KKKSSSSSSSSSSKKK...",
    "......KKWWWWWWWWWWKK....",
    "......KWWYYYYYYYYWWK....",
    "......KWWWWWWWWWWWWK....",
    ".....KDDDDDDDDDDDDDDK...",
    ".....KDDYDDDDDDDDYDDK...",
    "......KKKKKKKKKKKKKK....",
    ".......KSSK....KSSK.....",
    ".......KWWK....KWWK.....",
    ".......KKKK....KKKK.....",
    "........................",
    "........................",
]

LEN_F1 = [
    ".......KK..KK..KK.......",
    "......KYYKKYYKKYYK......",
    ".....KYYYYYYYYYYYYK.....",
    "....KYYYYYYYYYYYYYYK....",
    "....KYYKYYKYYYYKYYKYK...",
    "....KYYKSSSKYYKSSSKYK...",
    "....KYYKSMMKSSKMMSKYK...",
    "....KYYKSWWKSSKWWKSYK...",
    "....KYYKSPSSSSSSPSKYK...",
    ".....KKKSSSSSSSSSSKKK...",
    "......KKWWWWWWWWWWKK....",
    "......KWWYYYYYYYYWWK....",
    "......KWWYYOYYOYYWWK....",
    "......KWWWWWWWWWWWWK....",
    ".....KDDDDDDDDDDDDDDK...",
    ".....KDDDDKDDDDKDDDDK...",
    "......KKKK.KKKK.KKKK....",
    ".......KSSK....KSSK.....",
    ".......KDDK....KDDK.....",
    ".......KKKK....KKKK.....",
    "........................",
    "........................",
]

LUKA_F1 = [
    "........KKKKKK..........",
    "......KKFFFFFFKK........",
    ".....KFFFFFFFFFFK.......",
    "....KFFFFFFFFFFFFK......",
    "...KFFfKFFKFFFFKFFKfFK..",
    "..KFFfKSSSKFFFFKSSSKfFK.",
    "..KFFfKSMMKSSSSKMMSKfFK.",
    "..KFFfKSWWKSSSSKWWKSfFK.",
    "..KFFfKSPSSSSSSSSPSKfFK.",
    "..KFFfKKSSSSSSSSSSKKfFK.",
    "..KFFfK.KKDDDDDDKK.KfFK.",
    "..KFfFK.KDDAAAADDK.KfFK.",
    "..KFfFK.KDDDDDDDDK.KfFK.",
    "..KFfK..KDDDDDDDDK..KfK.",
    "...KK...KDDDDDDDDK...KK.",
    ".......KDDADDDDADDK.....",
    "........KKKKKKKKKK......",
    "........KSSK..KSSK......",
    "........KDDK..KDDK......",
    "........KKKK..KKKK......",
    "........................",
    "........................",
]

KAITO_F1 = [
    "........KKKKKK..........",
    "......KKCCCCCCKK........",
    ".....KCCCCCCCCCCK.......",
    "....KCCCCCCCCCCCCK......",
    "....KCcKCCKCCCCKCCKcCK..",
    "....KCcKSSSKCCKSSSKcCK..",
    "....KCcKSMMKSSKMMSKcCK..",
    "....KCcKSWWKSSKWWKSKCK..",
    "....KCcKSPSSSSSSPSKcCK..",
    ".....KKKSSSSSSSSSSKKK...",
    "......KKCCCCCCCCCCKK....",
    ".....KCCCCCCCCCCCCCCK...",
    ".....KCCKWCCCCCCWKCCK...",
    ".....KCCKWWWWWWWWKCCK...",
    ".....KCCKWWWWWWWWKCCK...",
    ".....KCCKWWCCCCWWKCCK...",
    ".....KCCKWWWWWWWWKCCK...",
    "......KK.KKKKKKKK.KK....",
    "........KDDK..KDDK......",
    "........KDDK..KDDK......",
    "........KKKK..KKKK......",
    "........................",
]

GUMI_F1 = [
    ".......KKKKKKKKKK.......",
    "......KOOWWWWWWOOK......",
    ".....KVVVVVVVVVVVVk.....",
    "....KVVVVVVVVVVVVVVK....",
    "....KVvKVVkVVVVkVVKvK...",
    "....KVvKSSSKVVKSSSKvK...",
    "....KVvKSLLKSSKLLSKvK...",
    "....KVvKSWWKSSKWWKSvK...",
    "....KVvKSPSSSSSSPSKvK...",
    ".....KKKSSSSSSSSSSKKK...",
    "......KKOOOOOOOOOOKK....",
    "......KOOVVVVVVVVOOK....",
    "......KOOOOOOOOOOOOK....",
    ".....KVVVVVVVVVVVVVVK...",
    ".....KVVOOVVVVVVOOVVK...",
    "......KKKKKKKKKKKKKK....",
    ".......KSSK....KSSK.....",
    ".......KOOK....KOOK.....",
    ".......KKKK....KKKK.....",
    "........................",
    "........................",
    "........................",
]


def save_anim_gif(f1: list[str], out_path: str, scale: int = 2, duration_ms: int = 230):
    max_w = max(len(r) for r in f1)
    norm_f1 = [r.ljust(max_w, ".") for r in f1]
    norm_f2 = shift_leg_frame(norm_f1)
    imgs = [render_grid(norm_f1, scale=scale), render_grid(norm_f2, scale=scale)]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        transparency=0,
    )


def generate_pastel_pixel_sky(out_path: str):
    """Generates a dreamy 2000s pixel-art pastel sky + clouds + cityscape."""
    w, h = 320, 180
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    # Sky gradient: cyan -> soft sky blue -> pastel pink horizon
    top_col = (94, 214, 248)
    mid_col = (164, 232, 252)
    bot_col = (255, 212, 232)
    for y in range(h):
        t = y / (h - 1)
        if t < 0.55:
            k = t / 0.55
            c = tuple(int(top_col[i] + (mid_col[i] - top_col[i]) * k) for i in range(3))
        else:
            k = (t - 0.55) / 0.45
            c = tuple(int(mid_col[i] + (bot_col[i] - mid_col[i]) * k) for i in range(3))
        draw.line([(0, y), (w, y)], fill=c)

    # Fluffy pastel pink & white pixel clouds
    clouds = [
        (36, 56, 22), (56, 48, 28), (78, 58, 20),
        (256, 62, 24), (278, 52, 26), (298, 64, 18),
        (20, 108, 30), (50, 96, 36), (82, 108, 24),
        (238, 102, 28), (268, 92, 34), (298, 106, 22),
    ]
    for cx, cy, r in clouds:
        draw.ellipse([cx - r, cy - r + 4, cx + r, cy + r + 4], fill=(255, 210, 228))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 244, 250))

    # Tiny 4-point pixel sparkles in sky & clouds
    sparkles = [(28, 34), (62, 74), (112, 26), (218, 38), (272, 46), (302, 82), (44, 92), (255, 98)]
    for sx, sy in sparkles:
        draw.point((sx, sy), fill=(255, 255, 255))
        draw.point((sx - 1, sy), fill=(255, 220, 238))
        draw.point((sx + 1, sy), fill=(255, 220, 238))
        draw.point((sx, sy - 1), fill=(255, 220, 238))
        draw.point((sx, sy + 1), fill=(255, 220, 238))

    # Pastel pixel cityscape at bottom-left and bottom-right
    buildings = [
        (0, 128, 26, 180, (124, 186, 246), (92, 152, 222)),
        (24, 142, 48, 180, (152, 206, 252), (112, 174, 236)),
        (46, 134, 68, 180, (116, 176, 238), (84, 144, 212)),
        (252, 136, 274, 180, (116, 176, 238), (84, 144, 212)),
        (272, 126, 298, 180, (152, 206, 252), (112, 174, 236)),
        (296, 140, 320, 180, (124, 186, 246), (92, 152, 222)),
    ]
    for x1, y1, x2, y2, body_c, roof_c in buildings:
        draw.rectangle([x1, y1, x2, y2], fill=body_c, outline=(74, 124, 188))
        draw.rectangle([x1, y1, x2, y1 + 4], fill=roof_c)
        for wy in range(y1 + 8, y2 - 4, 8):
            for wx in range(x1 + 4, x2 - 4, 7):
                draw.rectangle([wx, wy, wx + 3, wy + 4], fill=(224, 246, 255))

    big = img.resize((w * 4, h * 4), Image.Resampling.NEAREST)
    big.save(out_path, format="PNG")


if __name__ == "__main__":
    sprites = {
        "miku_walk.gif": MIKU_F1,
        "teto_walk.gif": TETO_F1,
        "rin_walk.gif": RIN_F1,
        "len_walk.gif": LEN_F1,
        "luka_walk.gif": LUKA_F1,
        "kaito_walk.gif": KAITO_F1,
        "gumi_walk.gif": GUMI_F1,
    }
    for fname, grid in sprites.items():
        save_anim_gif(grid, os.path.join(ASSETS_DIR, fname), scale=2, duration_ms=230)
    generate_pastel_pixel_sky(os.path.join(ASSETS_DIR, "bg_sky.png"))
    print("Generated sprites + bg_sky.png in", ASSETS_DIR)
