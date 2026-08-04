from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ROLLOUTS = ROOT / "rollouts"
OUTPUT = ROOT / "views"
FRAME = 64
SCALE = 4
LABEL_HEIGHT = 24


def font(size: int = 16) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT = font()
SMALL_FONT = font(14)


def frame_from_atlas(atlas: Image.Image, column: int, row: int = 0) -> Image.Image:
    left = column * FRAME
    top = row * FRAME
    return atlas.crop((left, top, left + FRAME, top + FRAME))


def labelled_frame(frame: Image.Image, title: str, lane: str | None = None) -> Image.Image:
    scaled = frame.resize((FRAME * SCALE, FRAME * SCALE), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (scaled.width, scaled.height + LABEL_HEIGHT), "#0b1020")
    canvas.paste(scaled, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 4), title, font=SMALL_FONT, fill="#f8fafc")
    if lane:
        bbox = draw.textbbox((0, 0), lane, font=SMALL_FONT)
        draw.text((canvas.width - (bbox[2] - bbox[0]) - 7, 4), lane, font=SMALL_FONT, fill="#9fb3c8")
    return canvas


def make_comparison_view(scenario: str, columns: list[tuple[int, str]]) -> None:
    atlas = Image.open(ROLLOUTS / "truth-vs-prediction" / f"{scenario}.png").convert("RGB")
    cells = []
    for row, lane in ((0, "vérité"), (1, "modèle")):
        lane_cells = [
            labelled_frame(frame_from_atlas(atlas, column, row), label, lane)
            for column, label in columns
        ]
        cells.append(lane_cells)

    width = len(columns) * cells[0][0].width
    height = 2 * cells[0][0].height
    sheet = Image.new("RGB", (width, height), "#0b1020")
    for row, lane_cells in enumerate(cells):
        for column, cell in enumerate(lane_cells):
            sheet.paste(cell, (column * cell.width, row * cell.height))
    sheet.save(OUTPUT / f"truth-vs-prediction-{scenario}.png", optimize=True)

    animation = []
    for column in range(7, atlas.width // FRAME):
        truth = labelled_frame(frame_from_atlas(atlas, column, 0), f"frame {column - 7}", "vérité")
        predicted = labelled_frame(frame_from_atlas(atlas, column, 1), f"frame {column - 7}", "modèle")
        paired = Image.new("RGB", (truth.width * 2, truth.height), "#0b1020")
        paired.paste(truth, (0, 0))
        paired.paste(predicted, (truth.width, 0))
        animation.append(paired)
    animation[0].save(
        OUTPUT / f"truth-vs-prediction-{scenario}.gif",
        save_all=True,
        append_images=animation[1:],
        duration=125,
        loop=0,
        optimize=False,
    )


def overlay_decoded(frame: Image.Image, values: list[float] | None) -> Image.Image:
    result = frame.copy()
    if values is None:
        return result
    draw = ImageDraw.Draw(result)
    player_x, player_y, puck_x, puck_y = values
    radius = 4
    draw.ellipse(
        (player_x - radius, player_y - radius, player_x + radius, player_y + radius),
        outline="#ff4fd8",
        width=2,
    )
    draw.line((player_x - 5, player_y, player_x + 5, player_y), fill="#ff4fd8", width=1)
    draw.line((player_x, player_y - 5, player_x, player_y + 5), fill="#ff4fd8", width=1)
    draw.ellipse(
        (puck_x - radius, puck_y - radius, puck_x + radius, puck_y + radius),
        outline="#32e6ff",
        width=2,
    )
    draw.line((puck_x - 5, puck_y, puck_x + 5, puck_y), fill="#32e6ff", width=1)
    draw.line((puck_x, puck_y - 5, puck_x, puck_y + 5), fill="#32e6ff", width=1)
    return result


def make_decoded_view(scenario: str) -> None:
    manifest = json.loads(
        (ROLLOUTS / "decoded-position-rollouts" / "manifest.json").read_text()
    )
    record = next(item for item in manifest["scenarios"] if item["id"] == scenario)
    decoded = record["decodedPositions"]
    atlas = Image.open(ROLLOUTS / "decoded-position-rollouts" / f"{scenario}.png").convert("RGB")
    columns = [7, 8, 11, 15, 19, 23, 27]
    labels = ["contexte", "t+1", "t+4", "t+8", "t+12", "t+16", "t+20"]
    cells = []
    for column, label in zip(columns, labels):
        frame = frame_from_atlas(atlas, column)
        cells.append(labelled_frame(overlay_decoded(frame, decoded[column]), label))
    sheet = Image.new("RGB", (len(cells) * cells[0].width, cells[0].height + 24), "#0b1020")
    for column, cell in enumerate(cells):
        sheet.paste(cell, (column * cell.width, 0))
    legend = ImageDraw.Draw(sheet)
    legend.text((8, cells[0].height + 3), "joueur décodé", font=SMALL_FONT, fill="#ff4fd8")
    legend.text((130, cells[0].height + 3), "palet décodé", font=SMALL_FONT, fill="#32e6ff")
    sheet.save(OUTPUT / f"decoded-position-{scenario}.png", optimize=True)

    animation = []
    for column in range(atlas.width // FRAME):
        rendered = overlay_decoded(frame_from_atlas(atlas, column), decoded[column])
        animation.append(labelled_frame(rendered, f"frame {column}", "décodage"))
    animation[0].save(
        OUTPUT / f"decoded-position-{scenario}.gif",
        save_all=True,
        append_images=animation[1:],
        duration=143,
        loop=0,
        optimize=False,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selections = {
        "collision": [(7, "contexte"), (8, "t+1"), (11, "t+4"), (14, "t+7 impact"), (19, "t+12"), (31, "t+24"), (47, "t+40"), (71, "t+64")],
        "wall-bounce": [(7, "contexte"), (8, "t+1"), (12, "t+5"), (17, "t+10 rebond"), (24, "t+17"), (32, "t+25 rebond"), (48, "t+41"), (71, "t+64")],
        "goal-reset": [(7, "contexte"), (8, "t+1 but"), (9, "t+2 remise"), (13, "t+6"), (19, "t+12"), (31, "t+24"), (47, "t+40"), (71, "t+64")],
    }
    for scenario, columns in selections.items():
        make_comparison_view(scenario, columns)
        make_decoded_view(scenario)


if __name__ == "__main__":
    main()
