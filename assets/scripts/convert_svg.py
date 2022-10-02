#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from lxml import etree


def get_hex(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if s is None:
        return None
    if s.startswith("#"):
        if len(s) == 7:
            return tuple(int(s[i : i + 2], 16) for i in (1, 3, 5))
        if len(s) == 4:
            return tuple(int(s[i : i + 1], 16) for i in (1, 2, 3))
        raise ValueError(f"Invalid hex code: {s}")
    if s.startswith("rgb"):
        m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+).*\)", s)
        if m is None:
            raise ValueError(f"Invalid hex code: {s}")
        return tuple([int(i) for i in m.groups()])
    if s == "none":
        return None
    raise ValueError(f"Unsupported hex code: {s}")


def get_style(node: etree.Element) -> Optional[Dict[str, str]]:
    style = node.get("style")
    if style is None:
        return None
    style = [k.strip().split(":") for k in style.split(";") if len(k.strip()) > 0]
    for e in style:
        if len(e) != 2:
            raise ValueError(f"Invalid style element: {e}")
    style = {k[0].strip(): k[1].strip() for k in style if len(k) == 2}
    return style


def is_black(h: Optional[Tuple[int, int, int]]) -> bool:
    if h is None:
        return False
    return all(i < 70 for i in h)


def is_white(h: Optional[Tuple[int, int, int]]) -> bool:
    if h is None:
        return False
    return all(i > 220 for i in h)


def get_colors(node: etree.Element) -> Set[Tuple[int, int, int]]:
    colors = set()

    style = get_style(node)
    if style is not None:
        if "fill" in style:
            colors.add(get_hex(style["fill"]))
        if "stroke" in style:
            colors.add(get_hex(style["stroke"]))

    colors.add(get_hex(node.get("fill")))
    colors.add(get_hex(node.get("stroke")))

    return {color for color in colors if color is not None and not is_black(color) and not is_white(color)}


def convert(
    node: etree.Element,
    convert_text: bool,
    color_map: Optional[Dict[Tuple[int, int, int], str]],
) -> None:
    classes = node.get("class") or ""
    classes = set([c.strip() for c in classes.split(" ")])

    if node.tag.endswith("use"):
        classes.add("dark-fill")

    style = get_style(node)
    if style is not None:
        if "fill" in style:
            color = get_hex(style["fill"])
            if is_black(color):
                classes.add("dark-fill")
                style.pop("fill")
            elif is_white(color):
                classes.add("light-fill")
                style.pop("fill")
            elif color_map is not None and color is not None:
                classes.add(f"{color_map[color]}-fill")
                style.pop("fill")
        if "stroke" in style:
            color = get_hex(style["stroke"])
            if is_black(color):
                classes.add("dark-stroke")
                style.pop("stroke")
            elif is_white(color):
                classes.add("light-stroke")
                style.pop("stroke")
            elif color_map is not None and color is not None:
                classes.add(f"{color_map[color]}-stroke")
                style.pop("stroke")
        if style:
            node.set("style", " ".join(f"{k}: {v};" for k, v in style.items()))
        else:
            node.attrib.pop("style")

    fill = node.get("fill")
    if fill is not None:
        color = get_hex(fill)
        if is_black(color):
            classes.add("dark-fill")
            node.attrib.pop("fill")
        elif is_white(color):
            classes.add("light-fill")
            node.attrib.pop("fill")
        elif color_map is not None and color is not None:
            classes.add(f"{color_map[color]}-fill")
            node.attrib.pop("fill")

    stroke = node.get("stroke")
    if stroke is not None:
        color = get_hex(stroke)
        if is_black(color):
            classes.add("dark-stroke")
            node.attrib.pop("stroke")
        elif is_white(color):
            classes.add("light-stroke")
            node.attrib.pop("stroke")
        elif color_map is not None and color is not None:
            classes.add(f"{color_map[color]}-stroke")
            node.attrib.pop("stroke")

    if convert_text:
        if node.tag.endswith("text"):
            classes.add("dark-fill")

    classes = " ".join(classes).strip()
    if classes:
        node.set("class", classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleans up SVG images")
    parser.add_argument("img", help="Path to the image")
    parser.add_argument(
        "-t",
        "--convert-text",
        default=False,
        action="store_true",
        help="If set, convert text as well",
    )
    parser.add_argument(
        "-c",
        "--convert-colors",
        default=False,
        action="store_true",
        help="If set, converts colors as well",
    )
    args = parser.parse_args()

    img = Path(args.img).absolute()
    assert img.exists(), f"Image not found: {img}"

    doc = etree.parse(str(img))

    # Removes comments.
    etree.strip_tags(doc, etree.Comment)

    if args.convert_colors:
        colors = set()
        for elem in doc.getiterator():
            colors.update(get_colors(elem))
        color_map = {color: f"color-{chr((i % 10) + ord('a'))}" for i, color in enumerate(colors)}
    else:
        color_map = None

    for elem in doc.getiterator():
        convert(elem, args.convert_text, color_map)

    with open(img, "wb") as f:
        doc.write(f, xml_declaration=False)


if __name__ == "__main__":
    main()
