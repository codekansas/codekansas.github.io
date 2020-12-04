#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from lxml import etree
from typing import Optional, Tuple


def get_hex(s: str) -> Optional[Tuple[int, int, int]]:
    if s.startswith("#"):
        if len(s) == 7:
            return tuple(int(s[i:i+2], 16) for i in (1, 3, 5))
        if len(s) == 4:
            return tuple(int(s[i:i+1], 16) for i in (1, 2, 3))
        raise ValueError(f"Invalid hex code: {s}")
    if s.startswith("rgb"):
        m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+).*\)", s)
        if m is None:
            raise ValueError(f"Invalid hex code: {s}")
        return tuple([int(i) for i in m.groups()])
    if s == "none":
        return None
    raise ValueError(f"Unsupported hex code: {s}")


def is_black(s: str) -> bool:
    h = get_hex(s)
    if h is None:
        return False
    return all(i < 10 for i in h)


def is_white(s: str) -> bool:
    h = get_hex(s)
    if h is None:
        return False
    return all(i > 245 for i in h)


def convert(node: etree.Element, convert_text: bool) -> None:
    classes = set()

    style = node.get("style")
    if style is not None:
        style = [
            k.strip().split(":")
            for k in style.split(";")
            if len(k.strip()) > 0
        ]
        for e in style:
            if len(e) != 2:
                raise ValueError(f"Invalid style element: {e}")
        style = {k[0].strip(): k[1].strip() for k in style if len(k) == 2}
        if "fill" in style:
            if is_black(style["fill"]):
                classes.add("dark-fill")
                style.pop("fill")
            elif is_white(style["fill"]):
                classes.add("light-fill")
                style.pop("fill")
        if "stroke" in style:
            if is_black(style["stroke"]):
                classes.add("dark-stroke")
                style.pop("stroke")
            elif is_white(style["stroke"]):
                classes.add("light-stroke")
                style.pop("stroke")
        node.set("style", " ".join(f"{k}: {v};" for k, v in style.items()))

    fill = node.get("fill")
    if fill is not None:
        if is_black(fill):
            classes.add("dark-fill")
            node.pop("fill")
        elif is_white(fill):
            classes.add("light-fill")
            node.pop("fill")

    stroke = node.get("stroke")
    if stroke is not None:
        if is_black(stroke):
            classes.add("dark-stroke")
            node.pop("stroke")
        elif is_white(stroke):
            classes.add("light-stroke")
            node.pop("stroke")

    if convert_text:
        if node.tag.endswith("text"):
            classes.add("dark-fill")

    if classes:
        node.set("class", " ".join(classes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleans up SVG images")
    parser.add_argument("img", help="Path to the image")
    parser.add_argument("-c", "--convert-text", default=False, action="store_true",
                        help="If set, convert text as well")
    args = parser.parse_args()

    img = Path(args.img).absolute()
    assert img.exists(), f"Image not found: {img}"

    doc = etree.parse(str(img))

    # Removes comments.
    etree.strip_tags(doc, etree.Comment)

    for elem in doc.getiterator():
        convert(elem, args.convert_text)

    with open(img, "wb") as f:
        doc.write(f)


if __name__ == "__main__":
    main()
