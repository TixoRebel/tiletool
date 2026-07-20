#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from PIL import Image, ImageChops
import hashlib
from dataclasses import dataclass
import argparse
from pathlib import Path
from functools import partial
from enum import Enum

@dataclass(frozen=True)
class TileInfo:
    index: int
    hflip: bool
    vflip: bool

class TiletoolException(Exception):
    pass

class TilesetFormat(Enum):
    F4BPP = 0
    F8BPP = 1

type TileRotations = frozenset[bytes]
type TileDict = dict[TileRotations, tuple[int, dict[bytes, tuple[bool, bool]]]]
type Tileset = list[Image.Image]
type Tilemap = list[TileInfo]
type Palette = list[int]

def combine_images_vertically(images: Tileset, palette: Palette) -> Image.Image:
    total_height = sum(img.height for img in images)

    combined_image = Image.new('P', (images[0].width, total_height))
    combined_image.putpalette(palette)

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    return combined_image

def make_tile_rotations(img_tile: Image.Image) -> tuple[TileRotations, bytes, bytes, bytes, bytes]:
    img_tile_hflip  = img_tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_tile_vflip  = img_tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img_tile_vhflip = img_tile_hflip.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    normal_hash = hashlib.sha256(img_tile.tobytes()).digest()
    hflip_hash  = hashlib.sha256(img_tile_hflip.tobytes()).digest()
    vflip_hash  = hashlib.sha256(img_tile_vflip.tobytes()).digest()
    vhflip_hash = hashlib.sha256(img_tile_vhflip.tobytes()).digest()

    return frozenset((normal_hash, vflip_hash, hflip_hash, vhflip_hash)), normal_hash, hflip_hash, vflip_hash, vhflip_hash

def make_tile_info(normal_hash: bytes, hflip_hash: bytes, vflip_hash: bytes, vhflip_hash: bytes) -> dict[bytes, tuple[bool, bool]]:
    return {vhflip_hash: (True, True), vflip_hash: (False, True), hflip_hash: (True, False), normal_hash: (False, False)}

def write_tilemap_bin(tilemap_list: list[TileInfo], pal_idx: int, tile_offset: int, path: str):
    with open(path, "wb") as f:
        for tile in tilemap_list:
            tile_data = ((tile.index + tile_offset) & 0x3FF) | ((1 if tile.hflip else 0) << 10) | ((1 if tile.vflip else 0) << 11) | ((pal_idx & 0xF) << 12)
            f.write(tile_data.to_bytes(2, byteorder="little"))

def make_tileset_tilemap(img: Image.Image) -> tuple[TileDict, Tileset, Tilemap]:
    tiles_dict = dict()
    tileset_list = []
    tilemap_list = []

    for y in range(0, img.height, 8):
        for x in range(0, img.width, 8):
            img_tile = img.crop((x, y, x + 8, y + 8))
            tile_rotations, normal_hash, hflip_hash, vflip_hash, vhflip_hash = make_tile_rotations(img_tile)

            tile_index, tile_info = tiles_dict.get(tile_rotations, (-1, None))
            if not tile_info:
                tileset_list.append(img_tile)
                tile_index = len(tileset_list) - 1
                tile_info = make_tile_info(normal_hash, hflip_hash, vflip_hash, vhflip_hash)
                tiles_dict[tile_rotations] = (tile_index, tile_info)

            hflip, vflip = tile_info[normal_hash]

            tilemap_list.append(TileInfo(tile_index, hflip, vflip))

    return (tiles_dict, tileset_list, tilemap_list)

def load_validate_image(path: str) -> tuple[Image.Image, Palette]:
    img = Image.open(path)

    if img.width % 8 != 0 or img.height % 8 != 0:
        raise TiletoolException(f"source image is not a multiple of 8 pixels wide / tall ({img.width}, {img.height})")

    if img.mode != 'P':
        raise TiletoolException(f"source image in indexed/palette mode")

    palette = img.getpalette()

    if not palette:
        raise TiletoolException(f"source image does not contain a palette")

    return (img, palette)

def load_tileset(path) -> tuple[TileDict, Tileset]:
    tiles_dict = dict()
    tileset_list = []

    tileset_img, _ = load_validate_image(path)
    for y in range(0, tileset_img.height, 8):
        for x in range(0, tileset_img.width, 8):
            img_tile = tileset_img.crop((x, y, x + 8, y + 8))
            tile_rotations, normal_hash, hflip_hash, vflip_hash, vhflip_hash = make_tile_rotations(img_tile)

            tileset_list.append(img_tile)
            tile_index = len(tileset_list) - 1
            tile_info = make_tile_info(normal_hash, hflip_hash, vflip_hash, vhflip_hash)
            tiles_dict[tile_rotations] = (tile_index, tile_info)

    return (tiles_dict, tileset_list)

def make_tilemap(img: Image.Image, tiles_dict: TileDict) -> Tilemap:
    tilemap_list = []
    for y in range(0, img.height, 8):
        for x in range(0, img.width, 8):
            img_tile = img.crop((x, y, x + 8, y + 8))
            tile_rotations, normal_hash, _, _, _ = make_tile_rotations(img_tile)
            tile_index, tile_info = tiles_dict.get(tile_rotations, (-1, None))
            if not tile_info:
                raise TiletoolException(f"source image contains tile not found in tileset at ({x}, {y})")

            hflip, vflip = tile_info[normal_hash]
            tilemap_list.append(TileInfo(tile_index, hflip, vflip))

    return tilemap_list

def handle_create(parser: argparse.ArgumentParser, args):
    try:
        if not args.output_tilemap and not args.output_tileset:
            parser.error("must supply either an output tilemap or tileset")

        if not args.image.exists():
            parser.error(f"source image '{args.image.absolute()}' does not exist")

        if args.format == None:
            format = TilesetFormat.F4BPP
        else:
            if args.format.lower() == '4bpp':
                format = TilesetFormat.F4BPP
            elif args.format.lower() == '8bpp':
                format = TilesetFormat.F8BPP
            else:
                parser.error(f"invalid format '{args.format}', valid options are: 4bpp, 8bpp")

        if args.palette == None:
            pal_idx = 0
        elif args.palette != 0 and format == TilesetFormat.F8BPP:
            parser.error("non-zero palette is not allowed for 8bpp tiles")
        elif args.palette >= 0 and args.palette < 16:
            pal_idx = args.palette
        else:
            parser.error(f"invalid palette '{args.palette}', valid options are 0-15")

        if args.offset == None:
            tile_offset = 0
        else:
            tile_offset = args.offset

        img, palette = load_validate_image(args.image)

        if args.input_tileset:
            if not args.input_tileset.exists():
                parser.error(f"source tileset '{args.input_tileset.absolute()}' does not exist")

            tiles_dict, tileset_list = load_tileset(args.input_tileset)
            tilemap_list = make_tilemap(img, tiles_dict)
        else:
            tiles_dict, tileset_list, tilemap_list = make_tileset_tilemap(img)

        if args.output_tileset:
            combine_images_vertically(tileset_list, palette).save(args.output_tileset)

        if args.output_tilemap:
            write_tilemap_bin(tilemap_list, pal_idx, tile_offset, args.output_tilemap)
    except TiletoolException as e:
        parser.error(str(e))

def main():
    parser = argparse.ArgumentParser(description="A tool for creating GBA tilemaps and tilesets.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_create = subparsers.add_parser("create", help="Create a tilemap, tileset, or both.")
    parser_create.add_argument("image", type=Path, help="The image to use to create the tilemap/tileset.")
    parser_create.add_argument("-is", "--input-tileset", type=Path, help="A tileset to use instead of auto-generating one from the source image.", metavar="input tileset")
    parser_create.add_argument("-om", "--output-tilemap", type=Path, help="The output path of the new tilemap.", metavar="output tilemap")
    parser_create.add_argument("-os", "--output-tileset", type=Path, help="The output path of the new tileset.", metavar="output tileset")
    parser_create.add_argument("-f", "--format", type=str, help="The format for the new tileset, either 8bpp or 4bpp (defaults to 4bpp if not specified).", metavar="tilemap format")
    parser_create.add_argument("-p", "--palette", type=lambda x: int(x, 0), help="The palette to use (defaults to 0 if not specified).", metavar="tilemap format")
    parser_create.add_argument("-s", "--offset", type=lambda x: int(x, 0), help="The offset at which the tiles will be loaded to in game.", metavar="offset")
    parser_create.set_defaults(func=partial(handle_create, parser=parser_create))

    args = parser.parse_args()

    args.func(args=args)

if __name__ == "__main__":
    main()
