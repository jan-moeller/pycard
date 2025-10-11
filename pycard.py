import base64
import mimetypes
import sys
import time
from abc import abstractmethod, ABCMeta
from enum import Enum
from math import log10, floor
from pathlib import Path
from types import TracebackType
from typing import Any, cast, final, override, Self

from click import command, argument, option, Choice
from playwright.sync_api import sync_playwright, FloatRect, BrowserType
from pyforma import Template, TemplateContext, DefaultTemplateContext
from ruamel.yaml import (
    YAML,
    CommentedSeq,
    CommentedMap,
    BaseConstructor,
    ScalarNode,
)
from ruamel.yaml.scalarstring import ScalarString

PATH_TAG = "!path"
TEMPLATE_TAG = "!template"


def construct_path(constructor: BaseConstructor, node: ScalarNode):
    path_str = cast(ScalarString, constructor.construct_scalar(node))  # pyright: ignore[reportUnknownMemberType]
    return Path(path_str)


def construct_template(constructor: BaseConstructor, node: ScalarNode):
    templates_str = cast(ScalarString, constructor.construct_scalar(node))  # pyright: ignore[reportUnknownMemberType]
    return Template(templates_str)


_yaml = YAML()
_yaml.constructor.add_constructor(PATH_TAG, construct_path)  # pyright: ignore[reportUnknownMemberType]
_yaml.constructor.add_constructor(TEMPLATE_TAG, construct_template)  # pyright: ignore[reportUnknownMemberType]

type Variables = dict[str, Any]


class RendererType(Enum):
    """Enumerates all supported PNG renderer types"""

    NONE = 0
    FIREFOX = 1
    CHROMIUM = 2
    WEBKIT = 3


class CardRenderer(metaclass=ABCMeta):  # pyright: ignore[report*]
    """Provides functionality to render a card as PNG."""

    @abstractmethod
    def render(self, svg: Path) -> bytes:
        """Renders the card as a PNG file and returns the result."""
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...


@final
class NoneCardRenderer(CardRenderer):
    """Pseudo PNG renderer that does nothing."""

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    @override
    def render(self, svg: Path) -> bytes:
        return b""


@final
class PlaywrightCardRenderer(CardRenderer):
    """PNG renderer using Playwright (i.e. a browser)."""

    class Browser(Enum):
        FIREFOX = 1
        CHROMIUM = 2
        WEBKIT = 3

        @staticmethod
        def from_renderer_type(
            renderer_type: RendererType,
        ) -> "PlaywrightCardRenderer.Browser":
            match renderer_type:
                case RendererType.FIREFOX:
                    return PlaywrightCardRenderer.Browser.FIREFOX
                case RendererType.CHROMIUM:
                    return PlaywrightCardRenderer.Browser.CHROMIUM
                case RendererType.WEBKIT:
                    return PlaywrightCardRenderer.Browser.WEBKIT
                case _:
                    raise ValueError(
                        f"Invalid renderer type for PlaywrightCardRenderer: {renderer_type}"
                    )

    def __init__(self, browser: Browser = Browser.FIREFOX, scale_factor: int = 1):
        super().__init__()
        self._scale_factor = scale_factor
        self._playwright = sync_playwright().start()
        pw_browser = cast(BrowserType, getattr(self._playwright, browser.name.lower()))
        self._browser = pw_browser.launch()
        self._page = self._browser.new_page(device_scale_factor=self._scale_factor)

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._browser.close()
        self._playwright.stop()
        return False

    @override
    def render(self, svg: Path) -> bytes:
        _ = self._page.goto(f"file://{svg}")
        self._page.evaluate("document.fonts.ready")
        # Measure SVG's bounding box
        box = cast(
            FloatRect,
            self._page.evaluate("""
            () => {
                const svg = document.querySelector('svg');
                const rect = svg.getBBox(); // in SVG coordinates
                const ctm = svg.getScreenCTM(); // convert to CSS pixels
    
                return {
                    x: rect.x * ctm.a + ctm.e,
                    y: rect.y * ctm.d + ctm.f,
                    width: rect.width * ctm.a,
                    height: rect.height * ctm.d
                };
            }
        """),
        )
        return self._page.screenshot(clip=box)


def create_renderer(renderer_type: RendererType) -> CardRenderer:
    """Creates a PNG card renderer based on configured type."""

    match renderer_type:
        case RendererType.NONE:
            return NoneCardRenderer()
        case RendererType.FIREFOX | RendererType.CHROMIUM | RendererType.WEBKIT:
            return PlaywrightCardRenderer(
                PlaywrightCardRenderer.Browser.from_renderer_type(renderer_type),
                scale_factor=5,  # TODO: Don't hardcode this
            )


def num_digits(i: int) -> int:
    """Helper function that returns the number of digits in an integer."""
    return int(log10(i)) + 1


def gen_cards(
    *,
    env: TemplateContext,
    out_path: Path,
    cards: list[Variables],
    renderer: CardRenderer,
):
    """Generates all cards (SVG and PNG)."""

    for i, variables in enumerate(cards):
        variables |= {"ordinal": i}

        card_id = "{{:0{}d}}".format(num_digits(len(cards))).format(i)
        print(f"Generating: {card_id}")

        # Render template
        template = str(variables["template"])
        svg_text = env.render(env.load_template(Path(template)), variables=variables)

        out_file_svg = Path(out_path) / (card_id + ".svg")
        _ = out_file_svg.write_text(svg_text)

        png_bytes = renderer.render(out_file_svg)

        if len(png_bytes) > 0:
            out_file_png = Path(out_path) / (card_id + ".png")
            _ = out_file_png.write_bytes(png_bytes)


def load_cards(cards_file: Path) -> list[Variables]:
    """Loads card information from a yaml file and returns a flat list of all cards to generate."""

    definitions: Any = _yaml.load(cards_file)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(definitions, CommentedSeq):
        raise ValueError(f"Invalid cards file: {cards_file}")

    def flatten(defs: CommentedSeq, vars: Variables) -> list[Variables]:
        cards: list[Variables] = []
        for elem in defs:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(elem, CommentedMap):
                raise ValueError(f"Invalid cards file: {cards_file}")

            if cast(str, elem.tag) == "!category":
                if not elem.keys() <= {"variables", "cards"}:  # pyright: ignore[reportUnknownMemberType]
                    raise ValueError(f"Invalid cards file: {cards_file}")

                new_vars: Any = elem.get("variables", CommentedMap())  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                if not isinstance(new_vars, CommentedMap):
                    raise ValueError(f"Invalid cards file: {cards_file}")
                vars |= new_vars

                new_cards: Any = elem.get("cards", CommentedSeq())  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                if not isinstance(new_cards, CommentedSeq):
                    raise ValueError(f"Invalid cards file: {cards_file}")
                cards += flatten(new_cards, vars)

            else:
                cards.append(vars | cast(Variables, elem))

        return cards

    cards = flatten(definitions, {})

    return cards


def prepare_cards(env: TemplateContext, cards: list[Variables]) -> list[Variables]:
    resolved_cards: list[Variables] = []
    for card in cards:
        template = env.load_template(Path(card["template"]))

        def collect_ids(value: Any) -> set[str]:
            match value:
                case Template():
                    return env.unresolved_identifiers(value)
                case dict() as d:  # pyright: ignore[reportUnknownVariableType]
                    r = set[str]()
                    for v in cast(dict[str, Any], d).values():
                        r = r | collect_ids(v)
                    return r
                case list() as l:  # pyright: ignore[reportUnknownVariableType]
                    r = set[str]()
                    for v in cast(list[Any], l):
                        r = r | collect_ids(v)
                    return r
                case _:
                    return set[str]()

        # collect all required IDs
        _required_variables = env.unresolved_identifiers(template)
        required_variables = set[str]()
        for variable_name in _required_variables:
            if variable_name not in card:
                raise ValueError(f"Required variable {variable_name} is not defined")

            required_variables |= collect_ids(card[variable_name])

        required_variables |= _required_variables

        resolved: dict[str, Any] = {}
        in_progress: set[str] = set()

        def _visit(identifier: str, value: Any) -> Any:
            match value:
                case Template():
                    unresolved = env.unresolved_identifiers(value) - resolved.keys()
                    additional = {_v: resolve(_v) for _v in unresolved}
                    return env.render(value, variables=resolved | additional)

                case dict() as d:  # pyright: ignore[reportUnknownVariableType]
                    return {
                        k: _visit(identifier, v)
                        for k, v in cast(dict[str, Any], d).items()
                    }

                case list() as l:  # pyright: ignore[reportUnknownVariableType]
                    return [_visit(identifier, v) for v in cast(list[Any], l)]

                case _:
                    return value

        def resolve(identifier: str) -> Any:
            if identifier in resolved:
                return resolved[identifier]
            if identifier in in_progress:
                raise ValueError(f"Variable {identifier} depends on itself")

            in_progress.add(identifier)

            value = card[identifier]

            resolved[identifier] = _visit(identifier, value)
            in_progress.remove(identifier)
            return resolved[identifier]

        for identifier in required_variables:
            _ = resolve(identifier)

        resolved_cards.append(resolved | {"template": Path(card["template"])})

    return resolved_cards


def b64decode(s: str) -> bytes:
    return base64.b64decode(s)


def b64encode(b: bytes) -> str:
    return base64.b64encode(b).decode()


def read_text(p: Path | str) -> str:
    return Path(p).read_text()


def read_bytes(p: Path | str) -> bytes:
    return Path(p).read_bytes()


def bytes2hex(b: bytes) -> str:
    return b.hex()


def hex2bytes(s: str) -> bytes:
    return bytes.fromhex(s)


def mimetype(p: Path | str) -> str:
    m = mimetypes.guess_file_type(p)[0]
    if m is None:
        raise ValueError(f"Unknown mimetype: {p}")
    return m


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        milliseconds = round(seconds * 1000)
        return f"{milliseconds} ms"

    total_seconds = floor(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if total_seconds >= 3600:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


@command()
@argument(
    "cards-file",
    type=Path,
)
@option(
    "-o",
    "--out-path",
    type=Path,
    default=Path.cwd(),
    help="Path to the output folder.",
)
@option(
    "-t",
    "--templates-path",
    type=Path,
    default=Path.cwd(),
    help="Path to the template folder.",
)
@option(
    "-a",
    "--assets-path",
    type=Path,
    default=Path.cwd(),
    help="Path to the assets folder.",
)
@option(
    "--png",
    type=Choice(RendererType, case_sensitive=False),
    default=RendererType.NONE,
    help="Render as PNG using the provided method.",
)
def main(
    cards_file: Path,
    out_path: Path,
    templates_path: Path,
    assets_path: Path,
    png: RendererType,
) -> int:
    """Main application entry point"""

    start = time.perf_counter()

    env = DefaultTemplateContext(base_path=templates_path)
    cards = load_cards(cards_file)
    globals = dict(
        out_path=out_path,
        assets_path=assets_path,
        num_cards=len(cards),
        # extra functions
        b64decode=b64decode,
        b64encode=b64encode,
        read_text=read_text,
        read_bytes=read_bytes,
        bytes2hex=bytes2hex,
        hex2bytes=hex2bytes,
        mimetype=mimetype,
    )
    cards = [globals | {"ordinal": i} | v for i, v in enumerate(cards)]
    cards = prepare_cards(env, cards)

    out_path.mkdir(parents=True, exist_ok=True)

    with create_renderer(png) as renderer:
        gen_cards(env=env, out_path=out_path, cards=cards, renderer=renderer)
    end = time.perf_counter()

    print(f"Done in {format_duration(end - start)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
