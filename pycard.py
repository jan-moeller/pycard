import base64
import mimetypes
import sys
from abc import abstractmethod, ABCMeta
from collections.abc import Callable
from enum import Enum
from math import log10
from pathlib import Path
from types import TracebackType
from typing import Any, cast, final, override, Self

from click import command, argument, option, Choice
from jinja2 import Template, Environment, FileSystemLoader
from playwright.sync_api import sync_playwright, FloatRect, BrowserType
from ruamel.yaml import YAML, CommentedSeq, CommentedMap

_yaml = YAML()

type Variables = dict[str, str | int | float | bool]


class RendererType(Enum):
    """Enumerates all supported PNG renderer types"""

    NONE = 0
    FIREFOX = 1
    CHROMIUM = 2
    WEBKIT = 3


def recursive_render(
    env: Environment,
    template: Template,
    variables: dict[str, Any],
    max_depth: int = 10,
) -> str:
    """Recursively renders a template and returns the final result."""

    if max_depth <= 0:
        raise ValueError("max_depth must be positive")

    prev_result = None
    for _ in range(max_depth):
        result = template.render(**variables)
        if result == prev_result:
            break
        prev_result = result
        template = env.from_string(result)
    else:
        raise ValueError("Reached maximum recursive template rendering depth - cycle?")

    return result


class CardRenderer(metaclass=ABCMeta):  # pyright: ignore[report*]
    """Provides functionality to render a card as PNG."""

    @abstractmethod
    def render(self, svg: str) -> bytes:
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
    def render(self, svg: str) -> bytes:
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
    def render(self, svg: str) -> bytes:
        self._page.set_content(svg)
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
    env: Environment,
    cards: list[Variables],
    renderer: CardRenderer,
):
    """Generates all cards (SVG and PNG)."""

    out_path = cast(Path, env.globals["out_path"])
    cache_path = out_path / "cache.yaml"
    cache: Any = _yaml.load(cache_path) if cache_path.exists() else None  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
    if not isinstance(cache, list):
        cache = cast(list[Variables], [])

    for i, variables in enumerate(cards):
        card_id = "{{:0{}d}}".format(num_digits(len(cards))).format(i)
        print(f"Generating: {card_id}")

        variables |= {"ordinal": i}
        out_file_svg = out_path / (card_id + ".svg")
        out_file_png = out_path / (card_id + ".png")

        definitions_changed = cast(
            bool,
            cache[i] != cards[i] if i < len(cache) else True,  # pyright: ignore[reportUnknownArgumentType]
        )

        # Skip generation if nothing changed
        svg_text = None
        if definitions_changed or not out_file_svg.exists():
            svg_text = recursive_render(
                env,
                env.get_template(str(variables["template"])),
                variables,
            )

            _ = out_file_svg.write_text(svg_text)

        if definitions_changed or (
            not out_file_png.exists() and not isinstance(renderer, NoneCardRenderer)
        ):
            if not svg_text:
                svg_text = out_file_svg.read_text()
            png_bytes = renderer.render(svg_text)
            _ = out_file_png.write_bytes(png_bytes)

    with open(cache_path, "w") as cache_file:
        _yaml.dump(cards, cache_file)  # pyright: ignore[reportUnknownMemberType]


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


def extend_environment_globals(env: Environment, **kwargs: Any):  # pyright: ignore[reportAny]
    """Extends environment global variables."""
    for key, value in kwargs.items():  # pyright: ignore[reportAny]
        env.globals[key] = value


def extend_environment_filters(env: Environment, **kwargs: Callable[[Any], Any]):
    """Extends environment filters."""
    for key, value in kwargs.items():
        env.filters[key] = value


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

    cards = load_cards(cards_file)

    out_path.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(templates_path), autoescape=False)
    extend_environment_globals(
        env,
        out_path=out_path,
        assets_path=assets_path,
        num_cards=len(cards),
    )
    extend_environment_filters(
        env,
        b64decode=lambda s: base64.b64decode(s),  # pyright: ignore[reportAny]
        b64encode=lambda b: base64.b64encode(b).decode(),  # pyright: ignore[reportAny]
        read_text=lambda p: Path(p).read_text(),  # pyright: ignore[reportAny]
        read_bytes=lambda p: Path(p).read_bytes(),  # pyright: ignore[reportAny]
        bytes2hex=lambda b: b.hex(),  # pyright: ignore[reportAny]
        hex2bytes=bytes.fromhex,
        mimetype=lambda p: mimetypes.guess_file_type(p)[0],  # pyright: ignore[reportAny]
    )

    with create_renderer(png) as renderer:
        gen_cards(env=env, cards=cards, renderer=renderer)

    print("Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pyright: ignore[reportAny]
