# Subtitle fonts

All fonts here except `bignoodletoooblique.ttf` come from [google/fonts](https://github.com/google/fonts)
under the SIL Open Font License 1.1 (the `OFL-*.txt` files next to this note). They back the
built-in subtitle presets (`podcast_reels_forge/utils/subtitle_presets.py`); every one covers Cyrillic.

| File | Family | Source |
|---|---|---|
| `MontserratBlack.ttf` | Montserrat Black | `ofl/montserrat/Montserrat[wght].ttf`, static instance at wght=900 |
| `OswaldBold.ttf` | Oswald Bold | `ofl/oswald/Oswald[wght].ttf`, static instance at wght=700 |
| `UnboundedBlack.ttf` | Unbounded Black | `ofl/unbounded/Unbounded[wght].ttf`, static instance at wght=900 |
| `RussoOne-Regular.ttf` | Russo One | `ofl/russoone/`, unmodified |
| `PressStart2P-Regular.ttf` | Press Start 2P | `ofl/pressstart2p/`, unmodified |
| `RubikMonoOne-Regular.ttf` | Rubik Mono One | `ofl/rubikmonoone/`, unmodified |

The three variable fonts were instanced with `fontTools.varLib.instancer` and given their own family
names, so libass and fontconfig pick the heavy weight without relying on variable-font support. The
modified fonts remain under the OFL; none of them carries a Reserved Font Name.
