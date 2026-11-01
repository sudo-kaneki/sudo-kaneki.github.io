# Agent Guidelines

Personal site of Writabrata Bhattacharya. Jekyll, custom retro-modern theme.
**Not al-folio any more** — do not consult al-folio docs or muscle memory; they
describe a theme and a project this repo no longer has.

## Architecture

Four routes: `/` (`_pages/home.md`), `/cv/`, `/projects/`, `/blog/`. Plus `/404.html`.

Five layouts: `base.liquid` (the shell: head, nav, main, footer) → `home.liquid`,
`listing.liquid` (content pages: cv, projects, blog index), `article.liquid`
(blog posts), `book.liquid` (the ML interview book reader). A layout can widen
`<main>` by setting `main_class` in its front matter (the book uses
`wrap wrap--book`).

Six includes: `site-head`, `site-nav`, `site-footer`, `timeline`, `project-card`,
`cv-section`.

Five SCSS partials, compiled by `assets/css/site.scss`:

| File                 | Holds                                                              |
| -------------------- | ------------------------------------------------------------------ |
| `_sass/_tokens.scss` | **Every** colour, size, and font. Nothing else may hardcode these. |
| `_sass/_base.scss`   | Reset and element defaults                                         |
| `_sass/_ui.scss`     | Components: `.btn`, `.card`, `.tag`, `.tabs`, `.timeline`          |
| `_sass/_pages.scss`  | Page-specific layout                                               |
| `_sass/_syntax.scss` | Code block / syntax highlighting                                   |

Four JS files: `assets/js/theme-toggle.js`, `assets/js/favicon-blink.js`,
`assets/js/blog-filter.js`, `assets/js/book-toc.js`.

Data: `_data/cv.yml`, `_data/socials.yml`, `_data/projects.yml`.

Seven Jekyll plugins, no Bootstrap, no jQuery, no framework.

## The one rule

**`_sass/_tokens.scss` is the only file allowed to contain a literal colour,
size, or font name.** Everything else uses `var(--token)`. Design changes —
including swapping in a pixel/bitmap display font via `--font-display` and
`--font-body` — must stay a single-file edit.

Retro cues, non-negotiable: `--radius: 0` (never a rounded corner), visible
1px borders, hard offset shadows (`4px 4px 0`, never blurred), and hover states
that translate by 2px so buttons depress like keys. All motion is disabled under
`prefers-reduced-motion`.

Dark is the default theme. Light must be checked too — both are first-class.

### Swapping in a pixel font

Set `--font-display` (and optionally `--font-body`) plus:

- `--fw-display: 400` — pixel/bitmap fonts ship exactly one weight; the
  browser fakes bold by smearing glyphs horizontally, which destroys the
  pixel grid.
- Re-pitch the `--fs-*` scale to multiples of 8px. Bitmap fonts only render
  crisply at integer multiples of their design size, and the current scale
  (14/20/28px, etc.) is not — glyphs would blur.
- Check `--lh-tight` and `--tracking-tight` too; a pixel face may need both
  loosened to avoid clipping or overlapping glyphs.

## Data

`_data/cv.yml` is the **single source of truth** for three outputs: the `/cv/`
page, the homepage timeline (`_includes/timeline.liquid` merges its
`education` and `experience` and sorts on a normalized zero-padded `YYYYMM`
key), and the CV PDF that `.github/workflows/render-cv.yml` regenerates via
RenderCV on every push touching it.

**Never hand-edit the CV PDF or duplicate career history anywhere.** Edit
`cv.yml`; everything else follows.

`_data/projects.yml` feeds `/projects/` and the homepage `featured` block.

## The ML interview book

"Cracking the ML Interview" is the `ml_interview` collection: one file per
chapter in `_ml_interview/`, rendered by `book.liquid` with a chapter sidebar.
It lives under `/blog/cracking-the-ml-interview/` until it is published as a
standalone book. Do not move it to `/book/` without being asked.

- Ordering comes from the `chapter` front-matter number: `0` is the preface
  (the book's landing page and its **single** entry on `/blog/`), `1`–`15` are
  chapters, `99` is the changelog. `permalink` is set per file.
- Display math is kramdown `$$ … $$` on its own lines (MathJax via
  `math: true`, set by the collection defaults).
- Diagrams not yet redrawn are `<figure class="figure-pending">` placeholders.
  Replace them with inline SVG that uses theme tokens (`currentColor`,
  `var(--accent)`), never hardcoded colours.
- "Further reading" lists that don't start at 1 are HTML `<ol start="N">`
  because kramdown ignores list start numbers. The numbers are cited in the
  text ("reference 39"), so keep them stable.
- Revising from feedback: readers file issues through
  `.github/ISSUE_TEMPLATE/book-feedback.yml` (linked, prefilled, from every
  chapter). To apply one: edit the chapter file, bump its `updated` date, and
  add a line to `_ml_interview/99-changelog.md`.

## Local development

Build with Docker Compose — this is the only supported build path:

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

**Do not run `docker compose up --build`.** It rebuilds the image and destroys
the pre-installed gems baked into it, breaking the build. Likewise, **never run
a bare `bundle install`** — it rewrites `Gemfile.lock` to versions the image
doesn't have. Both mistakes have broken this repo's dev environment repeatedly.
`docker compose up` (without `--build`) uses the prebuilt image and is fine for
clicking through the site at http://localhost:8080.

Before every commit: `npx prettier . --write` (CI enforces it).
