# Retro-Modern Site Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace al-folio's Bootstrap-based academic presentation layer with an owned retro-modern design system, collapsing the site from 11 routes to 4 (`home`, `cv`, `projects`, `blog`), with `_data/cv.yml` as the single source of truth for the CV page, the RenderCV PDF, and the homepage timeline.

**Architecture:** The new theme is built **alongside** the existing one under non-colliding filenames, previewed at `/preview/`, and only then cut over. This keeps `docker compose up --build` green at every single commit — the alternative (delete first, rebuild after) leaves the build red across many commits, which is precisely when you most need to see what you are doing. After cutover, al-folio's layouts, SCSS, vendored Bootstrap/MDB, and 11 Jekyll plugins are deleted.

**Tech Stack:** Jekyll 4 (Ruby), Liquid templates, Sass (dart-sass via `jekyll-sass-converter`), vanilla JS (no framework), Docker Compose for local dev, GitHub Actions → `gh-pages` for deploy.

**Spec:** [`docs/superpowers/specs/2026-07-11-retro-site-rebuild-design.md`](../specs/2026-07-11-retro-site-rebuild-design.md)

**Branch:** `retro-rebuild` (already created; spec committed as `7bd3330`)

---

## Global Constraints

These apply to **every** task. They are not repeated per-task.

1. **No `bust_file_cache` / `bust_css_cache` filters in any new template.** These come from `jekyll-cache-bust`, which Task 6 removes. Using them in new code creates a build failure that only appears three tasks later. Use plain `| relative_url`.
2. **No `site.third_party_libraries.*` references in any new template.** That hash lives in the 671-line `_config.yml` and is consumed by `jekyll-3rd-party-libraries`, both of which Task 5 replaces/removes. Hardcode CDN URLs in new templates.
3. **No hardcoded colours, sizes, or fonts outside `_sass/_tokens.scss`.** Every other file references `var(--token)`. This is the entire reason the token layer exists — the pixel-font swap must be a two-line change.
4. **`--radius: 0` everywhere. No `border-radius` in any new CSS.**
5. **Shadows are hard offsets** (`4px 4px 0`), never blurred.
6. **All hover transforms and transitions must be disabled under `prefers-reduced-motion: reduce`.**
7. **Dark is the default theme.** Light mode is fully supported and must be visually checked at every verification step, not just dark.
8. **Run `npx prettier . --write` before every commit.** The repo has a Prettier CI gate (`.github/workflows/prettier.yml`) with the `@shopify/prettier-plugin-liquid` plugin.
9. **Never `git add -A`.** The working tree has a pre-existing unstaged deletion (`assets/img/prof_pic_color.png`) that is not part of this work. Always `git add` explicit paths.

### Verification commands (used throughout)

**Build check** (scriptable, fast — use this for assertions):

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

Expected: exits 0, ends with `done in N seconds`. Any Liquid error, missing filter, or missing plugin fails loudly here.

**Serve check** (visual — use this for the human review gate):

```bash
docker compose up --build
# visit http://localhost:8080
# Ctrl-C, then:
docker compose down
```

---

## Naming: deviations from the spec (deliberate)

The spec named the new layouts `default`/`home`/`page`/`post` and the new SCSS `_tokens`/`_base`/`_components`/`_pages`. Three of those names **collide** with existing al-folio files, which would make coexistence impossible. The new names below are final — there is no later rename step.

| Spec said              | Plan uses                      | Why                                  |
| ---------------------- | ------------------------------ | ------------------------------------ |
| `_layouts/default`     | `_layouts/base.liquid`         | `default.liquid` exists (al-folio)   |
| `_layouts/page`        | `_layouts/listing.liquid`      | `page.liquid` exists (al-folio)      |
| `_layouts/post`        | `_layouts/article.liquid`      | `post.liquid` exists (al-folio)      |
| `_layouts/home`        | `_layouts/home.liquid`         | no collision — unchanged             |
| `_sass/_components`    | `_sass/_ui.scss`               | `_components.scss` exists (al-folio) |
| `assets/css/main.scss` | `assets/css/site.scss`         | `main.scss` exists (al-folio)        |
| `_includes/head`       | `_includes/site-head.liquid`   | `head.liquid` exists (al-folio)      |
| `_includes/footer`     | `_includes/site-footer.liquid` | `footer.liquid` exists (al-folio)    |

`listing.liquid` serves `/cv/`, `/projects/`, and `/blog/`. `article.liquid` serves blog posts.

One behavioural addition not in the spec: **the homepage "What I'm building" block hides itself when `_data/projects.yml` is empty**, exactly as the "Writing" block hides when `_posts/` is empty. You have not yet supplied project content, and shipping an empty-state apology on the homepage is worse than shipping nothing.

---

## Task 1: Purge demo content

Deletes ~40 files of al-folio boilerplate. Nothing new is built. The old theme still renders the remaining real pages, so the build stays green.

**Files:**

- Delete: `_projects/` (9 files), `_books/` (1), `_teachings/` (2), `_news/` (4)
- Delete: all 21 demo posts in `_posts/`
- Delete: `_pages/projects.md`, `_pages/books.md`, `_pages/teaching.md`, `_pages/repositories.md`, `_pages/profiles.md`, `_pages/news.md`, `_pages/dropdown.md`, `_pages/about_einstein.md`
- Delete: `assets/pdf/example_pdf.pdf`, `assets/rendercv/rendercv_output/Albert_Einstein_CV.pdf`, `CV_Writabrata (1).pdf` (repo root)
- Modify: `_config.yml` — remove the `collections:` entries for `books`, `news`, `projects`, `teachings`, and their `defaults:` scopes

**Interfaces:**

- Consumes: nothing
- Produces: a repo whose only content is `_pages/about.md`, `_pages/blog.md`, `_pages/cv.md`, `_pages/publications.md`, `_pages/404.md`, `_data/*`, `_bibliography/papers.bib`

<!-- prettier-ignore -->
- [ ] **Step 1: Delete the demo collections and posts**

```bash
cd /Users/kaneki/Documents/Code/sudo-kaneki.github.io
git rm -r --quiet _projects _books _teachings _news
git rm --quiet _posts/*.md
git rm --quiet _pages/projects.md _pages/books.md _pages/teaching.md \
  _pages/repositories.md _pages/profiles.md _pages/news.md \
  _pages/dropdown.md _pages/about_einstein.md
git rm --quiet assets/pdf/example_pdf.pdf \
  assets/rendercv/rendercv_output/Albert_Einstein_CV.pdf \
  "CV_Writabrata (1).pdf"
```

<!-- prettier-ignore -->
- [ ] **Step 2: Keep `_posts/` alive as a directory**

`git rm _posts/*.md` removes the directory entirely. Jekyll tolerates a missing `_posts/`, but `paginate-v2` on `_pages/blog.md` will error with `Pagination: Pagination is enabled, but I couldn't find any posts`. Add a `.gitkeep` so the directory survives, and disable pagination until there are posts (re-enabled in Task 4).

```bash
mkdir -p _posts && touch _posts/.gitkeep
git add _posts/.gitkeep
```

<!-- prettier-ignore -->
- [ ] **Step 3: Remove the dead collections from `_config.yml`**

Find the `collections:` block (around line 155) and reduce it to nothing. Delete these lines entirely:

```yaml
collections:
  books:
    output: true
  news:
    defaults:
      layout: post
    output: true
  projects:
    output: true
  teachings:
    output: true
```

Then find the `defaults:` block (around line 234) and delete any `scope:` entry whose `type:` is `books`, `news`, `projects`, or `teachings`. Leave `posts` and `pages` scopes intact.

<!-- prettier-ignore -->
- [ ] **Step 4: Verify the build is green**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

Expected: exit 0. If it fails with `Unknown collection`, a `defaults:` scope in Step 3 was missed.

Then confirm the demo routes are gone and the real ones survive:

```bash
test ! -d _site/projects && echo "OK: projects gone"
test ! -d _site/teaching && echo "OK: teaching gone"
test -f _site/index.html && echo "OK: home renders"
test -f _site/cv/index.html && echo "OK: cv renders"
```

Expected: four `OK:` lines.

<!-- prettier-ignore -->
- [ ] **Step 5: Commit**

```bash
npx prettier . --write
git add _config.yml _posts/.gitkeep
git commit -m "chore: delete al-folio demo content and dead pages

Removes the projects, books, teachings, and news collections, all 21
template blog posts, the eight pages that served them, and the stray
Einstein/example PDFs. None of this content was ours."
```

---

## Task 2: Design tokens and base stylesheet

Creates the token layer and a new stylesheet entry point that compiles independently of al-folio's `main.scss`. Nothing consumes it yet, so the site is unchanged and the build stays green.

**Files:**

- Create: `_sass/_tokens.scss`
- Create: `_sass/_base.scss`
- Create: `assets/css/site.scss`

**Interfaces:**

- Consumes: nothing
- Produces: `/assets/css/site.css`, exposing these CSS custom properties to all later tasks — `--bg`, `--surface`, `--fg`, `--fg-muted`, `--border`, `--accent`, `--accent-strong`, `--accent-ink`, `--font-display`, `--font-body`, `--fs-xs`…`--fs-3xl`, `--sp-1`…`--sp-12`, `--radius`, `--bw`, `--shadow`, `--shadow-sm`, `--w-content`, `--w-wide`, `--t`. Later tasks reference **only** these, never literal values.

<!-- prettier-ignore -->
- [ ] **Step 1: Create `_sass/_tokens.scss`**

Every design decision in the site lives in this one file.

```scss
/* =============================================================
   DESIGN TOKENS — the only file permitted to contain literal
   colours, sizes, or font names. Everything else uses var().
   ============================================================= */

:root {
  /* ---- Typography -------------------------------------------
     THE FONT SWAP HAPPENS HERE. To adopt a pixel/bitmap face,
     change --font-display (and optionally --font-body) and add
     the @font-face / <link> in _includes/site-head.liquid.
     Nothing else in the codebase needs to change.
     --------------------------------------------------------- */
  --font-display: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --font-body: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;

  --fs-xs: 0.75rem;
  --fs-sm: 0.875rem;
  --fs-base: 1rem;
  --fs-lg: 1.25rem;
  --fs-xl: 1.75rem;
  --fs-2xl: 2.5rem;
  --fs-3xl: 3.5rem;

  --lh-tight: 1.25;
  --lh-base: 1.7;

  /* ---- Spacing (4px base) ---------------------------------- */
  --sp-1: 0.25rem;
  --sp-2: 0.5rem;
  --sp-3: 0.75rem;
  --sp-4: 1rem;
  --sp-6: 1.5rem;
  --sp-8: 2rem;
  --sp-12: 3rem;
  --sp-16: 4rem;
  --sp-24: 6rem;

  /* ---- Retro hardware cues --------------------------------- */
  --radius: 0; /* never anything else */
  --bw: 1px;
  --shadow: 4px 4px 0 var(--border);
  --shadow-sm: 2px 2px 0 var(--border);

  /* ---- Layout ---------------------------------------------- */
  --w-content: 720px;
  --w-wide: 960px;

  /* ---- Motion ---------------------------------------------- */
  --t: 120ms ease-out;
}

/* ---- Dark theme (DEFAULT) ---------------------------------- */
:root,
:root[data-theme="dark"] {
  --bg: #0b0d0c;
  --surface: #121614;
  --fg: #e6e8e6;
  --fg-muted: #9aa39d;
  --border: #2c332e;
  --accent: #5ee2a0; /* phosphor green, desaturated for AA */
  --accent-strong: #7dffb8;
  --accent-ink: #0b0d0c; /* text ON an accent background */
}

/* ---- Light theme ------------------------------------------- */
:root[data-theme="light"] {
  --bg: #f4f3ee;
  --surface: #ffffff;
  --fg: #16191a;
  --fg-muted: #5b6360;
  --border: #d4d3cc;
  --accent: #0f7a4d;
  --accent-strong: #0a5c3a;
  --accent-ink: #ffffff;
}

/* Contrast ratios (verified against WCAG 2.1 AA, 4.5:1 body text):
   dark  — fg 15.0:1, fg-muted 7.2:1, accent 12.5:1
   light — fg 16.1:1, fg-muted  5.4:1, accent  4.8:1          */

@media (prefers-reduced-motion: reduce) {
  :root {
    --t: 0ms;
  }
}
```

<!-- prettier-ignore -->
- [ ] **Step 2: Create `_sass/_base.scss`**

```scss
/* Reset + element defaults. No component classes here. */

*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-body);
  font-size: var(--fs-base);
  line-height: var(--lh-base);
}

h1,
h2,
h3,
h4 {
  font-family: var(--font-display);
  line-height: var(--lh-tight);
  margin: 0 0 var(--sp-4);
  font-weight: 700;
}

h1 {
  font-size: var(--fs-2xl);
}
h2 {
  font-size: var(--fs-xl);
}
h3 {
  font-size: var(--fs-lg);
}

p,
ul,
ol {
  margin: 0 0 var(--sp-4);
}

a {
  color: var(--accent);
  text-decoration: underline;
  text-underline-offset: 3px;
  transition: color var(--t);
}

a:hover {
  color: var(--accent-strong);
}

/* Retro focus ring: thick, offset, unmistakable. */
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

img {
  max-width: 100%;
  height: auto;
}

hr {
  border: 0;
  border-top: var(--bw) solid var(--border);
  margin: var(--sp-12) 0;
}

code,
pre {
  font-family: var(--font-body);
  font-size: var(--fs-sm);
}

pre {
  background: var(--surface);
  border: var(--bw) solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-4);
  overflow-x: auto;
}

/* Long content must scroll inside itself, never the page body. */
table {
  display: block;
  overflow-x: auto;
  border-collapse: collapse;
  width: 100%;
}

.wrap {
  max-width: var(--w-content);
  margin: 0 auto;
  padding: var(--sp-12) var(--sp-4);
}

.skip-link {
  position: absolute;
  left: -9999px;
}

.skip-link:focus {
  left: var(--sp-4);
  top: var(--sp-4);
  z-index: 10;
  background: var(--accent);
  color: var(--accent-ink);
  padding: var(--sp-2) var(--sp-4);
}
```

<!-- prettier-ignore -->
- [ ] **Step 3: Create `assets/css/site.scss`**

Note the empty front matter — Jekyll requires it to process the file. This compiles to `/assets/css/site.css`.

```scss
---
---

@use "tokens";
@use "base";
```

<!-- prettier-ignore -->
- [ ] **Step 4: Verify it compiles**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
grep -q -- "--font-display" _site/assets/css/site.css && echo "OK: tokens compiled"
grep -q "prefers-reduced-motion" _site/assets/css/site.css && echo "OK: motion guard present"
```

Expected: two `OK:` lines. The old site is untouched and still renders.

<!-- prettier-ignore -->
- [ ] **Step 5: Commit**

```bash
npx prettier . --write
git add _sass/_tokens.scss _sass/_base.scss assets/css/site.scss
git commit -m "feat: add retro design tokens and base stylesheet

Single source of truth for every colour, size, and font in the new
theme. --font-display and --font-body are the swap points for the
pixel font. Compiles to /assets/css/site.css alongside the existing
al-folio main.css; nothing consumes it yet."
```

---

## Task 3: The shell — layouts, nav, footer, theme toggle

Builds the page shell and proves it renders at `/preview/`, with **the old site still fully working**. This is the first task where you can look at the design.

**Files:**

- Create: `_layouts/base.liquid`
- Create: `_includes/site-head.liquid`, `_includes/site-nav.liquid`, `_includes/site-footer.liquid`
- Create: `assets/js/theme-toggle.js`
- Create: `_sass/_ui.scss`
- Modify: `assets/css/site.scss` (add `@use "ui";`)
- Create: `_pages/preview.md` (temporary scaffold; deleted in Task 6)

**Interfaces:**

- Consumes: all tokens from Task 2
- Produces:
  - Layout `base` — expects `page.title` (string, optional). Renders `{{ content }}` inside `<main class="wrap">`.
  - CSS classes for later tasks: `.btn`, `.card`, `.tag`, `.tabs`, `.tab`, `.timeline`, `.timeline__item`, `.timeline__date`, `.timeline__body`, `.timeline__title`, `.timeline__org`, `.timeline__meta`
  - `data-theme` attribute on `<html>`, values `"dark"` | `"light"`

<!-- prettier-ignore -->
- [ ] **Step 1: Create `_includes/site-head.liquid`**

The theme-init script is **inline and blocking on purpose** — it must stamp `data-theme` before first paint or the page flashes the wrong theme. Do not move it to an external file and do not add `defer`.

```liquid
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

{% assign page_title = page.title | default: site.first_name | append: ' ' | append: site.last_name %}
<title>
  {{ page_title }}
  ·
  {{ site.first_name }}
  {{ site.last_name }}
</title>
<meta name="description" content="{{ page.description | default: site.description | strip_newlines | strip }}">
<link rel="canonical" href="{{ page.url | replace: 'index.html', '' | absolute_url }}">

<link
  rel="shortcut icon"
  href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>{{ site.icon }}</text></svg>"
>

<link rel="stylesheet" href="{{ '/assets/css/site.css' | relative_url }}">

<script>
  // Blocking on purpose: must run before first paint to avoid a
  // flash of the wrong theme. Dark is the default.
  (function () {
    var stored = null;
    try {
      stored = localStorage.getItem('theme');
    } catch (e) {}
    var prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    var theme = stored || (prefersLight ? 'light' : 'dark');
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>

{% if page.math %}
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [
          ['$', '$'],
          ['\\(', '\\)'],
        ],
      },
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
{% endif %}
```

<!-- prettier-ignore -->
- [ ] **Step 2: Create `_includes/site-nav.liquid`**

```liquid
<a class="skip-link" href="#main">Skip to content</a>

<header class="nav">
  <div class="nav__inner">
    <a class="nav__brand" href="{{ '/' | relative_url }}">
      {{- site.first_name | slice: 0, 1 -}}
      {{- site.last_name | slice: 0, 1 -}}
    </a>

    <nav class="nav__links" aria-label="Main">
      <a href="{{ '/' | relative_url }}">home</a>
      <a href="{{ '/cv/' | relative_url }}">cv</a>
      <a href="{{ '/projects/' | relative_url }}">projects</a>
      <a href="{{ '/blog/' | relative_url }}">blog</a>
    </nav>

    <button id="theme-toggle" class="nav__toggle" type="button" aria-label="Switch theme">☀</button>
  </div>
</header>
```

<!-- prettier-ignore -->
- [ ] **Step 3: Create `_includes/site-footer.liquid`**

```liquid
<footer class="footer">
  <div class="footer__inner">
    <span
      >© {{ 'now' | date: '%Y' }}
      {{ site.first_name }}
      {{ site.last_name -}}
    </span>
    <span class="footer__links">
      <a href="https://github.com/{{ site.data.socials.github_username }}">github</a>
      <a href="https://www.linkedin.com/in/{{ site.data.socials.linkedin_username }}">linkedin</a>
      <a href="mailto:{{ site.data.socials.email }}">email</a>
    </span>
  </div>
</footer>
```

<!-- prettier-ignore -->
- [ ] **Step 4: Create `_layouts/base.liquid`**

```liquid
<!doctype html>
<html lang="{{ site.lang | default: 'en' }}">
  <head>
    {% include site-head.liquid %}
  </head>
  <body>
    {% include site-nav.liquid %}
    <main id="main" class="wrap">
      {{ content }}
    </main>
    {% include site-footer.liquid %}
    <script defer src="{{ '/assets/js/theme-toggle.js' | relative_url }}"></script>
  </body>
</html>
```

<!-- prettier-ignore -->
- [ ] **Step 5: Create `assets/js/theme-toggle.js`**

```js
(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;

  function sync() {
    var isDark = root.getAttribute("data-theme") === "dark";
    btn.textContent = isDark ? "☀" : "☾";
    btn.setAttribute("aria-label", "Switch to " + (isDark ? "light" : "dark") + " theme");
  }

  btn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {}
    sync();
  });

  sync();
})();
```

<!-- prettier-ignore -->
- [ ] **Step 6: Create `_sass/_ui.scss`**

This is where the four retro rules become real. Note `.btn` and `.card`: hard border, hard offset shadow, and a hover that physically depresses the element like a key.

```scss
/* ---- Nav ---------------------------------------------------- */
.nav {
  border-bottom: var(--bw) solid var(--border);
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 5;
}

.nav__inner {
  max-width: var(--w-wide);
  margin: 0 auto;
  padding: var(--sp-4);
  display: flex;
  align-items: center;
  gap: var(--sp-6);
}

.nav__brand {
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--fg);
  text-decoration: none;
  border: var(--bw) solid var(--border);
  padding: var(--sp-1) var(--sp-2);
}

.nav__links {
  display: flex;
  gap: var(--sp-4);
  margin-left: auto;
  flex-wrap: wrap;
}

.nav__links a {
  color: var(--fg-muted);
  text-decoration: none;
}

.nav__links a:hover {
  color: var(--accent);
}

.nav__toggle {
  background: transparent;
  border: var(--bw) solid var(--border);
  border-radius: var(--radius);
  color: var(--fg);
  cursor: pointer;
  font-size: var(--fs-base);
  line-height: 1;
  padding: var(--sp-2);
}

/* ---- Button: the key-press interaction ---------------------- */
.btn {
  display: inline-block;
  font-family: var(--font-display);
  background: var(--surface);
  color: var(--fg);
  border: var(--bw) solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: var(--sp-3) var(--sp-6);
  text-decoration: none;
  transition:
    transform var(--t),
    box-shadow var(--t);
}

.btn:hover {
  color: var(--accent);
  transform: translate(2px, 2px);
  box-shadow: var(--shadow-sm);
}

.btn:active {
  transform: translate(4px, 4px);
  box-shadow: none;
}

.btn--accent {
  background: var(--accent);
  color: var(--accent-ink);
}

.btn--accent:hover {
  color: var(--accent-ink);
}

/* ---- Card --------------------------------------------------- */
.card {
  background: var(--surface);
  border: var(--bw) solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: var(--sp-6);
  transition:
    transform var(--t),
    box-shadow var(--t);
}

.card:hover {
  transform: translate(2px, 2px);
  box-shadow: var(--shadow-sm);
}

/* ---- Tag ---------------------------------------------------- */
.tag {
  display: inline-block;
  font-size: var(--fs-xs);
  color: var(--fg-muted);
  border: var(--bw) solid var(--border);
  border-radius: var(--radius);
  padding: 0 var(--sp-2);
  margin: 0 var(--sp-1) var(--sp-1) 0;
}

/* ---- Tabs (blog filter) ------------------------------------- */
.tabs {
  display: flex;
  border: var(--bw) solid var(--border);
  margin-bottom: var(--sp-8);
  width: fit-content;
}

.tab {
  background: transparent;
  border: 0;
  border-right: var(--bw) solid var(--border);
  border-radius: var(--radius);
  color: var(--fg-muted);
  cursor: pointer;
  font-family: var(--font-display);
  font-size: var(--fs-sm);
  padding: var(--sp-2) var(--sp-4);
}

.tab:last-child {
  border-right: 0;
}

.tab[aria-selected="true"] {
  background: var(--accent);
  color: var(--accent-ink);
}

/* ---- Timeline ----------------------------------------------- */
.timeline {
  list-style: none;
  margin: 0;
  padding: 0 0 0 var(--sp-6);
  border-left: var(--bw) solid var(--border);
}

.timeline__item {
  display: flex;
  gap: var(--sp-6);
  padding: var(--sp-4) 0;
  position: relative;
}

.timeline__item::before {
  content: "";
  position: absolute;
  left: calc(-1 * var(--sp-6) - 4px);
  top: var(--sp-6);
  width: 7px;
  height: 7px;
  background: var(--accent);
}

.timeline__date {
  font-family: var(--font-display);
  color: var(--accent);
  flex: 0 0 4ch;
}

.timeline__body {
  display: flex;
  flex-direction: column;
}

.timeline__title {
  font-family: var(--font-display);
  color: var(--fg);
}

.timeline__org {
  color: var(--fg-muted);
}

.timeline__meta {
  color: var(--fg-muted);
  font-size: var(--fs-xs);
}

/* ---- Footer ------------------------------------------------- */
.footer {
  border-top: var(--bw) solid var(--border);
  margin-top: var(--sp-24);
}

.footer__inner {
  max-width: var(--w-wide);
  margin: 0 auto;
  padding: var(--sp-6) var(--sp-4);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: var(--sp-4);
  color: var(--fg-muted);
  font-size: var(--fs-sm);
}

.footer__links {
  display: flex;
  gap: var(--sp-4);
}

@media (max-width: 640px) {
  .nav__inner {
    gap: var(--sp-3);
  }

  .timeline__item {
    flex-direction: column;
    gap: var(--sp-1);
  }
}
```

<!-- prettier-ignore -->
- [ ] **Step 7: Wire `_ui` into the stylesheet**

Replace the contents of `assets/css/site.scss`:

```scss
---
---

@use "tokens";
@use "base";
@use "ui";
```

<!-- prettier-ignore -->
- [ ] **Step 8: Create the temporary preview page**

`_pages/preview.md` — this is scaffolding and is deleted in Task 6.

```markdown
---
layout: base
permalink: /preview/
title: preview
---

# Shell preview

Body text renders here.

<a class="btn btn--accent" href="#">accent button</a>
<a class="btn" href="#">plain button</a>

<div class="card">A card. Hard border, hard shadow, no radius.</div>
```

<!-- prettier-ignore -->
- [ ] **Step 9: Verify the shell renders and the old site still works**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
grep -q 'id="theme-toggle"' _site/preview/index.html && echo "OK: toggle present"
grep -q 'site.css' _site/preview/index.html && echo "OK: new stylesheet linked"
grep -q 'bootstrap' _site/preview/index.html && echo "FAIL: bootstrap leaked into new shell" || echo "OK: no bootstrap in new shell"
test -f _site/index.html && echo "OK: old site still builds"
```

Expected: four `OK:` lines, no `FAIL`.

<!-- prettier-ignore -->
- [ ] **Step 10: Look at it (human gate)**

```bash
docker compose up --build
```

Visit `http://localhost:8080/preview/`. Confirm, in **both** themes (click the toggle):

- No rounded corners anywhere.
- Buttons visibly depress toward the bottom-right on hover.
- Hard-edged shadows, no blur.
- Reloading does **not** flash the wrong theme.
- At a 375px-wide viewport there is no horizontal scrollbar.

Then `Ctrl-C` and `docker compose down`.

<!-- prettier-ignore -->
- [ ] **Step 11: Commit**

```bash
npx prettier . --write
git add _layouts/base.liquid _includes/site-head.liquid _includes/site-nav.liquid \
  _includes/site-footer.liquid assets/js/theme-toggle.js _sass/_ui.scss \
  assets/css/site.scss _pages/preview.md
git commit -m "feat: add retro page shell (base layout, nav, footer, theme toggle)

Builds alongside the al-folio theme under non-colliding names; visible
at /preview/. Theme init is inline and blocking to prevent a flash of
the wrong theme. Old site untouched and still building."
```

---

## Task 4: Homepage

Builds the narrative homepage at `/preview-home/`. The timeline is generated from `cv.yml`, so the career history is never maintained twice.

**Files:**

- Create: `_layouts/home.liquid`
- Create: `_includes/timeline.liquid`, `_includes/project-card.liquid`
- Create: `_data/projects.yml`
- Create: `_pages/home.md`
- Modify: `_sass/_pages.scss` (new), `assets/css/site.scss`

**Interfaces:**

- Consumes: `base` layout, all `.btn` / `.card` / `.tag` / `.timeline*` classes from Task 3
- Produces:
  - `_data/projects.yml` schema, consumed by Task 5's `/projects/` page:
    ```yaml
    - title: string # required
      blurb: string # required, one line
      tags: [string] # optional
      featured: bool # optional; true → also shown on homepage
      links: # optional
        - label: string
          url: string
    ```
  - `_includes/project-card.liquid` — expects a variable named `project` matching the schema above.

<!-- prettier-ignore -->
- [ ] **Step 1: Create `_includes/timeline.liquid`**

`cv.yml` nests everything under a top-level `cv:` key, so the path is `site.data.cv.cv.sections`. Education and experience have **different keys** (`institution`/`degree` vs `company`/`position`) but both have `start_date`, so they can be concatenated and sorted on that, then discriminated at render time by testing for `item.company`.

`start_date` values are `YYYY-MM` strings (YAML does not parse these as dates), so a lexicographic sort is chronologically correct.

```liquid
{% assign edu = site.data.cv.cv.sections.education %}
{% assign exp = site.data.cv.cv.sections.experience %}
{% assign items = exp | concat: edu | sort: 'start_date' | reverse %}

<ol class="timeline">
  {% for item in items %}
    <li class="timeline__item">
      <span class="timeline__date">{{ item.start_date | slice: 0, 4 }}</span>
      <div class="timeline__body">
        {% if item.company %}
          <span class="timeline__title">{{ item.position }}</span>
          <span class="timeline__org">{{ item.company }}</span>
        {% else %}
          <span class="timeline__title">{{ item.degree }}, {{ item.area }}</span>
          <span class="timeline__org">{{ item.institution }}</span>
        {% endif %}
        <span class="timeline__meta">{{ item.start_date }} — {{ item.end_date }} · {{ item.location }}</span>
      </div>
    </li>
  {% endfor %}
</ol>
```

<!-- prettier-ignore -->
- [ ] **Step 2: Create `_includes/project-card.liquid`**

```liquid
<article class="card">
  <h3>{{ project.title }}</h3>
  <p>{{ project.blurb }}</p>
  {% if project.tags %}
    <p>
      {% for tag in project.tags -%}
        <span class="tag">{{ tag }}</span>
      {%- endfor %}
    </p>
  {% endif %}
  {% if project.links %}
    <p>
      {% for link in project.links %}
        <a href="{{ link.url }}">{{ link.label }}</a>
        {%- unless forloop.last %} · {% endunless %}
      {% endfor %}
    </p>
  {% endif %}
</article>
```

<!-- prettier-ignore -->
- [ ] **Step 3: Create `_data/projects.yml`**

Starts empty. **Do not invent projects.** Both the homepage block and the `/projects/` page handle the empty case (Step 5, and Task 5 Step 2).

```yaml
# Side projects, outside of work.
#
# Schema:
#   - title: string        (required)
#     blurb: string        (required, one line)
#     tags: [string]       (optional — tech used)
#     featured: bool       (optional — true also surfaces it on the homepage)
#     links:               (optional)
#       - label: string
#         url: string
#
# Deep write-ups belong in _posts/ — link to them from `links`.
```

<!-- prettier-ignore -->
- [ ] **Step 4: Create `_layouts/home.liquid`**

Note the two `{% if %}` guards: the "building" and "writing" blocks disappear entirely rather than render an empty state.

```liquid
---
layout: base
---
<section class="hero">
  <h1 class="hero__name">
    {{ site.first_name -}}
    <br>
    {{ site.last_name }}
  </h1>
  <p class="hero__headline">{{ site.data.cv.cv.headline }}</p>
  <p class="hero__actions">
    <a class="btn btn--accent" href="mailto:{{ site.data.socials.email }}">contact</a>
    <a class="btn" href="{{ '/cv/' | relative_url }}">cv</a>
  </p>
</section>

<section class="section">
  <h2>my story</h2>
  {{ content }}
</section>

<section class="section">
  <h2>the path so far</h2>
  {% include timeline.liquid %}
</section>

{% assign featured = site.data.projects | where: 'featured', true %}
{% if featured.size > 0 %}
  <section class="section">
    <h2>what i'm building</h2>
    <div class="grid">
      {% for project in featured %}
        {% include project-card.liquid project=project %}
      {% endfor %}
    </div>
  </section>
{% endif %}

{% if site.posts.size > 0 %}
  <section class="section">
    <h2>writing</h2>
    <ul class="post-list">
      {% for post in site.posts limit: 3 %}
        <li class="post-list__item">
          <span class="post-list__date">{{ post.date | date: '%Y-%m-%d' }}</span>
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </li>
      {% endfor %}
    </ul>
    <p><a href="{{ '/blog/' | relative_url }}">all posts →</a></p>
  </section>
{% endif %}

<section class="section">
  <h2>elsewhere</h2>
  <p>
    <a href="https://github.com/{{ site.data.socials.github_username }}">github</a> ·
    <a href="https://www.linkedin.com/in/{{ site.data.socials.linkedin_username }}">linkedin</a> ·
    <a href="mailto:{{ site.data.socials.email }}">{{ site.data.socials.email }}</a>
  </p>
</section>
```

<!-- prettier-ignore -->
- [ ] **Step 5: Create `_sass/_pages.scss`**

```scss
/* ---- Hero --------------------------------------------------- */
.hero {
  padding: var(--sp-16) 0 var(--sp-12);
  border-bottom: var(--bw) solid var(--border);
}

.hero__name {
  font-size: var(--fs-3xl);
  margin: 0 0 var(--sp-4);
  letter-spacing: -0.01em;
}

.hero__headline {
  color: var(--fg-muted);
  font-size: var(--fs-lg);
  margin-bottom: var(--sp-8);
}

.hero__actions {
  display: flex;
  gap: var(--sp-4);
  flex-wrap: wrap;
}

/* ---- Sections ----------------------------------------------- */
.section {
  padding: var(--sp-12) 0;
  border-bottom: var(--bw) solid var(--border);
}

.section:last-of-type {
  border-bottom: 0;
}

.section h2 {
  color: var(--accent);
  text-transform: lowercase;
}

/* ---- Grid --------------------------------------------------- */
.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--sp-6);
}

/* ---- Post list ---------------------------------------------- */
.post-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.post-list__item {
  display: flex;
  gap: var(--sp-6);
  padding: var(--sp-3) 0;
  border-bottom: var(--bw) solid var(--border);
}

.post-list__date {
  color: var(--fg-muted);
  font-size: var(--fs-sm);
  flex: 0 0 10ch;
}

/* ---- CV ----------------------------------------------------- */
.cv-entry {
  padding: var(--sp-6) 0;
  border-bottom: var(--bw) solid var(--border);
}

.cv-entry__head {
  display: flex;
  justify-content: space-between;
  gap: var(--sp-4);
  flex-wrap: wrap;
}

.cv-entry__title {
  font-family: var(--font-display);
  font-weight: 700;
}

.cv-entry__when {
  color: var(--fg-muted);
  font-size: var(--fs-sm);
}

@media (min-width: 640px) {
  .grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 640px) {
  .hero__name {
    font-size: var(--fs-2xl);
  }

  .post-list__item {
    flex-direction: column;
    gap: var(--sp-1);
  }
}
```

<!-- prettier-ignore -->
- [ ] **Step 6: Wire `_pages` into the stylesheet**

Replace the contents of `assets/css/site.scss`:

```scss
---
---

@use "tokens";
@use "base";
@use "ui";
@use "pages";
```

<!-- prettier-ignore -->
- [ ] **Step 7: Create `_pages/home.md` at a preview permalink**

The permalink becomes `/` in Task 6, once `about.md` is deleted. Both cannot claim `/` at once.

The body is the bio prose, ported from `_pages/about.md` and tightened — the first paragraph is dropped because the hero already states the role.

```markdown
---
layout: home
permalink: /preview-home/
title: home
---

My technical foundation is in **pure mathematics** — an M.Sc. from
[Chennai Mathematical Institute](https://www.cmi.ac.in/), where my thesis explored angle and
distance estimates and Kazhdan constants, and a Post Graduate Diploma in Business Analytics from
[IIT Kharagpur](https://www.iitkgp.ac.in/). That depth informs everything I build: from stochastic
process modelling and Bayesian inference to evaluation frameworks for production RAG systems.

Day to day I work on **LLMs and RAG** (advanced retrieval for knowledge-intensive domains,
fine-tuning LLaMA 70B and Qwen3-Embedding-8B with PEFT/LoRA, model merging), **speech AI** (VAD and
Mel-Spectrogram pipelines, TDNN language identification, Whisper fine-tuning for low-resource Indian
languages), and the **MLOps** that carries all of it to production.

I'm currently most interested in **world models** and **reinforcement learning** — particularly how
latent world representations enable better planning and decision-making agents. I'm also drawn to
game development and quantitative finance.

Outside research I've mentored Mathematics and Physics Olympiad students, led the editorial team at
the PGDBA annual AI magazine (AINA 4.0), and play football, basketball, padel, and guitar. 🎸
```

<!-- prettier-ignore -->
- [ ] **Step 8: Verify the timeline generated correctly from `cv.yml`**

This is the assertion that matters: the timeline must contain **all five** career entries, in descending date order, pulled from `cv.yml` and not hand-written.

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace

grep -c 'timeline__item' _site/preview-home/index.html
```

Expected: `5` (Inception 2024, BCG 2023, IIT KGP 2022, CMI MSc 2020, CMI BSc 2017).

```bash
grep -q 'Inception Institute of Artificial Intelligence' _site/preview-home/index.html && echo "OK: experience present"
grep -q 'Chennai Mathematical Institute' _site/preview-home/index.html && echo "OK: education present"
grep -q 'Applied Scientist | LLMs' _site/preview-home/index.html && echo "OK: headline from cv.yml"
grep -q 'what i.m building' _site/preview-home/index.html && echo "FAIL: empty projects block rendered" || echo "OK: empty projects block hidden"
grep -q 'writing' _site/preview-home/index.html && echo "FAIL: empty writing block rendered" || echo "OK: empty writing block hidden"
```

Expected: five `OK:` lines, no `FAIL`.

Confirm the sort is descending — the first date shown must be `2024`:

```bash
grep -o 'timeline__date">[0-9]\{4\}' _site/preview-home/index.html | head -1
```

Expected: `timeline__date">2024`

<!-- prettier-ignore -->
- [ ] **Step 9: Look at it (human gate)**

```bash
docker compose up --build
```

Visit `http://localhost:8080/preview-home/` in both themes and at 375px width. Then `Ctrl-C`, `docker compose down`.

<!-- prettier-ignore -->
- [ ] **Step 10: Commit**

```bash
npx prettier . --write
git add _layouts/home.liquid _includes/timeline.liquid _includes/project-card.liquid \
  _data/projects.yml _pages/home.md _sass/_pages.scss assets/css/site.scss
git commit -m "feat: add narrative homepage with cv.yml-generated timeline

Hero, story, timeline, building, writing, elsewhere. The timeline
concatenates cv.yml's education and experience and sorts by start_date,
so career history is never maintained in two places. The building and
writing blocks hide themselves while their data is empty."
```

---

## Task 5: CV, projects, and blog pages

Completes the four routes at preview permalinks. `/cv/` and `/blog/` and `/projects/` do not yet exist under their real paths — `_pages/cv.md` and `_pages/blog.md` still belong to al-folio until Task 6.

**Files:**

- Create: `_layouts/listing.liquid`, `_layouts/article.liquid`
- Create: `_includes/cv-section.liquid`
- Create: `_pages/cv-new.md`, `_pages/projects-new.md`, `_pages/blog-new.md`
- Create: `assets/js/blog-filter.js`
- Modify: `_data/socials.yml` (fix the stale CV PDF pointer)

**Interfaces:**

- Consumes: `base` layout, `.card` / `.tag` / `.tabs` / `.tab` / `.cv-entry*` / `.post-list*` classes
- Produces: layouts `listing` (content pages) and `article` (blog posts). `article` honours `page.math` (bool) to load MathJax.

<!-- prettier-ignore -->
- [ ] **Step 1: Fix the stale CV PDF pointer**

RenderCV writes to `assets/rendercv/rendercv_output/Writabrata_Bhattacharya_CV.pdf` on every push touching `cv.yml`, but `socials.yml` points at a hand-uploaded `assets/pdf/CV_Writabrata.pdf` that no longer reflects `cv.yml`.

In `_data/socials.yml`, change:

```yaml
cv_pdf: /assets/pdf/CV_Writabrata.pdf # path to your CV PDF file
```

to:

```yaml
# Generated by .github/workflows/render-cv.yml from _data/cv.yml on every push.
# Do not point this at a hand-uploaded PDF — it will silently go stale.
cv_pdf: /assets/rendercv/rendercv_output/Writabrata_Bhattacharya_CV.pdf
```

Then delete the stale copy:

```bash
git rm --quiet assets/pdf/CV_Writabrata.pdf
```

<!-- prettier-ignore -->
- [ ] **Step 2: Create `_layouts/listing.liquid`**

```liquid
---
layout: base
---
<h1>{{ page.title }}</h1>
{{ content }}
```

<!-- prettier-ignore -->
- [ ] **Step 3: Create `_layouts/article.liquid`**

```liquid
---
layout: base
---
<article>
  <h1>{{ page.title }}</h1>
  <p class="cv-entry__when">
    {{ page.date | date: '%Y-%m-%d' }}
    {% for cat in page.categories -%}
      <span class="tag">{{ cat }}</span>
    {%- endfor %}
  </p>
  {{ content }}
</article>

<p><a href="{{ '/blog/' | relative_url }}">← all posts</a></p>
```

<!-- prettier-ignore -->
- [ ] **Step 4: Create `_includes/cv-section.liquid`**

`cv.yml` has two shapes of section. `experience`/`education` are rich entries with `highlights`; `awards`/`skills`/`languages`/`interests` are flat `label`/`details` pairs. `publications` is a third shape. This include handles all three by discriminating on which keys are present.

Expects a variable `entries` (the array) and `kind` (string).

```liquid
{% for entry in entries %}
  <div class="cv-entry">
    <div class="cv-entry__head">
      {% if entry.company %}
        <span class="cv-entry__title">{{ entry.position }} · {{ entry.company }}</span>
        <span class="cv-entry__when">{{ entry.start_date }} — {{ entry.end_date }}</span>
      {% elsif entry.institution %}
        <span class="cv-entry__title">
          {{- entry.degree }}
          {{ entry.area }} · {{ entry.institution -}}
        </span>
        <span class="cv-entry__when">{{ entry.start_date }} — {{ entry.end_date }}</span>
      {% elsif entry.authors %}
        <span class="cv-entry__title">
          {% if entry.url -%}
            <a href="{{ entry.url }}">{{ entry.title }}</a>
          {%- else -%}
            {{- entry.title -}}
          {%- endif %}
        </span>
        <span class="cv-entry__when">{{ entry.date }}</span>
      {% else %}
        <span class="cv-entry__title">{{ entry.label }}</span>
        {% if entry.date -%}
          <span class="cv-entry__when">{{ entry.date }}</span>
        {%- endif %}
      {% endif %}
    </div>

    {% if entry.journal %}
      <p class="cv-entry__when">{{ entry.journal }}</p>
    {% endif %}
    {% if entry.details %}
      <p>{{ entry.details }}</p>
    {% endif %}

    {% if entry.highlights %}
      <ul>
        {% for h in entry.highlights %}
          <li>{{ h | markdownify | remove: '<p>' | remove: '</p>' }}</li>
        {% endfor %}
      </ul>
    {% endif %}
  </div>
{% endfor %}
```

<!-- prettier-ignore -->
- [ ] **Step 5: Create `_pages/cv-new.md`**

Permalink becomes `/cv/` in Task 6.

```markdown
---
layout: listing
permalink: /cv-new/
title: cv
---

<p>
  <a class="btn btn--accent" href="{{ site.data.socials.cv_pdf | relative_url }}">download pdf ↓</a>
</p>

{% assign s = site.data.cv.cv.sections %}

## experience

{% include cv-section.liquid entries=s.experience kind="experience" %}

## education

{% include cv-section.liquid entries=s.education kind="education" %}

## publications

{% include cv-section.liquid entries=s.publications kind="publications" %}

## awards

{% include cv-section.liquid entries=s.awards kind="awards" %}

## skills

{% include cv-section.liquid entries=s.skills kind="skills" %}

## languages

{% include cv-section.liquid entries=s.languages kind="languages" %}

## interests

{% include cv-section.liquid entries=s.interests kind="interests" %}
```

<!-- prettier-ignore -->
- [ ] **Step 6: Create `_pages/projects-new.md`**

Permalink becomes `/projects/` in Task 6.

```markdown
---
layout: listing
permalink: /projects-new/
title: projects
---

Things I build outside of work.

{% if site.data.projects.size > 0 %}

  <div class="grid">
    {% for project in site.data.projects %}
      {% include project-card.liquid project=project %}
    {% endfor %}
  </div>
{% else %}
  <p>Nothing here yet — add entries to <code>_data/projects.yml</code>.</p>
{% endif %}
```

<!-- prettier-ignore -->
- [ ] **Step 7: Create `_pages/blog-new.md`**

Permalink becomes `/blog/` in Task 6. Filtering is client-side over the already-rendered list, so it degrades to showing every post when JS is off.

```markdown
---
layout: listing
permalink: /blog-new/
title: blog
---

<div class="tabs" role="tablist">
  <button class="tab" role="tab" data-filter="all" aria-selected="true">all</button>
  <button class="tab" role="tab" data-filter="engineering" aria-selected="false">engineering</button>
  <button class="tab" role="tab" data-filter="reading" aria-selected="false">reading</button>
</div>

{% if site.posts.size > 0 %}

  <ul class="post-list" id="post-list">
    {% for post in site.posts %}
      <li class="post-list__item" data-categories="{{ post.categories | join: ' ' }}">
        <span class="post-list__date">{{ post.date | date: "%Y-%m-%d" }}</span>
        <span>
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {% for cat in post.categories %}<span class="tag">{{ cat }}</span>{% endfor %}
        </span>
      </li>
    {% endfor %}
  </ul>
{% else %}
  <p>No posts yet.</p>
{% endif %}

<script defer src="{{ '/assets/js/blog-filter.js' | relative_url }}"></script>
```

<!-- prettier-ignore -->
- [ ] **Step 8: Create `assets/js/blog-filter.js`**

```js
(function () {
  var tabs = document.querySelectorAll(".tab[data-filter]");
  var list = document.getElementById("post-list");
  if (!tabs.length || !list) return;

  var items = list.querySelectorAll(".post-list__item");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var filter = tab.getAttribute("data-filter");

      tabs.forEach(function (t) {
        t.setAttribute("aria-selected", String(t === tab));
      });

      items.forEach(function (item) {
        var cats = (item.getAttribute("data-categories") || "").split(" ");
        var show = filter === "all" || cats.indexOf(filter) !== -1;
        item.style.display = show ? "" : "none";
      });
    });
  });
})();
```

<!-- prettier-ignore -->
- [ ] **Step 9: Add a smoke-test post so the blog and filter are actually exercised**

An empty blog cannot demonstrate that filtering works. Create `_posts/2026-07-11-hello.md`:

```markdown
---
layout: article
title: Hello
date: 2026-07-11
categories: [engineering]
math: true
---

First post. Inline math renders: $e^{i\pi} + 1 = 0$.

\`\`\`python
print("code renders too")
\`\`\`
```

Replace the `\`\`\`` sequences above with three real backticks — they are escaped here only to keep this plan's code fence intact.

<!-- prettier-ignore -->
- [ ] **Step 10: Verify all four routes**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

CV must render every section from `cv.yml`, and point at the RenderCV PDF (not the deleted one):

```bash
grep -c 'cv-entry' _site/cv-new/index.html
```

Expected: `19` or more (2 experience + 3 education + 1 publication + 3 awards + 5 skills + 4 languages + 2 interests = 20 entries).

```bash
grep -q 'rendercv_output/Writabrata_Bhattacharya_CV.pdf' _site/cv-new/index.html && echo "OK: PDF points at RenderCV output"
grep -q 'assets/pdf/CV_Writabrata.pdf' _site/cv-new/index.html && echo "FAIL: still pointing at stale PDF" || echo "OK: stale PDF pointer gone"
grep -q 'Boston Consulting Group' _site/cv-new/index.html && echo "OK: experience rendered"
grep -q 'Kazhdan' _site/cv-new/index.html && echo "OK: education highlights rendered"
grep -q 'INSPIRE Scholarship' _site/cv-new/index.html && echo "OK: awards rendered"
grep -q 'Assessing The Capabilities' _site/cv-new/index.html && echo "OK: publications rendered"
grep -q 'Nothing here yet' _site/projects-new/index.html && echo "OK: projects empty state"
grep -q 'data-categories="engineering"' _site/blog-new/index.html && echo "OK: post is filterable"
grep -q 'MathJax' _site/blog/2026/hello/index.html && echo "OK: MathJax loaded on math post"
```

Expected: nine `OK:` lines, no `FAIL`.

<!-- prettier-ignore -->
- [ ] **Step 11: Look at it (human gate)**

```bash
docker compose up --build
```

Check `http://localhost:8080/cv-new/`, `/projects-new/`, `/blog-new/`, and the post. On the blog, click **engineering** and **reading** — the post should show under `all` and `engineering`, and vanish under `reading`. Check both themes and 375px width. Then `Ctrl-C`, `docker compose down`.

<!-- prettier-ignore -->
- [ ] **Step 12: Commit**

```bash
npx prettier . --write
git add _layouts/listing.liquid _layouts/article.liquid _includes/cv-section.liquid \
  _pages/cv-new.md _pages/projects-new.md _pages/blog-new.md \
  assets/js/blog-filter.js _data/socials.yml _posts/2026-07-11-hello.md
git commit -m "feat: add cv, projects, and blog pages

CV renders every section of cv.yml and links the RenderCV-generated PDF
instead of the stale hand-uploaded one. Blog filters by category client
-side, degrading to show-all without JS."
```

---

## Task 6: Cut over and demolish

The moment of truth. The new theme takes the real permalinks, and al-folio's presentation layer is deleted. **This task must not be split** — a half-swapped site does not build, and that is exactly why Tasks 2–5 built the replacement first.

**Files:**

- Delete: all 13 `_layouts/*.liquid` except `base`, `home`, `listing`, `article`
- Delete: all 13 `_sass/*.scss` except `_tokens`, `_base`, `_ui`, `_pages`; delete `_sass/font-awesome/`
- Delete: `assets/css/main.scss` and all vendored Bootstrap/MDB/academicons/jupyter/tikzjax CSS
- Delete: vendored Bootstrap/Masonry/Plotly/ECharts/Chart.js/Leaflet/Mermaid/diff2html/distill JS
- Delete: `_includes/` — every al-folio include
- Delete: `_pages/about.md`, `_pages/publications.md`, `_pages/preview.md`, `_bibliography/`
- Rename: `_pages/{home,cv-new,projects-new,blog-new}.md` → real permalinks
- Rewrite: `_config.yml` (671 lines → ~70)

**Interfaces:**

- Consumes: everything from Tasks 2–5
- Produces: the finished site at `/`, `/cv/`, `/projects/`, `/blog/`

<!-- prettier-ignore -->
- [ ] **Step 1: Take the real permalinks**

Delete the al-folio pages that own them, and the preview scaffold:

```bash
git rm --quiet _pages/about.md _pages/publications.md _pages/preview.md
git rm -r --quiet _bibliography
git mv _pages/cv.md _pages/cv-OLD.md && git rm --quiet _pages/cv-OLD.md
git mv _pages/blog.md _pages/blog-OLD.md && git rm --quiet _pages/blog-OLD.md
git mv _pages/cv-new.md _pages/cv.md
git mv _pages/projects-new.md _pages/projects.md
git mv _pages/blog-new.md _pages/blog.md
```

Now update the four permalinks in front matter:

- `_pages/home.md`: `permalink: /preview-home/` → `permalink: /`
- `_pages/cv.md`: `permalink: /cv-new/` → `permalink: /cv/`
- `_pages/projects.md`: `permalink: /projects-new/` → `permalink: /projects/`
- `_pages/blog.md`: `permalink: /blog-new/` → `permalink: /blog/`

<!-- prettier-ignore -->
- [ ] **Step 2: Delete al-folio's layouts, includes, and SCSS**

```bash
cd /Users/kaneki/Documents/Code/sudo-kaneki.github.io

# Layouts — keep only the four new ones
git rm --quiet _layouts/about.liquid _layouts/archive.liquid _layouts/bib.liquid \
  _layouts/book-review.liquid _layouts/book-shelf.liquid _layouts/course.liquid \
  _layouts/cv.liquid _layouts/default.liquid _layouts/distill.liquid \
  _layouts/none.liquid _layouts/page.liquid _layouts/post.liquid _layouts/profiles.liquid

# Includes — delete every al-folio include (the new ones are site-*, timeline,
# project-card, cv-section)
git rm -r --quiet _includes/cv _includes/repository
git rm --quiet _includes/audio.liquid _includes/bib_search.liquid _includes/calendar.liquid \
  _includes/citation.liquid _includes/course_schedule.liquid _includes/courses.liquid \
  _includes/disqus.liquid _includes/distill_scripts.liquid _includes/figure.liquid \
  _includes/footer.liquid _includes/giscus.liquid _includes/head.liquid \
  _includes/header.liquid _includes/latest_posts.liquid _includes/metadata.liquid \
  _includes/news.liquid _includes/newsletter.liquid _includes/pagination.liquid \
  _includes/projects.liquid _includes/projects_horizontal.liquid \
  _includes/related_posts.liquid _includes/scripts.liquid \
  _includes/selected_papers.liquid _includes/video.liquid

# SCSS — keep only _tokens, _base, _ui, _pages
git rm -r --quiet _sass/font-awesome
git rm --quiet _sass/_blog.scss _sass/_components.scss _sass/_cv.scss _sass/_distill.scss \
  _sass/_footer.scss _sass/_layout.scss _sass/_navbar.scss _sass/_publications.scss \
  _sass/_tabs.scss _sass/_teachings.scss _sass/_themes.scss _sass/_typograms.scss \
  _sass/_typography.scss _sass/_utilities.scss _sass/_variables.scss

# Stylesheet entry point
git rm --quiet assets/css/main.scss
```

<!-- prettier-ignore -->
- [ ] **Step 3: Delete vendored Bootstrap, MDB, and the unused JS libraries**

```bash
git rm --quiet assets/css/bootstrap.min.css assets/css/bootstrap.min.css.map \
  assets/css/mdb.min.css assets/css/mdb.min.css.map assets/css/bootstrap-toc.min.css \
  assets/css/academicons.min.css assets/css/scholar-icons.css assets/css/tikzjax.min.css \
  assets/css/jupyter.css assets/css/jupyter-grade3.css assets/css/jupyter-monokai.css

git rm -r --quiet assets/js/distillpub
git rm --quiet assets/js/bootstrap.bundle.min.js assets/js/bootstrap.bundle.min.js.map \
  assets/js/bootstrap-toc.min.js assets/js/bibsearch.js assets/js/masonry.js \
  assets/js/plotly-setup.js assets/js/echarts-setup.js assets/js/chartjs-setup.js \
  assets/js/leaflet-setup.js assets/js/mermaid-setup.js assets/js/diff2html-setup.js \
  assets/js/calendar-setup.js assets/js/newsletter.js assets/js/jupyter_new_tab.js \
  assets/js/theme.js assets/js/common.js assets/js/no_defer.js assets/js/mathjax-setup.js
```

Anything left in `assets/js/` that is not `theme-toggle.js`, `blog-filter.js`, or `copy_code.js` is also dead — list it and remove it:

```bash
ls assets/js/
```

Keep `copy_code.js`; `git rm` whatever else remains (`search/`, `search-*.js`, `shortcut-key.js`, `progress-bar.js`, `zoom.js`, `tooltips-setup.js`, `vanilla-back-to-top.min.js`, `typograms.js`, `tabs.min.js`, `tikzjax.min.js`, `venobox-setup.js`, `photoswipe-setup.js`, `pseudocode-setup.js`, `vega-setup.js`, `*-analytics-setup.js`, `cookie-consent-setup.js`).

<!-- prettier-ignore -->
- [ ] **Step 4: Rewrite `_config.yml`**

The existing file is 671 lines, ~250 of which are the `third_party_libraries` block that nothing will reference any more. Replace the **entire file** with this:

```yaml
# -----------------------------------------------------------------------------
# Site
# -----------------------------------------------------------------------------
title: blank
first_name: Writabrata
middle_name:
last_name: Bhattacharya
description: >
  Applied Scientist | LLMs, Speech Deep Learning, MLOps & World Models
keywords: machine-learning, deep-learning, LLM, speech, TTS, STT, MLOps, world-models, reinforcement-learning
lang: en
icon: ⚛️
url: https://sudo-kaneki.github.io
baseurl:

# -----------------------------------------------------------------------------
# Blog
# -----------------------------------------------------------------------------
blog_name: Writabrata's Blog
blog_description: Thoughts on ML, AI, Mathematics & beyond
permalink: /blog/:year/:title/

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
markdown: kramdown
highlighter: rouge
kramdown:
  input: GFM
  syntax_highlighter_opts:
    css_class: "highlight"
    block:
      line_numbers: false

plugins:
  - jekyll-email-protect
  - jekyll-feed
  - jekyll-link-attributes
  - jekyll-minifier
  - jekyll-paginate-v2
  - jekyll-sitemap
  - jekyll-terser
  - jemoji

include: ["_pages"]

exclude:
  - bin/
  - docs/
  - CONTRIBUTING.md
  - CUSTOMIZE.md
  - Dockerfile
  - docker-compose.yml
  - docker-compose-slim.yml
  - FAQ.md
  - Gemfile
  - Gemfile.lock
  - INSTALL.md
  - LICENSE
  - lighthouse_results/
  - package.json
  - package-lock.json
  - QUICKSTART.md
  - README.md
  - requirements.txt
  - SEO.md
  - TROUBLESHOOTING.md

defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: "article"

# -----------------------------------------------------------------------------
# Minification (production only)
# -----------------------------------------------------------------------------
jekyll-minifier:
  exclude: ["robots.txt"]
  compress_javascript: true
  compress_css: true
```

Note: `pagination:` is deliberately **absent**. `jekyll-paginate-v2` is installed for later, but enabling it with one post gains nothing and its `enabled: true` flag errors on an empty `_posts/`.

<!-- prettier-ignore -->
- [ ] **Step 5: Verify the cutover**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

Expected: exit 0. A failure here naming an undefined filter (`bust_file_cache`) or variable means a surviving template still references the old machinery — grep for it and remove.

Assert the four real routes exist and Bootstrap is gone from the output:

```bash
test -f _site/index.html && echo "OK: /"
test -f _site/cv/index.html && echo "OK: /cv/"
test -f _site/projects/index.html && echo "OK: /projects/"
test -f _site/blog/index.html && echo "OK: /blog/"
grep -rq "bootstrap" _site/ && echo "FAIL: bootstrap still in output" || echo "OK: no bootstrap anywhere"
grep -q 'timeline__item' _site/index.html && echo "OK: homepage timeline at /"
test ! -f _site/preview/index.html && echo "OK: preview scaffold gone"
test ! -f _site/preview-home/index.html && echo "OK: preview-home gone"
```

Expected: eight `OK:` lines, no `FAIL`.

Confirm the CSS actually shrank:

```bash
du -sh _site/assets/css/
```

Expected: well under 100KB (was ~500KB with Bootstrap + MDB).

<!-- prettier-ignore -->
- [ ] **Step 6: Look at it (human gate)**

```bash
docker compose up --build
```

Walk all four routes at `http://localhost:8080`, in both themes, at 375px and desktop width. Open the browser console — **there must be zero errors**. This is the first time the site is the real site.

Then `Ctrl-C`, `docker compose down`.

<!-- prettier-ignore -->
- [ ] **Step 7: Commit**

```bash
npx prettier . --write
git add -u
git add _config.yml _pages/
git commit -m "feat!: cut over to the retro theme, delete al-folio presentation layer

The new theme takes /, /cv/, /projects/, /blog/. Deletes 13 al-folio
layouts, 13 SCSS partials, every al-folio include, vendored Bootstrap +
MDB, and ~20 unused JS libraries. _config.yml drops from 671 lines to
~70, losing the 250-line third_party_libraries block entirely.

BREAKING: /publications/, /projects/<slug>, /news/, /teaching/,
/repositories/, /people/ no longer exist. Publications now live as a
section of /cv/, sourced from cv.yml."
```

---

## Task 7: Cull the dependencies and simplify CI

Nothing in the templates references the dead plugins any more, so they can now be removed safely. Doing this **before** Task 6 would have broken the al-folio templates that were still live.

**Files:**

- Modify: `Gemfile`, `Gemfile.lock`, `requirements.txt`
- Modify: `.github/workflows/deploy.yml`
- Delete: `purgecss.config.js`
- Delete: `.github/workflows/{update-citations,update-tocs,deploy-image,deploy-docker-tag,docker-slim,copilot-setup-steps}.yml`

**Interfaces:**

- Consumes: a site that no longer references any culled plugin
- Produces: the final dependency set and deploy pipeline

<!-- prettier-ignore -->
- [ ] **Step 1: Cull the Gemfile**

Replace the `:jekyll_plugins` and `:other_plugins` groups. The full new `Gemfile`:

```ruby
source 'https://rubygems.org'

gem 'jekyll'

group :jekyll_plugins do
    gem 'jekyll-email-protect'
    gem 'jekyll-feed'
    gem 'jekyll-link-attributes'
    gem 'jekyll-minifier'
    gem 'jekyll-paginate-v2'
    gem 'jekyll-sitemap'
    gem 'jekyll-terser', :git => "https://github.com/RobertoJBeltran/jekyll-terser.git"
    gem 'jemoji'
end
```

Dropped: `jekyll-3rd-party-libraries`, `jekyll-archives-v2`, `jekyll-cache-bust`, `jekyll-get-json`, `jekyll-imagemagick`, `jekyll-jupyter-notebook`, `jekyll-regex-replace`, `jekyll-scholar`, `jekyll-tabs`, `jekyll-toc`, `jekyll-twitter-plugin`, `classifier-reborn`, and the entire `:other_plugins` group (`css_parser`, `feedjira`, `httparty`, `observer`, `ostruct`).

<!-- prettier-ignore -->
- [ ] **Step 2: Regenerate the lockfile inside the container**

The lockfile must be generated with the container's Ruby, not the host's.

```bash
docker compose run --rm jekyll bundle install
```

Expected: writes a new `Gemfile.lock`. Confirm the culled gems are gone:

```bash
grep -q "jekyll-scholar" Gemfile.lock && echo "FAIL: scholar still locked" || echo "OK: scholar gone"
grep -q "jekyll-cache-bust" Gemfile.lock && echo "FAIL: cache-bust still locked" || echo "OK: cache-bust gone"
```

Expected: two `OK:` lines.

<!-- prettier-ignore -->
- [ ] **Step 3: Verify the build survives the cull**

This is the step that catches any template still secretly depending on a dead plugin.

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
```

Expected: exit 0. A `Liquid Exception: Unknown filter` or `Dependency Error` here names the exact plugin a template still needs — find the reference with `grep -rn "<filter-name>" _layouts _includes _pages` and remove it.

<!-- prettier-ignore -->
- [ ] **Step 4: Trim `requirements.txt`**

Only RenderCV is still needed (`nbconvert` served `jekyll-jupyter-notebook`; `scholarly` served the citations workflow; `pyyaml` was a transitive helper).

```
rendercv[full]
```

<!-- prettier-ignore -->
- [ ] **Step 5: Simplify `deploy.yml`**

The **`Deploy 🚀` step does not change** — this is still push-to-`main` → build → publish `_site/` to `gh-pages`. Three build steps are removed because the things they served are gone.

In `.github/workflows/deploy.yml`, delete:

- the entire `Setup Python 🐍` step (served `nbconvert`, for `jekyll-jupyter-notebook`)
- the entire `Update _config.yml ⚙️` step (set `giscus.repo`; giscus is gone)
- the entire `Purge unused CSS 🧹` step

and reduce the `Install and Build 🔧` step to:

```yaml
- name: Install and Build 🔧
  run: |
    export JEKYLL_ENV=production
    bundle exec jekyll build
```

(dropping `apt-get install imagemagick`, which served `jekyll-imagemagick`, and `pip3 install nbconvert`).

Removing purgecss is not merely cleanup. It strips CSS classes it cannot find referenced in the HTML — and `theme-toggle.js` and `blog-filter.js` apply classes that exist **only in JS**. Left in place, it would produce a site that works locally and breaks in production.

<!-- prettier-ignore -->
- [ ] **Step 6: Delete purgecss and the dead workflows**

```bash
git rm --quiet purgecss.config.js
git rm --quiet .github/workflows/update-citations.yml \
  .github/workflows/update-tocs.yml \
  .github/workflows/deploy-image.yml \
  .github/workflows/deploy-docker-tag.yml \
  .github/workflows/docker-slim.yml \
  .github/workflows/copilot-setup-steps.yml
```

`update-citations.yml` depended on `papers.bib` (deleted). `update-tocs.yml` depended on `jekyll-toc` (culled). The three Docker workflows published container images for the al-folio template itself, which is not this site's job.

**Kept:** `deploy.yml`, `render-cv.yml` (the `cv.yml` → PDF pipeline — now load-bearing), `prettier.yml`, `axe.yml`, `broken-links*.yml`, `codeql.yml`.

<!-- prettier-ignore -->
- [ ] **Step 7: Final full verification**

```bash
docker compose up --build
```

All four routes, both themes, 375px and desktop, zero console errors. Confirm the PDF download button on `/cv/` actually resolves (click it).

Then `Ctrl-C`, `docker compose down`.

<!-- prettier-ignore -->
- [ ] **Step 8: Commit**

```bash
npx prettier . --write
git add Gemfile Gemfile.lock requirements.txt .github/workflows/deploy.yml
git add -u
git commit -m "chore: cull 11 jekyll plugins, drop purgecss, simplify deploy

Gemfile drops jekyll-scholar, cache-bust, 3rd-party-libraries, toc,
tabs, archives, imagemagick, jupyter-notebook, twitter, get-json, and
regex-replace. deploy.yml loses the Python setup, the imagemagick
install, the giscus config step, and the purgecss step — the last of
which would have stripped the theme-toggle and blog-filter classes,
since they only ever appear in JS.

The deploy step itself is unchanged."
```

---

## Task 8: Documentation

The repo's `AGENTS.md`, `CLAUDE.md`, and `.github/` instruction files still describe al-folio. Leaving them is worse than deleting them — they will actively mislead the next agent (and you, in six months).

**Files:**

- Modify: `AGENTS.md`
- Delete: `CUSTOMIZE.md`, `FAQ.md`, `INSTALL.md`, `QUICKSTART.md`, `TROUBLESHOOTING.md`, `SEO.md`, `ANALYTICS.md`, `CONTRIBUTING.md`, `.github/instructions/`, `.github/agents/`

<!-- prettier-ignore -->
- [ ] **Step 1: Delete the al-folio documentation**

```bash
git rm --quiet CUSTOMIZE.md FAQ.md INSTALL.md QUICKSTART.md TROUBLESHOOTING.md \
  SEO.md ANALYTICS.md CONTRIBUTING.md
git rm -r --quiet .github/instructions .github/agents
```

<!-- prettier-ignore -->
- [ ] **Step 2: Rewrite `AGENTS.md`**

`CLAUDE.md` just does `@AGENTS.md`, so it needs no change.

````markdown
# Agent Guidelines

Personal site of Writabrata Bhattacharya. Jekyll, custom retro-modern theme.
**Not al-folio any more** — do not consult al-folio docs; they describe a theme
this repo no longer has.

## Architecture

Four routes: `/` (`_pages/home.md`), `/cv/`, `/projects/`, `/blog/`.

Four layouts: `base` (shell) → `home`, `listing` (content pages), `article` (posts).

Four SCSS partials, compiled by `assets/css/site.scss`:

| File                 | Holds                                                              |
| -------------------- | ------------------------------------------------------------------ |
| `_sass/_tokens.scss` | **Every** colour, size, and font. Nothing else may hardcode these. |
| `_sass/_base.scss`   | Reset and element defaults                                         |
| `_sass/_ui.scss`     | Components: `.btn`, `.card`, `.tag`, `.tabs`, `.timeline`          |
| `_sass/_pages.scss`  | Page-specific layout                                               |

## The one rule

**`_sass/_tokens.scss` is the only file allowed to contain a literal colour,
size, or font name.** Everything else uses `var(--token)`. Design changes —
including swapping in a pixel font via `--font-display` — must stay a
single-file edit.

Retro cues, non-negotiable: `--radius: 0` (never a rounded corner), visible
1px borders, hard offset shadows (`4px 4px 0`, never blurred), and hover states
that translate by 2px so buttons depress like keys. All motion is disabled under
`prefers-reduced-motion`.

Dark is the default theme. Light must be checked too — both are first-class.

## Data

`_data/cv.yml` is the **single source of truth** for three outputs: the `/cv/`
page, the homepage timeline (`_includes/timeline.liquid` concatenates its
`education` and `experience` and sorts by `start_date`), and the PDF that
`.github/workflows/render-cv.yml` generates via RenderCV on every push.

**Never hand-edit the CV PDF or duplicate career history anywhere.** Edit
`cv.yml`; everything else follows.

`_data/projects.yml` feeds `/projects/` and the homepage `featured` block.

## Local development

```bash
docker compose up --build   # http://localhost:8080
docker compose down
```

Before every commit: `npx prettier . --write` (CI enforces it).
````

<!-- prettier-ignore -->
- [ ] **Step 3: Verify the build is unaffected and commit**

```bash
docker compose run --rm jekyll bundle exec jekyll build --trace
npx prettier . --write
git add AGENTS.md
git add -u
git commit -m "docs: replace al-folio documentation with the real thing

Deletes eight al-folio docs and the .github instruction files, which now
describe a theme this repo no longer has. AGENTS.md documents the four
routes, the four-partial SCSS structure, and the token rule."
```

---

## Self-review

**Spec coverage.** Every section of the spec maps to a task:

| Spec section                     | Task    |
| -------------------------------- | ------- |
| §5 IA — four routes              | 4, 5, 6 |
| §5.1 Deletion inventory          | 1, 6, 7 |
| §6.1 Four retro rules            | 2, 3    |
| §6.2 Tokens (colour/type/layout) | 2       |
| §6.3 File plan                   | 2, 3    |
| §6.4 Theme toggle, no-FOUC       | 3       |
| §7.1 Homepage, six blocks        | 4       |
| §7.2 CV rendered from `cv.yml`   | 5       |
| §7.3 Projects from `_data`       | 4, 5    |
| §7.4 Blog filter tabs            | 5       |
| §8 Deploy simplification         | 7       |
| §8.2 Stale-PDF bug               | 5       |
| §9 Staged, verified commits      | all     |

**Gap found and closed:** the spec's §9 ordering (content → layouts → SCSS → plugins → workflows) would leave the build red for several commits, because deleting al-folio's layouts before writing replacements breaks every page. The plan inverts this: **build alongside, preview, cut over, then demolish.** The plugin cull correctly lands _after_ the cutover (Task 7), since al-folio's live templates depend on `bust_file_cache` and `site.third_party_libraries` right up until the moment they are deleted.

**Gap found and closed:** the spec did not say what the homepage does when `projects.yml` is empty. It now hides that block, matching the "Writing" block's behaviour.

**Gap found and closed:** the spec did not mention `AGENTS.md`/`CLAUDE.md`, which document al-folio and would misdirect every future agent. Added as Task 8.

**Type consistency.** `project` (the include variable) and the `_data/projects.yml` keys (`title`, `blurb`, `tags`, `featured`, `links[].label`, `links[].url`) match across Task 4 Step 2, Task 4 Step 3, and Task 5 Step 6. The `.timeline*` / `.cv-entry*` / `.post-list*` / `.tabs` / `.tab` classes defined in Task 3 Step 6 and Task 4 Step 5 are exactly the ones consumed in Tasks 4 and 5. The `data-theme` values (`"dark"` / `"light"`) are consistent across `_tokens.scss`, `site-head.liquid`, and `theme-toggle.js`.

**Known deferral (intentional, per spec §3):** the pixel font. `--font-display` and `--font-body` ship as monospace placeholders. Swapping them is the next task and touches exactly one file plus one `<link>`.
