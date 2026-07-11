# Retro-Modern Site Rebuild — Design

**Date:** 2026-07-11
**Status:** Approved (pending spec review)
**Repo:** `sudo-kaneki.github.io`

## 1. Context

The site is a near-stock [al-folio](https://github.com/alshedivat/al-folio) Jekyll academic theme. Its structure and visual language belong to the template, not to the owner.

An audit of the content found that **almost all of it is demo boilerplate**. The site's only real content is four assets:

| Asset                      | Status                                                     |
| -------------------------- | ---------------------------------------------------------- |
| `_pages/about.md`          | Real — hand-written bio prose                              |
| `_data/cv.yml`             | Real — full CV in RenderCV v2.8 schema                     |
| `_data/socials.yml`        | Real — email, GitHub, LinkedIn                             |
| `_bibliography/papers.bib` | Real — one paper (ADIPEC/SPE 2024), duplicated in `cv.yml` |

Everything else — 21 blog posts, 9 projects, the bookshelf, 2 teaching entries, 4 news announcements — ships with the template.

This means a rebuild is not a migration. It is building a site around four assets.

## 2. Goals

Rebuild the site as a **narrative personal site with a retro-modern visual identity**, structurally inspired by [raahul42.github.io](https://raahul42.github.io) and aesthetically pointed at the retro-computer feel of [guedia.me](https://guedia.me).

1. Replace the academic-template information architecture with a narrative one.
2. Replace Bootstrap and al-folio's presentation layer with a small, owned design system.
3. Preserve the data pipelines that work (`cv.yml` → PDF, deploy).
4. Leave the codebase ready for a pixel/bitmap display font to be dropped in as a **subsequent, separate step**.

## 3. Non-goals

- **Font selection is explicitly deferred.** This rebuild ships with placeholder display and body faces wired through two CSS custom properties. Choosing and loading the pixel font is a follow-up task.
- No CRT costume: no scanline overlay, no boot sequence, no prompt-styled navigation. (Considered and rejected — see §4.)
- No content writing beyond porting the existing bio. The blog starts empty by design.

## 4. Decisions

Each decision below was chosen against at least one considered alternative.

| #   | Decision                                                                                                                     | Rejected alternative                           | Why                                                                                                                                                                                                              |
| --- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | **Full rebuild** of the presentation layer                                                                                   | Restyle within al-folio; reskin over Bootstrap | Overriding Bootstrap means permanent `!important` wars and ~200KB of unused CSS. The template-isms leak at the edges regardless.                                                                                 |
| D2  | **Keep Jekyll, the `cv.yml` pipeline, and the deploy.** Delete Bootstrap, all al-folio SCSS, all al-folio layouts.           | Fresh bare Jekyll site                         | A bare site means rebuilding dark mode, SEO, RSS, syntax highlighting, and the deploy by hand — to arrive at the same place.                                                                                     |
| D3  | **Retro-modern hybrid** aesthetic                                                                                            | Full terminal/CRT; terminal home + clean rest  | A CRT costume fights readability on exactly the pages that must stay readable (CV, publications). Retro _hardware cues_ — hard edges, offset shadows, key-press hovers — carry the personality without the cost. |
| D4  | **Dark default, light mode fully supported**                                                                                 | Dark-only                                      | Light mode is a real, tested theme, not an afterthought.                                                                                                                                                         |
| D5  | **CV rendered from `cv.yml` to HTML**, with a download button                                                                | Embed/preview the PDF                          | See §4.1.                                                                                                                                                                                                        |
| D6  | **Publications merged into `/cv/`** as a section; **`cv.yml` is the only source**. Delete `papers.bib` and `jekyll-scholar`. | Keep `papers.bib` + `jekyll-scholar`           | See §4.2.                                                                                                                                                                                                        |
| D7  | **Projects as `_data/projects.yml`**, not a Jekyll collection                                                                | `_projects/` collection                        | A collection demands a page per project. Cards + "deep writeups become blog posts" is one content system fewer and degrades gracefully.                                                                          |
| D8  | **One `/blog/` with client-side category filter tabs**                                                                       | Two routes (`/blog/` + `/reading/`)            | Hard-splitting routes creates a filing dilemma for hybrid posts and adds a fifth nav item. Tabs give separation without forcing a taxonomy decision on every post.                                               |

### 4.1 Why the CV is rendered, not embedded

`.github/workflows/render-cv.yml` already runs `rendercv render _data/cv.yml` on every push touching `cv.yml`, and commits the resulting PDF back to the repo. `cv.yml` is therefore **already** the source of truth for the PDF.

Rendering the same file to HTML gives **one source, two outputs** — the page and the PDF cannot drift apart. Embedding the PDF instead would invert this: the page becomes an artifact you cannot style, while you still edit `cv.yml` to change it.

An embedded PDF also cannot take dark mode, cannot take the pixel font, cannot take the borders, is invisible to search engines, and does not scroll inside an iframe on iOS Safari.

Marginal cost is low: `cv.yml` is already parsed to build the homepage timeline.

### 4.2 Why `jekyll-scholar` goes

The ADIPEC paper currently exists in **both** `papers.bib` and `cv.yml` — the duplication is already there. Collapsing onto `cv.yml` removes it, and removes with it: `jekyll-scholar`, `_layouts/bib.liquid`, `_includes/citation.liquid`, `assets/js/bibsearch.js`, `_sass/_publications.scss`, and `assets/css/scholar-icons.css`.

Cost: no BibTeX-copy button, no abstract toggle. For one publication this is overkill; both can be re-added if the publication list grows to justify them.

## 5. Information architecture

Four routes, down from eleven.

| Route        | Source of truth                                          | Notes                                                                                           |
| ------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `/`          | `_pages/home.md` + `_data/cv.yml` + `_data/projects.yml` | Narrative single-scroll. Timeline generated from `cv.yml`.                                      |
| `/cv/`       | `_data/cv.yml`                                           | Experience, education, publications, awards, skills, languages, interests. Download-PDF button. |
| `/projects/` | `_data/projects.yml`                                     | Side projects outside work. Card grid.                                                          |
| `/blog/`     | `_posts/`                                                | Starts empty. Category filter tabs. MathJax + Rouge on posts.                                   |

Nav: `home · cv · projects · blog` plus a theme toggle.

### 5.1 Deletion inventory

**Content:** `_projects/` (9), `_books/` (1), `_teachings/` (2), `_news/` (4), all 21 demo posts in `_posts/`, `_bibliography/papers.bib`.

**Pages:** `_pages/{projects,books,teaching,repositories,profiles,news,dropdown,about_einstein}.md`. `_pages/about.md` is replaced by `_pages/home.md`.

**Layouts:** all 13 in `_layouts/`, replaced by 4 (§6.3).

**SCSS:** all 13 partials in `_sass/`, replaced by 4 (§6.3).

**Vendored CSS/JS:** `bootstrap.min.css`(+map), `mdb.min.css`(+map), `bootstrap-toc.min.css`, `academicons.min.css`, `scholar-icons.css`, `tikzjax.min.css`, `jupyter*.css`; `bootstrap.bundle.min.js`(+map), `bootstrap-toc.min.js`, `bibsearch.js`, `masonry.js`, `plotly-setup.js`, `echarts-setup.js`, `chartjs-setup.js`, `leaflet-setup.js`, `mermaid-setup.js`, `diff2html-setup.js`, `calendar-setup.js`, `newsletter.js`, `distillpub/`.

**Retained assets:** `mathjax-setup.js`, `copy_code.js`, and the Rouge/Pygments syntax-highlighting CSS.

**Plugins: 19 → 8.**

- **Keep (8):** `jekyll-feed`, `jekyll-sitemap`, `jekyll-email-protect`, `jekyll-minifier`, `jekyll-terser`, `jekyll-paginate-v2`, `jekyll-link-attributes`, `jemoji`.
- **Drop (11):** `jekyll-3rd-party-libraries`, `jekyll-archives-v2`, `jekyll-cache-bust`, `jekyll-get-json`, `jekyll-imagemagick`, `jekyll-jupyter-notebook`, `jekyll-regex-replace`, `jekyll-scholar`, `jekyll-tabs`, `jekyll-toc`, `jekyll-twitter-plugin`.
- Also drop the non-plugin gem `classifier-reborn` (used only for demo-content categorization).

`jekyll-archives-v2` is safe to drop because blog categories are filtered client-side (§7.4) rather than served as generated archive pages. `jekyll-paginate-v2` is retained for when the blog has enough posts to need it.

**Also dropped:** ninja-keys ⌘K search (a four-page site does not need one), giscus comments (never configured — `repo_id` is empty), Disqus, the newsletter form.

## 6. Design system

### 6.1 Principles

Every design decision lives as a CSS custom property in `_sass/_tokens.scss`. **Nothing else in the codebase hardcodes a colour or a size.** This is what makes the later font swap a two-line change rather than an archaeology expedition.

The retro read is produced by four rules, not by ornament:

1. `--radius: 0`. No rounded corners anywhere.
2. Borders are `1px solid` and **visible** — they are structure, not decoration.
3. Shadows are **hard offsets** (`4px 4px 0 var(--border)`), never blurred. This is the strongest "old computer" signal and it costs nothing.
4. Interactive elements **translate on hover** (`translate(2px, 2px)`, shadow shrinking to `2px 2px 0`), so buttons physically depress like keys.

### 6.2 Tokens

- **Colour.** Dark is the default and the primary design target. The existing green accent carries forward as a phosphor green, **desaturated enough to pass WCAG AA on body text** — true `#33FF33` CRT green is unreadable at 16px. Full-saturation green is reserved for accents, links, and focus rings.
- **Type.** `--font-display` (headings) and `--font-body` (prose). **These two variables are where the pixel font lands in the follow-up step.** Everything is built anticipating that swap: generous `line-height`, no tight tracking, and a type scale that still holds when the display face becomes a chunky bitmap font.
- **Layout.** 720px content column (down from 930px — better measure for prose). The hero may break wider.
- **Motion.** All hover transforms and transitions respect `prefers-reduced-motion`.

### 6.3 File plan

New `_sass/` (4 files, replacing 13):

- `_tokens.scss` — colours (light + dark), spacing scale, type scale, borders, shadows, motion
- `_base.scss` — reset, typography, links, headings
- `_components.scss` — button, card, tab bar, timeline, tag, nav, footer
- `_pages.scss` — home hero, cv, projects, blog
- (`assets/css/main.scss` imports these plus the Rouge theme)

New `_layouts/` (4, replacing 13): `default`, `home`, `page`, `post`.

New `_includes/`: `head`, `nav`, `footer`, `timeline`, `project-card`, `post-list`.

Target: **~600 lines of owned CSS replacing ~4000 lines of theme plus 200KB of Bootstrap.**

### 6.4 Theme toggle

A blocking inline script in `<head>` reads `localStorage`, falling back to `prefers-color-scheme`, and stamps `data-theme` on the root element **before first paint** to avoid a flash of the wrong theme. A toggle button in the nav writes `localStorage` and flips the attribute.

## 7. Page specifications

### 7.1 Home

Six blocks. Only one is hand-written; the rest read from existing data.

1. **Hero** — name in the display face at the site's largest size; the headline from `cv.yml` (`Applied Scientist | LLMs · Speech AI · MLOps · World Models`); two key-style buttons, `[ contact ]` and `[ cv ↓ ]`.
2. **Story** — the prose from the current `about.md`, tightened. The only hand-written block.
3. **Timeline** — generated by merging `cv.yml`'s `education` and `experience` sections and sorting by date descending. Renders as a hard-bordered vertical rail. **Never maintained separately from the CV.**
4. **What I'm building** — entries in `_data/projects.yml` marked `featured: true` (2–3).
5. **Writing** — the three most recent posts. **Hidden entirely when `_posts/` is empty**, so the page never ships an empty-state apology.
6. **Elsewhere** — socials from `_data/socials.yml`, plus email.

### 7.2 CV

Renders `cv.yml` section by section: experience, education, publications, awards, skills, languages, interests. `[ download pdf ↓ ]` pinned at the top.

### 7.3 Projects

Card grid from `_data/projects.yml`. Each entry: `title`, `blurb`, `tags` (tech), `links`, `featured` (bool).

If a project earns a deep writeup, that writeup is a **blog post** and the card links to it.

### 7.4 Blog

Post list under a retro tab bar: `ALL / ENGINEERING / READING`, driven by post `categories`.

Filtering is **client-side over the already-rendered list** — no page reloads, and it degrades to showing all posts when JS is off. Post pages retain MathJax and Rouge syntax highlighting.

## 8. Build and deploy

**Deployment is unchanged.** `deploy.yml` pushes to `main` → GitHub Actions runs `bundle exec jekyll build` → `JamesIves/github-pages-deploy-action` publishes `_site/` to `gh-pages`, which GitHub Pages serves. The deploy step itself is not modified.

The workflow gets **shorter**, because three build steps exist only to serve deleted things:

| Step                                         | Fate                | Reason                                                                                                                                                                                                              |
| -------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apt-get install imagemagick`                | Remove              | Only needed by `jekyll-imagemagick`.                                                                                                                                                                                |
| `Setup Python 🐍` + `pip3 install nbconvert` | Remove              | Only needed by `jekyll-jupyter-notebook`.                                                                                                                                                                           |
| `Update _config.yml` (giscus)                | Remove              | Giscus is dropped; it was never configured.                                                                                                                                                                         |
| `Purge unused CSS 🧹`                        | Remove              | Pointless against ~600 lines of owned CSS — **and actively dangerous**: purgecss strips classes it cannot see in the HTML, which would silently eat the theme-toggle and blog-filter classes that exist only in JS. |
| `Deploy 🚀`                                  | **Keep, unchanged** |                                                                                                                                                                                                                     |

`purgecss.config.js` is deleted.

### 8.1 Other workflows

**Keep:** `render-cv.yml` (the `cv.yml` → PDF pipeline — now more load-bearing than ever), `prettier.yml`, `axe.yml` (accessibility gate), `broken-links*.yml`.

**Drop:** `update-citations.yml` (depended on `papers.bib`), `update-tocs.yml` (depended on `jekyll-toc`), `deploy-image.yml`, `deploy-docker-tag.yml`, `docker-slim.yml` (Docker-image publishing for the al-folio template itself), `copilot-setup-steps.yml`.

`requirements.txt` reduces to `rendercv[full]` (dropping `nbconvert`, `scholarly`, `pyyaml`).

### 8.2 Bug fixed en route

RenderCV writes the CV to `assets/rendercv/rendercv_output/Writabrata_Bhattacharya_CV.pdf`, but `_data/socials.yml` sets `cv_pdf: /assets/pdf/CV_Writabrata.pdf` — a **stale, hand-uploaded copy**. The download button currently serves a PDF that does not reflect `cv.yml`.

Fix: point `cv_pdf` at the RenderCV output. Delete `assets/pdf/CV_Writabrata.pdf`, `assets/pdf/example_pdf.pdf`, `assets/rendercv/rendercv_output/Albert_Einstein_CV.pdf`, and the stray `CV_Writabrata (1).pdf` in the repo root.

## 9. Risks and execution

**The risk is not the CSS. It is the demolition.**

Jekyll fails loudly but unhelpfully when a layout references a plugin that no longer exists, and `_config.yml`, `purgecss.config.js`, `Gemfile.lock`, and the workflows all reference things being removed.

Therefore the teardown proceeds in **staged, individually verified commits**, not one big-bang change:

1. Delete demo content (collections, demo posts, dead pages).
2. Delete al-folio layouts and includes; add the 4 new layouts.
3. Delete al-folio SCSS and vendored Bootstrap/MDB; add the 4 new SCSS files.
4. Cull plugins in `Gemfile` and `_config.yml`; regenerate `Gemfile.lock`.
5. Simplify the workflows; delete `purgecss.config.js`.

**`docker compose up --build` must be green after every stage.**

### 9.1 Verification (per stage)

- All four routes render.
- Both themes render correctly; no flash of wrong theme on load.
- Mobile viewport (375px) has no horizontal overflow.
- No console errors.
- `npx prettier . --write` is clean.

Final gate: `axe.yml` passes, and the deployed `gh-pages` build serves correctly.

## 10. Follow-up (out of scope)

1. **Pixel/bitmap display font** — swap `--font-display` and `--font-body`. This is the next task and the reason the token layer exists.
2. Write real blog posts and populate `_data/projects.yml`.
3. Re-add BibTeX export / abstract toggles if the publication list grows to justify it.
