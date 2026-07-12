---
layout: listing
permalink: /cv/
title: cv
---

<p class="cv-download">
  <a class="btn btn--accent" href="{{ site.data.socials.cv_pdf | relative_url }}">download pdf ↓</a>
</p>

{% comment %}
Sections are iterated straight from \_data/cv.yml rather than listed here.
Add a section to cv.yml and it appears on this page AND in the RenderCV PDF
with no template change — which is the whole point of cv.yml being the single
source of truth. Hardcoding the list meant a new section silently vanished
from the page while still appearing in the PDF.

Liquid preserves the YAML key order, so the page order is the cv.yml order.
The heading is the section key with underscores turned into spaces.

Do NOT indent the HTML below. This is a Markdown file, and kramdown treats
any line indented four spaces as a code block — it would render the literal
text "<h2>education</h2>" inside a <pre> instead of a heading.
{% endcomment %}

{% for section in site.data.cv.cv.sections %}
{% assign key = section[0] %}
{% assign entries = section[1] %}
{% if entries and entries.size > 0 %}

<h2>{{ key | replace: "_", " " }}</h2>

{% include cv-section.liquid entries=entries kind=key %}
{% endif %}
{% endfor %}
