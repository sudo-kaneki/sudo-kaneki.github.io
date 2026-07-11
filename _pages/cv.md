---
layout: listing
permalink: /cv/
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
