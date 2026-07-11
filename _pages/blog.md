---
layout: listing
permalink: /blog/
title: blog
---

{% comment %}
The filter only renders when there is more than one category to filter by.
With zero or one category the buttons are pure noise — an "all" button that
filters nothing.
{% endcomment %}
{% assign cats = site.categories | sort %}
{% if cats.size > 1 %}

  <div class="tabs" role="group" aria-label="Filter posts by category">
    <button class="tab" type="button" data-filter="all" aria-pressed="true">all</button>
    {% for cat in cats %}
      <button class="tab" type="button" data-filter="{{ cat[0] | downcase }}" aria-pressed="false">{{ cat[0] }}</button>
    {% endfor %}
  </div>
{% endif %}

{% if site.posts.size > 0 %}

  <ul class="post-list" id="post-list" aria-live="polite">
    {% for post in site.posts %}
      <li class="post-list__item" data-categories="{{ post.categories | join: ',' | downcase }}">
        <span class="post-list__date">{{ post.date | date: "%Y-%m-%d" }}</span>
        <span>
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {% for cat in post.categories %}<span class="tag">{{ cat }}</span>{% endfor %}
        </span>
      </li>
    {% endfor %}
  </ul>
  <p id="post-empty" hidden>No posts in that category yet.</p>
{% else %}
  <p>No posts yet.</p>
{% endif %}

<script defer src="{{ '/assets/js/blog-filter.js' | relative_url }}"></script>
