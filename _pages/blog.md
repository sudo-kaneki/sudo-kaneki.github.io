---
layout: listing
permalink: /blog/
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
