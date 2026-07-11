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
