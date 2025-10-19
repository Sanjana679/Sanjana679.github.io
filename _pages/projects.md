---
layout: page
title: projects
permalink: /projects/
description: A growing collection of my cool projects.
nav: true
nav_order: 2
display_categories: [fun, school]
horizontal: true
---

<!-- pages/projects.md -->
<div class="projects">

{% if site.enable_project_categories and page.display_categories %}

  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>

  {% assign categorized_projects = site.projects | where: "category", category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}

  {% if page.horizontal %}
    <div class="container">
      {% for project in sorted_projects %}
        <div class="card mb-4 hoverable">
          <div class="row g-0 align-items-center">
            {% if project.thumbnail %}
            <div class="col-md-4">
              <img src="{{ project.thumbnail | relative_url }}" class="img-fluid rounded-start" alt="{{ project.title }}">
            </div>
            {% endif %}
            <div class="col-md-8">
              <div class="card-body">
                <h3 class="card-title">{{ project.title }}</h3>
                <p class="card-text">{{ project.description }}</p>
                <p class="card-meta">
                  {% if project.date %}
                    <i class="fa-solid fa-calendar fa-sm"></i> {{ project.date | date: "%B %Y" }}
                    &nbsp; &middot; &nbsp;
                  {% endif %}
                  {% if project.category %}
                    <i class="fa-solid fa-tag fa-sm"></i> {{ project.category }}
                  {% endif %}
                </p>
                {% if project.url %}
                  <a href="{{ project.url | relative_url }}" class="btn btn-sm btn-primary mt-2">View Project</a>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for project in sorted_projects %}
        {% include projects.liquid %}
      {% endfor %}
    </div>
  {% endif %}
  {% endfor %}

{% else %}

  {% assign sorted_projects = site.projects | sort: "importance" %}

  {% if page.horizontal %}
    <div class="container">
      {% for project in sorted_projects %}
        <div class="card mb-4 hoverable">
          <div class="row g-0 align-items-center">
            {% if project.thumbnail %}
            <div class="col-md-4">
              <img src="{{ project.thumbnail | relative_url }}" class="img-fluid rounded-start" alt="{{ project.title }}">
            </div>
            {% endif %}
            <div class="col-md-8">
              <div class="card-body">
                <h3 class="card-title">{{ project.title }}</h3>
                <p class="card-text">{{ project.description }}</p>
                <p class="card-meta">
                  {% if project.date %}
                    <i class="fa-solid fa-calendar fa-sm"></i> {{ project.date | date: "%B %Y" }}
                    &nbsp; &middot; &nbsp;
                  {% endif %}
                  {% if project.category %}
                    <i class="fa-solid fa-tag fa-sm"></i> {{ project.category }}
                  {% endif %}
                </p>
                {% if project.url %}
                  <a href="{{ project.url | relative_url }}" class="btn btn-sm btn-primary mt-2">View Project</a>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for project in sorted_projects %}
        {% include projects.liquid %}
      {% endfor %}
    </div>
  {% endif %}
{% endif %}
</div>