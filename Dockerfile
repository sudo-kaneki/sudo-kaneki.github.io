FROM ruby:3.3

LABEL description="Docker image for local Jekyll development"

ENV DEBIAN_FRONTEND=noninteractive

# nodejs provides the JS runtime ExecJS needs for uglifier (jekyll-minifier
# dependency) — without it, minification silently no-ops.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /srv/jekyll

COPY Gemfile Gemfile.lock ./
RUN bundle install

EXPOSE 8080 35729

CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--port", "8080", "--livereload", "--force_polling"]
