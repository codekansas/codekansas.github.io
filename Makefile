# Makefile

all:
	bundle exec jekyll serve
.PHONY: all

no-dev:
	JEKYLL_ENV=production bundle exec jekyll serve
.PHONY: no-dev

to-webp:
	cd images && ./to_webp.sh
.PHONY: to-webp
