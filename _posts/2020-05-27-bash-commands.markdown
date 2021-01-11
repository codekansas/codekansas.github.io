---
layout: post
title: "Useful bash / zsh Profile Additions"
category: 🖥️
excerpt: Some functions I found useful to add to my bash and zsh profile.
pinned: true
---

> Update 2020-01-11 - I switched a lot of this stuff over to using [DotBot][dotbot] after seeing [this repo][dotfiles-inspiration]. You can see my dotfile repo [here][dotfiles] - it should work in whatever setup you're using, although some stuff might be specific to my own setup.

These are a bunch of additions I like to add to my local profile which I've found make me more productive. As best I could I tried to make it so that the instructions can be followed pretty mindlessly (mostly because I sometimes have to copy-paste stuff into a new environment).

# Git Aliases

Some useful Git aliases:

## Give a nice-looking commit tree

This is similar to the `hg sl` command we use at Facebook.

{% highlight bash %}
git config --global alias.sl 'log --graph --decorate --oneline'
{% endhighlight %}

## ls

> Usage: more friendly `ls`

I guess this is pretty standard but the `ls` command has some useful modifiers.

{% highlight bash %}
alias ll='ls -ahl'
{% endhighlight %}

## Date

> Usage: `today` gives the current date and `now` gives the current time

{% highlight bash %}
alias today='date +"%Y-%m-%d"'
alias now='date +"%T"'
{% endhighlight %}

Example:

{% highlight bash %}
$ echo $(today)
2020-05-27
$ echo $(now)
11:01:07
{% endhighlight %}

## Google Drive

> Usage: `gdrive <fid> <fpath>` where `<fid>` is the Google Drive file identifier and `<fpath>` is the output path

{% highlight bash %}
function gdrive {
  if [[ $# -ne 2 ]]; then
    echo "Usage: gdrive <fid> <fpath>"
    exit 1
  fi
  FILEID="$1"
  FILENAME="$2"
  O=$(wget \
    --quiet \
    --save-cookies /tmp/cookies.txt \
    --keep-session-cookies \
    --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget \
    --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=${O}&id=${FILEID}" \
    -O $FILENAME
  rm -rf /tmp/cookies.txt
}
{% endhighlight %}

## tmux

[tmux](https://github.com/tmux/tmux/wiki) is a must for remote development. Here are some of my personal add-ons to make it work a bit better.

### List Open Sessions

> Usage: List open sessions on a machine on startup

{% highlight bash %}
if [[ -n $TMUX ]]; then
  echo -e "\033[1;31m----- TMUX session: $(tmux display-message -p '#S') -----\033[0m"
elif [[ ! -n $TMUX ]] && [[ $(tmux ls 2> /dev/null) ]]; then
  echo -e "\033[1;31m----- Open tmux sessions -----\033[0m"
  tmux ls
  echo -e "\033[1;31m------------------------------\033[0m"
fi
{% endhighlight %}

### Attach to Named Session in Control Mode

> Usage: `tmuxc <session>`, can use tab completion to get the named session

Using named sessions is very important for organizing multiple projects. Otherwise I found it really easy to lose track of where stuff is.

{% highlight bash %}
alias tmuxc='tmux -CC a -t'

## Provides tab completion for the tmuxc command.
_tmuxc_complete()
{
  local cur opts
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  opts="$(tmux list-sessions -F '#S' | paste -sd ' ')"
  COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
  return 0
}
complete -F _tmuxc_complete tmuxc
{% endhighlight %}

## Anaconda

> Usage: Shorthand and tab completion for activating a Conda environment

{% highlight bash %}
alias cenv='conda activate'

_conda_complete()
{
  local cur opts
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  opts="$(ls -1 ${HOME}/.conda/envs/ | paste -sd ' ')"
  COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
  return 0
}
complete -F _conda_complete 'cenv'
{% endhighlight %}

## uBlock Origin

This is more of a productivity tip. I have a blocklist that blocks the endless scrolling parts of some social media sites without blocking potentially informative posts.

[Click here to subscribe](abp:subscribe?location=https://ben.bolte.cc/assets/scripts/ublock_list.txt&title=Social Media Posts)

[dotfiles]: https://github.com/codekansas/dotfiles
[dotfiles-inspiration]: https://github.com/mikejqzhang/dotfiles
[dotbot]: https://github.com/anishathalye/dotbot
