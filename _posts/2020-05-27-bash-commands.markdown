---
layout: post
title: "Useful bash / zsh Profile Additions"
category: 🖥️
excerpt: Some functions I found useful to add to my bash and zsh profile.
---

These are a bunch of additions I like to add to my local profile which I've found make me more productive. As best I could I tried to make it so that the instructions can be followed pretty mindlessly (mostly because I sometimes have to copy-paste stuff into a new environment).

## .binaries

> Usage: `binit <fname>` to create a new script, `bedit <fname>` to edit existing script, `brun <fname>` to run script, `bdelete <fname>` to delete script

`.binaries` is a small set of tools that I wrote for managing bash scripts. It is basically just a directory for bash scripts and provides a quick way to add, edit, delete or run them from the command line, with tab completion. I combined all of this into a single command that can be run using:

{% highlight bash %}
curl https://ben.bolte.cc/assets/scripts/setup_binary.sh | sh
{% endhighlight %}

Here are more details about this tool.

<details>
<summary>Directory Structure</summary>
{% highlight bash %}
~/.binaries/
├── bin
│   ├── bdelete
│   ├── bedit
│   ├── binit
│   └── brun
├── etc
│   └── bcomplete
└── scripts
    ├── one_off_command
    ├── project1
    │   ├── generate
    │   ├── score
    │   └── train
    └── project2
        ├── dist
        └── train
{% endhighlight %}
</details>

<details>
<summary>bin/bdelete</summary>

{% highlight bash %}
#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: bdelete <name_of_script_to_delete>"
    exit 1
fi

filename=$1
shift

scriptspath="${HOME}/.binaries/scripts"
filepath="${scriptspath}/${filename}"

if [ ! -f "${filepath}" ]; then
    echo "[ ${filename} ] doesnt exist! Available:"
    find $scriptspath -type f | cut -c$((${#scriptspath}+2))-
else
    rm $filepath
fi
{% endhighlight %}
</details>

<details>
<summary>bin/brun</summary>

{% highlight bash %}
#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: brun <script_to_run>"
    exit 1
fi

## Gets the name of the script to edit.
filename=$1
shift

scriptspath="${HOME}/.binaries/scripts"
filepath="${scriptspath}/${filename}"

if [ ! -f "${filepath}" ]; then
    echo "[ ${filename} ] is not a runable script. Available:"
    find $scriptspath -type f | cut -c$((${#scriptspath}+2))-
else
    ${filepath}
fi
{% endhighlight %}
</details>

<details>
<summary>bin/bedit</summary>

{% highlight bash %}
#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: bedit <script_to_edit>"
    exit 1
fi

# Gets the name of the script to edit.
filename=$1
shift

scriptspath="${HOME}/.binaries/scripts"
filepath="${scriptspath}/${filename}"

if [ ! -f "${filepath}" ]; then
    echo "[ ${filename} ] is not an editable script. Available:"
    find $scriptspath -type f | cut -c$((${#scriptspath}+2))-
else
    $EDITOR "${filepath}"
fi
{% endhighlight %}
</details>

<details>
<summary>bin/binit</summary>

{% highlight bash %}
#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: binit <name_of_script_to_create>"
    exit 1
fi

filename=$1
shift

scriptspath="${HOME}/.binaries/scripts"
filepath="${scriptspath}/${filename}"
mkdir -p $(dirname "$filepath")

if [ -f "${filepath}" ]; then
    echo "[ ${filename} ] already exists! Choose a different name, not one of:"
    find $scriptspath -type f | cut -c$((${#scriptspath}+2))-
    exit 1
else
    echo "#!/bin/bash" > ${filepath}
    echo "" >> ${filepath}
    echo "" >> ${filepath}
    chmod +x "${filepath}"
fi

$EDITOR + "${filepath}"
{% endhighlight %}
</details>

<details>
<summary>etc/bcomplete</summary>

{% highlight bash %}
_binary_complete()
{
    # Path to the scripts directory.
    SCRIPTDIR="${HOME}/.binaries/scripts/"

    local cur opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts="$(find $SCRIPTDIR -type f | cut -c$((${#SCRIPTDIR}+1))- | paste -sd " " -)"

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _binary_complete bedit
complete -F _binary_complete brun
complete -F _binary_complete bdelete
{% endhighlight %}
</details>

Make sure to add this to the profile:

{% highlight bash %}
export PATH=$PATH:${HOME}/.binaries/bin
source ${HOME}/.binaries/etc/bcomplete
{% endhighlight %}

# Git Aliases

Some useful Git aliases:

## Give a nice-looking commit tree

This is similar to the `hg sl` command we use at Facebook.

```
git config --global alias.sl 'log --graph --decorate --oneline'
```

## .vimrc

My preferred Vim setup uses the [badwolf](https://vimawesome.com/plugin/badwolf) colorscheme for [pathogen](https://github.com/tpope/vim-pathogen). I combined this all into a command that can be run using:


{% highlight bash %}
curl https://ben.bolte.cc/assets/scripts/setup_vimrc.sh | sh
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
if [[ $(tmux ls 2> /dev/null) ]]; then
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
