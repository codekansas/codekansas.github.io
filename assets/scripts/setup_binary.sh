# Script for quickly installing .binaries extension.
# Run this with:
#   curl https://ben.bolte.cc/assets/scripts/setup_binary.sh | sh

set -e

ROOT_DIR=$HOME/.binaries

mkdir -p $ROOT_DIR/bin $ROOT_DIR/etc $ROOT_DIR/scripts

echo "Writing $ROOT_DIR/bin/bdelete"
cat > "$ROOT_DIR/bin/bdelete" <<- EOM
#!/bin/bash

if [[ \$# -ne 1 ]]; then
    echo "Usage: bdelete <name_of_script_to_delete>"
    exit 1
fi

filename=\$1
shift

scriptspath="\${HOME}/.binaries/scripts"
filepath="\${scriptspath}/\${filename}"

if [ ! -f "\${filepath}" ]; then
    echo "[ \${filename} ] doesnt exist! Available:"
    find \$scriptspath -type f | cut -c\$((\${#scriptspath}+2))-
else
    rm \$filepath
fi
EOM

echo "Writing $ROOT_DIR/bin/brun"
cat > "$ROOT_DIR/bin/brun" <<- EOM
#!/bin/bash

if [[ \$# -ne 1 ]]; then
    echo "Usage: brun <script_to_run>"
    exit 1
fi

# Gets the name of the script to edit.
filename=\$1
shift

scriptspath="\${HOME}/.binaries/scripts"
filepath="\${scriptspath}/\${filename}"

if [ ! -f "\${filepath}" ]; then
    echo "[ \${filename} ] is not a runable script. Available:"
    find \$scriptspath -type f | cut -c\$((\${#scriptspath}+2))-
else
    \${filepath}
fi
EOM

echo "Writing $ROOT_DIR/bin/bedit"
cat > "$ROOT_DIR/bin/bedit" <<- EOM
#!/bin/bash

if [[ \$# -ne 1 ]]; then
    echo "Usage: bedit <script_to_edit>"
    exit 1
fi

# Gets the name of the script to edit.
filename=\$1
shift

scriptspath="\${HOME}/.binaries/scripts"
filepath="\${scriptspath}/\${filename}"

if [ ! -f "\${filepath}" ]; then
    echo "[ \${filename} ] is not an editable script. Available:"
    find \$scriptspath -type f | cut -c\$((\${#scriptspath}+2))-
else
    \$EDITOR "\${filepath}"
fi
EOM

echo "Writing $ROOT_DIR/bin/binit"
cat > "$ROOT_DIR/bin/binit" <<- EOM
#!/bin/bash

if [[ \$# -ne 1 ]]; then
    echo "Usage: binit <name_of_script_to_create>"
    exit 1
fi

filename=\$1
shift

scriptspath="\${HOME}/.binaries/scripts"
filepath="\${scriptspath}/\${filename}"
mkdir -p \$(dirname "\$filepath")

if [ -f "\${filepath}" ]; then
    echo "[ \${filename} ] already exists! Choose a different name, not one of:"
    find \$scriptspath -type f | cut -c\$((\${#scriptspath}+2))-
    exit 1
else
    echo "#!/bin/bash" > \${filepath}
    echo "" >> \${filepath}
    echo "" >> \${filepath}
    chmod +x "\${filepath}"
fi

\$EDITOR + "\${filepath}"
EOM

echo "Writing $ROOT_DIR/etc/bcomplete"
cat > "$ROOT_DIR/etc/bcomplete" <<- EOM
_binary_complete()
{
    # Path to the scripts directory.
    SCRIPTDIR="\${HOME}/.binaries/scripts/"

    local cur opts

    COMPREPLY=()
    cur="\${COMP_WORDS[COMP_CWORD]}"
    opts="\$(find \$SCRIPTDIR -type f | cut -c\$((\${#SCRIPTDIR}+1))- | paste -sd " " -)"

    COMPREPLY=( \$(compgen -W "\${opts}" -- \${cur}) )
    return 0
}
complete -F _binary_complete bedit
complete -F _binary_complete brun
complete -F _binary_complete bdelete
EOM

tree $ROOT_DIR

echo -e "Setup complete! Add this to your \033[1;31mprofile\033[0m (.bashrc, .zshrc, .profile) file:"
echo -e "\033[1;32mexport PATH=\$PATH:$ROOT_DIR/bin"
echo -e "source $ROOT_DIR/etc/bcomplete\033[0m"

