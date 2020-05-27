# Script for quickly installing my .vimrc file.
# Run this with:
#   curl https://ben.bolte.cc/assets/scripts/setup_vimrc.sh | sh

set -e

echo "Writing .vimrc"
cat > "$HOME/.vimrc" <<- EOM
execute pathogen#infect()
colorscheme badwolf

" turns of syntax highlighting
syntax enable

" use spaces not tabs
set tabstop=8 softtabstop=0 expandtab shiftwidth=2 smarttab

" show line numbers
set relativenumber

" show command in bottom bar
set showcmd

" highlight current line
set cursorline

" load filetype-specific indent files
filetype indent on

" highlight matches
set showmatch

" search
set incsearch
set hlsearch

" turn of highlighting
nnoremap <leader><space> :nohlsearch<CR>
EOM

echo "Downloading pathogen"
mkdir -p $HOME/.vim/autoload ~/.vim/bundle
if [ ! -f "$HOME/.vim/autoload/pathogen.vim" ] ; then
    curl -LSso $HOME/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim
fi

echo "Downloading badwolf"
if [ ! -d "$HOME/.vim/bundle/badwolf" ] ; then
    git clone https://github.com/sjl/badwolf $HOME/.vim/bundle/badwolf
else
    cd $HOME/.vim/bundle/badwolf
    git pull
fi
