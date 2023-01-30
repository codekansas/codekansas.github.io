---
layout: post
title: More
no_comments: true
excerpt: Some more links to miscellaneous things.
---

- Productivity
  - [uBlock]({{ site.url }}/assets/scripts/ublock_list.txt) - My uBlock Origin filter list
  - [Daylio](https://daylio.net/) - My preferred journaling app
  - [Tailscale](https://tailscale.com/) - Easy WireGuard mesh network, lets you easily SSH into different computers from anywhere
  - MacOS Ergonomics
    - [Mac Mouse Fix](https://mousefix.org/) - Lets you use third-party mouse buttons to control stuff (e.g., moving between screens with side buttons and showing Mission Control with the center button). FOSS, source code is on [Github](https://github.com/noah-nuebling/mac-mouse-fix)
    - [Rectangle](https://rectangleapp.com/) - Lets you snap to parts of your screen. Also FOSS, source code is on [Github](https://github.com/rxhanson/Rectangle) with a pro version for $10
- Hobbies
  - [Goodreads](https://www.goodreads.com/review/list/56667319-benjamin?shelf=favorites) - My favorite books
  - [Lichess](https://lichess.org/@/bkbolte18) - My lichess profile
  - [Codeforces](https://codeforces.com/profile/codekansas) - My Codeforces profile (warning, very inactive)
- Media
  - [Cool Youtube Videos](https://www.youtube.com/playlist?list=PLGukhZ1bCGDiwUPP0ze59FOGjZr21Aicp) - A YouTube playlist of random videos I thought were cool
  - [Lofi Work Music](https://www.youtube.com/watch?v=jfKfPfyJRdk)
  - [Other Lofi Work Music](https://open.spotify.com/artist/7sKOw5KIGmCldJ8wkQhGQo?si=O0LluUKzSX6gmwzeZVQAzg)
- Tools
  - 3D Printer
    - Don't use Creality, even though it is cheap you will never get back the hours of your life wasted priming it before every print
    - I've had good experiences with Prusa i3 (both RepRap and MK3S+), Octoprint is a must
  - A subset of my VSCode `settings.json` file

{% highlight jsonc %}
{

  /* Editor features */
  "editor.quickSuggestionsDelay": 10,
  "editor.rulers": [
    {
      "column": 80,
      "color": "#00ff2255"
    },
    {
      "column": 88,
      "color": "#00ff2222"
    },
    {
      "column": 120,
      "color": "#00e1ff55"
    }
  ],
  "editor.acceptSuggestionOnEnter": "smart",
  "editor.suggestSelection": "recentlyUsed",
  "editor.minimap.enabled": false,
  "editor.maxTokenizationLineLength": 512,
  "editor.cursorSurroundingLines": 15,
  
  /* Default extensions */
  "remote.SSH.defaultExtensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-python.vscode-pylance",
    "eamodio.gitlens",
    "xaver.clang-format",
    "visualstudioexptteam.vscodeintellicode",
    "tyriar.sort-lines",
    "sirtori.indenticator",
    "oderwat.indent-rainbow",
    "esbenp.prettier-vscode",
    "yzhang.markdown-all-in-one"
  ],
}
{% endhighlight %}

- Misc
  - [RSS Feed]({{ site.url }}/feed.xml) - RSS feed for this blog

Here are some quotes that I have found interesting.

> Why do people have to be this lonely? What's the point of it all? Millions of people in this world, all of them yearning, looking to others to satisfy them, yet isolating themselves. Why? Was the earth put here just to nourish human loneliness? - Haruki Murakami, Sputnik Sweetheart

> Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds, because you know that the testing of your faith produces perseverance. Let perseverance finish its work so that you may be mature and complete, not lacking anything. - James 1:2-4

> I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. I will face my fear. I will permit it to pass over me and through me. And when it has gone past I will turn the inner eye to see its path. Where the fear has gone there will be nothing. Only I will remain. - Frank Herbert, Dune
