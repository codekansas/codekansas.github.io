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
  - [Dotfiles](https://github.com/codekansas/dotfiles) - Repository for all of my dotfiles (various Bash commands and aliases)
  - [ML Template](https://github.com/codekansas/ml-template) - My basic template for starting ML projects, similar to [PyTorch Lightning](https://www.pytorchlightning.ai/) or [Fairseq](https://github.com/facebookresearch/fairseq)
    - Uses [Omegaconf](https://omegaconf.readthedocs.io/en/2.3_branch/) for config management
    - Also check out [this project](https://github.com/ashleve/lightning-hydra-template) which uses Hydra for config management
    - Only supports launching jobs locally and on Slurm clusters at the moment, since that's mainly how I train models
- Learning ML / Self-Driving / Robotics
  - [Geoff Hinton Coursera Course (2012)](https://www.cs.toronto.edu/~hinton/coursera_lectures.html) - Doesn't get as much attention as Andrew Ng's course but in my opinion is superior in explaining concepts behind neural networks, especially for people motivated by the biological analogs.
  - [Coursera Self-Driving Specialization](https://www.coursera.org/specializations/self-driving-cars) - I did half of this before I started working at Tesla, and I think if more Tesla engineers took it, the car would probably drive better
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

  - At one point I was trying to force myself to use an ergonomic keyboard, but after a few months of using it I had a suspicion that it wasn't making me type any faster and was actually making me have more errors. I did some typing speed tests with a couple different keyboards to confirm this, ultimately finding that my random wired Mac keyboard was the best (see below). For what it's worth, this basically jives with my experience watching other people type on ergonomic keyboards as well.

| Keyboard   | Trial         | Words per Minute | Errors per Minute | Adjusted Words per Minute |
| ---------- | ------------- | ---------------- | ----------------- | ------------------------- |
| Apple      | 1             | 97               | 2                 | 95                        |
| Apple      | 2             | 99               | 1                 | 98                        |
| Apple      | 3             | 88               | 5                 | 83                        |
| Apple      | 4             | 103              | 3                 | 100                       |
| Apple      | 5             | 92               | 2                 | 90                        |
| Apple      | Aggregate     | 95.8 +/- 2.6     | 2.6 +/- 0.7       | 93.2 +/- 3.1              |
| Mechanical | 1             | 87               | 4                 | 83                        |
| Mechanical | 2             | 90               | 5                 | 85                        |
| Mechanical | 3             | 87               | 3                 | 84                        |
| Mechanical | 4             | 89               | 1                 | 88                        |
| Mechanical | 5             | 90               | 3                 | 87                        |
| Mechanical | Aggregate     | 88.6 +/- 0.7     | 3.2 +/- 0.7       | 85.4 +/- 0.9              |
| Ergonomic  | 1             | 95               | 1                 | 94                        |
| Ergonomic  | 2             | 80               | 1                 | 79                        |
| Ergonomic  | 3             | 92               | 3                 | 89                        |
| Ergonomic  | 4             | 85               | 3                 | 82                        |
| Ergonomic  | 5             | 101              | 0                 | 101                       |
| Ergonomic  | Aggregate     | 90.6 +/- 3.7     | 1.6 +/- 0.6       | 89.0 +/- 4.0              |

- Misc
  - [RSS Feed]({{ site.url }}/feed.xml) - RSS feed for this blog

Here are some quotes that I have found interesting.

> Why do people have to be this lonely? What's the point of it all? Millions of people in this world, all of them yearning, looking to others to satisfy them, yet isolating themselves. Why? Was the earth put here just to nourish human loneliness? - Haruki Murakami, Sputnik Sweetheart

> If you only read the books that everyone else is reading, you can only think what everyone else is thinking. - Haruki Murakami, and also a tote bag at the Strand

> Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds, because you know that the testing of your faith produces perseverance. Let perseverance finish its work so that you may be mature and complete, not lacking anything. - James 1:2-4

> I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. I will face my fear. I will permit it to pass over me and through me. And when it has gone past I will turn the inner eye to see its path. Where the fear has gone there will be nothing. Only I will remain. - Frank Herbert, Dune
