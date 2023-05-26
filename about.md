---
layout: post
title: About
no_comments: true
---

- Biography
  - I'm a software engineer working on robotics, currently based out of NYC
  - [Resume]({{ site.url }}/resume)
  - [Calendar]({{ site.url }}/calendar)
  - I have some more interactive posts [here](https://lightning.bolte.cc/#/)
- Productivity
  - [uBlock]({{ site.url }}/assets/scripts/ublock_list.txt) - My uBlock Origin filter list
  - [Daylio](https://daylio.net/) - My preferred journaling app
  - [Tailscale](https://tailscale.com/) - Easy WireGuard mesh network, lets you easily SSH into different computers from anywhere
  - MacOS Ergonomics
    - [Mac Mouse Fix](https://mousefix.org/) - Lets you use third-party mouse buttons to control stuff (e.g., moving between screens with side buttons and showing Mission Control with the center button). FOSS, source code is on [Github](https://github.com/noah-nuebling/mac-mouse-fix)
    - [Rectangle](https://rectangleapp.com/) - Lets you snap to parts of your screen. Also FOSS, source code is on [Github](https://github.com/rxhanson/Rectangle) with a pro version for $10
  - [Dotfiles](https://github.com/codekansas/dotfiles) - Repository for all of my dotfiles (various Bash commands and aliases, Vim configuration, etc.)
- Learning ML / Self-Driving / Robotics
  - [Geoff Hinton Coursera Course (2012)](https://www.cs.toronto.edu/~hinton/coursera_lectures.html) - Doesn't get as much attention as Andrew Ng's course but in my opinion is superior in explaining concepts behind neural networks, especially for people motivated by the biological analogs.
  - [Coursera Self-Driving Specialization](https://www.coursera.org/specializations/self-driving-cars) - I did half of this before I started working at Tesla, and I think if more Tesla engineers took it, the car would probably drive better
  - [ML Template](https://github.com/codekansas/ml-template) - My basic template for starting ML projects, similar to [PyTorch Lightning](https://www.pytorchlightning.ai/) or [Fairseq](https://github.com/facebookresearch/fairseq)
    - Uses [Omegaconf](https://omegaconf.readthedocs.io/en/2.3_branch/) for config management
    - Also check out [this project](https://github.com/ashleve/lightning-hydra-template) which uses Hydra for config management
    - Mine only supports launching jobs locally and on Slurm clusters at the moment, since that's mainly how I train models
- Hobbies
  - [Goodreads](https://www.goodreads.com/review/list/56667319-benjamin?shelf=favorites) - My favorite books
  - [Lichess](https://lichess.org/@/bkbolte18) - My lichess profile
  - [Codeforces](https://codeforces.com/profile/codekansas) - My Codeforces profile (warning, very inactive)
  - [PCPartPicker](https://pcpartpicker.com/user/codekansas/) - My PCPartPicker profile
- Media
  - [Cool Youtube Videos](https://www.youtube.com/playlist?list=PLGukhZ1bCGDiwUPP0ze59FOGjZr21Aicp) - A YouTube playlist of random videos I thought were cool
  - [Lofi Work Music](https://www.youtube.com/watch?v=jfKfPfyJRdk)
  - [Other Lofi Work Music](https://open.spotify.com/artist/7sKOw5KIGmCldJ8wkQhGQo?si=O0LluUKzSX6gmwzeZVQAzg)
- Tools
  - 3D Printer
    - Don't use Creality, even though it is cheap you will never get back the hours of your life wasted priming it before every print
    - I've had good experiences with Prusa i3 (both RepRap and MK3S+), Octoprint is a must
  - Power tools
    - Mitre saw, drill, sawzall, and jigsaw are from Dewalt
    - Drill press and bench grinder are from Wen
  - Keyboard: At one point I was trying to force myself to use an ergonomic keyboard, but after a few months of using it I had a suspicion that it wasn't making me type any faster and was actually making me have more errors. So now I mainly just use an Apple keyboard. I have some numbers about my typing speeds for some different keyboards near the bottom of this page.
  - Editor: For scripting I use Vim. My Vim setup is in my dotfiles repository [here](https://github.com/codekansas/dotfiles). But for most projects I currently use VSCode.
- Misc
  - [Directory]({{ site.url }}/directory) - A list of all the posts on this blog
  - [RSS Feed]({{ site.url }}/feed.xml) - RSS feed for this blog

<details>
<summary style="margin: auto; text-align: center;">VSCode settings</summary>
<div style="margin-top: 0.5em;">
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
</div>
</details>

<details>
<summary style="margin: auto; text-align: center;">Keyboard speed test results</summary>
<div>
<ul>
  <li><a>Apple</a>: Apple Magic Keyboard (Wired)</li>
  <li><a>Mechanical</a>: Das Keyboard Model S</li>
  <li><a>Ergonomic</a>: Perixx Periduo-406</li>
</ul>
<table style="margin-top: 1em;">
  <thead>
    <tr>
      <th>Keyboard</th>
      <th>Trial</th>
      <th>Words per Minute</th>
      <th>Errors per Minute</th>
      <th>Adjusted Words per Minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>1</td>
      <td>97</td>
      <td>2</td>
      <td>95</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>2</td>
      <td>99</td>
      <td>1</td>
      <td>98</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>3</td>
      <td>88</td>
      <td>5</td>
      <td>83</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>4</td>
      <td>103</td>
      <td>3</td>
      <td>100</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>5</td>
      <td>92</td>
      <td>2</td>
      <td>90</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>Aggregate</td>
      <td>95.8 +/- 2.6</td>
      <td>2.6 +/- 0.7</td>
      <td>93.2 +/- 3.1</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>1</td>
      <td>87</td>
      <td>4</td>
      <td>83</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>2</td>
      <td>90</td>
      <td>5</td>
      <td>85</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>3</td>
      <td>87</td>
      <td>3</td>
      <td>84</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>4</td>
      <td>89</td>
      <td>1</td>
      <td>88</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>5</td>
      <td>90</td>
      <td>3</td>
      <td>87</td>
    </tr>
    <tr>
      <td>Mechanical</td>
      <td>Aggregate</td>
      <td>88.6 +/- 0.7</td>
      <td>3.2 +/- 0.7</td>
      <td>85.4 +/- 0.9</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>1</td>
      <td>95</td>
      <td>1</td>
      <td>94</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>2</td>
      <td>80</td>
      <td>1</td>
      <td>79</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>3</td>
      <td>92</td>
      <td>3</td>
      <td>89</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>4</td>
      <td>85</td>
      <td>3</td>
      <td>82</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>5</td>
      <td>101</td>
      <td>0</td>
      <td>101</td>
    </tr>
    <tr>
      <td>Ergonomic</td>
      <td>Aggregate</td>
      <td>90.6 +/- 3.7</td>
      <td>1.6 +/- 0.6</td>
      <td>89.0 +/- 4.0</td>
    </tr>
  </tbody>
</table>
</div>
</details>

Here are some quotes that I have found interesting.

> Why do people have to be this lonely? What's the point of it all? Millions of people in this world, all of them yearning, looking to others to satisfy them, yet isolating themselves. Why? Was the earth put here just to nourish human loneliness? - Haruki Murakami, Sputnik Sweetheart

> Training is nothing, will is everything - the will to act. - Batman Begins

> If you only read the books that everyone else is reading, you can only think what everyone else is thinking. - Haruki Murakami, and also a tote bag at the Strand

> Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds, because you know that the testing of your faith produces perseverance. Let perseverance finish its work so that you may be mature and complete, not lacking anything. - James 1:2-4

> I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. I will face my fear. I will permit it to pass over me and through me. And when it has gone past I will turn the inner eye to see its path. Where the fear has gone there will be nothing. Only I will remain. - Frank Herbert, Dune
