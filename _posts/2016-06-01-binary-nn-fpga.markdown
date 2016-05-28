---
layout: post
title: "BinaryConnect on an FPGA"
date: 2016-05-25 12:00:00
categories: ml
---

This post provides a tutorial on implementing the BinaryConnect algorithm on an FPGA. It is currently a work in progress, so I will be adding sections as I work on them (I find that explaining things in words helps me clarify my thought process some). Hopefully I'll finish this in about a month. The Github repository where I will be adding code is located [here](https://github.com/codekansas/binary-ml).

* TOC
{:toc}

<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
<script type="text/javascript">
// Turn all headers into links back to the table of contents
$(document).ready(function() {
    $("article").find("h1, h2, h3, h4, h5, h6").each(function(index) {
        var content = $(this).text();
        $(this).html("<a href=\"#markdown-toc\" style=\"color: black;\">" + content + "</a>");
    });
});
</script>

[bengio]: http://arxiv.org/pdf/1511.00363v3.pdf
[original code]: https://github.com/MatthieuCourbariaux/BinaryConnect