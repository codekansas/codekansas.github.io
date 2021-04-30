---
layout: post
title: Create Video from Stream of Numpy Arrays in Matplotlib
category: 💻
excerpt: Short post with code snippits for creating videos from Numpy arrays in Matplotlib.
---

While it's really easy to show an image in Matplotlib, I find that rendering videos quickly from PyTorch tensors or Numpy arrays seems to be a constant problem. I figured I'd write a short code snippet about how to do it quickly, for anyone else that is in the same situation.

{% highlight python %}
from typing import Iterator, Optional, Tuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def write_animation(
    itr: Iterator[np.array],
    out_file: Path,
    dpi: int = 50,
    fps: int = 30,
    title: str = "Animation",
    comment: Optional[str] = None,
    writer: str = "ffmpeg",
) -> None:
    """Function that writes an animation from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib animation writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    first_img = next(itr)
    height, width, _ = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Ensures that there's no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    Writer = mpl.animation.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = Writer(
        fps=fps,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(first_img, interpolation="nearest")
        mpl_writer.grab_frame()

        for img in itr:
            im.set_data(img)
            mpl_writer.grab_frame()
{% endhighlight %}

This makes it easy and memory-efficient to write a video from a coroutine, for example:

{% highlight python %}
def dummy_image_generator() -> Iterator[np.array]:
    for _ in range(100):
        yield np.random.rand(480, 640, 3)

write_animation(dummy_image_generator(), "test.mp4")
{% endhighlight %}

Hope this helps!

