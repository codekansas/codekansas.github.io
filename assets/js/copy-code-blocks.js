(function () {
  const copyLinkLabel = "copy";

  let blocks = document.querySelectorAll("pre");

  blocks.forEach((block) => {
    if (navigator.clipboard) {
      let copy_link = document.createElement("a");
      copy_link.classList.add("code-link")
      copy_link.innerText = copyLinkLabel;

      let copy_div = document.createElement("div");
      copy_div.classList.add("code-div");
      copy_div.appendChild(copy_link);
      block.prepend(copy_div);

      copy_link.addEventListener("click", async () => {
        await copyCode(block, copy_link);
      });
    }
  });

  async function copyCode(block, copy_link) {
    let code = block.querySelector("code");
    let text = code.innerText;
    await navigator.clipboard.writeText(text);

    // Changes inner text and code block color.
    copy_link.innerText = "copied!";

    setTimeout(() => {
      copy_link.innerText = copyLinkLabel;
    }, 1000);
  }
})();
