(function () {
  const copyButtonLabel = "copy";

  let blocks = document.querySelectorAll("pre");

  blocks.forEach((block) => {
    if (navigator.clipboard) {
      let button = document.createElement("button");

      button.classList.add("copy-button");
      button.innerText = copyButtonLabel;
      block.parentElement.appendChild(button);

      button.addEventListener("click", async () => {
        await copyCode(block, button);
      });
    }
  });

  async function copyCode(block, button) {
    let code = block.querySelector("code");
    let text = code.innerText;
    await navigator.clipboard.writeText(text);

    // Changes inner text and code block color.
    button.innerText = "copied!";

    setTimeout(() => {
      button.innerText = copyButtonLabel;
    }, 1000);
  }
})();
