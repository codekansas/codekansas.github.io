(function () {
  let blocks = document.querySelectorAll("pre");

  blocks.forEach((block) => {
    if (block.offsetHeight < 400) return;

    let button = document.createElement("button");

    button.classList.add("collapse-button");
    button.innerText = "expand";
    block.parentElement.appendChild(button);

    block.classList.add("collapsed");

    button.addEventListener("click", async () => {
      await collapseCode(block, button);
    });
  });

  async function collapseCode(block, button) {
    if (block.classList.contains("collapsed")) {
      block.classList.remove("collapsed");
      button.innerText = "collapse";
    } else {
      block.classList.add("collapsed");
      button.innerText = "expand";
    }
  }
})();
