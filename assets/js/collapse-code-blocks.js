(function () {
  let blocks = document.querySelectorAll("pre");

  blocks.forEach((block) => {
    if (block.offsetHeight < 400) return;

    let expand_link = document.createElement("a");
    expand_link.classList.add("code-link");
    expand_link.innerText = "expand";
    block.firstChild.append(expand_link);

    block.classList.add("collapsed");

    expand_link.addEventListener("click", async () => {
      await collapseCode(block, expand_link);
    });
  });

  async function collapseCode(block, expand_link) {
    if (block.classList.contains("collapsed")) {
      block.classList.remove("collapsed");
      expand_link.innerText = "collapse";
    } else {
      block.classList.add("collapsed");
      expand_link.innerText = "expand";
    }
  }
})();
