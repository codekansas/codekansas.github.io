(function() {
  let container = document.querySelector("#blog-contents");
  let hs = container.querySelectorAll("h2, h3, h4, h5");

  if (hs.length <= 1) {
    return;
  }

  let toc = document.createElement("details");
  toc.classList.add("table-of-contents");

  let summary = document.createElement("summary");
  summary.innerHTML = "Table of Contents";
  toc.appendChild(summary);

  let tocInner = document.createElement("div");

  hs.forEach((element) => {
    let newElem = document.createElement("div")
    let newLink = document.createElement("a");

    newLink.innerHTML = element.innerHTML;
    newLink.href = `#${element.id}`;

    let dep = 0;
    if (element.localName === "h3") dep = 1;
    if (element.localName === "h4") dep = 2;
    if (element.localName === "h5") dep = 3;
    newElem.style.marginLeft = `${dep}em`;

    newElem.appendChild(newLink);
    tocInner.appendChild(newElem);
  });

  toc.appendChild(tocInner);

  container.insertBefore(toc, container.firstChild);
})();
