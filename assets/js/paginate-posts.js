(function () {
  var posts = document.querySelectorAll(".blog-posts li");
  var posts_per_page = 10;
  if (posts.length < posts_per_page) {
    return;
  }
  var pages = Math.ceil(posts.length / posts_per_page);
  var current_page = 1;
  var page_links = document.querySelector(".page-links");
  var page_link_list = [];

  var prev_page_link = document.createElement("a");
  prev_page_link.setAttribute("href", "#");
  prev_page_link.setAttribute("class", "page-link");
  prev_page_link.setAttribute("data-page", "prev");
  prev_page_link.innerHTML = "Prev";
  page_link_list.push(prev_page_link);

  for (var i = 1; i <= pages; i++) {
    var page_link = document.createElement("a");
    page_link.setAttribute("href", "#");
    page_link.setAttribute("data-page", i);
    page_link.setAttribute("class", "page-link");
    page_link.innerHTML = i;
    page_link_list.push(page_link);
  }

  var next_page_link = document.createElement("a");
  next_page_link.setAttribute("href", "#");
  next_page_link.setAttribute("class", "page-link");
  next_page_link.setAttribute("data-page", "next");
  next_page_link.innerHTML = "Next";
  page_link_list.push(next_page_link);

  page_link_list.forEach(function (page_link) {
    page_links.appendChild(page_link);
  });

  for (var i = 0; i < page_link_list.length; i++) {
    page_link_list[i].addEventListener("click", function (e) {
      e.preventDefault();
      var page = this.getAttribute("data-page");
      console.log(current_page, page);
      if (page == "prev") {
        if (current_page <= 1) {
          return;
        }
        current_page--;
      } else if (page == "next") {
        if (current_page >= pages) {
          return;
        }
        current_page++;
      } else {
        current_page = parseInt(page);
      }
      show_page(current_page);
    });
  }

  function show_page(page) {
    for (var i = 1; i <= pages; i++) {
      if (i == page) {
        page_link_list[i].classList.add("active");
      } else {
        page_link_list[i].classList.remove("active");
      }
    }

    for (var i = 0; i < posts.length; i++) {
      posts[i].style.display = "none";
    }

    var start = (page - 1) * posts_per_page;
    var end = Math.min(start + posts_per_page, posts.length);

    for (var i = start; i < end; i++) {
      posts[i].style.display = "flex";
    }
  }

  show_page(current_page);
})();
