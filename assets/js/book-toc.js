(function () {
  var panel = document.querySelector("[data-book-toc]");
  if (!panel) return;

  // The contents panel is a <details>: collapsed above the article on narrow
  // screens, always open as a sidebar on wide ones. The breakpoint is read
  // from CSS (--bp-book on the panel) so it is defined in one place only.
  var bp = getComputedStyle(panel).getPropertyValue("--bp-book").trim();
  if (bp && window.matchMedia) {
    var wide = window.matchMedia("(min-width: " + bp + ")");
    var sync = function () {
      panel.open = wide.matches;
    };
    sync();
    if (wide.addEventListener) wide.addEventListener("change", sync);
  }

  // "On this page": list the current chapter's sections under its entry.
  var list = document.querySelector("[data-book-sections]");
  var headings = document.querySelectorAll(".book__article h2[id]");
  if (!list || !headings.length) return;

  headings.forEach(function (h) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    list.appendChild(li);
  });
  list.hidden = false;
})();
